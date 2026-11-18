"""Amendment 4 (research_r2_v4/amendment4.py): append-only authorisation of J4b seeds 1-2 and the three-seed
confirmation vs J2. Real-file tests (amendment 3 unchanged and still refusing seeds 1-2, historical hashes, J4b = J2 +
photometric only, production refusals) and synthetic CPU tests (three seeds across two child bases, fail-closed inputs,
H_bal-only decision on all six folds)."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

from research_r2.data import sha256_file  # noqa: E402
from research_r2_v4 import amendment4 as a4  # noqa: E402
from research_r2_v4 import child_gate as cg  # noqa: E402
from research_r2_v4 import child_inputs as ci  # noqa: E402
from research_r2_v4 import child_run as cr  # noqa: E402
from research_r2_v4 import jrun  # noqa: E402
from research_r2_v4.common import INNER, V4, env  # noqa: E402

PARENT, CHILD = "Jsyn_P", "Jsyn_C"
ERR = (jrun.JCheckError, cg.GateError)
REVIEWED = "d9ed896"  # reviewed checkpoint (J4b seed-0 screen)
_S = {}


def pcfg():
    return jrun.JConfig(PARENT, backbone="synthetic", trainable="synthetic", epochs=1, batch=256, lr_backbone=1e-2, workers=0)


def ccfg():
    c = pcfg(); c.row = CHILD; c.source_balanced = True
    return c


def setUpModule():
    frame, bench, F = env()
    tmp = Path(tempfile.mkdtemp(prefix="r2v4_a4_"))
    acc = jrun.ImageAccess(frame, None, loader=jrun.synthetic_loader(frame))
    for s in (0, 1, 2):
        st = jrun.run_row(pcfg(), s, [F[f] for f in INNER], frame, bench, jrun.SyntheticBackend(), acc, tmp / "parent", device="cpu", log=lambda *_: None)
        assert all(v == "ok" for v in st.values()), st
    for s, base in ((0, tmp / "childA"), (1, tmp / "childB"), (2, tmp / "childB")):
        st = cr._run(ccfg(), s, base, frame, bench, F, jrun.SyntheticBackend(), acc, parent_row=PARENT, device="cpu",
                     worktree_check=lambda: True, log=lambda *_: None)
        assert st == {fn: "ok" for fn in INNER}, st
    _S.update(frame=frame, F=F, tmp=tmp)


def tearDownModule():
    shutil.rmtree(_S["tmp"], ignore_errors=True)


def bases(tag):
    """Fresh copies of the two synthetic child bases for one negative case."""
    d = Path(tempfile.mkdtemp(prefix=f"r2v4_a4_{tag}_", dir=_S["tmp"]))
    shutil.copytree(_S["tmp"] / "childA", d / "A"); shutil.copytree(_S["tmp"] / "childB", d / "B")
    return d / "A", d / "B"


def pair(A, B, child_bases=None, child_cfg=None):
    cb = child_bases if child_bases is not None else {0: A, 1: B, 2: B}
    par = _S["tmp"] / "parent"
    return a4._confirmation_pair(cb, {0: par, 1: par, 2: par}, CHILD, PARENT, child_cfg or ccfg(), pcfg(), _S["frame"], _S["F"])


def adir(base, seed, fn):
    return base / CHILD / f"seed{seed}" / fn / "attempt1"


def edit_json(p, fn):
    x = json.loads(Path(p).read_text()); fn(x); Path(p).write_text(json.dumps(x))


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)


# ============================================================================= real files
class TestAmendment3Preserved(unittest.TestCase):
    def test_amendment_3_unchanged_and_still_verifies(self):
        for f, h in a4.A3_SHA256.items():
            self.assertEqual(sha256_file(ROOT / f), h, f)
        self.assertEqual(a4.a3_problems(), [])  # pinned tool: amendment-3 files and its 24 code pins
        pinned = sorted(json.loads((ROOT / "research_r2_v4/amendment_3.json").read_text())["code_pins"])
        self.assertEqual(len(pinned), 24)
        r = git("diff", "--quiet", REVIEWED, "--", *pinned, *a4.A3_SHA256)
        self.assertEqual(r.returncode, 0, "an amendment-3 pinned file differs from the reviewed commit")

    def test_amendment_3_runner_still_refuses_seeds_1_and_2(self):
        for s in (1, 2, 3):
            with self.assertRaisesRegex(cr.ChildRunError, f"seed {s} is not authorised by amendment 3"):
                cr.check_declaration(a4.CHILD_ROW, s)
        cr.check_declaration(a4.CHILD_ROW, 0)

    def test_historical_artifact_hashes_unchanged(self):
        arts = ROOT / "artifacts/r2/v4"
        # Files published as sanitised derivatives (local paths replaced) are recorded with both hashes in the
        # publication map: the public copy must match its recorded public hash, and the recorded private original
        # must be the hash in the stage manifest. Every other file must match the manifest exactly.
        sanitised = dict(json.loads((arts / "PUBLICATION_SANITIZATION_MAP.json").read_text())["files"])
        n = 0
        for stage in ("stage_j_phase1_20261114", "stage_j_phase2_20261115", "stage_hrt_20261116", "stage_hrt_20261116_post_review",
                      "stage_hrt_20261116_post_review_v3", "stage_j4b_20261117"):
            for line in (arts / stage / "SHA256SUMS").read_text().splitlines():
                h, _, rel = line.partition("  ")
                path = arts / stage / rel
                entry = sanitised.get(path.resolve().relative_to(ROOT.resolve()).as_posix())
                if entry is None:
                    self.assertEqual(sha256_file(path), h, f"{stage}/{rel}")
                else:
                    self.assertEqual(entry["private_original_sha256"], h, f"{stage}/{rel} (map vs manifest)")
                    self.assertEqual(sha256_file(path), entry["public_sanitized_sha256"], f"{stage}/{rel} (public copy)")
                n += 1
        self.assertGreater(n, 100)
        if not a4.CHILD_BASES[0].exists():
            self.skipTest("private J4b run files are only on the project machine; public artifacts verified")
        for line in (arts / "stage_j4b_20261117/private_files_SHA256SUMS.txt").read_text().splitlines():
            h, _, rel = line.partition("  ")
            self.assertEqual(sha256_file(a4.CHILD_BASES[0] / rel), h, rel)

    def test_j4b_is_j2_plus_photometric_only(self):
        c, p = asdict(ci.declared_child_config(a4.CHILD_ROW)), asdict(jrun.j_rows()[ci.PARENT_ROW])
        self.assertEqual(ci.modification(c, p), ("augment", "photometric"))
        self.assertEqual({k: v for k, v in c.items() if k not in ("row", "augment")}, {k: v for k, v in p.items() if k not in ("row", "augment")})
        self.assertEqual((c["epochs"], c["batch"], c["lr_backbone"], c["lr_head"], c["weight_decay"], c["trainable"]),
                         (5, 32, 1e-5, 1e-3, 0.05, p["trainable"]))


class TestProductionRefusals(unittest.TestCase):
    def test_unauthorised_rows_seeds_and_directories(self):
        with self.assertRaisesRegex(a4.A4Error, "not the amendment-4 child row"):
            a4.run_child("J5_best_source_balanced", 1, None)
        for s in (0, 3, -1):
            with self.assertRaises(a4.A4Error):
                a4.run_child(a4.CHILD_ROW, s, None)
        for p in (a4.SEED0_STAGE, a4.CHILD_BASES[0], a4.CHILD_BASES[0] / a4.CHILD_ROW / "seed1", ci.PARENT_BASES[1], V4 / "stage_hrt_20261116",
                  ROOT / "artifacts" / "r2", V4):
            with self.assertRaisesRegex(a4.A4Error, "overlaps protected evidence"):
                a4.check_out_dir(p)
        a4.check_out_dir(a4.CHILD_BASES[1])

    def test_real_declaration_state(self):
        declared = (ROOT / a4.DECLARATION_REL).exists()
        if not declared:  # before the declaration commit: nothing is authorised
            for s in (1, 2):
                with self.assertRaisesRegex(a4.A4Error, "amendment 4 is not declared"):
                    a4.check_authorised(a4.CHILD_ROW, s, "run")
            with self.assertRaisesRegex(a4.A4Error, "amendment 4 is not declared"):
                a4.check_authorised(a4.CHILD_ROW, None, "evaluate")
            return
        for s in (1, 2):
            a4.check_authorised(a4.CHILD_ROW, s, "run")
        a4.check_authorised(a4.CHILD_ROW, None, "evaluate")
        for s in (0, 3):
            with self.assertRaisesRegex(a4.A4Error, f"seed {s} is not authorised by amendment 4"):
                a4.check_authorised(a4.CHILD_ROW, s, "run")
        with self.assertRaisesRegex(a4.A4Error, "not authorised by amendment 4"):
            a4.check_authorised("J5_best_source_balanced", 1, "run")
        frame, _, F = env()
        a4.check_pinned(frame, F)  # fold identities, configurations, J2 and J4b seed-0 evidence equal the pins


class TestAuthorisationRecord(unittest.TestCase):
    """amendment-4 declaration logic on a temporary root (copies of the real amendment-3 files + one pinned file)."""

    def root(self, seeds=(1, 2), evaluation=None):
        r = Path(tempfile.mkdtemp(prefix="r2v4_a4_root_", dir=_S["tmp"]))
        (r / "research_r2_v4").mkdir()
        for f in ("research_r2_v4/amendment_3.json", "research_r2_v4/DECLARATION_AMENDMENT_3.json", "research_r2_v4/DECLARATION.json"):
            shutil.copy(ROOT / f, r / f)
        (r / "pinned.py").write_text("x = 1\n")
        am = {"amendment": 4, "extends": {"amendment_3_sha256": a4.A3_SHA256["research_r2_v4/amendment_3.json"]},
              "authorised_now": {"rows": [a4.CHILD_ROW], "seeds": list(seeds),
                                 "evaluation": evaluation or {"mode": "confirm", "seeds": [0, 1, 2], "folds": list(INNER)}},
              "code_pins": {"pinned.py": sha256_file(r / "pinned.py")}}
        (r / a4.AMENDMENT_REL).write_text(json.dumps(am))
        a4.declare(r)
        return r

    def check(self, r, row, seed, purpose="run", a3=lambda: []):
        return a4.check_authorised(row, seed, purpose, root=r, a3_check=a3)

    def test_seeds_1_2_only_and_the_declared_evaluation(self):
        r = self.root()
        for s in (1, 2):
            self.check(r, a4.CHILD_ROW, s)
        self.check(r, a4.CHILD_ROW, None, "evaluate")
        for s in (0, 3, "1", None):
            with self.assertRaisesRegex(a4.A4Error, "is not authorised by amendment 4"):
                self.check(r, a4.CHILD_ROW, s)
        for row in ("J5_best_source_balanced", "J4a_best_geometric_aug", ci.PARENT_ROW):
            with self.assertRaisesRegex(a4.A4Error, "not authorised by amendment 4"):
                self.check(r, row, 1)
        with self.assertRaisesRegex(a4.A4Error, "unknown purpose"):
            self.check(r, a4.CHILD_ROW, 1, "screen")
        r2 = self.root(evaluation={"mode": "confirm", "seeds": [0, 1], "folds": list(INNER)})
        with self.assertRaisesRegex(a4.A4Error, "does not authorise the three-seed confirmation"):
            self.check(r2, a4.CHILD_ROW, None, "evaluate")
        r3 = self.root(evaluation={"mode": "confirm", "seeds": [0, 1, 2], "folds": list(INNER[:5])})
        with self.assertRaisesRegex(a4.A4Error, "does not authorise the three-seed confirmation"):
            self.check(r3, a4.CHILD_ROW, None, "evaluate")

    def test_tampering_and_missing_declarations_fail_closed(self):
        r = self.root()
        edit_json(r / a4.AMENDMENT_REL, lambda a: a["authorised_now"].update(seeds=[1, 2, 3]))  # widen after declaring
        for s in (1, 3):
            with self.assertRaisesRegex(a4.A4Error, "declared file changed: research_r2_v4/amendment_4.json"):
                self.check(r, a4.CHILD_ROW, s)
        r = self.root(); (r / "pinned.py").write_text("x = 2\n")
        with self.assertRaisesRegex(a4.A4Error, "code pin pinned.py"):
            self.check(r, a4.CHILD_ROW, 1)
        r = self.root(); (r / "research_r2_v4/amendment_3.json").write_text("{}")
        with self.assertRaisesRegex(a4.A4Error, "declared file changed: research_r2_v4/amendment_3.json"):
            self.check(r, a4.CHILD_ROW, 1)
        r = self.root()
        with self.assertRaisesRegex(a4.A4Error, "amendment 3 does not verify"):
            self.check(r, a4.CHILD_ROW, 1, a3=lambda: ["amendment 3 does not verify: CHANGED"])
        r = self.root(); (r / a4.DECLARATION_REL).unlink()
        with self.assertRaisesRegex(a4.A4Error, "not declared"):
            self.check(r, a4.CHILD_ROW, 1)
        r = self.root()
        with self.assertRaisesRegex(a4.A4Error, "already exists"):
            a4.declare(r)
        r = self.root(); edit_json(r / a4.AMENDMENT_REL, lambda a: a["extends"].update(amendment_3_sha256="0" * 64)); (r / a4.DECLARATION_REL).unlink()
        a4.declare(r)
        with self.assertRaisesRegex(a4.A4Error, "does not extend the recorded amendment 3"):
            self.check(r, a4.CHILD_ROW, 1)

    def test_real_a3_problems_detect_a_changed_amendment_3(self):
        r = Path(tempfile.mkdtemp(prefix="r2v4_a4_a3_", dir=_S["tmp"]))
        (r / "research_r2_v4").mkdir()
        shutil.copy(ROOT / "research_r2_v4/amendment_3.json", r / "research_r2_v4/amendment_3.json")
        (r / "research_r2_v4/DECLARATION_AMENDMENT_3.json").write_text("{}")
        p = a4.a3_problems(r)
        self.assertTrue(any("DECLARATION_AMENDMENT_3.json changed" in x for x in p), p)
        self.assertTrue(any("amendment 3 does not verify" in x for x in p), p)


# ============================================================================= synthetic confirmation
class TestConfirmationPair(unittest.TestCase):
    def test_good_three_seed_pair_across_two_bases(self):
        C, P, meta = pair(_S["tmp"] / "childA", _S["tmp"] / "childB")
        self.assertEqual(sorted(C), [0, 1, 2]); self.assertEqual(sorted(P), [0, 1, 2])
        self.assertEqual(meta["modification"], ("source_balanced", True))
        for s in (0, 1, 2):
            self.assertTrue(C[s].identity["rederived"] and ci.is_sealed(C[s]) and C[s].identity["seed"] == s)
        dec = a4.confirmation_decision(C, P)
        self.assertEqual(dec["mode"], "confirm"); self.assertEqual(dec["gate"]["seeds"], [0, 1, 2])
        self.assertEqual(dec["gate"]["folds_required"], 5); self.assertEqual(sorted(dec["gate"]["per_fold_delta_pp"]), sorted(INNER))
        self.assertEqual(cg.primary_gate(C, P, INNER, "confirm", reps=10)["gate"], dec["gate"])

    def test_confirmation_report_on_synthetic_pair(self):
        import r2v4_a4  # the CLI imports cleanly and evaluates through amendment4.confirmation_report
        self.assertIs(r2v4_a4.a4, a4)
        C, P, meta = pair(_S["tmp"] / "childA", _S["tmp"] / "childB")
        res = a4.confirmation_report(C, P, meta, _S["frame"], _S["F"], ccfg(), reps=20)
        json.dumps(res, default=float)
        g = res["primary"]["gate"]
        self.assertEqual(res["decision"]["gate"], g); self.assertEqual(res["primary"]["endpoint"], "H_bal")
        self.assertEqual(sorted(res["primary"]["per_seed"]), [0, 1, 2]); self.assertEqual(sorted(res["child_receipts"]), [0, 1, 2])
        for s in (0, 1, 2):
            self.assertEqual(sorted(res["child_receipts"][s]), sorted(INNER))
        per_fold_mean = {k: sum(res["primary"]["per_seed"][s]["per_fold_delta_pp"][k] for s in (0, 1, 2)) / 3 for k in INNER}
        for k in INNER:
            self.assertAlmostEqual(per_fold_mean[k], g["per_fold_delta_pp"][k], places=9)
            hb = res["diagnostic_lofo_metrics_seed_mean"][k]["H_bal"]
            self.assertAlmostEqual(hb["delta_pp"], g["per_fold_delta_pp"][k], places=9)  # same LOFO choice as the gate
        for m in ("H_micro", "seen_top1", "seen_top5", "unseen_top1_micro", "unseen_top5_micro", "unseen_crop_top1", "unseen_disease_top1",
                  "unseen_withincrop_macro", "unseen_healthy_shortcut"):
            self.assertIn(m, res["diagnostic_lofo_metrics_overall"])
        self.assertNotIn("pass", res["diagnostics"])
        self.assertEqual(set(res["diagnostic_source_strata_seen"]["seed_mean"]), {"dominant", "other_fitting_source", "novel_source"})
        self.assertIn("unique_images", res["diagnostic_source_strata_seen"]["units"]["other_fitting_source"])

    def test_h_bal_is_the_only_endpoint_and_six_folds_three_seeds_required(self):
        C, P, _ = pair(_S["tmp"] / "childA", _S["tmp"] / "childB")
        for e in cg.DIAGNOSTIC_ENDPOINTS:
            with self.assertRaisesRegex(cg.GateError, "primary endpoint is H_bal"):
                cg.primary_gate(C, P, INNER, "confirm", endpoint=e, reps=10)
        with self.assertRaisesRegex(a4.A4Error, "needs all six folds"):
            a4.confirmation_decision(C, P, folds=INNER[:5])
        with self.assertRaisesRegex(a4.A4Error, "needs all six folds"):
            a4.confirmation_decision(C, P, folds=list(reversed(INNER)))
        with self.assertRaisesRegex(cg.GateError, "exactly seeds"):
            a4.confirmation_decision({s: C[s] for s in (0, 1)}, P)
        with self.assertRaisesRegex(cg.GateError, "exactly seeds"):
            a4.confirmation_decision(C, {s: P[s] for s in (0, 2)})
        with self.assertRaisesRegex(cg.GateError, "identified as seed"):
            a4.confirmation_decision({0: C[0], 1: C[2], 2: C[1]}, P)
        with self.assertRaisesRegex(cg.GateError, "not a result"):
            cg.promotion_decision({"role": "primary", "gate": {"pass": True}}, P, "confirm")

    def test_seed_sets_and_base_contents(self):
        A, B = bases("seedset")
        with self.assertRaisesRegex(a4.A4Error, "needs exactly"):
            pair(A, B, child_bases={0: A, 1: B})
        with self.assertRaisesRegex(a4.A4Error, "needs exactly"):
            pair(A, B, child_bases={0: A, 1: B, 2: B, 3: B})
        shutil.copytree(B / CHILD / "seed1", A / CHILD / "seed1")  # an extra seed in the seed-0 base
        with self.assertRaisesRegex(a4.A4Error, r"holds Jsyn_C seeds \[0, 1\], declared for seeds \[0\]"):
            pair(A, B)
        A, B = bases("seedmissing")
        shutil.rmtree(B / CHILD / "seed2")
        with self.assertRaisesRegex(a4.A4Error, r"holds Jsyn_C seeds \[1\], declared for seeds \[1, 2\]"):
            pair(A, B)

    def test_wrong_configuration_refused(self):
        with self.assertRaisesRegex(ci.ChildInputError, "exactly one registered field"):
            pair(_S["tmp"] / "childA", _S["tmp"] / "childB", child_cfg=pcfg())
        c = ccfg(); c.epochs = 2
        with self.assertRaisesRegex(ci.ChildInputError, "exactly one registered field"):
            pair(_S["tmp"] / "childA", _S["tmp"] / "childB", child_cfg=c)

    def test_fail_closed_inputs(self):
        cases = {
            "missing_sidecar": (lambda A, B: (adir(B, 2, "I3") / ci.SIDECAR).unlink(), "missing child provenance"),
            "identity_mismatch": (lambda A, B: edit_json(adir(B, 1, "I1") / ci.SIDECAR, lambda s: s.update(identity_sha256="0" * 64)),
                                  "child provenance does not match"),
            "stale_cache": (lambda A, B: shutil.copy(adir(B, 2, "I2") / "features.pt", adir(B, 1, "I2") / "features.pt"),
                            "features.pt missing or changed"),
            "wrong_checkpoint": (lambda A, B: shutil.copy(_S["tmp"] / "parent" / PARENT / "seed2" / "I5" / "attempt1" / "checkpoint.pt",
                                                          adir(B, 2, "I5") / "checkpoint.pt"), "checkpoint.pt missing or changed"),
            "missing_receipt": (lambda A, B: (adir(B, 1, "I4") / "receipt.json").unlink(), r"incomplete, missing folds \['I4'\]"),
            "failed_attempt_only": (lambda A, B: edit_json(adir(A, 0, "I2") / "receipt.json", lambda r: r.update(status="failed")),
                                    r"incomplete, missing folds \['I2'\]"),
            "missing_fold": (lambda A, B: shutil.rmtree(B / CHILD / "seed2" / "I6"), r"incomplete, missing folds \['I6'\]"),
            "duplicate_attempt": (lambda A, B: shutil.copytree(adir(B, 1, "I3"), adir(B, 1, "I3").parent / "attempt2"), "duplicate run"),
            "packet_changed": (lambda A, B: shutil.copy(adir(B, 2, "I1") / "I1_scores.npz", adir(B, 1, "I1") / "I1_scores.npz"),
                               "I1_scores.npz missing or changed"),
        }
        for name, (tamper, msg) in cases.items():
            with self.subTest(name):
                A, B = bases(name)
                tamper(A, B)
                with self.assertRaisesRegex(ERR, msg):
                    pair(A, B)

    def test_rederivation_mismatch_refused(self):
        from unittest import mock
        with mock.patch.object(ci, "rederive_packet", side_effect=ci.ChildInputError("simulated re-derivation mismatch")):
            with self.assertRaisesRegex(ci.ChildInputError, "simulated re-derivation mismatch"):
                pair(_S["tmp"] / "childA", _S["tmp"] / "childB")


if __name__ == "__main__":
    unittest.main()
