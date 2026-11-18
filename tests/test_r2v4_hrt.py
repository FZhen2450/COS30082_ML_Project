"""H_RT workflow tests (CPU, synthetic backend, no images; real folds and real saved Stage H index sets).

End to end: a synthetic row is trained at seed 0 on the six folds (the 'base', as Phase 1), then retrained for the
registered pair (I2, CPDD) in both arms. Negative tests cover every refusal of research_r2_v4.hrt."""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from research_r2.candidates import derive_fold  # noqa: E402
from research_r2_v4 import confirm, hrt, jrun  # noqa: E402
from research_r2_v4.common import INNER, env  # noqa: E402

ROW = "Jsyn_H"
FN, SRC = "I2", "CPDD"
_S = {}


def syn_cfg():
    return jrun.JConfig(ROW, backbone="synthetic", trainable="synthetic", epochs=1, batch=256, lr_backbone=1e-2, workers=0)


class FrozenMutating(jrun.SyntheticBackend):
    @staticmethod
    def embed(model, x):
        import torch
        with torch.no_grad():
            model[1].weight.add_(1e-6)  # illegal: changes a frozen parameter
        return model(x.float()).float()


def setUpModule():
    frame, bench, F = env()
    tmp = Path(tempfile.mkdtemp(prefix="r2v4_hrt_"))
    base, out = tmp / "base", tmp / "hrt"
    acc = jrun.ImageAccess(frame, None, loader=jrun.synthetic_loader(frame))
    st = jrun.run_row(syn_cfg(), 0, [F[f] for f in INNER], frame, bench, jrun.SyntheticBackend(), acc, base, device="cpu", log=lambda *_: None)
    assert all(v == "ok" for v in st.values()), st
    status = {arm: hrt.run_arm(ROW, F[FN], SRC, arm, frame, bench, jrun.SyntheticBackend(), out, loader=jrun.synthetic_loader(frame),
                               device="cpu", phase1=base, declared=syn_cfg(), log=lambda *_: None) for arm in hrt.ARMS}
    _S.update(frame=frame, bench=bench, F=F, tmp=tmp, base=base, out=out, status=status)


def tearDownModule():
    shutil.rmtree(_S["tmp"], ignore_errors=True)


def verify(out, arm="ablated"):
    return hrt.verify_arm(out, ROW, _S["F"][FN], SRC, arm, _S["frame"], phase1=_S["base"], declared=syn_cfg())


def attempt(out, arm="ablated", n=1):
    return Path(out) / ROW / f"{FN}_{SRC}" / arm / f"attempt{n}"


def copy_out():
    d = Path(tempfile.mkdtemp(prefix="r2v4_hrt_case_")) / "hrt"
    shutil.copytree(_S["out"], d)
    return d


def edit_json(p, fn):
    x = json.loads(Path(p).read_text()); fn(x); Path(p).write_text(json.dumps(x))


def reseal(a):
    """Re-hash receipt.json into the success marker so that only the intended check can trip."""
    edit_json(a / "hrt_receipt.json", lambda h: h.update(receipt_sha256=jrun.sha256_file(a / "receipt.json")))


class TestEndToEnd(unittest.TestCase):
    def test_both_arms_complete_and_evaluate(self):
        self.assertEqual(_S["status"], {"ablated": "ok", "random": "ok"})
        frame, F = _S["frame"], _S["F"]
        src = frame.source.to_numpy()
        for arm in hrt.ARMS:
            adir, rec, h, f_arm = verify(_S["out"], arm)
            self.assertEqual(h["opened_rows_excluded"], [])
            self.assertGreater(h["opened_rows_main_process"], 0)
            self.assertEqual(rec["n_fitting"], len(F[FN]["fitting_rows"]) - int((src[F[FN]["fitting_rows"]] == SRC).sum()))
        _, _, _, f_abl = verify(_S["out"], "ablated")
        self.assertEqual(int((src[f_abl["fitting_rows"]] == SRC).sum()), 0)
        import r2v4_stage_j as sj
        v = confirm.verified_row_seed(_S["base"], ROW, 0, INNER, syn_cfg())
        g = float(hrt.GAMMA_GRID[sj.table_from_packets(frame, F, v, ROW).choose(INNER)[1]])
        res = hrt.evaluate_row(ROW, [(FN, SRC)], _S["out"], frame, F, g, phase1=_S["base"], declared=syn_cfg(), reps=50)
        p = res[f"{FN}_{SRC}"]
        self.assertEqual(p["evaluation"], {"universe": 296, "seen_rows": 855, "unseen_rows": 309})
        self.assertEqual(p["n_novel_source_seen"], 39)
        self.assertEqual(set(p["outcomes"]), {"base", "ablated", "random"})
        gk = f"g{g:g}"
        e = p["effects"][gk]["novel_seen_top1"]
        o = p["outcomes"]
        self.assertAlmostEqual(e["source_effect"], o["ablated"][gk]["novel_seen_top1"] - o["random"][gk]["novel_seen_top1"])
        self.assertAlmostEqual(p["bootstrap_ablated_minus"]["random"]["novel_seen_top1"]["point_pp"], 100 * e["source_effect"], places=9)
        self.assertEqual(p["support"]["random"]["emptied_compositions"], [])
        self.assertEqual(len(p["support"]["ablated"]["emptied_compositions"]), 5)


class TestScopeAndFolds(unittest.TestCase):
    def test_registered_pairs(self):
        self.assertEqual(hrt.declared_pairs(), [("I6", "CDDM"), ("I1", "MTID"), ("I2", "CPDD")])
        fake = Path(tempfile.mkdtemp()) / "a2.json"
        a2 = json.loads(hrt.AMENDMENT_2.read_text()); a2["source_co_endpoint"]["H_RT"]["pairs"][2] = ["I3", "MTID"]
        fake.write_text(json.dumps(a2))
        with mock.patch.object(hrt, "AMENDMENT_2", fake), self.assertRaisesRegex(hrt.HRTError, "re-derived pair rule"):
            hrt.declared_pairs()

    def test_index_set_must_reproduce_its_manifest(self):
        frame, F = _S["frame"], _S["F"]
        _, removed, prov = hrt.arm_fold(frame, F[FN], SRC, "random")
        self.assertEqual(prov["index_set"], "I2_CPDD_seed0.json")
        d = Path(tempfile.mkdtemp())
        shutil.copy(hrt.INDEXSETS / "I2_CPDD_seed0.json", d / "I2_CPDD_seed0.json")
        sums = d / "SUMS"; sums.write_text(f"{jrun.sha256_file(d / 'I2_CPDD_seed0.json')}  I2_CPDD_seed0.json\n")
        hrt.arm_fold(frame, F[FN], SRC, "random", indexsets=d, sums=sums)  # faithful copy passes
        edit_json(d / "I2_CPDD_seed0.json", lambda s: s["removed_rows"].__setitem__(0, s["removed_rows"][0] + 1))
        with self.assertRaisesRegex(hrt.HRTError, "does not match the Stage H manifest"):
            hrt.arm_fold(frame, F[FN], SRC, "random", indexsets=d, sums=sums)
        sums.write_text(f"{jrun.sha256_file(d / 'I2_CPDD_seed0.json')}  I2_CPDD_seed0.json\n")
        with self.assertRaisesRegex(hrt.HRTError, "does not reproduce the saved index set"):
            hrt.arm_fold(frame, F[FN], SRC, "random", indexsets=d, sums=sums)
        edit_json(d / "I2_CPDD_seed0.json", lambda s: s.update(source="MTID"))
        sums.write_text(f"{jrun.sha256_file(d / 'I2_CPDD_seed0.json')}  I2_CPDD_seed0.json\n")
        with self.assertRaisesRegex(hrt.HRTError, "names I2/MTID"):
            hrt.arm_fold(frame, F[FN], SRC, "random", indexsets=d, sums=sums)

    def test_incomplete_source_removal_refused(self):
        frame, F = _S["frame"], _S["F"]
        fold = F[FN]; src = frame.source.to_numpy()
        s_rows = [r for r in fold["fitting_rows"] if src[r] == SRC]
        removed = s_rows[1:]  # one CPDD image left in the fitting rows
        f_bad = derive_fold(frame, fold, sorted(set(fold["fitting_rows"]) - set(removed)))
        with self.assertRaisesRegex(hrt.HRTError, "incomplete source removal"):
            hrt.verify_arm_fold(frame, fold, f_bad, SRC, "ablated", removed)

    def test_invalid_composition_support_refused(self):
        frame, F = _S["frame"], _S["F"]
        fold = F[FN]; src, comp = frame.source.to_numpy(), frame.comp_id.to_numpy()
        fit = np.asarray(fold["fitting_rows"]); n_s = int((src[fit] == SRC).sum())
        other = fit[src[fit] != SRC]
        c, cnt = np.unique(comp[other], return_counts=True)
        small = c[np.argmin(cnt)]
        whole = [int(r) for r in fit[comp[fit] == small]]
        if any(src[r] == SRC for r in whole):
            self.skipTest("smallest composition contains the source")
        rest = [int(r) for r in other if comp[r] != small][: n_s - len(whole)]
        removed = sorted(whole + rest)
        f_bad = derive_fold(frame, fold, sorted(set(fit.tolist()) - set(removed)))
        with self.assertRaisesRegex(hrt.HRTError, "emptied a composition"):
            hrt.verify_arm_fold(frame, fold, f_bad, SRC, "random", removed)
        with_src = sorted([int(r) for r in fit[src[fit] == SRC]][:1] + [int(r) for r in other][: n_s - 1])
        with self.assertRaisesRegex(hrt.HRTError, "contains CPDD rows"):
            hrt.verify_arm_fold(frame, fold, derive_fold(frame, fold, sorted(set(fit.tolist()) - set(with_src))), SRC, "random", with_src)
        short = [int(r) for r in other if comp[r] != small][: n_s - 5]
        with self.assertRaisesRegex(hrt.HRTError, "removal size"):
            hrt.verify_arm_fold(frame, fold, derive_fold(frame, fold, sorted(set(fit.tolist()) - set(short))), SRC, "random", short)


class TestExclusionAccess(unittest.TestCase):
    def test_excluded_rows_cannot_be_opened(self):
        frame, F = _S["frame"], _S["F"]
        _, removed, _ = hrt.arm_fold(frame, F[FN], SRC, "ablated")
        acc = hrt.ExclusionAccess(frame, None, removed, loader=jrun.synthetic_loader(frame))
        with self.assertRaisesRegex(hrt.HRTError, "would allow"):
            acc.allow("train", F[FN]["fitting_rows"])
        jrun.ImageAccess.allow(acc, "train", removed[:3])  # even if a phase allow-list were wrong
        with self.assertRaisesRegex(hrt.HRTError, "excluded in this H_RT arm"):
            acc.get(removed[0])
        self.assertEqual(acc.opened, 0)

    def test_training_on_the_unablated_fold_is_refused_before_any_image(self):
        frame, bench, F = _S["frame"], _S["bench"], _S["F"]
        _, removed, _ = hrt.arm_fold(frame, F[FN], SRC, "ablated")
        acc = hrt.ExclusionAccess(frame, None, removed, loader=jrun.synthetic_loader(frame))
        with self.assertRaises(hrt.HRTError):
            jrun.run_fold(syn_cfg(), 0, F[FN], INNER.index(FN), frame, bench, jrun.SyntheticBackend(), acc, Path(tempfile.mkdtemp()), "cpu")
        self.assertEqual(acc.opened, 0)


class TestVerificationRefusals(unittest.TestCase):
    def setUp(self):
        self.out = copy_out()

    def tearDown(self):
        shutil.rmtree(self.out.parent, ignore_errors=True)

    def refuse(self, msg, arm="ablated"):
        with self.assertRaisesRegex(jrun.JCheckError, msg):
            verify(self.out, arm)

    def test_duplicate_attempts(self):
        shutil.copytree(attempt(self.out), attempt(self.out, n=2))
        self.refuse("duplicate run, 2 successful attempts")

    def test_mismatched_candidate_universe(self):
        a = attempt(self.out)
        z = dict(np.load(a / f"{FN}_scores.npz"))
        z["cand_ids"] = z["cand_ids"][::-1].copy()
        np.savez_compressed(a / f"{FN}_scores.npz", **z)
        edit_json(a / "receipt.json", lambda r: r.update(packet_sha256=jrun.sha256_file(a / f"{FN}_scores.npz"))); reseal(a)
        self.refuse("score packet")

    def test_feature_cache_from_the_other_arm(self):
        a, b = attempt(self.out), attempt(self.out, "random")
        shutil.copy(b / "features.pt", a / "features.pt")
        edit_json(a / "receipt.json", lambda r: r.update(features_sha256=jrun.sha256_file(a / "features.pt"))); reseal(a)
        self.refuse("feature cache")

    def test_checkpoint_from_the_other_arm(self):
        a, b = attempt(self.out), attempt(self.out, "random")
        shutil.copy(b / "checkpoint.pt", a / "checkpoint.pt")
        edit_json(a / "receipt.json", lambda r: r.update(checkpoint_sha256=jrun.sha256_file(a / "checkpoint.pt"))); reseal(a)
        self.refuse("checkpoint metadata row does not name this arm")

    def test_receipt_changed_after_marker(self):
        edit_json(attempt(self.out) / "receipt.json", lambda r: r.update(seconds=1.0))
        self.refuse("changed since the H_RT success marker")

    def test_marker_for_another_arm(self):
        a = attempt(self.out)
        edit_json(a / "hrt_receipt.json", lambda h: h.update(arm="random"))
        self.refuse("names another row, pair or arm")

    def test_identity_and_integrity_mutations(self):
        cases = [
            (lambda r: r.update(torch_seed=7), "torch_seed"),
            (lambda r: r["config"].update(lr_backbone=5e-3) or r.update(config_sha256=jrun.JConfig(**r["config"]).sha()), "config_sha256|declared row"),
            (lambda r: r["grad_checks"].update(finite_checks=0), "gradient checks"),
            (lambda r: r["grad_checks"].update(steps=0), "step count"),
            (lambda r: r.update(frozen_digest_after="0" * 64), "frozen-parameter"),
            (lambda r: r.update(reload_max_abs_diff=1e-4), "reload difference"),
            (lambda r: r.update(frozen_digest_before="1" * 64, frozen_digest_after="1" * 64), "initialisation"),
            (lambda r: r.update(n_fitting=r["n_fitting"] + 1), "number of fitting rows"),
        ]
        for mutate, msg in cases:
            with self.subTest(msg=msg):
                d = copy_out(); a = attempt(d)
                edit_json(a / "receipt.json", mutate); reseal(a)
                with self.assertRaisesRegex(jrun.JCheckError, msg):
                    verify(d)
                shutil.rmtree(d.parent, ignore_errors=True)

    def test_marker_mutations(self):
        for mutate, msg in ((lambda h: h.update(opened_rows_excluded=[123]), "excluded rows were opened"),
                            (lambda h: h["support"].update(removed_rows_sha256="0" * 64), "registered derivation"),
                            (lambda h: h.update(base_checkpoint_sha256="0" * 64), "another base model")):
            with self.subTest(msg=msg):
                d = copy_out()
                edit_json(attempt(d) / "hrt_receipt.json", mutate)
                with self.assertRaisesRegex(jrun.JCheckError, msg):
                    verify(d)
                shutil.rmtree(d.parent, ignore_errors=True)

    def test_zero_steps_refused_by_confirmation_checks(self):
        rec = json.loads((attempt(self.out) / "receipt.json").read_text())
        rec["grad_checks"]["steps"] = 0
        self.assertTrue(any("step count" in x for x in confirm.receipt_problems(rec, rec["row"], 0, FN, INNER.index(FN), None)))


class TestFailureHandling(unittest.TestCase):
    def test_failure_is_recorded_and_retry_gets_a_new_attempt(self):
        frame, bench, F = _S["frame"], _S["bench"], _S["F"]
        d = Path(tempfile.mkdtemp()) / "hrt"
        kw = dict(loader=jrun.synthetic_loader(frame), device="cpu", phase1=_S["base"], declared=syn_cfg(), log=lambda *_: None)
        r = hrt.run_arm(ROW, F[FN], SRC, "ablated", frame, bench, FrozenMutating(), d, **kw)
        self.assertTrue(r.startswith("failed"), r)
        a1 = attempt(d)
        self.assertTrue((a1 / "failure.json").exists())
        self.assertFalse((a1 / "hrt_receipt.json").exists())
        with self.assertRaisesRegex(hrt.HRTError, "no successful attempt"):
            verify(d)
        r2 = hrt.run_arm(ROW, F[FN], SRC, "ablated", frame, bench, jrun.SyntheticBackend(), d, **kw)
        self.assertEqual(r2, "ok")
        self.assertTrue((a1 / "failure.json").exists(), "the failed attempt is kept")
        self.assertEqual(verify(d)[0].name, "attempt2")
        self.assertEqual(hrt.run_arm(ROW, F[FN], SRC, "ablated", frame, bench, jrun.SyntheticBackend(), d, **kw), "ok (existing)")
        shutil.rmtree(d.parent, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
