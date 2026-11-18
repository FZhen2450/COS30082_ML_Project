"""Three-seed confirmation (Phase 2 evaluator repair): negative regression tests for every refusal, a positive
end-to-end check with independent recomputation, and a demonstration that the evaluator before the repair
(scripts/r2v4_stage_j.py at e96d471) accepted some of these invalid fixtures.

Fixture: one synthetic row trained for real (CPU, synthetic backend, no images) at seeds 0, 1 and 2 on the six
inner folds. The control is a deliberately weak random-score table so that the synthetic row passes the seed-0
screen and the cross-seed checks are reached."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from research_r2.packets import fold_contract  # noqa: E402
from research_r2_v4 import confirm, jrun  # noqa: E402
from research_r2_v4 import evaluate as ev  # noqa: E402
from research_r2_v4.common import INNER, env, scorer  # noqa: E402

import r2v4_stage_j as sj  # noqa: E402

ROW = "Jsyn_A"
OLD_COMMIT = "e96d471"
_S = {}


def syn_cfg():
    return jrun.JConfig(ROW, backbone="synthetic", trainable="synthetic", epochs=1, batch=256, lr_backbone=1e-2, workers=0)


def weak_control(frame, F):
    def arm(fn, seed):
        con = fold_contract(frame, F[fn]); rng = np.random.default_rng(seed); k = len(con["cand_ids"])
        return scorer(frame, {"seen": rng.normal(size=(len(con["seen_truth"]), k)), "unseen": rng.normal(size=(len(con["unseen_truth"]), k)),
                              "seen_truth": con["seen_truth"], "unseen_truth": con["unseen_truth"], "cand_ids": con["cand_ids"], "fold": F[fn]})
    return ev.RowTable(sj.D1, {fn: [arm(fn, 900 + j)] for j, fn in enumerate(INNER)}, ["single"])


def setUpModule():
    frame, bench, F = env()
    good = Path(tempfile.mkdtemp(prefix="r2v4_confirm_"))
    access = jrun.ImageAccess(frame, None, loader=jrun.synthetic_loader(frame))
    for s in confirm.CONFIRM_SEEDS:
        st = jrun.run_row(syn_cfg(), s, [F[f] for f in INNER], frame, bench, jrun.SyntheticBackend(), access, good, device="cpu", log=lambda *_: None)
        assert all(v == "ok" for v in st.values()), st
    _S.update(frame=frame, F=F, good=good, ctrl=weak_control(frame, F))


def tearDownModule():
    shutil.rmtree(_S["good"], ignore_errors=True)


def fast():
    """Fewer bootstrap resamples in tests (the statistic is unchanged; only its Monte Carlo precision)."""
    st = ExitStack()
    pb, pbs = ev.paired_bootstrap, confirm.paired_bootstrap_seeds
    st.enter_context(mock.patch.object(ev, "paired_bootstrap", lambda *a, **k: pb(*a, **{**k, "reps": 40})))
    st.enter_context(mock.patch.object(confirm, "paired_bootstrap_seeds", lambda *a, **k: pbs(*a, **{**k, "reps": 40})))
    return st


def attempt(base, seed, fold):
    return Path(base) / ROW / f"seed{seed}" / fold / "attempt1"


def edit_receipt(adir, fn):
    p = Path(adir) / "receipt.json"
    rec = json.loads(p.read_text()); fn(rec); p.write_text(json.dumps(rec))


def relabel_seed(seed_dir, new_seed):
    """Make a copied seed directory internally consistent with another seed number (receipt-level forgery)."""
    for i, fold in enumerate(INNER):
        def f(rec):
            rec["seed"] = new_seed; rec["torch_seed"] = 1000 * new_seed + i; rec["feature_cache_meta"]["seed"] = new_seed
        edit_receipt(Path(seed_dir) / fold / "attempt1", f)


class Case:
    def __init__(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="r2v4_case_"))
        self.runs = self.tmp / "runs"
        shutil.copytree(_S["good"], self.runs)
        self.out = self.tmp / "eval"

    def evaluate(self, runs=None):
        with fast():
            return sj.cmd_evaluate(types.SimpleNamespace(out=str(self.out), runs=[str(r) for r in (runs or [self.runs])], parent=None),
                                   d1_table=_S["ctrl"])

    def old_evaluate(self):
        """The evaluator as it was before the repair, from git, run on the same fixture (single run directory)."""
        src = subprocess.run(["git", "show", f"{OLD_COMMIT}:scripts/r2v4_stage_j.py"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
        mod = types.ModuleType("r2v4_stage_j_old")
        mod.__file__ = str(ROOT / "scripts" / "r2v4_stage_j_old.py")
        exec(compile(src, mod.__file__, "exec"), mod.__dict__)
        with fast():
            return mod.cmd_evaluate(types.SimpleNamespace(out=str(self.runs), parent=None), d1_table=_S["ctrl"])

    def close(self):
        shutil.rmtree(self.tmp, ignore_errors=True)


class TestConfirmationNegative(unittest.TestCase):
    def setUp(self):
        self.c = Case()

    def tearDown(self):
        self.c.close()

    def assertRefused(self, res, contains):
        ts = res["rows"][ROW]["three_seed"]
        self.assertEqual(ts["status"], "refused", ts)
        self.assertIn(contains, ts["reason"])
        self.assertNotIn("gate_vs_D1", ts)

    # --- seed sets
    def test_seed_set_0_1_3_refused(self):
        (self.c.runs / ROW / "seed2").rename(self.c.runs / ROW / "seed3")
        self.assertRefused(self.c.evaluate(), "needs exactly [0, 1, 2]")

    def test_extra_seed_refused(self):
        shutil.copytree(self.c.runs / ROW / "seed2", self.c.runs / ROW / "seed3")
        relabel_seed(self.c.runs / ROW / "seed3", 3)
        self.assertRefused(self.c.evaluate(), "needs exactly [0, 1, 2]")

    def test_missing_seed0_refuses_screen_and_confirmation(self):
        (self.c.runs / ROW / "seed0").rename(self.c.runs / ROW / "seed3")
        relabel_seed(self.c.runs / ROW / "seed3", 3)
        res = self.c.evaluate()
        self.assertNotIn(ROW, res["rows"])
        self.assertIn("no verified seed-0 run", res["refused_incomplete"][f"{ROW}/screen"])

    def test_non_canonical_seed_alias_refused(self):
        shutil.copytree(self.c.runs / ROW / "seed1", self.c.runs / ROW / "seed01")
        res = self.c.evaluate()
        self.assertNotIn(ROW, res["rows"])
        self.assertIn("non-canonical", res["refused_incomplete"][ROW])

    # --- incomplete / duplicate
    def test_incomplete_fold_refused(self):
        (attempt(self.c.runs, 2, "I4") / "I4_scores.npz").unlink()
        with self.assertRaises(jrun.JCheckError):
            confirm.verified_row_seed(self.c.runs, ROW, 2, INNER)
        shutil.rmtree(self.c.runs / ROW / "seed2" / "I4")
        res = self.c.evaluate()
        self.assertIn("missing folds ['I4']", res["refused_incomplete"][f"{ROW}/seed2"])
        self.assertRefused(res, "seeds [2] refused")

    def test_failed_fold_refused(self):
        edit_receipt(attempt(self.c.runs, 1, "I3"), lambda r: r.update(status="failed"))
        res = self.c.evaluate()
        self.assertIn("missing folds ['I3']", res["refused_incomplete"][f"{ROW}/seed1"])
        self.assertRefused(res, "refused")

    def test_two_successful_attempts_refused(self):
        shutil.copytree(attempt(self.c.runs, 1, "I2"), self.c.runs / ROW / "seed1" / "I2" / "attempt2")
        res = self.c.evaluate()
        self.assertIn("duplicate run, 2 successful attempts", res["refused_incomplete"][f"{ROW}/seed1"])
        self.assertRefused(res, "refused")

    def test_seed_copied_and_relabelled_refused(self):
        shutil.rmtree(self.c.runs / ROW / "seed2")
        shutil.copytree(self.c.runs / ROW / "seed1", self.c.runs / ROW / "seed2")
        relabel_seed(self.c.runs / ROW / "seed2", 2)
        self.assertRefused(self.c.evaluate(), "duplicate run, checkpoint_sha256 repeats")

    def test_seed_copied_without_relabel_refused(self):
        shutil.rmtree(self.c.runs / ROW / "seed2")
        shutil.copytree(self.c.runs / ROW / "seed1", self.c.runs / ROW / "seed2")
        res = self.c.evaluate()
        self.assertIn("receipt seed=1, expected 2", res["refused_incomplete"][f"{ROW}/seed2"])
        self.assertRefused(res, "refused")

    def test_same_seed_in_two_run_directories_refused(self):
        other = self.c.tmp / "runs_b"
        shutil.copytree(self.c.runs / ROW / "seed1", other / ROW / "seed1")
        res = self.c.evaluate(runs=[self.c.runs, other])
        self.assertIn("duplicate run", res["refused_incomplete"][f"{ROW}/seed1"])
        self.assertRefused(res, "refused")

    # --- configuration / code / candidates
    def test_incompatible_configuration_refused(self):
        for fold in INNER:
            def f(rec):
                rec["config"]["lr_backbone"] = 5e-3
                rec["config_sha256"] = jrun.JConfig(**rec["config"]).sha(); rec["feature_cache_meta"]["config_sha256"] = rec["config_sha256"]
            edit_receipt(attempt(self.c.runs, 2, fold), f)
        self.assertEqual(len(confirm.verified_row_seed(self.c.runs, ROW, 2, INNER)), 6)  # each seed alone is consistent
        self.assertRefused(self.c.evaluate(), "incompatible configurations across seeds")

    def test_config_hash_mismatch_refused(self):
        edit_receipt(attempt(self.c.runs, 1, "I5"), lambda r: r["config"].update(epochs=2))
        res = self.c.evaluate()
        self.assertIn("config_sha256 does not match", res["refused_incomplete"][f"{ROW}/seed1"])

    def test_declared_configuration_enforced(self):
        declared = syn_cfg(); declared.lr_head = 1.0
        with self.assertRaisesRegex(confirm.ConfirmError, r"differs from the declared row in \['lr_head'\]"):
            confirm.verified_row_seed(self.c.runs, ROW, 0, INNER, declared)

    def test_smoke_run_refused(self):
        for fold in INNER:
            def f(rec):
                rec["config"]["max_steps"] = 20
                rec["config_sha256"] = jrun.JConfig(**rec["config"]).sha(); rec["feature_cache_meta"]["config_sha256"] = rec["config_sha256"]
            edit_receipt(attempt(self.c.runs, 2, fold), f)
        res = self.c.evaluate()
        self.assertIn("smoke run", res["refused_incomplete"][f"{ROW}/seed2"])

    def test_training_code_differs_refused(self):
        for fold in INNER:
            edit_receipt(attempt(self.c.runs, 2, fold), lambda r: r["code"].update(jrun_sha256="0" * 64))
        self.assertRefused(self.c.evaluate(), "training code differs across seeds")

    def test_mismatched_candidate_mapping_refused(self):
        a = attempt(self.c.runs, 2, "I1")
        z = dict(np.load(a / "I1_scores.npz"))
        z["cand_ids"] = z["cand_ids"][::-1].copy(); z["seen"] = z["seen"][:, ::-1].copy(); z["unseen"] = z["unseen"][:, ::-1].copy()
        np.savez_compressed(a / "I1_scores.npz", **z)
        edit_receipt(a, lambda r: r.update(packet_sha256=jrun.sha256_file(a / "I1_scores.npz")))
        res = self.c.evaluate()
        self.assertIn("PacketError", res["refused_incomplete"][f"{ROW}/seed2"])
        self.assertRefused(res, "refused")

    # --- provenance / receipts
    def test_missing_checkpoint_provenance_refused(self):
        a = attempt(self.c.runs, 1, "I6")
        edit_receipt(a, lambda r: r["feature_cache_meta"].update(checkpoint_sha256="f" * 64))
        res = self.c.evaluate()
        self.assertIn("feature_cache_meta checkpoint_sha256", res["refused_incomplete"][f"{ROW}/seed1"])
        rec = json.loads((a / "receipt.json").read_text()); del rec["checkpoint_sha256"]
        self.assertTrue(any("missing checkpoint provenance" in x for x in confirm.receipt_problems(rec, ROW, 1, "I6", 5, None)))

    def test_invalid_receipts_refused(self):
        rec = json.loads((attempt(self.c.runs, 0, "I2") / "receipt.json").read_text())
        self.assertEqual(confirm.receipt_problems(rec, ROW, 0, "I2", 1, syn_cfg()), [])
        for mutate, msg in ((lambda r: r.update(fold="I3"), "receipt fold"), (lambda r: r.update(row="Jsyn_B"), "receipt row"),
                            (lambda r: r.update(torch_seed=5), "torch_seed"), (lambda r: r.update(reload_max_abs_diff=1e-3), "reload difference"),
                            (lambda r: r.update(frozen_digest_after="x"), "frozen-parameter"),
                            (lambda r: r.update(trainable_tensors_changed="1/2"), "trainable tensors"),
                            (lambda r: r["grad_checks"].update(finite_checks=0), "gradient checks"),
                            (lambda r: r["grad_checks"].update(skipped_steps=10 ** 6), "AMP skipped"),
                            (lambda r: r.pop("code"), "code identity"), (lambda r: r.pop("config"), "no configuration")):
            bad = json.loads(json.dumps(rec)); mutate(bad)
            self.assertTrue(any(msg in x for x in confirm.receipt_problems(bad, ROW, 0, "I2", 1, None)), msg)
        clip = json.loads(json.dumps(rec)); clip["config"]["backbone"] = "clip_l14"; clip["code"]["worktree_clean"] = False
        self.assertTrue(any("dirty worktree" in x for x in confirm.receipt_problems(clip, ROW, 0, "I2", 1, None)))

    def test_screen_fail_blocks_confirmation(self):
        strong = _S["ctrl"]
        res_ok = self.c.evaluate()
        self.assertTrue(res_ok["rows"][ROW]["seed0"]["gate_vs_D1"]["pass"])
        t0 = sj.table_from_packets(_S["frame"], _S["F"], confirm.verified_row_seed(self.c.runs, ROW, 0, INNER), "ctl")
        _S["ctrl"] = t0  # control == the row itself: the screen cannot pass (0 pp)
        try:
            self.assertRefused(self.c.evaluate(), "seed-0 screen did not pass")
        finally:
            _S["ctrl"] = strong


class TestConfirmationPositive(unittest.TestCase):
    def test_good_fixture_and_independent_recomputation(self):
        c = Case()
        try:
            res = c.evaluate()
            ts = res["rows"][ROW]["three_seed"]
            self.assertEqual(ts["status"], "evaluated", ts)
            self.assertEqual(ts["seeds"], [0, 1, 2])
            self.assertEqual(res["refused_incomplete"], {})
            tables = {s: sj.table_from_packets(_S["frame"], _S["F"], confirm.verified_row_seed(c.runs, ROW, s, INNER), f"s{s}") for s in (0, 1, 2)}
            Sd = _S["ctrl"].summary(INNER)
            S = {s: t.summary(INNER) for s, t in tables.items()}
            # gate on the per-fold seed mean, recomputed by hand
            per = {f: np.mean([S[s]["per"][f]["H_bal"] for s in S]) for f in INNER}
            d = [per[f] - Sd["per"][f]["H_bal"] for f in INNER]
            ausuc = np.mean([np.mean([S[s]["per"][f]["ausuc"] for s in S]) for f in INNER])
            hs = np.mean([np.mean([S[s]["per"][f]["unseen_healthy_shortcut"] for s in S]) for f in INNER])
            g = ts["gate_vs_D1"]
            self.assertAlmostEqual(g["mean_gain_pp"], 100 * np.mean(d), places=9)
            self.assertEqual(g["folds_improved"], sum(x > 0 for x in d))
            self.assertAlmostEqual(g["ausuc_delta"], ausuc - Sd["ausuc"], places=9)
            self.assertAlmostEqual(g["shortcut_delta_pp"], 100 * (hs - Sd["hs"]), places=9)
            self.assertEqual(g["pass"], bool(100 * np.mean(d) >= 1 and sum(x > 0 for x in d) >= 5 and ausuc >= Sd["ausuc"] and 100 * (hs - Sd["hs"]) <= 5))
            # each seed: its own LOFO choice and its own nested estimate; the confirmation averages them
            nest = {s: sj.nested_with_parent_rule({sj.D1: _S["ctrl"], ROW: tables[s]}, sj.D1, [ROW], INNER, {}) for s in tables}
            for s in tables:
                self.assertEqual(ts["per_seed"][s]["per_fold_gamma"], {f: S[s]["per"][f]["gamma"] for f in INNER})
            n3 = ts["nested_vs_D1"]
            self.assertAlmostEqual(n3["mean_diff_pp"], np.mean([nest[s]["mean_diff_pp"] for s in nest]), places=9)
            pf = {f: np.mean([100 * (h["H_bal"] - h["control_H_bal"]) for s in nest for h in nest[s]["held"] if h["held_fold"] == f]) for f in INNER}
            self.assertEqual(n3["folds_better"], sum(v > 0 for v in pf.values()))
            self.assertEqual(ts["pass"], g["pass"] and n3["meets_gate_thresholds"])
            self.assertEqual(ts["verdict"], "confirmed" if ts["pass"] else "not confirmed")
            self.assertIn("stage_nested_three_seed", res)
            self.assertAlmostEqual(res["stage_nested_three_seed"]["mean_diff_pp"], n3["mean_diff_pp"], places=9)  # one candidate
            # outputs kept apart and labelled
            self.assertEqual(res["rows"][ROW]["seed0_label"], "seed-0 screen")
            self.assertEqual(res["stage_nested"]["label"], "seed-0 screen tables")
        finally:
            c.close()

    def test_bootstrap_seeds_reduces_to_single_seed(self):
        t = sj.table_from_packets(_S["frame"], _S["F"], confirm.verified_row_seed(_S["good"], ROW, 0, INNER), "s0")
        a = ev.paired_bootstrap(t, _S["ctrl"], INNER, reps=60)
        b = confirm.paired_bootstrap_seeds([t], [_S["ctrl"]], INNER, reps=60)
        self.assertAlmostEqual(a["point_pp"], b["point_pp"], places=9)
        np.testing.assert_allclose(a["ci95_pp"], b["ci95_pp"], atol=1e-9)

    def test_confirmation_requires_exact_seeds(self):
        t = {s: sj.table_from_packets(_S["frame"], _S["F"], confirm.verified_row_seed(_S["good"], ROW, s, INNER), f"s{s}") for s in (0, 1, 2)}
        with self.assertRaises(confirm.ConfirmError):
            confirm.three_seed_confirmation(ROW, {0: t[0], 1: t[1], 3: t[2]}, _S["ctrl"], sj.D1, INNER, sj.nested_with_parent_rule)


class TestPreviousEvaluatorAcceptedInvalidFixtures(unittest.TestCase):
    """The evaluator before the repair produced a three-seed block for fixtures the repaired one refuses."""

    def run_both(self, mutate):
        c = Case()
        try:
            mutate(c)
            old = c.old_evaluate()
            new = c.evaluate()
            return old, new
        finally:
            c.close()

    def test_seed_set_0_1_3(self):
        def m(c): (c.runs / ROW / "seed2").rename(c.runs / ROW / "seed3")
        old, new = self.run_both(m)
        self.assertIn("three_seed", old["rows"][ROW])
        self.assertEqual(sorted(old["rows"][ROW]["three_seed"]["nested_per_seed"]), [0, 1, 3])  # accepted
        self.assertEqual(new["rows"][ROW]["three_seed"]["status"], "refused")

    def test_seed1_used_as_screen_when_seed0_missing(self):
        def m(c):
            (c.runs / ROW / "seed0").rename(c.runs / ROW / "seed3")
        old, new = self.run_both(m)
        self.assertEqual(old["rows"][ROW]["seeds"], [1, 2, 3])
        self.assertIn("three_seed", old["rows"][ROW])  # accepted, with seed 1 standing in for the seed-0 screen
        self.assertNotIn(ROW, new["rows"])

    def test_duplicate_seed_copy(self):
        def m(c):
            shutil.rmtree(c.runs / ROW / "seed2")
            shutil.copytree(c.runs / ROW / "seed1", c.runs / ROW / "seed2")
        old, new = self.run_both(m)
        ns = old["rows"][ROW]["three_seed"]["nested_per_seed"]
        self.assertEqual(ns[1], ns[2])  # the copied run counted twice
        self.assertEqual(new["rows"][ROW]["three_seed"]["status"], "refused")


if __name__ == "__main__":
    unittest.main()
