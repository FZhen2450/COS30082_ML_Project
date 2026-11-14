"""v4 tests (CPU, no features, no images): matched controls, common-universe restriction, LOFO equivalence,
evaluation-view isolation, and the J pilot's data boundary."""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from research_r2.candidates import derive_fold  # noqa: E402
from research_r2.metrics import GAMMA_GRID, FoldScorer, lofo_calibrated  # noqa: E402
from research_r2.packets import PacketError, fold_contract  # noqa: E402
from research_r2.runner import make_fold  # noqa: E402
from research_r2_v4 import evaluate as ev  # noqa: E402
from research_r2_v4.common import env, restrict, scorer  # noqa: E402


class _Env:
    frame = bench = F = None

    @classmethod
    def get(cls):
        if cls.frame is None:
            cls.frame, cls.bench, cls.F = env()
        return cls.frame, cls.bench, cls.F


def random_arm(frame, fold, seed=0):
    con = fold_contract(frame, fold)
    rng = np.random.default_rng(seed)
    k = len(con["cand_ids"])
    return {"seen": rng.normal(size=(len(con["seen_truth"]), k)), "unseen": rng.normal(size=(len(con["unseen_truth"]), k)),
            "seen_truth": con["seen_truth"], "unseen_truth": con["unseen_truth"], "cand_ids": con["cand_ids"], "fold": fold}


class TestMatchedRemoval(unittest.TestCase):
    def test_invariants(self):
        from r2v4_stage_h import matched_removal
        frame, _, F = _Env.get()
        fold = F["I1"]
        for src in ("CPDD", "MTID"):
            n_s = int((frame.source.to_numpy()[fold["fitting_rows"]] == src).sum())
            kept, picks, lv = matched_removal(frame, fold["fitting_rows"], src, seed=0)
            self.assertEqual(len(picks), n_s)
            self.assertFalse((frame.source.to_numpy()[picks] == src).any(), "control removed images of the ablated source")
            self.assertEqual(set(kept) | set(picks), set(fold["fitting_rows"]))
            self.assertEqual(set(frame.comp_id[kept]), set(frame.comp_id[fold["fitting_rows"]]), "control emptied a composition")
            self.assertEqual(sum(lv.values()), n_s)
            again = matched_removal(frame, fold["fitting_rows"], src, seed=0)[1]
            self.assertEqual(picks, again, "fixed seed must give the same index set")


class TestRestrict(unittest.TestCase):
    def test_restrict_matches_by_id_and_validates(self):
        frame, _, F = _Env.get()
        fold = F["I1"]
        fit2 = [r for r in fold["fitting_rows"] if frame.source[r] != "CPDD"]
        f2 = derive_fold(frame, fold, fit2)
        self.assertLess(len(f2["candidate_composition_ids"]), len(fold["candidate_composition_ids"]))
        arm = random_arm(frame, fold)
        sc = restrict(frame, arm, f2)
        ids = list(arm["cand_ids"])
        con2 = fold_contract(frame, f2)
        rows = [fold["seen_val_rows"].index(r) for r in f2["seen_val_rows"]]
        cols = [ids.index(int(c)) for c in con2["cand_ids"]]
        np.testing.assert_array_equal(sc.parts["seen"]["scores"], arm["seen"][np.ix_(rows, cols)])
        np.testing.assert_array_equal(sc.parts["seen"]["truth"], con2["seen_truth"])

    def test_restrict_refuses_missing_candidates(self):
        frame, _, F = _Env.get()
        fold = F["I1"]
        f2 = derive_fold(frame, fold, [r for r in fold["fitting_rows"] if frame.source[r] != "CPDD"])
        arm = random_arm(frame, f2)  # smaller universe cannot be expanded to the base fold
        with self.assertRaises((ValueError, KeyError, PacketError)):
            restrict(frame, arm, fold)


class TestLofoEquivalence(unittest.TestCase):
    def test_single_config_matches_v3_lofo(self):
        frame, _, F = _Env.get()
        sc = {f: [scorer(frame, random_arm(frame, F[f], seed=i))] for i, f in enumerate(["I1", "I2", "I3"])}
        t = ev.RowTable("x", sc, ["single"])
        ref = lofo_calibrated({f: sc[f][0].curve() for f in sc})
        s = t.summary(list(sc))
        for f in sc:
            self.assertAlmostEqual(s["per"][f]["H_bal"], ref[f]["H_bal"], places=12)
            self.assertAlmostEqual(s["per"][f]["gamma"], ref[f]["gamma"], places=12)

    def test_config_zero_wins_ties(self):
        frame, _, F = _Env.get()
        a = scorer(frame, random_arm(frame, F["I1"]))
        t = ev.RowTable("x", {"I1": [a, a], "I2": [a, a]}, ["g1=0", "g1=1"])
        self.assertEqual(t.choose(["I2"])[0], 0)


class TestEvaluationViewIsolation(unittest.TestCase):
    def test_only_fitting_rows_use_extra_views(self):
        frame, bench, F = _Env.get()
        fold = F["I1"]
        x = torch.randn(len(frame), 12, 8)
        nonfit = sorted(set(range(len(frame))) - set(fold["fitting_rows"]))
        x[nonfit, 1:] = float("nan")  # any use of a non-fitting extra view would surface as NaN
        fd, _, _ = make_fold(frame, fold, x, None)
        self.assertTrue(torch.isfinite(fd.x_fit).all() and torch.isfinite(fd.x_seen).all() and torch.isfinite(fd.x_uns).all())


class TestPilotBoundary(unittest.TestCase):
    def test_pilot_split_is_fitting_only_and_lineage_disjoint(self):
        from r2v4_j_pilot import pilot_split, pilot_fold
        frame, bench, F = _Env.get()
        fn = pilot_fold()
        tr, va = pilot_split(frame, F[fn], seed=0)
        fit = set(F[fn]["fitting_rows"])
        self.assertTrue(set(tr) <= fit and set(va) <= fit and not (set(tr) & set(va)))
        group = np.array(json.loads((ROOT / "artifacts/r2/lineage_groups_v1.json").read_text())["group"])
        self.assertFalse(set(group[tr]) & set(group[va]), "lineage group split across pilot train/validation")
        forbidden = set(F[fn]["seen_val_rows"]) | set(F[fn]["unseen_val_rows"]) | set(bench["outer_exclusion_rows"])
        self.assertFalse((set(tr) | set(va)) & forbidden)


if __name__ == "__main__":
    unittest.main()
