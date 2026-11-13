"""Round 2 research tests (unittest; CPU; synthetic plus optional real-benchmark replay)."""
import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from evaluation.metrics import topk_accuracy
from research_r2 import benchmark as B
from research_r2.data import COLUMNS, assert_not_official_test, load_training_source
from research_r2.lineage import lineage_groups
from research_r2.methods import LeaceEraser, synth_crop_shift, comp_balanced_weight
from research_r2.metrics import FoldScorer, lofo_calibrated, harmonic

ROOT = Path(__file__).resolve().parents[1]
REAL_BENCH = ROOT.parents[1] / "experiments/r2-compositional/benchmark/benchmark.json"


def synthetic_frame(seed=0):
    rng = np.random.default_rng(seed)
    crops = [f"c{i}" for i in range(6)]
    diseases = ["healthy"] + [f"d{i}" for i in range(6)]
    rows, comp = [], 0
    pairs = [(c, "healthy") for c in crops] + [(crops[i % 6], diseases[1 + (i % 6)]) for i in range(12)] + \
            [(crops[(i + 2) % 6], diseases[1 + (i % 6)]) for i in range(6)]
    pairs = list(dict.fromkeys(pairs))
    for cid, (c, d) in enumerate(pairs):
        for k in range(int(rng.integers(5, 15))):
            rows.append([f"SRC{cid % 3}_{c}_{d}_{k}.jpg", crops.index(c), diseases.index(d), cid, c, d, f"{c}_{d}"])
    f = pd.DataFrame(rows, columns=COLUMNS)
    f["source"] = f.file.str.split("_").str[0]
    f["healthy"] = f.disease.eq("healthy")
    f["row"] = f.index
    return f


class TestGuards(unittest.TestCase):
    def test_refuses_official_test_csvs(self):
        for name in ("DPD_Small_test_seen.csv", "DPD_Small_test_unseen.csv", "x_test.csv"):
            with self.assertRaises(PermissionError):
                assert_not_official_test(name)
            with self.assertRaises(PermissionError):
                load_training_source(name)

    def test_training_hash_enforced(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "DPD_Small_train.csv"
            p.write_text("a.jpg,0,0,0,c,d,c_d\n")
            with self.assertRaises(ValueError):
                load_training_source(p)
            self.assertEqual(len(load_training_source(p, expected_sha256=None)), 1)


class TestBenchmark(unittest.TestCase):
    def setUp(self):
        self.f = synthetic_frame()
        n = len(self.f)
        shas = [f"s{i}" for i in range(n)]
        shas[3] = shas[2]  # exact duplicate inside one composition
        self.group, _ = lineage_groups(shas, list(range(0, 4 * n, 4)), threshold=0)

    def test_eligibility_excludes_healthy_and_unshared(self):
        el = B.eligible_compositions(self.f, min_images=1)
        sub = self.f[self.f.comp_id.isin(el)]
        self.assertFalse((sub.disease == "healthy").any())
        for d in set(sub.disease):
            self.assertGreaterEqual(self.f[self.f.disease == d].crop.nunique(), 2)

    def test_heldout_validity_detects_lost_disease(self):
        comps = self.f[self.f.disease == "d0"].comp_id.unique()
        self.assertFalse(B.heldout_set_valid(self.f, set(int(c) for c in comps)))

    def test_build_and_validate(self):
        b = B.build_benchmark(self.f, self.group, seed=1, n_folds=3, n_screen=2, min_images=1, seen_fraction=0.2)
        checks = B.validate_benchmark(self.f, self.group, b)
        self.assertEqual(len(checks), 4)
        self.assertEqual([x["role"] for x in b["folds"]], ["screen", "screen", "locked"])
        b2 = B.build_benchmark(self.f, self.group, seed=1, n_folds=3, n_screen=2, min_images=1, seen_fraction=0.2)
        self.assertEqual(b["semantic_sha256"], b2["semantic_sha256"])  # deterministic

    def test_lineage_group_not_split(self):
        b = B.build_benchmark(self.f, self.group, seed=1, n_folds=3, n_screen=2, min_images=1, seen_fraction=0.2)
        g = self.group
        for fold in b["folds"]:
            fit = set(g[fold["fitting_rows"]])
            self.assertFalse(fit & set(g[fold["seen_val_rows"]]))
        self.assertEqual(g[2], g[3])


class TestMetrics(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.K = 12
        self.fit_mask = np.array([True] * 9 + [False] * 3)
        self.ss = np.round(rng.normal(size=(40, self.K)), 1)  # rounding forces exact ties
        self.su = np.round(rng.normal(size=(30, self.K)), 1)
        self.ts = rng.integers(0, 9, 40)
        self.tu = rng.integers(9, 12, 30)
        self.sc = FoldScorer(self.ss, self.ts, self.su, self.tu, self.fit_mask,
                             np.arange(self.K) % 4, np.arange(self.K) % 5, np.arange(self.K) == 0)

    def test_top1_parity_with_wp5_contract_under_gamma(self):
        cands = list(range(100, 100 + self.K))  # ascending global IDs
        for g in (0.0, 0.3, 1.0, 2.5):
            r = self.sc.at(g, top5=True)
            adj_s = self.ss - g * self.fit_mask
            adj_u = self.su - g * self.fit_mask
            self.assertAlmostEqual(r["seen_top1"], topk_accuracy(adj_s, [cands[t] for t in self.ts], cands, 1))
            self.assertAlmostEqual(r["unseen_top1_micro"], topk_accuracy(adj_u, [cands[t] for t in self.tu], cands, 1))
            self.assertAlmostEqual(r["seen_top5"], topk_accuracy(adj_s, [cands[t] for t in self.ts], cands, 5))

    def test_macro_weights_each_composition_equally(self):
        r = self.sc.at(0.0)
        pu = self.sc.top1_pred("unseen", 0.0)
        per = [np.mean(pu[self.tu == c] == c) for c in np.unique(self.tu)]
        self.assertAlmostEqual(r["unseen_top1_macro"], float(np.mean(per)))
        self.assertAlmostEqual(r["H_bal"], harmonic(r["seen_top1"], r["unseen_top1_macro"]))

    def test_ausuc_bounds(self):
        perfect = FoldScorer(np.eye(self.K)[self.ts] * 10, self.ts, np.eye(self.K)[self.tu] * 10, self.tu,
                             self.fit_mask, np.arange(self.K), np.arange(self.K), np.zeros(self.K, bool))
        self.assertAlmostEqual(perfect.ausuc(), 1.0)
        self.assertTrue(0 <= self.sc.ausuc() <= 1)

    def test_lofo_never_uses_own_fold(self):
        grid = np.array([0.0, 1.0])
        curves = {"A": [{"H_bal": 0.9}, {"H_bal": 0.1}], "B": [{"H_bal": 0.1}, {"H_bal": 0.2}], "C": [{"H_bal": 0.1}, {"H_bal": 0.2}]}
        cal = lofo_calibrated(curves, grid)
        self.assertEqual(cal["A"]["H_bal"], 0.1)  # B and C prefer gamma index 1
        self.assertEqual(cal["B"]["H_bal"], 0.1)  # A and C mean prefers index 0 (0.5 vs 0.2)


class TestMethods(unittest.TestCase):
    def test_leace_removes_linear_crop_information(self):
        g = torch.Generator().manual_seed(0)
        z = torch.randint(0, 4, (600,), generator=g)
        x = torch.randn(600, 16, generator=g, dtype=torch.float64)
        x[:, :3] += torch.nn.functional.one_hot(z, 4)[:, :3].double() * 3
        er = LeaceEraser(x, torch.nn.functional.one_hot(z, 4), ridge=0.0)
        xe = er(x)
        zc = torch.nn.functional.one_hot(z, 4).double()
        cov = (xe - xe.mean(0)).T @ (zc - zc.mean(0)) / len(z)
        self.assertLess(cov.abs().max().item(), 1e-8)
        self.assertGreater((x - xe).norm().item(), 1.0)

    def test_synth_targets_other_crops(self):
        x = torch.randn(50, 8)
        crop = torch.arange(50) % 5
        dis = torch.arange(50) % 3
        comp = crop * 3 + dis
        xs, ds, cs = synth_crop_shift(x, crop, dis, comp, 5, per_image=3)
        self.assertTrue(torch.all(cs != crop.repeat(3)))
        self.assertTrue(torch.equal(ds, dis.repeat(3)))

    def test_balanced_weight_sums_per_composition(self):
        comp = torch.tensor([0, 0, 0, 1, 2, 2])
        w = comp_balanced_weight(comp)
        for c in (0, 1, 2):
            self.assertAlmostEqual(w[comp == c].sum().item(), 1.0)


@unittest.skipUnless(REAL_BENCH.exists() and (ROOT / "DPD_Small_train.csv").exists(), "real benchmark not built")
class TestRealBenchmark(unittest.TestCase):
    def test_replay_real_benchmark(self):
        frame = load_training_source(ROOT / "DPD_Small_train.csv")
        bench = json.loads(REAL_BENCH.read_text())
        group = np.array(json.loads((REAL_BENCH.parent / "lineage_groups.json").read_text())["group"])
        checks = B.validate_benchmark(frame, group, bench)
        self.assertEqual(len(checks), 10)
        declared = json.loads((ROOT / "research_r2/protocol.json").read_text())["benchmark_semantic_sha256"]
        self.assertEqual(bench["semantic_sha256"], declared)
        legacy = json.loads((ROOT / "artifacts/data/heldout_composition_seed42.json").read_text())
        d0 = next(f for f in bench["folds"] if f["name"] == "D0")
        self.assertEqual(d0["heldout_composition_ids"], sorted(legacy["validation_composition_ids"]))


if __name__ == "__main__":
    unittest.main()
