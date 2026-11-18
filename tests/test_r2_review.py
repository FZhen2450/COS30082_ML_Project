"""Tests added in response to the Round 2 review (guards, v2 isolation, poisoning, selection)."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from research_r2 import guards
from research_r2.benchmark_v2 import build_benchmark_v2, validate_benchmark_v2
from research_r2.data import load_training_source, sha256_file
from research_r2.lineage import lineage_groups
from research_r2.metrics import FoldScorer

ROOT = Path(__file__).resolve().parents[1]
BENCH_V1 = ROOT / "artifacts/r2/benchmark_v1.json"
BENCH_V2 = ROOT / "artifacts/r2/benchmark_v2.json"


class TestFrozenRecipeGuard(unittest.TestCase):
    def setUp(self):
        self.bench = json.loads(BENCH_V1.read_text())

    def test_registered_recipe_accepted(self):
        e = guards.verify_frozen("stage_e_frozen_v1", ROOT / "research_r2/matrices/stage_e_frozen_v1.json", self.bench)
        self.assertEqual(e["protected_folds"], ["L1", "L2", "L3"])

    def test_arbitrary_matrix_with_its_own_hash_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "stage_e_frozen_v1.json"
            p.write_text(json.dumps([{"id": "E_selected_D2", "backbones": ["clip_l14"], "recipe": "flat"}]))
            with self.assertRaises(guards.FrozenRecipeError):  # wrong path even though caller "knows" its hash
                guards.verify_frozen("stage_e_frozen_v1", p, self.bench)
            reg = {"x": {"matrix": str(p), "matrix_sha256": "0" * 64, "benchmark_semantic_sha256": self.bench["semantic_sha256"],
                         "row_ids": ["E_selected_D2"], "protected_folds": ["L1"]}}
            with self.assertRaises(guards.FrozenRecipeError):  # right path, hash does not match registry
                guards.verify_frozen("x", p, self.bench, registry=reg)
        with self.assertRaises(guards.FrozenRecipeError):
            guards.verify_frozen("not_registered", ROOT / "research_r2/matrices/stage_e_frozen_v1.json", self.bench)

    def test_wrong_benchmark_rejected(self):
        b = dict(self.bench, semantic_sha256="f" * 64)
        with self.assertRaises(guards.FrozenRecipeError):
            guards.verify_frozen("stage_e_frozen_v1", ROOT / "research_r2/matrices/stage_e_frozen_v1.json", b)

    def test_protected_folds_cannot_be_requested_directly(self):
        self.assertNotIn("L1", guards.select_folds(self.bench))
        with self.assertRaises(guards.FrozenRecipeError):
            guards.select_folds(self.bench, ["S1", "L2"])
        self.assertEqual(guards.select_folds(self.bench, ["S1", "D0"]), ["S1", "D0"])

    def test_registry_matrix_hash_matches_committed_file(self):
        for name, e in guards.load_registry().items():
            self.assertEqual(sha256_file(ROOT / e["matrix"]), e["matrix_sha256"], name)


class TestBenchmarkV2Synthetic(unittest.TestCase):
    def test_outer_isolated_from_inner(self):
        rng = np.random.default_rng(0)
        import pandas as pd
        from research_r2.data import COLUMNS
        crops, dis = [f"c{i}" for i in range(10)], ["healthy"] + [f"d{i}" for i in range(10)]
        rows, cid = [], 0
        pairs = [(c, "healthy") for c in crops] + [(crops[i], dis[1 + j]) for i in range(10) for j in range(10) if (i + j) % 3 == 0]
        for c, d in pairs:
            for k in range(int(rng.integers(6, 12))):
                rows.append([f"S{cid % 4}_{c}_{d}_{k}.jpg", crops.index(c), dis.index(d), cid, c, d, f"{c}_{d}"])
            cid += 1
        f = pd.DataFrame(rows, columns=COLUMNS)
        f["source"] = f.file.str.split("_").str[0]; f["healthy"] = f.disease.eq("healthy"); f["row"] = f.index
        group, _ = lineage_groups([f"s{i}" for i in range(len(f))], list(range(0, 4 * len(f), 4)), threshold=0)
        b = build_benchmark_v2(f, group, seed=5, outer_group_size=2, n_inner=3)
        checks = validate_benchmark_v2(f, group, b)
        self.assertTrue(any("zero outer rows" in c for c in checks))
        X = set(b["outer_exclusion_rows"])
        for fold in b["folds"]:
            if fold["role"] == "inner":
                for k in ("fitting_rows", "seen_val_rows", "unseen_val_rows"):
                    self.assertFalse(X & set(fold[k]))


class TestPoisoning(unittest.TestCase):
    def test_nan_scores_rejected(self):
        s = np.zeros((2, 3)); s[0, 0] = np.nan
        with self.assertRaises(ValueError):
            FoldScorer(s, [0, 1], np.zeros((1, 3)), [2], np.array([1, 1, 0], bool), np.arange(3), np.arange(3), np.zeros(3, bool))


class TestSelectionRule(unittest.TestCase):
    def test_chain(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("sel", ROOT / "scripts/r2_select_v2.py")
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        folds = ("I1", "I2")

        def summ(h, ausuc=0.5, hs=0.0):
            return {"screen_mean": {"primary_H_bal_lofo": float(np.mean(h)), "ausuc": ausuc, "healthy_shortcut_lofo": hs},
                    "per_fold": {f: {"lofo": {"H_bal": v}} for f, v in zip(folds, h)}}
        S = {"B": summ([0.50, 0.50]), "X": summ([0.60, 0.60]), "Y": summ([0.70, 0.40]), "Z": summ([0.65, 0.66])}
        sel, path, _ = mod.select(S, {"B": ["X", "Y"], "X": ["Z"], "Y": [], "Z": []}, folds, start="B")
        self.assertEqual(path, ["B", "X", "Z"])  # Y fails the paired-fold condition; Z passes vs X
        self.assertEqual(sel, "Z")


@unittest.skipUnless(BENCH_V2.exists(), "v2 benchmark not built")
class TestRealBenchmarkV2(unittest.TestCase):
    def test_replay_v2(self):
        frame = load_training_source(ROOT / "DPD_Small_train.csv")
        group = np.array(json.loads((ROOT / "artifacts/r2/lineage_groups_v1.json").read_text())["group"])
        bench = json.loads(BENCH_V2.read_text())
        validate_benchmark_v2(frame, group, bench)
        declared = json.loads((ROOT / "research_r2/protocol_v2.json").read_text())["benchmark_semantic_sha256"]
        self.assertEqual(bench["semantic_sha256"], declared)
        rebuilt = build_benchmark_v2(frame, group)
        self.assertEqual(rebuilt["semantic_sha256"], declared)  # deterministic from committed inputs

    def test_v1_audit_reproduces_reviewer_counts(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("aud", ROOT / "scripts/r2_audit_v1_independence.py")
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        rep = mod.audit(json.loads(BENCH_V1.read_text()))
        got = {k: v["unseen_in_any_screen_fitting"] for k, v in rep["folds"].items()}
        self.assertEqual(got, {"L1": 628, "L2": 292, "L3": 293})
        self.assertEqual(sum(v["unseen_untouched_by_screening"] for v in rep["folds"].values()), 0)


if __name__ == "__main__":
    unittest.main()
