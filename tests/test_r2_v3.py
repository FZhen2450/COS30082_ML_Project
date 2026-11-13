"""Tests for the explicit candidate-universe policy (benchmark v3) and Phase 2 method helpers."""
import json
import unittest
from pathlib import Path

import numpy as np
import torch

from research_r2.candidates import POLICY, candidate_universe, derive_fold
from research_r2.data import load_training_source
from research_r2.methods import LeaceEraser, style_augment, style_bank, within_concept_source_weight

ROOT = Path(__file__).resolve().parents[1]
V2, V3 = ROOT / "artifacts/r2/benchmark_v2.json", ROOT / "artifacts/r2/benchmark_v3.json"


class TestHelpers(unittest.TestCase):
    def test_within_concept_source_weight_preserves_concept_totals_and_equalises_sources(self):
        concept = torch.tensor([0, 0, 0, 0, 1, 1]); src = torch.tensor([0, 0, 0, 1, 2, 2])
        w = within_concept_source_weight(concept, src)
        self.assertAlmostEqual(float(w[concept == 0].sum()), 4.0)
        self.assertAlmostEqual(float(w[(concept == 0) & (src == 0)].sum()), float(w[(concept == 0) & (src == 1)].sum()))
        self.assertTrue(torch.allclose(w[concept == 1], torch.ones(2, dtype=torch.float64)))

    def test_style_bank_is_composition_centred(self):
        g = torch.Generator().manual_seed(0)
        x = torch.randn(40, 5, generator=g, dtype=torch.float64)
        comp = torch.tensor([0] * 20 + [1] * 20); src = torch.tensor(([0] * 10 + [1] * 10) * 2)
        bank = style_bank(x, comp, src)
        self.assertEqual(bank.shape, (4, 5))
        self.assertTrue(torch.allclose(bank[:2].sum(0), torch.zeros(5, dtype=torch.float64), atol=1e-12))
        xa, (ya,), n = style_augment(x, (comp,), bank, k=2)
        self.assertEqual((len(xa), len(ya), n), (120, 120, 80))
        self.assertTrue(torch.equal(ya[40:80], comp))

    def test_subspace_shift_is_a_projection(self):
        g = torch.Generator().manual_seed(1)
        z = torch.randint(0, 4, (500,), generator=g)
        x = torch.randn(500, 12, generator=g, dtype=torch.float64)
        x[:, :3] += torch.nn.functional.one_hot(z, 4)[:, :3].double() * 2
        P = LeaceEraser(x, torch.nn.functional.one_hot(z, 4)).proj
        v = torch.randn(7, 12, generator=g, dtype=torch.float64)
        self.assertTrue(torch.allclose((v @ P) @ P, v @ P, atol=1e-8))


@unittest.skipUnless(V2.exists() and V3.exists(), "benchmarks not built")
class TestCandidatePolicy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame = load_training_source(ROOT / "DPD_Small_train.csv")
        cls.v2 = json.loads(V2.read_text()); cls.v3 = json.loads(V3.read_text())

    def test_v2_inner_folds_had_22_phantom_candidates(self):
        outer = set(self.v2["outer_composition_ids"])
        for f in self.v2["folds"]:
            held = set(f["heldout_composition_ids"]); fit = set(self.frame.comp_id.loc[f["fitting_rows"]])
            phantom = {int(c) for c in self.frame.comp_id.unique() if c not in held and c not in fit}
            self.assertEqual(phantom, outer if f["role"] == "inner" else set(), f["name"])

    def test_v3_universe_has_no_phantoms(self):
        for f in self.v3["folds"]:
            cu = candidate_universe(self.frame, f)
            self.assertEqual(cu["universe"], f["candidate_composition_ids"])
            self.assertEqual(f["candidate_policy"], POLICY)
            fit = set(self.frame.comp_id.loc[f["fitting_rows"]])
            self.assertTrue(all(c in fit for c in cu["fitting"]))
            self.assertFalse(set(cu["universe"]) & set(f["excluded_composition_ids"]))

    def test_make_fold_v3_columns_mask_and_text(self):
        from research_r2.runner import make_fold
        f = self.v3["folds"][0]
        n = len(self.frame)
        feats = torch.randn(n, 2, 8)
        cids = sorted(int(c) for c in self.frame.comp_id.unique())
        text = {"crop_ids": sorted(self.frame.crop_id.unique()), "disease_ids": sorted(self.frame.disease_id.unique()),
                "comp_ids": cids, "crop": torch.randn(55, 8), "disease": torch.randn(175, 8),
                "comp": torch.arange(len(cids)).float()[:, None].repeat(1, 8), "logit_scale": 100.0}
        fd, truth, healthy = make_fold(self.frame, f, feats, text)
        ids = fd.extra["cand_ids"]
        self.assertEqual(ids.tolist(), f["candidate_composition_ids"])
        self.assertEqual(int((~fd.fit_mask).sum()), len(f["heldout_composition_ids"]))
        self.assertTrue(torch.equal(fd.text["comp"][:, 0], torch.tensor([float(cids.index(c)) for c in ids])))
        self.assertEqual(len(healthy), len(ids))
        self.assertTrue(set(ids[truth["unseen"]]) <= set(f["heldout_composition_ids"]))

    def test_tampered_universe_rejected(self):
        from research_r2.runner import make_fold
        f = dict(self.v3["folds"][0]); f["candidate_composition_ids"] = f["candidate_composition_ids"] + [self.v3["outer_composition_ids"][0]]
        with self.assertRaises(ValueError):
            make_fold(self.frame, f, torch.randn(len(self.frame), 2, 8), None)

    def test_derive_fold_moves_emptied_compositions_to_excluded(self):
        f = self.v3["folds"][0]
        fit = self.frame.loc[f["fitting_rows"]]
        victim = int(fit.comp_id.value_counts().index[-1])  # a small fitting composition
        f2 = derive_fold(self.frame, f, fit.index[fit.comp_id != victim])
        self.assertNotIn(victim, f2["candidate_composition_ids"])
        self.assertTrue(all(int(self.frame.comp_id[r]) != victim for r in f2["seen_val_rows"]))


if __name__ == "__main__":
    unittest.main()
