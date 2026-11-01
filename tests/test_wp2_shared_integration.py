"""Compatibility checks against merged WP1/WP5; metadata and synthetic scores only."""
import copy
import hashlib
import json
import unittest
from pathlib import Path

import torch

from development_data import build_development_split
from shared_protocol import build_protocol, candidate_table, validate_protocol
from evaluation import metrics as wp5
from test_notebook import helper_namespace

ROOT = Path(__file__).resolve().parents[1]
LEGACY_SOURCE = "9f1b729ab08f120c9e8d3a5681aa1ccbdf1d9210f9aaa6143959363dcb106a7d"
LEGACY_CANDIDATE = "4cb0a24a96cf647f32311817fa7b082a46d3cf00c20ae61f4287e83cc898dfaf"
LEGACY_SPLIT = "f0c7ba212449fd68f1d19c0f84cc91bf4cc1af29ae405128e5198bce89f25f85"


class WP2SharedIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = helper_namespace()
        cls.split = build_development_split(ROOT / "DPD_Small_train.csv")
        cls.spec = {"ids": cls.split.candidate_ids, "pairs": cls.split.candidate_pairs,
                    "sha256": cls.split.candidate_sha256}
        cls.reference = {key: getattr(cls.split, name).image_name.tolist()
                         for key, name in (("fitting_images", "fit"),
                                           ("seen_validation_images", "val_seen"),
                                           ("unseen_validation_images", "val_unseen"))}

    def test_historical_csv_bytes_and_existing_manifest_source_hashes(self):
        expected = {"DPD_Small_train.csv": LEGACY_SOURCE,
                    "DPD_Small_test_seen.csv": "45d31d48d2260f56cc28871b44434b5243100dc90dc95652c28c377b35b45e83",
                    "DPD_Small_test_unseen.csv": "757e3c1fdcc894004a69c21dbc00d6b818f563aa671db9ab8d315c4790c2490b"}
        for name, digest in expected.items():
            self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), digest)
        for name in ("stratified_seed42.json", "heldout_composition_seed42.json"):
            self.assertEqual(json.loads((ROOT / "artifacts/data" / name).read_text())["source_sha256"], LEGACY_SOURCE)

    def test_wp1_reconciles_exact_ordered_wp2_filenames(self):
        membership = {name: getattr(self.split, name).image_name.tolist()
                      for name in ("fit", "val_seen", "val_unseen")}
        digest = hashlib.sha256(json.dumps(membership, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        self.assertEqual(digest, LEGACY_SPLIT)
        protocol = build_protocol(ROOT / "DPD_Small_train.csv", reference=self.reference)
        self.assertEqual(protocol["historical_reconciliation"], "ordered_filenames_match_supplied_reference")
        self.assertEqual(protocol["counts"], [4868, 1238, 1894])
        folds = validate_protocol(protocol, ROOT / "DPD_Small_train.csv")
        for actual, name in zip(folds, ("fit", "val_seen", "val_unseen")):
            self.assertEqual(actual.image_name.tolist(), getattr(self.split, name).image_name.tolist())

    def test_changed_filename_order_is_rejected(self):
        changed = copy.deepcopy(self.reference)
        changed["fitting_images"][0], changed["fitting_images"][1] = changed["fitting_images"][1], changed["fitting_images"][0]
        with self.assertRaises(ValueError):
            build_protocol(ROOT / "DPD_Small_train.csv", reference=changed)

    def test_hash_namespaces_differ_but_candidate_columns_are_identical(self):
        protocol = build_protocol(ROOT / "DPD_Small_train.csv", reference=self.reference)
        self.assertEqual(self.split.candidate_sha256, LEGACY_CANDIDATE)
        self.assertNotEqual(protocol["candidate_sha256"], LEGACY_CANDIDATE)
        self.assertNotEqual(protocol["split_sha256"], LEGACY_SPLIT)
        self.assertEqual([x["composition_id"] for x in protocol["candidates"]], list(self.split.candidate_ids))
        self.assertEqual([(x["crop_id"], x["disease_id"]) for x in protocol["candidates"]], list(self.split.candidate_pairs))
        self.assertNotEqual(list(self.split.candidate_ids), list(range(323)))

    def test_wp5_metrics_consume_unchanged_canonical_log_scores(self):
        generator = torch.Generator().manual_seed(42)
        crop = torch.randn(4, 55, generator=generator)
        disease = torch.randn(4, 175, generator=generator)
        scores = self.ns["composition_scores"](crop, disease, self.spec)
        self.assertEqual(scores.dtype, torch.float64)
        truth = torch.tensor([self.split.candidate_ids[i] for i in (0, 10, 100, 322)])
        result = wp5.evaluate_compositions(scores, truth, self.split.candidate_ids)
        self.assertEqual(result["count"], 4)
        self.assertEqual(result["top1"], self.ns["composition_topk"](scores, truth, self.spec, 1))
        self.assertEqual(result["top5"], self.ns["composition_topk"](scores, truth, self.spec, 5))

    def test_exact_ties_and_top5_cutoff_keep_ascending_global_ids(self):
        ids = (10, 75, 99, 200, 250, 332)
        spec = {"ids": ids, "pairs": ((0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1))}
        scores, truth = torch.zeros(2, 6, dtype=torch.float64), torch.tensor([10, 332])
        for k in (1, 5):
            self.assertEqual(wp5.topk_accuracy(scores, truth, ids, k), self.ns["composition_topk"](scores, truth, spec, k))
            self.assertEqual(wp5.topk_accuracy(scores, truth, ids, k), .5)

    def test_near_equal_scores_preserve_one_ulp_ordering(self):
        ids = (10, 75, 99, 200, 250, 332)
        spec = {"ids": ids, "pairs": ((0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1))}
        scores = torch.ones(1, 6, dtype=torch.float64)
        scores[0, 1] = torch.nextafter(scores[0, 1], torch.tensor(float("inf"), dtype=torch.float64))
        truth = torch.tensor([75])
        self.assertEqual(wp5.topk_accuracy(scores, truth, ids, 1), 1.)
        self.assertEqual(self.ns["composition_topk"](scores, truth, spec, 1), 1.)

    def test_accuracy_units_and_operational_harmonic_mean(self):
        scores = torch.tensor([[1., 0.], [1., 0.], [1., 0.], [1., 0.]], dtype=torch.float64)
        self.assertEqual(wp5.topk_accuracy(scores, [10, 75, 75, 75], [10, 75], 1), .25)
        for s, u in ((0., 0.), (.5, .25), (1/1238, 1/1894), (610/1238, 44/1894), (1., 1.)):
            # Diagnostic comparison only; no tolerance is added to metric ranking,
            # checkpoint selection, or historical result values.
            self.assertAlmostEqual(self.ns["harmonic_mean"](s, u), wp5.harmonic_mean(s, u), delta=1e-15)
        with self.assertRaises(ValueError):
            wp5.harmonic_mean(75, 50)

    def test_optional_product_underflow_is_not_substituted(self):
        ids = (10, 75, 99, 200, 250, 332)
        pairs = ((0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1))
        crop = torch.full((1, 55), -2000.)
        crop[0, 0], crop[0, 1], crop[0, 2], crop[0, 54] = -1000., -999., -1001., 1000.
        disease = torch.full((1, 175), -2000.)
        disease[0, 0], disease[0, 1], disease[0, 174] = 0., -1., 1000.
        canonical = self.ns["composition_scores"](crop, disease, {"ids": ids, "pairs": pairs})
        product = wp5.composition_scores_from_logits(crop, disease, pairs)
        self.assertTrue(torch.isfinite(canonical).all())
        self.assertTrue(torch.equal(product, torch.zeros_like(product)))
        self.assertEqual(wp5.topk_accuracy(canonical, [99], ids, 1), 1.)
        self.assertEqual(wp5.topk_accuracy(product, [99], ids, 1), 0.)

    def test_tiny_harmonic_edge_case_is_documented_not_rewritten(self):
        self.assertEqual(self.ns["harmonic_mean"](1e-300, 1e-300), 1e-300)
        self.assertEqual(wp5.harmonic_mean(1e-300, 1e-300), 0.)

    def test_callers_reject_unsorted_candidate_contract(self):
        bad = {"ids": (75, 10), "pairs": ((0, 0), (0, 1))}
        with self.assertRaises(ValueError):
            self.ns["validate_candidate_spec"](bad)


if __name__ == "__main__":
    unittest.main()
