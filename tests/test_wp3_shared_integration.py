"""WP3A compatibility with merged protocol/metrics; no CLIP weights or real images."""
import json
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from evaluation import metrics as wp5
from shared_protocol import build_protocol
from development_data import build_development_split
from test_wp3_clip import helper_namespace

ROOT = Path(__file__).resolve().parents[1]


class WP3SharedIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = helper_namespace()
        cls.split = build_development_split(ROOT / "DPD_Small_train.csv")
        cls.table = cls.ns["build_prompt_table"](cls.split)

    def test_wp1_candidate_columns_match_prompt_global_ids(self):
        s = self.split
        reference = {key: getattr(s, name).image_name.tolist()
                     for key, name in (("fitting_images", "fit"), ("seen_validation_images", "val_seen"),
                                       ("unseen_validation_images", "val_unseen"))}
        p = build_protocol(ROOT / "DPD_Small_train.csv", reference=reference)
        self.assertEqual(p["historical_reconciliation"], "ordered_filenames_match_supplied_reference")
        self.assertEqual([c["composition_id"] for c in p["candidates"]], self.table.global_composition_id.tolist())
        self.assertEqual([(c["crop_id"], c["disease_id"]) for c in p["candidates"]],
                         list(zip(self.table.crop_id, self.table.disease_id)))

    def test_final_v2_prompt_hash_and_native_token_contract(self):
        ns = self.ns
        digest = ns["json_hash"]({"policy": ns["PROMPT_POLICY"], "rows": self.table.to_dict("records")})
        self.assertEqual(digest, "753672c7964bcb20a1346145f21a72b40a19f55590a006b5783823f23c13f537")
        self.assertEqual(ns["PROMPT_POLICY"]["version"], "label_only_single_template_v2")
        tokens = ns["tokenize_candidates"](self.table)
        self.assertEqual(tuple(tokens.shape), (323, 77))
        self.assertEqual(tokens.dtype, torch.int32)

    def test_native_preprocessing_is_not_imagenet(self):
        from clip.clip import _transform
        transform = _transform(224)
        output = transform(Image.new("RGB", (300, 200), (255, 0, 0)))
        self.assertEqual(tuple(output.shape), (3, 224, 224))
        self.assertAlmostEqual(output[0, 0, 0].item(), (1-.48145466)/.26862954, places=5)
        self.assertIn("CenterCrop", repr(transform))
        self.assertIn("bicubic", repr(transform).lower())

    def test_full_candidate_cosine_scores_match_wp5_metrics(self):
        generator = torch.Generator().manual_seed(42)
        images = torch.randn(4, 8, generator=generator)
        texts = torch.randn(323, 8, generator=generator)
        scores = self.ns["cosine_candidate_scores"](images, texts)
        self.assertEqual(scores.dtype, torch.float64)
        truths = torch.tensor([self.split.candidate_ids[i] for i in (0, 42, 100, 322)])
        local, _ = self.ns["fold_metrics"](scores, truths, self.split.candidate_ids)
        shared = wp5.evaluate_compositions(scores, truths, self.split.candidate_ids)
        self.assertEqual(local["sample_count"], shared["count"])
        self.assertEqual(local["top1"], shared["top1"])
        self.assertEqual(local["top5"], shared["top5"])

    def test_exact_ties_and_cutoff_match_shared_stable_sort(self):
        ids = (10, 75, 99, 200, 250, 332)
        scores, truths = torch.ones(2, 6, dtype=torch.float64), torch.tensor([10, 332])
        local, ranking = self.ns["fold_metrics"](scores, truths, ids)
        shared = wp5.evaluate_compositions(scores, truths, ids)
        self.assertEqual(ranking[0].tolist(), list(ids))
        self.assertEqual((local["top1"], local["top5"]), (.5, .5))
        self.assertEqual((local["top1"], local["top5"]), (shared["top1"], shared["top5"]))

    def test_near_tie_one_ulp_is_not_rounded(self):
        ids = (10, 75, 99, 200, 250, 332)
        scores = torch.ones(1, 6, dtype=torch.float64)
        scores[0, 1] = torch.nextafter(scores[0, 1], torch.tensor(float("inf"), dtype=torch.float64))
        local, _ = self.ns["fold_metrics"](scores, torch.tensor([75]), ids)
        self.assertEqual(local["top1"], 1.)
        self.assertEqual(wp5.topk_accuracy(scores, [75], ids, 1), 1.)

    def test_metric_units_h1_h5_and_zero_zero(self):
        seen = {"sample_count": 4, "top1": .25, "top5": .75}
        unseen = {"sample_count": 2, "top1": .5, "top5": 1.}
        local = self.ns["development_metrics"](seen, unseen)
        shared = wp5.summarize_seen_unseen(
            {"count": 4, "top1": .25, "top5": .75}, {"count": 2, "top1": .5, "top5": 1.})
        self.assertEqual(local["h1"], shared["harmonic_mean_top1"])
        self.assertEqual(local["h5"], wp5.harmonic_mean(.75, 1.))
        self.assertEqual(self.ns["harmonic_mean"](0., 0.), wp5.harmonic_mean(0., 0.))
        self.assertLess(local["h1"], 1.)

    def test_historical_source_guard_accepts_unchanged_crlf_identity(self):
        identity = self.ns["split_identity"](self.split)
        self.assertEqual(identity["source_sha256"], "9f1b729ab08f120c9e8d3a5681aa1ccbdf1d9210f9aaa6143959363dcb106a7d")
        record = {"split": {**identity, "candidate_ids": list(self.split.candidate_ids)},
                  "selected": {"epoch": 1, "purpose": "synthetic metadata compatibility fixture"}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run_summary.json"
            path.write_text(json.dumps(record))
            self.assertEqual(self.ns["read_wp2_reference"](path, identity, self.split.candidate_ids), record)

    def test_ascending_candidate_precondition_is_enforced(self):
        with self.assertRaises(ValueError):
            self.ns["rank_global_ids"](torch.ones(1, 2), (75, 10))


if __name__ == "__main__":
    unittest.main()
