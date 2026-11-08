"""Final-333 readiness regressions. Synthetic metadata only: no official CSV, image or model."""
import csv
import hashlib
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import numpy as np

from evaluation.confusion_artifacts import confusion_files
from evaluation.final_manifest import final_manifest, inference_order
from evaluation.model_adapters import MODEL_KINDS
from evaluation.reporting import (Candidate, CandidateSpec, EvaluationManifest, FINAL_NAMESPACE, RunProvenance,
                                  Sample, WP1_NAMESPACE, build_report, candidate_digest, fingerprint,
                                  records_from_scores)
from scripts.import_wp5_final import import_final
from test_development_import import synthetic_inputs, write_json

# Invented classes: 5 crops, one disease per class, disease 0 "healthy". Not the frozen labels.
CANDIDATES = tuple(Candidate(i, i % 5, i, f"crop{i % 5}", "healthy" if i == 0 else f"disease{i}", f"pair{i}")
                   for i in range(333))
SEEN_TRUTHS = (0, 1, 1, 2, 7, 258, 300, 322)       # training compositions (incl. dev-unseen ones)
UNSEEN_TRUTHS = (323, 324, 330, 332, 332)          # the 10 extra synthetic classes only


def development():
    dev = CANDIDATES[:323]
    spec = CandidateSpec("synthetic-development", "1", "development", dev, candidate_digest(dev, WP1_NAMESPACE),
                         WP1_NAMESPACE, fingerprint("synthetic training source"), fingerprint("synthetic split"),
                         "synthetic-split", "synthetic labels", (0,), synthetic=True)
    samples = (tuple(Sample(f"fit{i}.jpg", "fitting", i) for i in range(258))
               + (Sample("dev_s0.jpg", "seen", 0), Sample("dev_u0.jpg", "unseen", 300)))
    return EvaluationManifest(spec, samples, tuple(range(258)), tuple(range(258, 323)), 1, 1, "synthetic")


def write_csv(path, rows, *, bom=False):
    with path.open("w", newline="", encoding="utf-8-sig" if bom else "utf-8") as stream:
        csv.writer(stream, lineterminator="\r\n").writerows(rows)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows_for(prefix, truths):
    by_id = {c.composition_id: c for c in CANDIDATES}
    return [[f"{prefix}{n}.jpg", by_id[t].crop_id, by_id[t].disease_id, t, by_id[t].crop_name,
             by_id[t].disease_name, by_id[t].composition_name] for n, t in enumerate(truths)]


class FinalManifestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.seen_rows, self.unseen_rows = rows_for("ts", SEEN_TRUTHS), rows_for("tu", UNSEEN_TRUTHS)
        self.payload = {"schema": FINAL_NAMESPACE, "rows": [asdict(c) for c in CANDIDATES]}
        self.prompts = {"policy": {"version": "synthetic"}, "rows": [{"score_column": i} for i in range(333)]}

    def build(self, *, seen=None, unseen=None, bom=False, sha=None, seen_path=None, **kwargs):
        default_seen, unseen_path = self.root / "DPD_Small_test_seen.csv", self.root / "DPD_Small_test_unseen.csv"
        digests = {"seen": write_csv(default_seen, self.seen_rows if seen is None else seen, bom=bom),
                   "unseen": write_csv(unseen_path, self.unseen_rows if unseen is None else unseen)}
        if seen_path is not None:
            seen_path.write_bytes(default_seen.read_bytes())
        options = {"candidate_payload": self.payload, "prompt_payload": self.prompts,
                   "expected_sha256": sha or digests, "approval_reference": "SYNTHETIC policy fixture",
                   "evaluation_authorization": "SYNTHETIC authorization fixture", "synthetic": True, **kwargs}
        return final_manifest(development(), seen_path or default_seen, unseen_path, **options)

    def test_manifest_has_323_seen_by_333_columns_and_blind_order_has_no_truths(self):
        manifest = self.build()
        self.assertEqual((len(manifest.spec.ids), len(manifest.seen_truth_ids), manifest.unseen_truth_ids),
                         (333, 323, tuple(range(323, 333))))
        self.assertEqual((manifest.expected_seen, manifest.expected_unseen), (8, 5))
        self.assertEqual(sum(s.fold == "fitting" for s in manifest.samples), 258)
        order = inference_order(manifest)
        self.assertEqual(order["folds"]["seen"], [r[0] for r in self.seen_rows])
        self.assertEqual(order["split_sha256"], manifest.sha256)
        text = json.dumps(order)
        self.assertNotIn("truth_id", text)
        self.assertNotIn("pair1", text)  # no class labels either

    def test_rejections_report_counts_without_official_filenames_or_labels(self):
        bad_label = [row[:] for row in self.seen_rows]
        bad_label[2][5] = "renamed-disease"
        header = [["image", "crop_id", "disease_id", "composition_id", "crop", "disease", "pair"]] + self.seen_rows
        cases = {"inconsistent": dict(seen=bad_label), "malformed": dict(seen=header),
                 "duplicate": dict(seen=self.seen_rows + self.seen_rows[:1]),
                 "membership": dict(seen=rows_for("ts", (0, 325)))}
        for name, kwargs in cases.items():
            with self.subTest(name), self.assertRaises(ValueError) as caught:
                self.build(**kwargs)
            self.assertNotIn("ts", str(caught.exception))
            self.assertNotIn("renamed", str(caught.exception))
        with self.assertRaisesRegex(ValueError, "Duplicate or missing manifest sample"):
            self.build(seen=[["fit3.jpg", *self.seen_rows[0][1:]]])
        with self.assertRaisesRegex(ValueError, "raw SHA-256 differs"):
            self.build(sha={"seen": "0" * 64, "unseen": "0" * 64})
        with self.assertRaisesRegex(ValueError, "authorization"):
            self.build(evaluation_authorization=" ")
        with self.assertRaisesRegex(ValueError, "Synthetic flag"):
            self.build(synthetic=False)
        with self.assertRaisesRegex(ValueError, "Expected DPD_Small_test_seen.csv"):
            self.build(seen_path=self.root / "renamed.csv")

    def test_utf8_bom_is_stripped_and_recorded(self):
        manifest = self.build(bom=True)
        self.assertEqual(manifest.samples[258].sample_id, "ts0.jpg")
        self.assertTrue(json.loads(manifest.identity_source)["official_sources"][0]["utf8_bom_stripped"])

    def test_confusion_artifacts_keep_counts_and_allow_unseen_only_predictions(self):
        manifest = self.build()
        provenance = RunProvenance("m", "r", "c", fingerprint("c"), "p", "v", "e", "s", "float64", "budget")
        samples = [s for s in manifest.samples if s.fold != "fitting"]
        scores = np.zeros((len(samples), 333))
        for row, sample in enumerate(samples):
            scores[row, sample.truth_id] = 1.0
        scores[1, 1], scores[1, 330] = 0.0, 2.0     # seen truth 1 -> unseen-only class 330
        scores[2, 1], scores[2, 330] = 0.0, 2.0
        scores[4, 7], scores[4, 2] = 0.0, 2.0       # seen truth 7 -> class 2 (same crop)
        report = build_report(manifest, records_from_scores(manifest.spec, provenance, samples, scores,
                                                            manifest.spec.ids))
        files = confusion_files(report, fold="seen")
        self.assertTrue(files["seen_confusion_heatmap.png"].startswith(b"\x89PNG"))
        top = list(csv.DictReader(files["seen_confusion_top_pairs.csv"].decode().splitlines()))
        self.assertEqual([(r["truth_id"], r["predicted_id"], r["count"]) for r in top],
                         [("1", "330", "2"), ("7", "2", "1")])
        self.assertEqual((top[0]["prediction_in_truth_axis"], top[1]["same_crop"]), ("False", "True"))
        crop = list(csv.DictReader(files["seen_confusion_crop_level.csv"].decode().splitlines()))
        self.assertEqual(sum(int(r["count"]) for r in crop), len(SEEN_TRUTHS))


class FinalImportTests(unittest.TestCase):
    def setUp(self):
        helper = FinalManifestTests("test_utf8_bom_is_stripped_and_recorded")
        helper.setUp()
        self.addCleanup(helper.temp.cleanup)
        self.root, self.unseen_rows = helper.root, helper.unseen_rows
        self.manifest = helper.build()
        provenances, packets = synthetic_inputs(self.manifest)
        for kind in MODEL_KINDS:
            for packet in packets[kind]:
                del packet["ground_truth_ids"]  # blind export: owners never hold official labels
        self.packets = packets
        self.bundle = {"schema": "wp5-final-owner-handoff-v1",
                       "final_policy_reference": "SYNTHETIC Issue #8 fixture",
                       "evaluation_authorization_reference": "SYNTHETIC authorization fixture",
                       "owner_approval_reference": "SYNTHETIC owner fixture",
                       "identity_manifest": write_json(self.root / "final_manifest.json", self.manifest.to_dict()),
                       "development_manifest": write_json(self.root / "dev_manifest.json", development().to_dict()),
                       "final_candidates": write_json(self.root / "candidates.json", helper.payload),
                       "final_prompts": write_json(self.root / "prompts.json", helper.prompts),
                       "official_csv_sha256": {f: hashlib.sha256((self.root / n).read_bytes()).hexdigest()
                                               for f, n in (("seen", "DPD_Small_test_seen.csv"),
                                                            ("unseen", "DPD_Small_test_unseen.csv"))},
                       "manifest_sha256": self.manifest.sha256,
                       "models": {kind: {"provenance": asdict(provenances[kind]),
                                         "packet_files": [write_json(self.root / f"{kind}-{i}.json", packet)
                                                          for i, packet in enumerate(packets[kind])]}
                                  for kind in MODEL_KINDS}}
        self.output = self.root / "final-results"

    def run_import(self, **kwargs):
        entry = write_json(self.root / "handoff.json", self.bundle)
        return import_final(self.root / "handoff.json", entry["file_sha256"], self.output,
                            seen_csv=self.root / "DPD_Small_test_seen.csv",
                            unseen_csv=self.root / "DPD_Small_test_unseen.csv", **{"allow_synthetic": True, **kwargs})

    def test_end_to_end_blind_packets_reports_and_confusion_outputs(self):
        receipt = self.run_import()
        self.assertEqual(receipt["task_mode"], "final")
        self.assertEqual(receipt["blind_packets_truth_attached"], {kind: 2 for kind in MODEL_KINDS})
        for kind in MODEL_KINDS:
            report = json.loads((self.output / kind / "report.json").read_text(encoding="utf-8"))
            matrix = report["seen"]["confusion_matrix"]
            self.assertEqual((len(matrix["truth_ids"]), len(matrix["prediction_ids"]), matrix["count"]), (323, 333, 8))
            self.assertTrue((self.output / "confusion" / kind / "seen_confusion_heatmap.png").is_file())
        self.assertIn("SYNTHETIC", (self.output / "caption.txt").read_text(encoding="utf-8"))
        self.assertTrue((self.output / "receipt.json").is_file())

    def test_gates_reject_before_any_output(self):
        for key in ("evaluation_authorization_reference", "final_policy_reference"):
            with self.subTest(key):
                saved, self.bundle[key] = self.bundle[key], ""
                with self.assertRaisesRegex(ValueError, key):
                    self.run_import()
                self.bundle[key] = saved
        with self.assertRaisesRegex(ValueError, "allow-synthetic"):
            self.run_import(allow_synthetic=False)
        write_csv(self.root / "DPD_Small_test_unseen.csv", self.unseen_rows[:-1])  # tampered after approval
        with self.assertRaisesRegex(ValueError, "raw SHA-256 differs"):
            self.run_import()
        self.assertFalse(self.output.exists())

    def replace_packet(self, kind, index, packet):
        self.bundle["models"][kind]["packet_files"][index] = write_json(self.root / f"{kind}-{index}.json", packet)

    def assert_rejected(self, pattern):
        with self.assertRaisesRegex(ValueError, pattern) as caught:
            self.run_import()
        self.assertFalse(self.output.exists())
        # Synthetic class names (pairN, diseaseN, cropN) and truth lists never appear in errors.
        self.assertNotRegex(str(caught.exception), r"(pair|disease|crop)[0-9]|ground_truth_ids\s*[:=]")

    def test_packets_carrying_labels_are_rejected(self):
        kind = MODEL_KINDS[2]
        packet = dict(self.packets[kind][0])
        packet["ground_truth_ids"] = [s.truth_id for s in self.manifest.samples if s.fold == "seen"]
        self.replace_packet(kind, 0, packet)
        self.assert_rejected("must be blind")

    def test_incomplete_or_mismatched_packets_are_rejected(self):
        kind = MODEL_KINDS[1]
        original = self.packets[kind][0]
        cases = {
            "missing image": ({**original, "image_names": original["image_names"][:-1],
                               "scores": original["scores"][:-1]}, "Missing 1 expected prediction row"),
            "outside manifest": ({**original, "image_names": ["other.jpg", *original["image_names"][1:]]},
                                 "outside the final manifest"),
            "duplicate image": ({**original,
                                 "image_names": [original["image_names"][0], *original["image_names"][:-1]]},
                                "duplicate"),
            "manifest digest": ({**original, "manifest_sha256": "0" * 64, "split_sha256": "0" * 64}, "manifest_sha256"),
            "candidate digest": ({**original, "candidate_sha256": "0" * 64}, "candidate digest"),
            "candidate order": ({**original, "candidate_ids": original["candidate_ids"][::-1]}, "candidate columns"),
        }
        for name, (packet, pattern) in cases.items():
            with self.subTest(name):
                self.replace_packet(kind, 0, packet)
                self.assert_rejected(pattern)
        self.replace_packet(kind, 0, original)
        self.bundle["models"][kind]["packet_files"] = self.bundle["models"][kind]["packet_files"][:1]  # no unseen
        self.assert_rejected("Missing 5 expected prediction rows")
        del self.bundle["models"][kind]
        self.assert_rejected("all four selected models")

    def test_development_manifest_is_not_accepted_as_final(self):
        self.bundle["identity_manifest"] = self.bundle["development_manifest"]
        self.bundle["manifest_sha256"] = development().sha256
        with self.assertRaisesRegex(ValueError, "final-333 only"):
            self.run_import()
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
