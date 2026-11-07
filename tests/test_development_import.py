"""File import and presentation regressions using explicitly synthetic predictions."""
import csv
import hashlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from dataclasses import FrozenInstanceError, asdict, replace
from itertools import combinations
from pathlib import Path
from unittest.mock import patch

import numpy as np

from evaluation.model_adapters import MODEL_KINDS
from evaluation.reporting import Sample, development_manifest
from evaluation.result_artifacts import VerifiedResults, presentation_files, result_scope
from scripts.import_wp5_development import ROOT, checked_json, import_development, validate_development_source
from shared_protocol import build_protocol
from test_evaluation_reporting import fixture
from test_model_adapters import decoded_packet, selected_packet, sized_manifest


def synthetic_inputs(manifest):
    provenance = fixture()[1]
    provenances, packets = {}, {}
    for kind in MODEL_KINDS:
        packets[kind] = []
        for fold in ("seen", "unseen"):
            samples = [s for s in manifest.samples if s.fold == fold]
            scores = np.zeros((len(samples), len(manifest.spec.ids)), dtype=np.float64)
            # All ties exercise original ascending ID order; not empirical scores.
            adapter, packet = selected_packet(manifest, provenance, scores, kind, fold)
            provenances[kind] = adapter.provenance
            packets[kind].append(decoded_packet(packet))
    return provenances, packets


def write_json(path, value):
    raw = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
    path.write_bytes(raw)
    return {"path": path.name, "file_sha256": hashlib.sha256(raw).hexdigest()}


class DevelopmentImportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / "new-results"
        self.manifest = sized_manifest(323)
        self.provenances, self.packets = synthetic_inputs(self.manifest)
        self.bundle = {"schema": "wp5-owner-handoff-v1",
                       "owner_approval_reference": "SYNTHETIC fixture authorization only",
                       "identity_manifest": write_json(self.root / "identities.json", self.manifest.to_dict()),
                       "manifest_sha256": self.manifest.sha256, "models": {}}
        for kind in MODEL_KINDS:
            self.bundle["models"][kind] = {"provenance": asdict(self.provenances[kind]),
                "packet_files": [write_json(self.root / f"{kind}-{i}.json", packet)
                                 for i, packet in enumerate(self.packets[kind])]}
        self.handoff = self.root / "handoff.json"

    def run_import(self, **kwargs):
        entry = write_json(self.handoff, self.bundle)
        return import_development(self.handoff, entry["file_sha256"], self.output, allow_synthetic=True, **kwargs)

    def reject(self, pattern, **kwargs):
        with self.assertRaisesRegex((ValueError, TypeError, KeyError, FileNotFoundError), pattern):
            self.run_import(**kwargs)
        self.assertFalse(self.output.exists())

    def update_packet(self, kind, index, packet):
        entry = self.bundle["models"][kind]["packet_files"][index]
        entry.update(write_json(self.root / entry["path"], packet))

    def test_full_import_four_reports_six_pairs_and_hash_receipt(self):
        receipt = self.run_import()
        self.assertTrue(receipt["synthetic"])
        self.assertEqual(receipt["fold_counts"], {"fitting": 258, "seen": 2, "unseen": 2})
        self.assertEqual(len(receipt["verified_inputs"]), 10)
        for kind in MODEL_KINDS:
            report = json.loads((self.output / kind / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["seen"]["correct_top1"], 1)
            self.assertEqual(report["seen"]["expected"], 2)
            self.assertEqual(report["unseen"]["correct_top5"], 0)
        for a, b in combinations(MODEL_KINDS, 2):
            self.assertTrue((self.output / "paired" / (a + "__" + b) / "report.json").is_file())
        files = {p.relative_to(self.output).as_posix() for p in self.output.rglob("*") if p.is_file()}
        self.assertEqual({r["path"] for r in receipt["outputs"]}, files - {"receipt.json"})
        for row in receipt["outputs"]:
            self.assertEqual(hashlib.sha256((self.output / row["path"]).read_bytes()).hexdigest(), row["file_sha256"])
        self.assertEqual(json.loads((self.output / "receipt.json").read_text()), receipt)
        before = (self.output / "receipt.json").read_bytes()
        with self.assertRaises(FileExistsError):
            self.run_import()
        self.assertEqual((self.output / "receipt.json").read_bytes(), before)

    def test_cli_from_another_directory_uses_handoff_relative_paths(self):
        entry = write_json(self.handoff, self.bundle)
        result = subprocess.run([sys.executable, "-B", str(ROOT / "scripts/import_wp5_development.py"),
                                 "--handoff", str(self.handoff), "--handoff-sha256", entry["file_sha256"],
                                 "--output", str(self.output), "--allow-synthetic"],
                                cwd=self.root, capture_output=True, text=True, timeout=90)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("synthetic=True", result.stdout)
        self.assertTrue((self.output / "receipt.json").is_file())

    def test_wrong_outer_hash_rejected_before_decoding(self):
        self.handoff.write_text("not JSON")
        with self.assertRaisesRegex(ValueError, "raw file SHA-256"):
            import_development(self.handoff, "0" * 64, self.output)
        self.assertFalse(self.output.exists())

    def test_all_packet_raw_hashes_required(self):
        kind = MODEL_KINDS[-1]
        self.bundle["models"][kind]["packet_files"][-1]["file_sha256"] = "0" * 64
        self.reject("raw file SHA-256")

    def test_missing_approval_and_nonstring_approval_rejected(self):
        for approval in (None, "", [], 1):
            self.bundle["owner_approval_reference"] = approval
            self.reject("approval reference")

    def test_synthetic_requires_explicit_opt_in(self):
        entry = write_json(self.handoff, self.bundle)
        with self.assertRaisesRegex(ValueError, "allow-synthetic"):
            import_development(self.handoff, entry["file_sha256"], self.output)
        self.assertFalse(self.output.exists())

    def test_semantic_manifest_hash_cannot_be_raw_file_hash(self):
        self.bundle["manifest_sha256"] = self.bundle["identity_manifest"]["file_sha256"]
        self.reject("semantic manifest")

    def test_all_four_models_required(self):
        del self.bundle["models"][MODEL_KINDS[-1]]
        self.reject("all four")

    def test_final_manifest_rejected_before_packet_files_are_opened(self):
        self.bundle["identity_manifest"] = write_json(self.root / "identities.json", sized_manifest(333).to_dict())
        with patch("scripts.import_wp5_development.checked_json", wraps=checked_json) as reads:
            self.reject("development-323 only")
        self.assertEqual(reads.call_count, 2)

    def test_small_candidate_fixture_rejected(self):
        self.bundle["identity_manifest"] = write_json(self.root / "identities.json", fixture()[0].to_dict())
        self.reject("development-323 only")

    def test_missing_packet_file_rejected(self):
        self.bundle["models"][MODEL_KINDS[0]]["packet_files"][0]["path"] = "absent.json"
        self.reject("absent.json")

    def test_duplicate_records_rejected_without_partial_output(self):
        files = self.bundle["models"][MODEL_KINDS[-1]]["packet_files"]
        files.append(files[0])
        self.reject("Duplicate|duplicate")

    def test_incomplete_coverage_rejected(self):
        self.bundle["models"][MODEL_KINDS[-1]]["packet_files"].pop()
        self.reject("Missing|missing|coverage")

    def test_wrong_original_filename_and_truth_rejected(self):
        kind = MODEL_KINDS[0]
        for field, value in (("image_names", ["wrong-name", "s1"]), ("ground_truth_ids", [1, 0])):
            packet = dict(self.packets[kind][0], **{field: value})
            self.update_packet(kind, 0, packet)
            self.reject("identity, truth or fold")

    def test_candidate_order_and_comparison_budget_rejected(self):
        kind = MODEL_KINDS[0]
        packet = dict(self.packets[kind][0], candidate_ids=list(reversed(self.manifest.spec.ids)))
        self.update_packet(kind, 0, packet)
        self.reject("candidate columns")
        self.update_packet(kind, 0, self.packets[kind][0])
        self.bundle["models"][kind]["provenance"]["comparison_budget"] = "different budget"
        self.reject("comparison budgets")

    def test_json_duplicate_fields_and_nonfinite_constants_rejected(self):
        for raw in (b'{"schema":1,"schema":2}', b'{"score":NaN}', b'{"score":Infinity}'):
            self.handoff.write_bytes(raw)
            with self.assertRaises(ValueError):
                checked_json(self.handoff, hashlib.sha256(raw).hexdigest(), [], "test")

    def test_plot_failure_leaves_no_output(self):
        with patch("scripts.import_wp5_development.presentation_files", side_effect=ValueError("plot failed")):
            self.reject("plot failed")

    def test_exact_declared_precision_is_checked_through_file_boundary(self):
        kind = MODEL_KINDS[0]
        self.bundle["models"][kind]["provenance"]["score_precision"] = "float32"
        for i, original in enumerate(self.packets[kind]):
            packet = dict(original, score_precision="float32")
            if i == 0:
                packet["scores"][0][1] = float(np.nextafter(1., np.inf))
            self.update_packet(kind, i, packet)
        self.reject("exactly representable as float32")

    def test_existing_file_output_is_preserved_before_handoff_is_read(self):
        self.output.write_bytes(b"historical result")
        with self.assertRaises(FileExistsError):
            import_development(self.root / "absent-handoff.json", "0" * 64, self.output)
        self.assertEqual(self.output.read_bytes(), b"historical result")


class SourceValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = ROOT / "DPD_Small_train.csv"
        protocol = build_protocol(cls.source)
        reference = {key: protocol[key] for key in
                     ("fitting_images", "seen_validation_images", "unseen_validation_images")}
        # Test-only reconciliation of the deterministic source, no approval file written.
        protocol = build_protocol(cls.source, reference=reference)
        cls.manifest = development_manifest(protocol, cls.source, healthy_disease_names=[])

    def test_training_only_real_development_manifest_exact_join(self):
        result = validate_development_source(self.manifest, self.source)
        self.assertEqual(result["file_sha256"], self.manifest.spec.source_sha256)
        self.assertEqual(len(self.manifest.spec.ids), 323)
        self.assertEqual((self.manifest.expected_seen, self.manifest.expected_unseen), (1238, 1894))

    def test_fold_order_truth_and_split_hash_tampering_rejected(self):
        samples = list(self.manifest.samples)
        samples[0], samples[1] = samples[1], samples[0]
        with self.assertRaisesRegex(ValueError, "ordered filename fold"):
            validate_development_source(replace(self.manifest, samples=tuple(samples)), self.source)
        samples = list(self.manifest.samples)
        sample = next(s for s in samples if s.fold == "seen")
        other = next(s for s in samples if s.fold == "seen" and s.truth_id != sample.truth_id)
        samples[samples.index(sample)] = Sample(sample.sample_id, sample.fold, other.truth_id)
        with self.assertRaisesRegex(ValueError, "ground-truth join"):
            validate_development_source(replace(self.manifest, samples=tuple(samples)), self.source)
        wrong = replace(self.manifest, spec=replace(self.manifest.spec, split_sha256="0" * 64))
        with self.assertRaisesRegex(ValueError, "split fingerprint"):
            validate_development_source(wrong, self.source)

    def test_official_metadata_path_rejected_before_source_open(self):
        with patch("shared_protocol.build_protocol", side_effect=AssertionError("must not read")):
            with self.assertRaisesRegex(ValueError, "training metadata"):
                validate_development_source(self.manifest, ROOT / "DPD_Small_test_seen.csv")


class PresentationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        manifest = fixture()[0]
        provenances, packets = synthetic_inputs(manifest)
        cls.verified = VerifiedResults.from_packets(manifest, provenances, packets)
        cls.files = presentation_files(cls.verified)

    def test_only_verified_immutable_results_accepted(self):
        with self.assertRaises(TypeError):
            VerifiedResults()
        with self.assertRaises(TypeError):
            presentation_files(self.verified.to_dict())
        with self.assertRaises(FrozenInstanceError):
            self.verified._json = "{}"
        copied = self.verified.to_dict()
        copied["models"].clear()
        self.assertEqual(len(self.verified.to_dict()["models"]), 4)

    def test_comparison_values_denominators_and_synthetic_captions(self):
        rows = list(csv.DictReader(io.StringIO(self.files["comparison.csv"].decode())))
        self.assertEqual(len(rows), 4)
        for row in rows:
            self.assertEqual((row["seen_n"], row["unseen_n"]), ("2", "2"))
            self.assertEqual((float(row["seen_top1"]), float(row["unseen_top1"])), (.5, 0))
            self.assertEqual(float(row["H@1"]), 0)
            self.assertIn("SYNTHETIC", row["scope"])
            self.assertEqual(row["results_sha256"], self.verified.sha256)
        self.assertIn(b"SYNTHETIC", self.files["comparison.md"])
        self.assertIn(b"Seen denominator: 2; unseen denominator: 2", self.files["caption.txt"])
        self.assertTrue(self.files["comparison.png"].startswith(b"\x89PNG\r\n\x1a\n"))

    def test_paired_wins_and_health_class_error_denominators(self):
        def rows(name):
            return list(csv.DictReader(io.StringIO(self.files[name].decode())))
        pairs = rows("paired_wins.csv")
        self.assertEqual(len(pairs), 24)
        for row in pairs:
            self.assertEqual(int(row["a_wins"]) + int(row["b_wins"]) + int(row["both_correct"]) +
                             int(row["neither_correct"]), int(row["denominator"]))
            self.assertEqual((row["a_wins"], row["b_wins"]), ("0", "0"))
        for name in ("health_errors.csv", "per_class_errors.csv"):
            for row in rows(name):
                for k in (1, 5):
                    self.assertLessEqual(int(row["errors_top" + str(k)]), int(row["denominator"]))
        self.assertTrue(any(row["denominator"] == "0" for row in rows("per_class_errors.csv")))

    def test_presentation_is_reproducible(self):
        self.assertEqual(presentation_files(self.verified), self.files)

    def test_scope_never_confuses_synthetic_development_or_official(self):
        self.assertIn("SYNTHETIC", result_scope({"synthetic": True, "task_mode": "final"}))
        self.assertIn("TRAINING-DERIVED", result_scope({"synthetic": False, "task_mode": "development"}))
        self.assertIn("OFFICIAL TEST", result_scope({"synthetic": False, "task_mode": "final"}))

    def test_verified_rank_only_packets_preserve_ties_on_both_candidate_interfaces(self):
        for count in (323, 333):
            manifest = sized_manifest(count)
            provenances, packets = synthetic_inputs(manifest)
            for batches in packets.values():
                for packet in batches:
                    del packet["scores"]
                    packet["available_k"] = 5
                    packet["ranked_ids"] = [list(manifest.spec.ids[:5]) for _ in packet["image_names"]]
            result = VerifiedResults.from_packets(manifest, provenances, packets).to_dict()
            self.assertEqual(len(result["paired"]), 6)
            for report in result["models"].values():
                self.assertEqual(report["seen"]["correct_top1"], 1)
                self.assertEqual(report["seen"]["correct_top5"], 2)
                self.assertTrue(report["candidate_spec"]["synthetic"])


if __name__ == "__main__":
    unittest.main()
