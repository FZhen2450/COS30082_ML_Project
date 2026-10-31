"""Regression tests for training-only protocol and provenance failure modes."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

from shared_protocol import FOLDS, build_protocol, digest, validate_protocol, write_new_protocol

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "DPD_Small_train.csv"


class ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = build_protocol(SOURCE)

    def test_training_only_counts_and_component_coverage(self):
        folds = validate_protocol(self.protocol, SOURCE)
        self.assertEqual([len(f) for f in folds], [4868, 1238, 1894])
        self.assertEqual([f.composition_id.nunique() for f in folds], [258, 225, 65])
        self.assertEqual([folds[0].crop_id.nunique(), folds[0].disease_id.nunique()], [55, 175])
        self.assertEqual(len(self.protocol["candidates"]), 323)
        self.assertEqual(self.protocol["historical_reconciliation"], "pending_ordered_filename_reference")

    def test_deterministic_export_and_global_candidate_order(self):
        self.assertEqual(self.protocol, build_protocol(SOURCE))
        ids = [row["composition_id"] for row in self.protocol["candidates"]]
        self.assertEqual(ids, sorted(set(ids)))
        self.assertGreater(max(ids), 322)

    def test_reference_requires_exact_ordered_filenames(self):
        reference = {key: self.protocol[key] for key in FOLDS}
        p = build_protocol(SOURCE, reference=reference)
        self.assertEqual(p["historical_reconciliation"], "ordered_filenames_match_supplied_reference")
        bad = copy.deepcopy(reference)
        bad[FOLDS[0]].reverse()
        with self.assertRaisesRegex(ValueError, "Historical reference differs"):
            build_protocol(SOURCE, reference=bad)

    def test_rejects_candidate_reordering(self):
        p = copy.deepcopy(self.protocol)
        p["candidates"].reverse()
        p["candidate_sha256"] = digest(p["candidates"])
        with self.assertRaisesRegex(ValueError, "Candidate order"):
            validate_protocol(p, SOURCE)

    def test_rejects_filename_overlap_even_with_rehashed_payload(self):
        p = copy.deepcopy(self.protocol)
        p[FOLDS[1]][0] = p[FOLDS[0]][0]
        p["split_sha256"] = digest({key: p[key] for key in FOLDS})
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_protocol(p, SOURCE)

    def test_rejects_source_byte_changes(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "source.csv"
            path.write_bytes(SOURCE.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "Source-byte hash"):
                validate_protocol(self.protocol, path)

    def test_export_preserves_existing_different_manifest(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "protocol.json"
            write_new_protocol(self.protocol, path)
            original = path.read_bytes()
            write_new_protocol(self.protocol, path)
            changed = copy.deepcopy(self.protocol)
            changed["seed"] = 7
            with self.assertRaises(FileExistsError):
                write_new_protocol(changed, path)
            self.assertEqual(path.read_bytes(), original)

    def test_hash_namespace_is_separate_from_historical_issue_hashes(self):
        self.assertEqual(digest(self.protocol["candidates"]), self.protocol["candidate_sha256"])
        self.assertNotEqual(self.protocol["candidate_sha256"],
                            "4cb0a24a96cf647f32311817fa7b082a46d3cf00c20ae61f4287e83cc898dfaf")


if __name__ == "__main__":
    unittest.main()
