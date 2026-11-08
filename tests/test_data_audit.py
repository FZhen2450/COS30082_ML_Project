"""Unit tests for Issue #7 development data and augmentation audit.

Ensures CPU-only execution, synthetic fixtures, and strict isolation of official test data.
"""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from PIL import Image

from scripts.audit_development_data import (
    audit_augmentation_policy,
    audit_bounded_near_duplicates,
    audit_file_duplicates,
    audit_metadata_partitions,
    compute_average_hash,
    generate_full_audit,
    verify_source_safety,
)


class DataAuditTests(unittest.TestCase):
    def test_official_test_source_is_strictly_rejected(self):
        """Official test CSVs must trigger PermissionError."""
        for forbidden in ["DPD_Small_test_seen.csv", "DPD_Small_test_unseen.csv", "test_data.csv"]:
            with self.assertRaises(PermissionError):
                verify_source_safety(forbidden)

    def test_audit_metadata_partitions_real_protocol(self):
        """Audit the real checked-in training protocol against DPD_Small_train.csv."""
        protocol_path = Path("artifacts/data/development_seed42_wp1_v1.json")
        source_csv = Path("DPD_Small_train.csv")
        if not protocol_path.is_file() or not source_csv.is_file():
            self.skipTest("Checked-in protocol or train CSV missing.")

        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        summary = audit_metadata_partitions(protocol, source_csv)

        self.assertEqual(summary["auditor"], "Andy Wei Jian Chu (@AA-49)")
        counts = summary["partition_counts"]
        self.assertEqual(counts["total_development_images"], 8000)
        self.assertEqual(counts["fitting_images"], 4868)
        self.assertEqual(counts["seen_validation_images"], 1238)
        self.assertEqual(counts["unseen_validation_images"], 1894)

        comps = summary["composition_support"]
        self.assertEqual(comps["fitting_compositions"], 258)
        self.assertEqual(comps["unseen_val_heldout_compositions"], 65)
        self.assertTrue(summary["component_coverage"]["all_components_retained"])

    def test_duplicate_file_detection_synthetic(self):
        """Synthetic test checking intra-fold and cross-fold duplicate detection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            img1 = tmppath / "img1.jpg"
            img2 = tmppath / "img2.jpg"
            img3 = tmppath / "img3.jpg"

            content_a = b"dummy_content_alpha"
            content_b = b"dummy_content_beta"

            img1.write_bytes(content_a)
            img2.write_bytes(content_a)
            img3.write_bytes(content_b)

            protocol = {
                "fitting_images": ["img1.jpg"],
                "seen_validation_images": [],
                "unseen_validation_images": ["img2.jpg", "img3.jpg"],
            }

            res = audit_file_duplicates(protocol, tmppath)
            self.assertEqual(res["status"], "completed")
            self.assertEqual(res["duplicate_groups_total"], 1)
            self.assertTrue(res["fold_leakage_detected"])

    def test_bounded_near_duplicate_screening_synthetic(self):
        """Synthetic test checking bounded near-duplicate detection with aHash."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            # Create two nearly identical images (all white, vs white with 1 grey pixel)
            im1 = Image.new("L", (16, 16), color=255)
            im2 = Image.new("L", (16, 16), color=255)
            im2.putpixel((0, 0), 200)
            im3 = Image.new("L", (16, 16), color=0)  # Completely black

            im1.save(tmppath / "leaf1.png")
            im2.save(tmppath / "leaf2.png")
            im3.save(tmppath / "leaf3.png")

            protocol = {
                "fitting_images": ["leaf1.png"],
                "seen_validation_images": ["leaf2.png"],
                "unseen_validation_images": ["leaf3.png"],
            }

            res = audit_bounded_near_duplicates(protocol, tmppath, max_hamming_distance=2)
            self.assertEqual(res["status"], "completed")
            self.assertGreaterEqual(res["total_candidate_pairs"], 1)
            pair = res["candidate_near_duplicate_pairs"][0]
            self.assertTrue(pair["review_candidate_only"])
            self.assertTrue(pair["human_confirmation_required"])

    def test_augmentation_policy_review_status(self):
        """Test augmentation audit records human approval."""
        res = audit_augmentation_policy()
        self.assertEqual(res["human_review_status"], "approved")
        self.assertEqual(res["human_reviewer"], "Andy Wei Jian Chu (@AA-49)")

    def test_full_audit_generation(self):
        """Verify full audit report JSON generation."""
        protocol_path = Path("artifacts/data/development_seed42_wp1_v1.json")
        source_csv = Path("DPD_Small_train.csv")
        if not protocol_path.is_file() or not source_csv.is_file():
            self.skipTest("Missing base artifacts.")

        report = generate_full_audit(protocol_path, source_csv)
        self.assertIn("partition_counts", report)
        self.assertIn("file_duplicate_audit", report)
        self.assertIn("bounded_near_duplicate_audit", report)
        self.assertIn("augmentation_audit", report)
        self.assertIn("proposed_robustness_evaluation", report)

    def test_unsafe_and_redirected_image_paths_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);images=root/'images';images.mkdir();outside=root/'outside.jpg';outside.write_bytes(b'private')
            for name in ('../outside.jpg',str(outside),'a\\outside.jpg'):
                protocol={'fitting_images':[name],'seen_validation_images':[],'unseen_validation_images':[]}
                with self.subTest(name=name),self.assertRaises((ValueError,PermissionError)):
                    audit_file_duplicates(protocol,images)
            (images/'redirect.jpg').symlink_to(outside)
            protocol={'fitting_images':['redirect.jpg'],'seen_validation_images':[],'unseen_validation_images':[]}
            with self.assertRaises((ValueError,PermissionError)):audit_file_duplicates(protocol,images)
    def test_near_duplicate_failures_are_reported_without_claiming_complete_decode(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'broken.jpg').write_bytes(b'not an image')
            protocol={'fitting_images':['broken.jpg','missing.jpg'],'seen_validation_images':[],'unseen_validation_images':[]}
            result=audit_bounded_near_duplicates(protocol,root)
            self.assertEqual(result.get('unreadable_files'),['broken.jpg','missing.jpg'])
            self.assertEqual(result.get('successfully_hashed_images'),0)
    def test_preview_approval_does_not_claim_production_transform_equivalence(self):
        result=audit_augmentation_policy()
        self.assertIs(result.get('production_transform_equivalence_verified'),False)
        self.assertEqual(result.get('preview_resolution'),192)


if __name__ == "__main__":
    unittest.main()
