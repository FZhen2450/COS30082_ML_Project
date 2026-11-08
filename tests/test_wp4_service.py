"""Verify retrieval fold boundaries and global-ID presentation without models."""
import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from shared_protocol import FOLDS, candidate_table, digest
from data_utils import load_metadata
from wp4.service import Catalog, open_rgb, present_prediction


class Array:
    def __init__(self, values):
        self.values = values

    def tolist(self):
        return self.values


class WP4ServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        rows = []
        for i, (cid, crop, disease) in enumerate([(0, 0, 0), (0, 0, 0), (4, 1, 1),
                                                  (4, 1, 1), (9, 1, 0), (9, 1, 0)]):
            name = f"{i}.png"
            Image.new("RGB", (16, 16), (i * 35, 80, 50)).save(self.root / name)
            rows.append([name, crop, disease, cid, f"crop{crop}", f"disease{disease}", f"pair{cid}"])
        self.source = self.root / "train.csv"
        with self.source.open("w", newline="", encoding="utf-8") as stream:
            csv.writer(stream).writerows(rows)
        p = {"schema": "wp1-development-v1", "source_sha256": hashlib.sha256(self.source.read_bytes()).hexdigest(),
             "historical_reconciliation": "pending_ordered_filename_reference", "counts": [2, 2, 2],
             "candidates": candidate_table(load_metadata(self.source)),
             FOLDS[0]: ["0.png", "2.png"], FOLDS[1]: ["1.png", "3.png"], FOLDS[2]: ["4.png", "5.png"]}
        p["candidate_sha256"] = digest(p["candidates"])
        p["split_sha256"] = digest({key: p[key] for key in FOLDS})
        self.protocol = self.root / "protocol.json"
        self.protocol.write_text(json.dumps(p), encoding="utf-8")
        self.catalog = Catalog(self.protocol, self.source, self.root)

    def test_exact_and_fallback_references_use_fitting_only(self):
        exact = self.catalog.references(0)
        self.assertEqual([r["filename"] for r in exact["items"]], ["0.png"])
        fallback = self.catalog.references(9)
        self.assertEqual([r["filename"] for r in fallback["items"]], ["0.png"])
        self.assertEqual(fallback["relation"], "Same disease; different crop")
        self.assertFalse({"1.png", "3.png", "4.png", "5.png"} &
                         {r["filename"] for r in exact["items"] + fallback["items"]})

    def test_query_itself_is_excluded_even_when_copied(self):
        image = open_rgb(self.root / "0.png")
        image.save(self.root / "upload.png")
        found = self.catalog.references(0, query=open_rgb(self.root / "upload.png"))
        self.assertEqual(found["items"], [])

    def test_missing_reference_reported_and_path_escape_rejected(self):
        (self.root / "0.png").unlink()
        found = self.catalog.references(0)
        self.assertEqual(found["unavailable"], 1)
        self.assertEqual(found["items"], [])
        with self.assertRaisesRegex(ValueError, "inside"):
            self.catalog.image_path("../outside.jpg")

    def test_global_ids_and_ties_not_column_indices(self):
        result = {"candidate_ids": Array([0, 4, 9]), "scores": Array([[0., 2., 2.]]),
                  "candidate_sha256": self.catalog.protocol["candidate_sha256"]}
        rows = present_prediction(result, self.catalog)
        self.assertEqual([r["composition_id"] for r in rows], [4, 9, 0])
        self.assertEqual(rows[1]["crop_name"], "crop1")

    def test_mismatch_and_invalid_scores_rejected(self):
        result = {"candidate_ids": Array([0, 4, 9]), "scores": Array([[0., 2., 2.]]),
                  "candidate_sha256": "wrong"}
        with self.assertRaisesRegex(ValueError, "fingerprints"):
            present_prediction(result, self.catalog)
        result["candidate_sha256"] = self.catalog.protocol["candidate_sha256"]
        result["scores"] = Array([[float("nan"), 0., 1.]])
        with self.assertRaisesRegex(ValueError, "non-finite"):
            present_prediction(result, self.catalog)
        result["candidate_ids"] = Array([0, 9, 4])
        with self.assertRaisesRegex(ValueError, "columns"):
            present_prediction(result, self.catalog)


if __name__ == "__main__":
    unittest.main()
