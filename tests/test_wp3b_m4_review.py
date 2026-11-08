"""Regression reproductions for PR #14 review; synthetic CPU and temporary files."""
import hashlib
from pathlib import Path
import unittest
from unittest.mock import patch
import torch
import test_wp3b_m4 as m4


class ReviewNumericsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = m4.namespace()

    def test_nonfinite_learned_residual_rejected_in_both_arms(self):
        for cap in (None, .2):
            for value in (float("nan"), float("inf"), -float("inf")):
                with self.subTest(cap=cap, value=value):
                    model = self.ns["ResidualVisualAdapter"](cap)
                    with torch.no_grad():
                        model.up.bias.fill_(value)
                    with self.assertRaises(ValueError):
                        model(torch.ones(2, 768))

    def test_finite_residual_with_overflowed_cap_norm_rejected(self):
        model = self.ns["ResidualVisualAdapter"](.2)
        with torch.no_grad():
            model.up.bias.fill_(3e20)
        with self.assertRaises(ValueError):
            model(torch.ones(2, 768))

    def test_unbounded_nonfinite_output_rejected(self):
        model = self.ns["ResidualVisualAdapter"](None)
        with torch.no_grad():
            model.up.bias.fill_(1e22)
        with self.assertRaises(ValueError):
            model(torch.full((2, 768), 1e17))


class CurrentImageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = m4.namespace()

    def setUp(self):
        self.directory = m4.reference.reference.workspace_temporary_directory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.images = self.root / "images"
        self.images.mkdir()
        names = ("fit.jpg", "seen.jpg", "unseen.jpg")
        for name in names:
            (self.images / name).write_bytes(b"synthetic-image-bytes:" + name.encode())
        self.cache = {"folds": {role: {"filenames": [name]} for role, name in
                      zip(("fit", "val_seen", "val_unseen"), names)},
                      "image_hashes": {name: hashlib.sha256((self.images / name).read_bytes()).hexdigest()
                                       for name in names}}
        self.path = self.root / "cache.pt"
        self.ns["save_feature_cache"](self.path, self.cache)
        self.sha = self.ns["sha256_file"](self.path)
        self.config = {"image_directory": str(self.images)}

    def load(self):
        # Full cache structure/order is covered by CacheTests; this tiny fixture isolates real file I/O.
        with patch.dict(self.ns, {"validate_feature_cache": lambda *args: True}):
            return self.ns["load_feature_cache"](self.path, self.sha, None, None, self.config)

    def test_unchanged_current_images_load(self):
        self.assertEqual(self.load()["image_hashes"], self.cache["image_hashes"])

    def test_replaced_current_image_rejected_even_when_cache_hash_matches(self):
        (self.images / "fit.jpg").write_bytes(b"replacement")
        with self.assertRaises(ValueError):
            self.load()

    def test_missing_current_image_rejected(self):
        (self.images / "seen.jpg").unlink()
        with self.assertRaises(ValueError):
            self.load()

    def test_image_symlink_outside_approved_directory_rejected(self):
        outside = self.root / "outside.jpg"
        outside.write_bytes((self.images / "fit.jpg").read_bytes())
        (self.images / "fit.jpg").unlink()
        (self.images / "fit.jpg").symlink_to(outside)
        with self.assertRaises(ValueError):
            self.load()

    def test_unlisted_files_are_not_opened(self):
        (self.images / "unlisted.jpg").symlink_to(self.root / "missing-file")
        self.assertEqual(self.load()["image_hashes"], self.cache["image_hashes"])


class SelectionDenominatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = m4.namespace()

    def result(self, count):
        return {"sample_count": count, "loss": 1., "metrics": {"top1": .25, "top5": .5}}

    def test_partial_and_noninteger_development_counts_rejected(self):
        for seen, unseen in ((1, 1894), (1238, 1), (1894, 1238), (1238., 1894), (True, 1894)):
            with self.subTest(seen=seen, unseen=unseen):
                with self.assertRaises(ValueError):
                    self.ns["selection_record"](1, self.result(seen), self.result(unseen))

    def test_full_fold_counts_and_full_precision_policy_retained(self):
        seen, unseen = self.result(1238), self.result(1894)
        unseen["loss"] = 3.
        record = self.ns["selection_record"](2, seen, unseen)
        self.assertEqual((record["seen_count"], record["unseen_count"]), (1238, 1894))
        self.assertEqual(record["combined_validation_loss"], (1238 + 1894 * 3) / 3132)
        self.assertEqual((record["h1"], record["h5"]), (.25, .5))
        self.assertTrue(record["selection_policy"]["full_precision"])


class NativeTransformPersistenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        m4.CacheTests.setUpClass()
        cls.reference = m4.CacheTests

    def legacy_cache(self):
        import copy, re
        cache = copy.deepcopy(self.reference.cache)
        transform = cache["spec"]["native_transform"]
        transform = re.sub(r"<function _convert_image_to_rgb(?: at 0x[0-9a-fA-F]+)?>",
                           "<function _convert_image_to_rgb at 0x123456abcdef>", transform)
        cache["spec"]["native_transform"] = transform
        cache["spec_sha256"] = self.reference.ns["json_hash"](cache["spec"])
        cache["model_provenance"]["native_preprocess"] = transform
        return cache

    def test_legacy_cache_loads_across_function_memory_address_changes(self):
        ref = self.reference
        cache = self.legacy_cache()
        try:
            accepted = ref.ns["validate_feature_cache"](cache, ref.split, ref.table, ref.config)
        except ValueError as exc:
            self.fail("Unchanged legacy preprocessing must survive a new process: " + str(exc))
        self.assertTrue(accepted)

    def test_legacy_address_compatibility_does_not_accept_changed_interpolation(self):
        ref = self.reference
        cache = self.legacy_cache()
        cache["spec"]["native_transform"] = cache["spec"]["native_transform"].replace("bicubic", "bilinear")
        cache["spec_sha256"] = ref.ns["json_hash"](cache["spec"])
        with self.assertRaises(ValueError):
            ref.ns["validate_feature_cache"](cache, ref.split, ref.table, ref.config)

    def test_legacy_raw_spec_hash_remains_enforced(self):
        ref = self.reference
        cache = self.legacy_cache()
        cache["spec_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            ref.ns["validate_feature_cache"](cache, ref.split, ref.table, ref.config)


if __name__ == "__main__":
    unittest.main()
