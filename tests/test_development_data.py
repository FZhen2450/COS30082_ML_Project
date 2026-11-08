"""Train-only M3A split checks and synthetic CPU DataLoader integration."""

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import RandomSampler, SequentialSampler

import data_utils
from data_utils import COLUMN_NAMES, DEFAULT_CSVS, load_metadata
from development_data import build_development_split, get_development_dataloaders


ROOT = Path(__file__).resolve().parents[1]


def synthetic_training_csv(root, write_images=False):
    rows = []
    for comp, crop, disease in ((10, 0, 0), (20, 0, 1), (75, 1, 0), (332, 1, 1)):
        for index in range(8):
            name = f"synthetic_{comp}_{index}.png"
            rows.append([name, crop, disease, comp, f"crop{crop}",
                         f"disease{disease}", f"crop{crop}_disease{disease}"])
            if write_images:
                Image.new("L", (23, 17), 25 * index).save(root / name)
    path = root / "train.csv"
    pd.DataFrame(rows, columns=COLUMN_NAMES).to_csv(path, header=False, index=False)
    return path


class OfficialDevelopmentSplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = ROOT / DEFAULT_CSVS[0]
        cls.source = load_metadata(cls.path)
        cls.split = build_development_split(cls.path)

    def test_expected_image_and_composition_counts(self):
        for name, images, compositions in (
            ("fit", 4868, 258), ("val_seen", 1238, 225), ("val_unseen", 1894, 65)
        ):
            with self.subTest(partition=name):
                frame = getattr(self.split, name)
                self.assertEqual(len(frame), images)
                self.assertEqual(frame.composition_id.nunique(), compositions)

    def test_outer_partition_matches_existing_seed42_manifest(self):
        manifest = json.loads(
            (ROOT / "artifacts/data/heldout_composition_seed42.json").read_text()
        )
        self.assertEqual(self.split.val_unseen.image_name.tolist(),
                         manifest["validation_images"])
        retained = set(self.split.fit.image_name) | set(self.split.val_seen.image_name)
        self.assertEqual(retained, set(manifest["train_images"]))

    def test_filename_partitions_are_disjoint_and_complete(self):
        partitions = [set(getattr(self.split, name).image_name)
                      for name in ("fit", "val_seen", "val_unseen")]
        for index, names in enumerate(partitions):
            for other in partitions[index + 1:]:
                self.assertFalse(names & other)
        self.assertEqual(set.union(*partitions), set(self.source.image_name))
        self.assertEqual(sum(map(len, partitions)), 8000)

    def test_seen_coverage_and_heldout_isolation(self):
        fit = set(self.split.fit.composition_id)
        seen = set(self.split.val_seen.composition_id)
        unseen = set(self.split.val_unseen.composition_id)
        self.assertLessEqual(seen, fit)
        self.assertFalse(unseen & fit)
        self.assertFalse(unseen & seen)
        self.assertEqual(fit | unseen, set(self.source.composition_id))

    def test_components_and_original_label_rows_are_preserved(self):
        for column, count in (("crop_id", 55), ("disease_id", 175)):
            self.assertEqual(set(self.split.fit[column]), set(range(count)))
            self.assertLessEqual(set(self.split.val_unseen[column]),
                                 set(self.split.fit[column]))
        restored = pd.concat([self.split.fit, self.split.val_seen, self.split.val_unseen])
        pd.testing.assert_frame_equal(
            restored.sort_values("image_name").reset_index(drop=True),
            self.source.sort_values("image_name").reset_index(drop=True),
        )
        for name in ("fit", "val_seen", "val_unseen"):
            for column in ("crop_id", "disease_id", "composition_id"):
                self.assertEqual(str(getattr(self.split, name)[column].dtype), "int64")

    def test_same_seed_is_deterministic_and_another_seed_changes_membership(self):
        repeated = build_development_split(self.path)
        for name in ("fit", "val_seen", "val_unseen"):
            pd.testing.assert_frame_equal(getattr(self.split, name), getattr(repeated, name))
        alternative = build_development_split(self.path, seed=7)
        self.assertNotEqual(self.split.val_unseen.image_name.tolist(),
                            alternative.val_unseen.image_name.tolist())

    def test_all_323_candidates_have_deterministic_global_id_pair_mapping(self):
        split = self.split
        self.assertEqual(len(split.candidate_ids), 323)
        self.assertEqual(split.candidate_ids, tuple(sorted(self.source.composition_id.unique())))
        self.assertNotEqual(split.candidate_ids, tuple(range(323)))
        self.assertEqual(max(split.candidate_ids), 332)
        self.assertEqual(len(set(split.candidate_pairs)), 323)
        for index, (comp, pair) in enumerate(zip(split.candidate_ids, split.candidate_pairs)):
            self.assertIs(type(comp), int)
            self.assertTrue(all(type(value) is int for value in pair))
            self.assertEqual(split.mappings["composition_id_to_pair"][comp], pair)
            self.assertEqual(split.mappings["composition_id_to_index"][comp], index)
            self.assertEqual(split.mappings["composition_index_to_id"][index], comp)
            row = self.source[self.source.composition_id == comp].iloc[0]
            self.assertEqual(pair, (int(row.crop_id), int(row.disease_id)))
        self.assertLessEqual(set(split.val_unseen.composition_id), set(split.candidate_ids))
        self.assertEqual(set(split.mappings["crop_id_to_name"]), set(range(55)))
        self.assertEqual(set(split.mappings["disease_id_to_name"]), set(range(175)))
        self.assertEqual(split.mappings["num_compositions"], 323)

    def test_provenance_captures_source_parameters_and_candidate_order(self):
        split = self.split
        self.assertEqual(split.source_sha256, hashlib.sha256(self.path.read_bytes()).hexdigest())
        records = [[comp, *pair] for comp, pair in zip(split.candidate_ids, split.candidate_pairs)]
        expected = hashlib.sha256(
            json.dumps(records, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        self.assertEqual(split.candidate_sha256, expected)
        self.assertEqual((split.seed, split.heldout_fraction, split.seen_fraction), (42, 0.2, 0.2))
        self.assertIn("pytorch-v1", split.split_version)


class SyntheticDevelopmentDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.path = synthetic_training_csv(self.root, write_images=True)

    def tearDown(self):
        self.temp.cleanup()

    def build(self, **kwargs):
        return build_development_split(self.path, heldout_fraction=0.25, **kwargs)

    def test_construction_reads_only_the_designated_training_metadata(self):
        with (
            patch("development_data.load_metadata", wraps=load_metadata) as reader,
            patch.object(data_utils, "build_class_mappings",
                         side_effect=AssertionError("Official-test mapping wrapper called")),
            patch("dataset.Image.open", side_effect=AssertionError("Unexpected image read")),
        ):
            split = self.build()
        reader.assert_called_once_with(self.path)
        self.assertEqual([len(split.fit), len(split.val_seen), len(split.val_unseen)], [18, 6, 8])
        self.assertEqual(split.candidate_ids, (10, 20, 75, 332))

    def test_invalid_training_metadata_is_rejected(self):
        original = load_metadata(self.path)
        duplicate = pd.concat([original, original.iloc[:1]], ignore_index=True)
        bad_name = original.copy()
        bad_name.loc[0, "crop_name"] = "inconsistent_crop"
        bad_pair = original.copy()
        bad_pair.loc[0, "disease_id"] = 1
        bad_pair.loc[0, "disease_name"] = "disease1"
        for frame in (duplicate, bad_name, bad_pair):
            with self.subTest(rows=len(frame)):
                path = self.root / "bad_train.csv"
                frame.to_csv(path, header=False, index=False)
                with self.assertRaises(ValueError):
                    build_development_split(path, heldout_fraction=0.25)

    def test_invalid_parameters_and_official_test_paths_are_rejected(self):
        for kwargs in (
            {"seed": True}, {"seed": -1}, {"seed": 1.5},
            {"heldout_fraction": 0}, {"heldout_fraction": 1},
            {"seen_fraction": float("nan")}, {"seen_fraction": 0},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises((TypeError, ValueError)):
                build_development_split(self.path, **kwargs)
        for name in DEFAULT_CSVS[1:]:
            with self.subTest(path=name), patch(
                "development_data.load_metadata", side_effect=AssertionError("Test metadata read")
            ):
                with self.assertRaises(ValueError):
                    build_development_split(self.root / name)

    def test_dataloaders_preserve_tensor_targets_and_global_composition_ids(self):
        split = self.build()
        for as_dict in (True, False):
            with self.subTest(as_dict=as_dict):
                loaders = get_development_dataloaders(
                    split, self.root, batch_size=2, image_size=16,
                    as_dict_targets=as_dict, augment_fit=False,
                )
                self.assertEqual(len(loaders), 3)
                self.assertIsInstance(loaders[0].sampler, RandomSampler)
                self.assertTrue(all(isinstance(loader.sampler, SequentialSampler)
                                    for loader in loaders[1:]))
                for loader, frame in zip(loaders, (split.fit, split.val_seen, split.val_unseen)):
                    self.assertEqual(len(loader.dataset), len(frame))
                    images, targets = next(iter(loader))
                    self.assertEqual(tuple(images.shape), (2, 3, 16, 16))
                    self.assertEqual(images.dtype, torch.float32)
                    self.assertTrue(torch.isfinite(images).all().item())
                    values = ([targets[key] for key in ("crop", "disease", "composition")]
                              if as_dict else targets)
                    self.assertTrue(all(value.dtype == torch.int64 for value in values))
                    for crop, disease, comp in zip(*[value.tolist() for value in values]):
                        self.assertEqual(split.mappings["composition_id_to_pair"][comp],
                                         (crop, disease))
                    validation_images = next(iter(loaders[1]))[0]
                    torch.testing.assert_close(validation_images, next(iter(loaders[1]))[0])

    def test_fit_augmentation_is_reproducible_when_reconstructing_a_run(self):
        split = self.build()
        batches = []
        for _ in range(2):
            loaders = get_development_dataloaders(
                split, self.root, batch_size=2, image_size=16, augment_fit=True
            )
            batches.append(next(iter(loaders[0])))
        torch.testing.assert_close(batches[0][0], batches[1][0])
        for key in batches[0][1]:
            torch.testing.assert_close(batches[0][1][key], batches[1][1][key])

    def test_loader_creation_is_lazy_and_writes_no_artifacts(self):
        split = self.build()
        before = set(self.root.iterdir())
        with patch("dataset.Image.open", side_effect=AssertionError("Eager image iteration")):
            loaders = get_development_dataloaders(split, self.root)
        self.assertEqual([len(loader.dataset) for loader in loaders], [18, 6, 8])
        self.assertEqual(set(self.root.iterdir()), before)

    def test_invalid_loader_arguments_are_rejected(self):
        split = self.build()
        for kwargs in ({"batch_size": 0}, {"batch_size": True}, {"num_workers": -1},
                       {"num_workers": 0.5}, {"image_size": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises((TypeError, ValueError)):
                get_development_dataloaders(split, self.root, **kwargs)
        with self.assertRaises(FileNotFoundError):
            get_development_dataloaders(split, self.root / "missing")


class DevelopmentImportSafetyTests(unittest.TestCase):
    def test_import_has_no_data_training_network_or_artifact_side_effects(self):
        script = r"""
import importlib
import os
import sys
from unittest.mock import patch
import data_utils
import dataset
import torch

def reject_side_effects(event, args):
    if event == "open":
        _, mode, flags = args
        if ((isinstance(mode, str) and any(c in mode for c in "wax+"))
                or (isinstance(flags, int) and flags & (
                    os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))):
            raise AssertionError("Unexpected write")
    if (event.startswith("socket.") or event in {
        "os.mkdir", "os.remove", "os.rename", "os.rmdir", "os.link", "os.symlink",
        "subprocess.Popen", "os.system", "urllib.Request"
    }):
        raise AssertionError(f"Unexpected side effect: {event}")

sys.addaudithook(reject_side_effects)
with (
    patch.object(data_utils, "load_metadata", side_effect=AssertionError("Metadata read")),
    patch.object(data_utils, "make_validation_split", side_effect=AssertionError("Split run")),
    patch.object(data_utils, "build_class_mappings", side_effect=AssertionError("Test metadata")),
    patch.object(dataset.DPDataset, "__getitem__", side_effect=AssertionError("Dataset iteration")),
    patch.object(torch.utils.data.DataLoader, "__iter__", side_effect=AssertionError("Loader run")),
    patch.object(torch.hub, "download_url_to_file", side_effect=AssertionError("Download")),
    patch.object(torch.autograd, "backward", side_effect=AssertionError("Backward pass")),
):
    importlib.import_module("development_data")
print("IMPORT_SAFE")
"""
        result = subprocess.run(
            [sys.executable, "-B", "-c", script], cwd=ROOT,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": ""},
            capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(result.stdout.strip(), "IMPORT_SAFE")


if __name__ == "__main__":
    unittest.main()
