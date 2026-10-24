"""Data contract tests use official CSVs and temporary synthetic images only."""

import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import torch
from PIL import Image

from data_utils import (
    COLUMN_NAMES, DEFAULT_CSVS, audit_images, build_class_mappings, load_metadata,
    make_validation_split, save_split_manifest, validate_metadata,
)
from dataset import DPDataset, get_dataloaders, seed_everything

ROOT = Path(__file__).resolve().parents[1]


class MetadataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frames = [load_metadata(ROOT / name) for name in DEFAULT_CSVS]

    def test_official_splits_and_mappings(self):
        report = validate_metadata(*self.frames)
        self.assertEqual(report['split_sizes'], [8000, 2000, 1448])
        mapping = build_class_mappings(ROOT / DEFAULT_CSVS[0], ROOT / DEFAULT_CSVS[2])
        self.assertEqual(mapping['num_compositions'], 333)
        self.assertEqual(len(mapping['seen_comp_id_to_index']), 323)
        self.assertGreater(max(mapping['seen_comp_id_to_index']), 322)
        for comp_id, index in mapping['seen_comp_id_to_index'].items():
            self.assertEqual(mapping['seen_comp_index_to_id'][index], comp_id)

    def test_stratified_retains_singletons_and_all_classes(self):
        frame = self.frames[0]
        train, val = make_validation_split(frame)
        self.assertFalse(set(train.image_name) & set(val.image_name))
        self.assertEqual(set(train.image_name) | set(val.image_name), set(frame.image_name))
        self.assertEqual(set(train.composition_id), set(frame.composition_id))
        counts = frame.composition_id.value_counts()
        self.assertFalse(set(counts[counts == 1].index) & set(val.composition_id))
        self.assertEqual(set(counts[counts > 1].index), set(val.composition_id))
        train2, val2 = make_validation_split(frame)
        pd.testing.assert_frame_equal(train, train2)
        pd.testing.assert_frame_equal(val, val2)
        self.assertNotEqual(val.image_name.tolist(), make_validation_split(frame, seed=7)[1].image_name.tolist())

    def test_heldout_has_no_composition_leakage_and_preserves_components(self):
        frame = self.frames[0]
        for seed in (42, 7):
            train, val = make_validation_split(frame, mode='heldout_composition', seed=seed)
            self.assertFalse(set(train.composition_id) & set(val.composition_id))
            self.assertEqual(val.composition_id.nunique(), 65)
            self.assertEqual(len(train) + len(val), 8000)
            for col in ('crop_id', 'disease_id'):
                self.assertEqual(set(train[col]), set(frame[col]))

    def test_metadata_rejects_leakage_and_inconsistent_mapping(self):
        train, seen, unseen = self.frames
        bad = seen.copy()
        bad.loc[0, 'image_name'] = train.iloc[0].image_name
        with self.assertRaisesRegex(ValueError, 'overlap'):
            validate_metadata(train, bad, unseen)
        bad = seen.copy()
        bad.loc[0, 'crop_name'] = 'incorrect_name'
        with self.assertRaisesRegex(ValueError, 'mapping'):
            validate_metadata(train, bad, unseen)

    def test_manifest_checksum_and_partition(self):
        train, val = make_validation_split(self.frames[0])
        with tempfile.TemporaryDirectory() as folder:
            path = save_split_manifest(train, val, ROOT / DEFAULT_CSVS[0], folder, 'stratified', 42, .2)
            manifest = json.loads(path.read_text())
            self.assertEqual(len(manifest['source_sha256']), 64)
            self.assertEqual(manifest['validation_images'], val.image_name.tolist())

    def test_eda_summary_does_not_claim_unrun_or_failed_audit_passed(self):
        from eda import generate_summary_report
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as folder:
            try:
                os.chdir(folder)
                with contextlib.redirect_stdout(io.StringIO()):
                    generate_summary_report(*self.frames, [])
                summary = Path('eda_summary.txt').read_text(encoding='utf-8')
                self.assertIn('NOT RUN', summary)
                self.assertNotIn('All 11,448 verified', summary)
                with contextlib.redirect_stdout(io.StringIO()):
                    generate_summary_report(*self.frames, [(16, 16)],
                                            failures=[{'image_name': 'broken.jpg'}], images_checked=True)
                summary = Path('eda_summary.txt').read_text(encoding='utf-8')
                self.assertIn('1 decoded; 1 failed', summary)
                self.assertIn('not regenerated', summary)
            finally:
                os.chdir(previous)

    def test_invalid_split_arguments(self):
        for fraction in (0, 1, -0.1):
            with self.assertRaises(ValueError):
                make_validation_split(self.frames[0], val_split=fraction)
        with self.assertRaises(ValueError):
            make_validation_split(self.frames[0], mode='unknown')


class ImageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.frame = pd.DataFrame([
            ['gray.png', 0, 0, 332, 'crop', 'disease', 'crop_disease'],
        ], columns=COLUMN_NAMES)
        Image.new('L', (23, 17), 128).save(self.root / 'gray.png')

    def tearDown(self):
        self.temp.cleanup()

    def test_rgb_dtype_global_id_and_normalization(self):
        dataset = DPDataset(self.frame, self.root, image_size=32)
        image, targets = dataset[0]
        self.assertEqual(tuple(image.shape), (3, 32, 32))
        self.assertEqual(image.dtype, torch.float32)
        self.assertTrue(torch.isfinite(image).all())
        self.assertEqual(targets['composition'].item(), 332)
        self.assertTrue(all(value.dtype == torch.int64 for value in targets.values()))
        self.assertAlmostEqual(image[0, 0, 0].item(), (128 / 255 - .485) / .229, places=5)
        torch.testing.assert_close(image, dataset[0][0])
        _, targets = DPDataset(self.frame, self.root, as_dict_targets=False)[0]
        self.assertEqual(len(targets), 3)

    def test_augmentation_reproducible_and_finite(self):
        dataset = DPDataset(self.frame, self.root, image_size=32, augment=True)
        seed_everything(7)
        image, targets = dataset[0]
        seed_everything(7)
        torch.testing.assert_close(image, dataset[0][0])
        self.assertTrue(torch.isfinite(image).all())
        self.assertEqual(targets['composition'].item(), 332)

    def test_audit_checks_images_after_first_fifty(self):
        (self.root / 'broken.jpg').write_bytes(b'broken')
        frame = pd.concat([self.frame] * 51, ignore_index=True)
        frame.loc[50, 'image_name'] = 'broken.jpg'
        empty = frame.iloc[:0]
        failures, sizes = audit_images([frame, empty, empty], self.root)
        self.assertEqual(len(failures), 1)
        self.assertEqual(len(sizes), 50)
        self.assertEqual(failures[0]['image_name'], 'broken.jpg')

    def test_missing_image_has_path(self):
        self.frame.loc[0, 'image_name'] = 'missing.jpg'
        with self.assertRaisesRegex(RuntimeError, 'missing.jpg'):
            DPDataset(self.frame, self.root)[0]

    def test_loader_integration_including_windows_workers(self):
        # Synthetic fixture with two seen compositions and a heldout cross-pair.
        rows = []
        for i in range(8):
            crop = i % 2
            name = f'train{i}.png'
            Image.new('RGB', (16, 16), (i * 20, 80, 120)).save(self.root / name)
            rows.append([name, crop, crop, crop, f'crop{crop}', f'disease{crop}', f'pair{crop}'])
        seen = [['seen.png', 0, 0, 0, 'crop0', 'disease0', 'pair0']]
        unseen = [['unseen.png', 0, 1, 2, 'crop0', 'disease1', 'pair2']]
        for name in ('seen.png', 'unseen.png'):
            Image.new('RGB', (16, 16), 'green').save(self.root / name)
        for filename, values in zip(DEFAULT_CSVS, [rows, seen, unseen]):
            pd.DataFrame(values).to_csv(self.root / filename, index=False, header=False)
        for workers in (0, 1):
            loaders = get_dataloaders(data_dir=self.root, img_dir='.', batch_size=2,
                                      image_size=16, num_workers=workers)
            images, targets = next(iter(loaders[0]))
            self.assertEqual(tuple(images.shape), (2, 3, 16, 16))
            self.assertEqual(targets['crop'].dtype, torch.int64)
            val_image = next(iter(loaders[1]))[0]
            torch.testing.assert_close(val_image, next(iter(loaders[1]))[0])


if __name__ == '__main__':
    unittest.main()
