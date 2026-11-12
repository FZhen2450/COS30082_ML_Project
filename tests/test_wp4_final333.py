"""Final GUI contract, training-only fallback and native-score alignment checks."""
from pathlib import Path
import hashlib
import json
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from PIL import Image

from wp4.service import Catalog, present_prediction, load_independent
from wp4.clip_predictor import ResidualVisualAdapter, unit_features
from wp4.final_predictor import FinalCLIPPredictor

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT/'artifacts/data/final_full8000_333_v1.json'
SOURCE = ROOT/'DPD_Small_train.csv'


class NativeFixture(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.text = torch.randn(333, 768, generator=torch.Generator().manual_seed(33))
        self.eval().requires_grad_(False)
    def encode_text(self, tokens):
        return self.text[tokens[:, 0].long()]
    def encode_image(self, images):
        return torch.arange(1, 769, dtype=torch.float32)[None].repeat(len(images), 1)


def tokenize(prompts, context_length, truncate):
    assert len(prompts) == 333 and context_length == 77 and truncate is False
    result = torch.zeros(333, 77, dtype=torch.int64)
    result[:, 0] = torch.arange(333)
    return result


class FinalGUI333Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_catalog_contains_333_candidates_and_all_8000_training_references(self):
        catalog = Catalog(PROTOCOL, SOURCE, ROOT/'DPD_Small_dataset')
        self.assertTrue(catalog.is_final)
        self.assertEqual(len(catalog.candidates), 333)
        self.assertEqual(len(catalog.fitting), 8000)
        self.assertEqual(len({r['composition_id'] for r in catalog.fitting}), 323)

    def test_final_gui_refuses_missing_registry_and_changed_checkpoint_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            crop,disease=root/'crop.pt',root/'disease.pt'
            crop.write_bytes(b'synthetic crop fixture')
            disease.write_bytes(b'synthetic disease fixture')
            registry=root/'release.json'
            catalog=SimpleNamespace(is_final=True,protocol={'training_manifest_sha256':'training-fixture',
                                                           'candidate_sha256':'candidate-fixture'})
            with patch('wp4.service.FINAL_WP1_REGISTRY',registry),patch('wp1_final.FinalIndependentPredictor') as loader:
                with self.assertRaisesRegex(ValueError,'Register'):
                    load_independent(catalog,crop,disease)
                registry.write_text(json.dumps({'schema':'wp1-full8000-release-v1',
                    'training_manifest_sha256':'training-fixture','candidate_sha256':'candidate-fixture',
                    'checkpoints':{'crop':{'sha256':hashlib.sha256(crop.read_bytes()).hexdigest()},
                                   'disease':{'sha256':'changed-disease-file'}}}))
                with self.assertRaisesRegex(ValueError,'differ'):
                    load_independent(catalog,crop,disease)
                loader.assert_not_called()

    def test_unseen_reference_uses_labelled_same_disease_training_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            catalog = Catalog(PROTOCOL, SOURCE, directory)
            target = catalog.candidates[39]
            rows = [row for row in catalog.fitting if row['disease_id'] == target['disease_id']][:4]
            self.assertTrue(rows)
            for index, row in enumerate(rows):
                Image.new('RGB', (18, 14), (20+index*25, 100, 60)).save(Path(directory)/row['image_name'])
            result = catalog.references(39)
            self.assertEqual(result['relation'], 'Same disease; different crop')
            self.assertEqual(len(result['items']), len(rows))
            by_name = {row['image_name']:row for row in catalog.fitting}
            for item in result['items']:
                self.assertIn(item['filename'], by_name)
                self.assertNotEqual(by_name[item['filename']]['composition_id'], 39)
            query = result['items'][0]['image'].copy()
            # The original full-resolution query, not its reference thumbnail.
            query = Image.open(Path(directory)/rows[0]['image_name']).convert('RGB')
            excluded = catalog.references(39, query, Path(directory)/rows[0]['image_name'])
            self.assertNotIn(rows[0]['image_name'], [row['filename'] for row in excluded['items']])

    def test_final_frozen_and_historical_adapter_score_all_333_native_columns(self):
        catalog = Catalog(PROTOCOL, SOURCE, ROOT/'DPD_Small_dataset')
        def adapter(path, expected):
            self.assertEqual(len(expected['ids']), 323)
            return ResidualVisualAdapter().eval().requires_grad_(False), torch.ones(323, 768), {}
        for use_adapter in (False, True):
            native = NativeFixture()
            official_fixture = unit_features(native.text)
            with self.subTest(adapter=use_adapter), patch('wp4.final_predictor.load_native_clip',
                    return_value=(native, lambda im: torch.zeros(3, 224, 224), tokenize)), patch(
                    'wp4.final_predictor.load_m6_checkpoint', side_effect=adapter), patch(
                    'wp4.final_predictor.load_official_text_bank',return_value=(official_fixture,
                        {'text_bank_file_sha256':'synthetic-bank-file','direct_text_sha256':'synthetic-bank-features'})), patch.object(
                    native,'encode_text',side_effect=AssertionError('The official bank must not be re-encoded')):
                predictor = FinalCLIPPredictor(catalog, 'fixture', 'adapter-fixture' if use_adapter else None)
                result = predictor.predict([Image.new('RGB', (20, 20))])
                self.assertEqual(tuple(result['scores'].shape), (1, 333))
                expected = unit_features(native.encode_image(torch.zeros(1, 3, 224, 224))) @ official_fixture.T
                torch.testing.assert_close(result['scores'], expected, rtol=0, atol=0)
                self.assertEqual(result['candidate_ids'].tolist(), list(range(333)))
                self.assertEqual(result['candidate_sha256'], catalog.protocol['candidate_sha256'])
                self.assertEqual(len(present_prediction(result, catalog)), 5)
                self.assertTrue(predictor.text_bank_is_canonical)
                torch.testing.assert_close(predictor.direct,official_fixture,rtol=0,atol=0)
                if use_adapter:
                    self.assertEqual(predictor.provenance['adapter_training_scope'], 'historical_development_4868')
                    self.assertFalse(predictor.provenance['final_full8000_adapter_available'])


if __name__ == '__main__':
    unittest.main()
