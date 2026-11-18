"""Full-training identity, fixed-budget and final-333 scorer regression checks."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from PIL import Image

from final_protocol import load_final_protocol, validate_final_protocol
from wp1_final import FinalIndependentPredictor, train_final_task

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "artifacts/data/final_full8000_333_v1.json"
SOURCE = ROOT / "DPD_Small_train.csv"
CONFIG = json.loads((ROOT/"configs/wp1/resnet50_full8000.json").read_text())


class FullTrainingTests(unittest.TestCase):
    def setUp(self):
        self.protocol, self.frame = load_final_protocol(PROTOCOL, SOURCE)

    def test_full_training_has_8000_records_323_training_and_333_candidate_classes(self):
        self.assertEqual(len(self.frame), 8000)
        self.assertEqual(self.frame.composition_id.nunique(), 323)
        self.assertEqual([r['composition_id'] for r in self.protocol['candidates']], list(range(333)))
        self.assertEqual(len(set(self.protocol['training_images'])), 8000)

    def test_reordered_membership_missing_candidate_or_changed_prompt_is_rejected(self):
        for field in ('training_images', 'candidates', 'prompt_sha256', 'training_count'):
            value = copy.deepcopy(self.protocol)
            if field == 'training_images':
                value[field] = value[field][::-1]
            elif field == 'candidates':
                value[field].pop()
            elif field == 'training_count':
                value[field] = 4868
            else:
                value[field] = '0'*64
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_final_protocol(value, SOURCE)

    def test_crop_training_saves_fixed_epoch_two_without_validation_access(self):
        with tempfile.TemporaryDirectory() as directory, patch('wp1_final.build_classifier',
                return_value=torch.nn.Linear(1, 1)), patch('wp1_final.run_epoch',
                return_value={'count': 8000, 'loss': 1., 'top1': .2, 'top5': .4}) as epoch:
            output = Path(directory)/'crop'
            summary = train_final_task(CONFIG, 'crop', PROTOCOL, SOURCE, ROOT/'DPD_Small_dataset', output, 'cpu')
            self.assertEqual(epoch.call_count, 2)
            self.assertEqual(summary['epochs_completed'], 2)
            payload = torch.load(output/'final.pt', map_location='cpu', weights_only=True)
            self.assertEqual(payload['metadata']['training_count'], 8000)
            self.assertEqual(payload['metadata']['selected_epoch'], 2)
            self.assertFalse((output/'best.pt').exists())
            self.assertTrue(all('seen_validation' not in row for row in json.loads((output/'history.json').read_text())))

    def test_final_predictor_scores_unseen_candidate_and_rejects_development_weights(self):
        protocol = self.protocol
        target = protocol['candidates'][39]
        class Head(torch.nn.Module):
            def __init__(self, task):
                super().__init__()
                self.task = task
            def forward(self, batch):
                logits = torch.zeros(len(batch), 55 if self.task == 'crop' else 175)
                logits[:, target[self.task+'_id']] = 10
                return logits
        def checkpoint(path, device):
            task = str(path)
            meta = {**{k: protocol[k] for k in ('source_sha256', 'training_manifest_sha256',
                        'candidate_sha256', 'prompt_sha256', 'training_count')},
                    'task': task, 'training_scope': 'full_8000', 'backbone': 'resnet50',
                    'config': CONFIG, 'selected_epoch': CONFIG['epochs_by_task'][task]}
            return Head(task), meta
        with patch('wp1_final.load_checkpoint', side_effect=checkpoint), patch('wp1_final.preprocessing',
                return_value=lambda image: torch.zeros(3, 2, 2)):
            predictor = FinalIndependentPredictor('crop', 'disease', PROTOCOL, SOURCE)
            result = predictor.predict([Image.new('RGB', (2, 2))])
            self.assertEqual(tuple(result['scores'].shape), (1, 333))
            self.assertEqual(result['scores'].dtype, torch.float32)
            self.assertEqual(result['topk_composition_ids'][0, 0], 39)
        def legacy(path, device):
            model, meta = checkpoint(path, device)
            meta['training_scope'] = 'development_4868'
            return model, meta
        with patch('wp1_final.load_checkpoint', side_effect=legacy), self.assertRaises(ValueError):
            FinalIndependentPredictor('crop', 'disease', PROTOCOL, SOURCE)


if __name__ == '__main__':
    unittest.main()
