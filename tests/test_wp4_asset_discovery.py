"""Discovery checks use small byte fixtures, never real model deserialization."""
import hashlib
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from wp4.asset_discovery import discover_weight_files


class AssetDiscoveryTests(unittest.TestCase):
    def test_final_registry_selects_new_full_training_weights_and_never_legacy_defaults(self):
        from final_protocol import SOURCE_SHA256
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            files={task:root/(task+'.pt') for task in ('crop','disease')}
            for task,path in files.items():
                path.write_bytes(('full-training-'+task).encode())
            with patch('wp4.asset_discovery.ROOT',root):
                missing=discover_weight_files('wp1',final_training_sha='fixture-full-training')
                self.assertEqual(missing['paths'],{})
                registry=root/'artifacts/models/wp1_full8000_release.json'
                registry.parent.mkdir(parents=True)
                record={'schema':'wp1-full8000-release-v1','training_count':8000,
                        'source_sha256':SOURCE_SHA256,'training_manifest_sha256':'fixture-full-training',
                        'checkpoints':{task:{'path':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
                                       for task,path in files.items()}}
                registry.write_text(json.dumps(record))
                found=discover_weight_files('wp1',final_training_sha='fixture-full-training')
                self.assertEqual(found['paths'],{task:str(path.resolve()) for task,path in files.items()})
                wrong=discover_weight_files('wp1',final_training_sha='wrong-training-release')
                self.assertEqual(wrong['paths'],{})

    def test_renamed_files_are_matched_by_hash_and_wrong_named_files_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'ViT-L-14.pt').write_bytes(b'wrong model')
            renamed = root / 'renamed.pth'
            renamed.write_bytes(b'approved base')
            adapter = root / 'another-name.pt'
            adapter.write_bytes(b'approved adapter')
            assets = {
                'clip_weights': (hashlib.sha256(b'approved base').hexdigest(), 'unused'),
                'adapter_checkpoint': (hashlib.sha256(b'approved adapter').hexdigest(), 'unused')}
            with patch('wp4.asset_discovery.ASSETS', assets):
                result = discover_weight_files('m6', [root])
            self.assertEqual(result['paths'], {'clip_weights': str(renamed.resolve()),
                                               'adapter_checkpoint': str(adapter.resolve())})
            self.assertEqual(result['missing'], [])

    def test_missing_weights_and_cancellation_are_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            result = discover_weight_files('wp1', [directory])
            self.assertEqual(result['missing'], ['crop', 'disease'])
            stop = threading.Event()
            stop.set()
            result = discover_weight_files('m6', [directory], stop)
            self.assertTrue(result['cancelled'])
            self.assertEqual(result['paths'], {})

    def test_modified_file_does_not_reuse_cached_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'weights.pt'
            path.write_bytes(b'approved')
            with patch('wp4.asset_discovery.ASSETS', {
                    'clip_weights': (hashlib.sha256(b'approved').hexdigest(), 'unused')}):
                self.assertTrue(discover_weight_files('frozen_clip', [directory])['paths'])
                path.write_bytes(b'replaced and no longer approved')
                result = discover_weight_files('frozen_clip', [directory])
            self.assertEqual(result['paths'], {})


if __name__ == '__main__':
    unittest.main()
