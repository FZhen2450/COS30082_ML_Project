"""Discovery checks use small byte fixtures, never real model deserialization."""
import hashlib
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from wp4.asset_discovery import discover_weight_files


class AssetDiscoveryTests(unittest.TestCase):
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
