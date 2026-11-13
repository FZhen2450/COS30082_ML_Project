"""Synthetic-only regression coverage for mandatory full-8000 raw hash pins."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from evaluation.packet_io import read_json
from scripts.import_wp5_final import import_full8000_wp2


class ManifestReached(Exception):
    """Sentinel: the authenticated JSON boundary was crossed successfully."""


class RequiredRawPinTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='SYNTHETIC_ONLY_AUTH_PINS_')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manifest = self.root / 'manifest.json'
        self.manifest.write_text('{}', encoding='utf-8')
        self.bundle = dict(
            schema='wp5-full8000-wp2-import-v1',
            protocol_id='cos30082-official-shared333',
            protocol_version='full8000-20261112-v1',
            identity_manifest=dict(path='manifest.json', file_sha256=self.digest(self.manifest)),
            manifest_sha256='1'*64, inference_order_sha256='2'*64,
            owner_packet={}, provenance={}, reviewed_owner_metadata={},
            final_policy_reference='SYNTHETIC_ONLY_POLICY',
            evaluation_authorization_reference='SYNTHETIC_ONLY_AUTHORIZATION',
            owner_approval_reference='SYNTHETIC_ONLY_REVIEW')

    @staticmethod
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def handoff(self, bundle=None):
        path = self.root / 'handoff.json'
        path.write_text(json.dumps(self.bundle if bundle is None else bundle), encoding='utf-8')
        return path

    def invoke(self, path, pin):
        return import_full8000_wp2(path, pin, self.root/'WP5_OUTPUT',
                                  registry_path=self.root/'REGISTRY', allow_synthetic=True)

    def assert_no_output(self):
        self.assertFalse((self.root/'WP5_OUTPUT').exists())
        self.assertFalse((self.root/'REGISTRY').exists())

    def test_null_external_pin_rejected_before_input_IO(self):
        with patch('evaluation.full8000_packets.read_json') as reader:
            reader.side_effect = AssertionError('No input IO before required pin validation')
            with self.assertRaisesRegex(ValueError, 'SHA-256'):
                self.invoke(self.root/'not_read.json', None)
            reader.assert_not_called()
        self.assert_no_output()

    def test_malformed_external_pins_rejected_before_input_IO(self):
        for pin in ('', False, 42, 'A'*64, '0'*63, 'g'*64):
            with self.subTest(pin=pin), patch('evaluation.full8000_packets.read_json') as reader:
                reader.side_effect = AssertionError('No input IO before required pin validation')
                with self.assertRaisesRegex(ValueError, 'SHA-256'):
                    self.invoke(self.root/'not_read.json', pin)
                reader.assert_not_called()
        self.assert_no_output()

    def assert_bad_manifest_pin(self, pin):
        bundle = copy.deepcopy(self.bundle)
        bundle['identity_manifest']['file_sha256'] = pin
        with patch('evaluation.full8000_packets.read_json') as reader:
            reader.side_effect = [bundle, AssertionError('Unpinned manifest must not be read')]
            with self.assertRaisesRegex(ValueError, 'SHA-256'):
                self.invoke(self.root/'synthetic.json', '0'*64)
            self.assertEqual(reader.call_count, 1)
        self.assert_no_output()

    def test_null_manifest_pin_rejected_before_manifest_IO(self):
        self.assert_bad_manifest_pin(None)

    def test_malformed_manifest_pins_rejected_before_manifest_IO(self):
        for pin in ('', False, 42, 'A'*64, '0'*63, 'g'*64):
            with self.subTest(pin=pin):
                self.assert_bad_manifest_pin(pin)

    def test_valid_raw_pins_reach_existing_semantic_validator(self):
        path = self.handoff()
        with patch('evaluation.full8000_packets.load_full8000_manifest') as loader:
            loader.side_effect = ManifestReached
            with self.assertRaises(ManifestReached):
                self.invoke(path, self.digest(path))
            loader.assert_called_once_with({}, expected_semantic_sha256='1'*64, allow_synthetic=True)
        self.assert_no_output()

    def test_wrong_external_raw_bytes_rejected(self):
        path = self.handoff()
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            self.invoke(path, '0'*64)
        self.assert_no_output()

    def test_wrong_manifest_raw_bytes_rejected(self):
        path = self.handoff()
        self.manifest.write_text('{"synthetic_change":true}', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            self.invoke(path, self.digest(path))
        self.assert_no_output()

    def test_missing_manifest_pin_field_rejected(self):
        bundle = copy.deepcopy(self.bundle)
        del bundle['identity_manifest']['file_sha256']
        path = self.handoff(bundle)
        with self.assertRaisesRegex(ValueError, 'raw file hash'):
            self.invoke(path, self.digest(path))
        self.assert_no_output()

    def test_omitted_external_pin_rejected_before_input_IO(self):
        with patch('evaluation.full8000_packets.read_json') as reader:
            reader.side_effect = AssertionError('Missing argument must not read input')
            with self.assertRaisesRegex(TypeError, 'handoff_sha256'):
                import_full8000_wp2(self.root/'unused.json', output_path=self.root/'WP5_OUTPUT',
                                    registry_path=self.root/'REGISTRY', allow_synthetic=True)
            reader.assert_not_called()
        self.assert_no_output()

    def test_invalid_pins_never_enter_protected_intake_or_output(self):
        for case in ('null_external', 'invalid_external', 'null_manifest', 'missing_manifest'):
            with self.subTest(case=case):
                bundle = copy.deepcopy(self.bundle)
                if case == 'null_manifest':
                    bundle['identity_manifest']['file_sha256'] = None
                elif case == 'missing_manifest':
                    del bundle['identity_manifest']['file_sha256']
                path = self.handoff(bundle)
                pin = None if case == 'null_external' else self.digest(path)
                if case == 'invalid_external':
                    pin = 'NOT_A_SHA256'
                with patch('evaluation.full8000_packets.load_full8000_manifest') as loader, \
                     patch('evaluation.full8000_packets.packet_directory') as packet, \
                     patch('evaluation.full8000_packets.write_new_directory') as writer:
                    for operation in (loader, packet, writer):
                        operation.side_effect = AssertionError('Invalid pins must fail before protected intake')
                    with self.assertRaises(ValueError):
                        self.invoke(path, pin)
                    for operation in (loader, packet, writer):
                        operation.assert_not_called()
                self.assert_no_output()

    def test_optional_generic_reader_remains_available(self):
        self.assertEqual(read_json(self.manifest), {})
        self.assertEqual(read_json(self.manifest, self.digest(self.manifest)), {})


if __name__ == '__main__':
    unittest.main(verbosity=2)
