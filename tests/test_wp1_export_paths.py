"""Portable receipt lookup; no weights, model inference or artifact mutation."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location(
    "wp1_export_verifier", Path(__file__).resolve().parents[1] / "scripts/verify_wp1_development_export.py")
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)


class ReceiptPathTests(unittest.TestCase):
    def test_windows_and_posix_keys_identify_the_same_source_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "evaluation" / "reporting.py"
            source.parent.mkdir()
            source.write_bytes(b"unchanged source bytes\r\n")
            self.assertEqual(verifier.receipt_path(root, "evaluation/reporting.py"), source.resolve())
            self.assertEqual(verifier.receipt_path(root, "evaluation\\reporting.py"), source.resolve())
            self.assertEqual(source.read_bytes(), b"unchanged source bytes\r\n")

    def test_absolute_drive_and_traversal_paths_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in ("../outside.py", "..\\outside.py", "/outside.py", "C:\\outside.py",
                         "C:outside.py", "\\\\server\\share\\outside.py", "a/../outside.py"):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    verifier.receipt_path(directory, name)

    def test_empty_and_non_string_keys_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in ("", " ", None, 42):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    verifier.receipt_path(directory, name)

    def test_missing_file_and_directory_are_not_valid_source_entries(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                verifier.receipt_path(directory, "missing.py")
            with self.assertRaises(ValueError):
                verifier.receipt_path(directory, ".")


if __name__ == "__main__":
    unittest.main()
