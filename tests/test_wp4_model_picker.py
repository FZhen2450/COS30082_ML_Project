"""Model dispatch plus real Tk event-path tests with explicitly synthetic models."""
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch

from PIL import Image
import torch

from wp4.model_loader import MODEL_SPECS, load_model, validate_model_inputs
from test_wp4_clip_predictor import make_catalog


class ModelLoaderTests(unittest.TestCase):
    def test_wp1_dispatch_is_preserved_and_frozen_clip_ignores_stale_adapter_path(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.pt"
            path.write_bytes(b"synthetic path validation only")
            paths = {key: str(path) for key in ("crop", "disease", "clip_weights", "adapter_checkpoint")}
            catalog = object()
            with patch("wp4.service.load_independent", return_value="wp1") as independent:
                self.assertEqual(load_model(catalog, "wp1", paths), "wp1")
                independent.assert_called_once_with(catalog, str(path), str(path), "cpu")
            with patch("wp4.clip_predictor.CLIPPredictor", return_value="clip") as clip:
                self.assertEqual(load_model(catalog, "frozen_clip", paths), "clip")
                clip.assert_called_once_with(catalog, str(path), None, "cpu")
            with patch("wp4.clip_predictor.CLIPPredictor", return_value="m6") as clip:
                self.assertEqual(load_model(catalog, "m6", paths, "cuda"), "m6")
                clip.assert_called_once_with(catalog, str(path), str(path), "cuda")

    def test_missing_files_and_unknown_choices_rejected(self):
        for kind in MODEL_SPECS:
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                validate_model_inputs(kind, {}, "cpu")
        with self.assertRaises(ValueError):
            validate_model_inputs("unknown", {}, "cpu")
        with self.assertRaises(ValueError):
            validate_model_inputs("wp1", {}, "automatic")


class SyntheticPredictor:
    def __init__(self, catalog, variant):
        self.catalog, self.variant = catalog, variant

    def predict(self, images, topk=5):
        assert len(images) == 1 and images[0].mode == "RGB"
        ids = torch.tensor(list(self.catalog.candidates))
        scores = -torch.arange(len(ids), dtype=torch.float64)[None, :]
        return {"candidate_ids": ids, "scores": scores,
                "candidate_sha256": self.catalog.protocol["candidate_sha256"],
                "score_semantics": "synthetic_ui_fixture_only",
                "model_provenance": {"variant": self.variant, "synthetic": True,
                    "protocol_bridge": {"catalog_candidate": {"sha256": self.catalog.protocol["candidate_sha256"]},
                                        "model_candidate": {"sha256": "synthetic-historical-namespace"}}}}


class ModelPickerTkTests(unittest.TestCase):
    def setUp(self):
        try:
            import tkinter as tk
            self.root = tk.Tk()
        except (ImportError, RuntimeError) as exc:
            self.skipTest("A local Tk display is required: " + str(exc))
        except Exception as exc:
            # TclError is not importable on machines without Tk.
            if type(exc).__name__ == "TclError":
                self.skipTest("A local Tk display is required: " + str(exc))
            raise
        self.root.withdraw()
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.catalog = make_catalog(self.directory)
        first_id = next(iter(self.catalog.candidates))
        fitting = [row for row in self.catalog.fitting if row["composition_id"] == first_id]
        self.assertGreaterEqual(len(fitting), 4)
        for i, row in enumerate(fitting[:4]):
            Image.new("RGB", (20, 18), (20 + i * 40, 90, 30)).save(
                self.directory / row["image_name"], format="PNG")
        self.query = self.directory / "synthetic-query.png"
        Image.new("RGB", (20, 18), (200, 20, 80)).save(self.query)
        self.weight = self.directory / "synthetic-picker-file.pt"
        self.weight.write_bytes(b"path fixture, not model weights")
        self.errors = patch("wp4.gui.messagebox.showerror").start()
        self.addCleanup(patch.stopall)
        from wp4.gui import PlantApp
        self.app = PlantApp(self.root, self.catalog.protocol_path, self.directory, self.catalog.source_csv)
        self.wait_for(lambda: self.app.catalog is not None and not self.app.busy)

    def tearDown(self):
        if hasattr(self, "app"):
            self.app.close()
        elif hasattr(self, "root"):
            self.root.destroy()
        if hasattr(self, "temp"):
            self.temp.cleanup()

    def wait_for(self, condition):
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            self.root.update()
            if condition():
                return
            time.sleep(.01)
        self.fail("Timed out waiting for GUI event path; errors=" + repr(self.errors.call_args_list))

    def connect(self, kind):
        self.app.configure_model()
        dialog = self.app.model_dialog
        dialog.choice.current(dialog.kinds.index(kind))
        dialog.choice.event_generate("<<ComboboxSelected>>")
        self.root.update()
        for key, _ in MODEL_SPECS[kind]["fields"]:
            dialog.paths[key].set(str(self.weight))
        dialog.load()
        return dialog

    def test_clean_launch_m6_picker_prediction_references_export_and_model_switch(self):
        calls = []
        def loader(catalog, kind, paths, device):
            calls.append((kind, paths, device, threading.current_thread().name))
            return SyntheticPredictor(catalog, kind)
        with patch("wp4.gui.load_model", side_effect=loader):
            self.assertIsNone(self.app.predictor)
            self.assertEqual(self.app.identify_button["state"], "disabled")
            self.connect("m6")
            self.wait_for(lambda: self.app.predictor is not None and not self.app.busy)
            self.assertEqual(calls[0][0], "m6")
            self.assertNotEqual(calls[0][3], threading.current_thread().name)
            self.assertIn("M6", self.app.model_status.cget("text"))
            with patch("wp4.gui.filedialog.askopenfilename", return_value=str(self.query)):
                self.app.open_image()
            self.wait_for(lambda: self.app.query is not None and not self.app.busy)
            self.app.identify_button.invoke()
            self.wait_for(lambda: self.app.export_data is not None and len(self.app.photos) == 4 and not self.app.busy)
            self.assertEqual(len(self.app.predictions), 5)
            self.assertEqual([p["composition_id"] for p in self.app.predictions], list(self.catalog.candidates)[:5])
            self.assertIn("Fitting images only", self.app.reference_note.cget("text"))
            output = self.directory / "synthetic-export.json"
            with patch("wp4.gui.filedialog.asksaveasfilename", return_value=str(output)):
                self.app.export()
            exported = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(exported, self.app.export_data)
            self.assertTrue(exported["model_provenance"]["synthetic"])
            self.assertEqual(exported["image"], self.query.name)
            bridge = exported["model_provenance"]["protocol_bridge"]
            self.assertNotEqual(bridge["catalog_candidate"]["sha256"], bridge["model_candidate"]["sha256"])
            self.connect("frozen_clip")
            self.wait_for(lambda: self.app.model_kind == "frozen_clip" and not self.app.busy)
            self.assertIsNone(self.app.export_data)
            self.assertEqual(self.app.predictions, [])
            self.assertEqual(self.app.export_button["state"], "disabled")
            self.assertEqual(set(calls[-1][1]), {"clip_weights"})
        self.errors.assert_not_called()

    def test_missing_files_do_not_start_load_and_failed_load_retains_correct_old_model(self):
        with patch("wp4.gui.load_model", side_effect=lambda catalog, kind, paths, device: SyntheticPredictor(catalog, kind)):
            self.connect("wp1")
            self.wait_for(lambda: self.app.predictor is not None and not self.app.busy)
        previous = self.app.predictor
        self.app.configure_model()
        dialog = self.app.model_dialog
        dialog.choice.current(dialog.kinds.index("m6"))
        dialog.render_fields()
        with patch("wp4.gui.load_model") as loader:
            dialog.load()
            loader.assert_not_called()
        self.assertTrue(dialog.winfo_exists())
        self.errors.assert_called_once()
        self.errors.reset_mock()
        for key, _ in MODEL_SPECS["m6"]["fields"]:
            dialog.paths[key].set(str(self.weight))
        with patch("wp4.gui.load_model", side_effect=ValueError("synthetic invalid checkpoint")):
            dialog.load()
            self.wait_for(lambda: not self.app.busy)
        self.assertIs(self.app.predictor, previous)
        self.assertEqual(self.app.model_kind, "wp1")
        self.assertIn("WP1", self.app.model_status.cget("text"))
        self.errors.assert_called_once()


if __name__ == "__main__":
    unittest.main()
