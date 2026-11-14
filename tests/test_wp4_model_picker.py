"""Model dispatch plus real Tk event-path tests with explicitly synthetic models."""
import json
import hashlib
import inspect
from pathlib import Path
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from PIL import Image
import torch

from wp4.model_loader import MODEL_SPECS, load_model, validate_model_inputs
from test_wp4_clip_predictor import make_catalog


class ModelLoaderTests(unittest.TestCase):
    def test_wp1_export_provenance_binds_files_and_rejects_a_loading_time_change(self):
        from wp4.service import load_independent
        with tempfile.TemporaryDirectory() as directory:
            crop, disease = Path(directory)/'crop.pt', Path(directory)/'disease.pt'
            crop.write_bytes(b'synthetic crop'); disease.write_bytes(b'synthetic disease')
            catalog = SimpleNamespace(protocol_path='synthetic.json', source_csv='synthetic.csv',
                                      protocol={key: key for key in ('candidate_sha256','split_sha256','source_sha256')})
            predictor = SimpleNamespace(crop_transform='native crop', disease_transform='native disease')
            with patch('independent_baseline.IndependentPredictor', return_value=predictor):
                actual = load_independent(catalog, crop, disease)
            self.assertEqual(actual.provenance['variant'], 'wp1')
            self.assertEqual(actual.provenance['checkpoints']['crop']['file_sha256'],
                             hashlib.sha256(crop.read_bytes()).hexdigest())
            def changed(*args, **kwargs):
                disease.write_bytes(b'changed synthetic disease')
                return predictor
            with patch('independent_baseline.IndependentPredictor', side_effect=changed):
                with self.assertRaisesRegex(ValueError, 'changed while loading'):
                    load_independent(catalog, crop, disease)

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
        patch("wp4.model_picker.discover_weight_files", return_value={
            "paths": {}, "missing": [], "warnings": [], "cancelled": False}).start()
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
            self.assertEqual(exported['input_policy']['batch_size'], 1)
            self.assertEqual(exported['input_policy']['rgb_pixel_sha256'],
                             hashlib.sha256(self.app.query.tobytes()).hexdigest())
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

    def test_worker_target_does_not_own_tk_callbacks_and_picker_drops_variables(self):
        targets = []
        real_thread = threading.Thread
        def make_thread(*args, **kwargs):
            targets.append(kwargs['target'])
            return real_thread(*args, **kwargs)
        with patch('wp4.gui.threading.Thread', side_effect=make_thread), patch(
                'wp4.gui.load_model', side_effect=lambda catalog, kind, paths, device: SyntheticPredictor(catalog, kind)):
            self.connect('wp1')
            self.wait_for(lambda: self.app.predictor is not None and not self.app.busy)
        closures = [inspect.getclosurevars(target).nonlocals for target in targets]
        self.assertTrue(all('self' not in closure for closure in closures))
        closure = next(closure for closure in closures if 'action' in closure)
        self.assertNotIn('self', closure)
        self.assertNotIn('done', closure)
        self.assertNotIn('self', inspect.getclosurevars(closure['action']).nonlocals)
        dialog = self.app.model_dialog
        self.assertTrue(dialog.paths)
        self.assertIs(dialog.winfo_toplevel(), self.root)
        self.app.close()
        self.assertEqual(dialog.paths, {})
        self.assertIsNone(dialog.device)
        self.assertIsNone(dialog.on_load)
        dialog.destroy()  # Closing an already destroyed picker is safe.

    def test_search_fills_selected_files_without_overwriting_manual_paths(self):
        with patch("wp4.model_picker.discover_weight_files", return_value={
                "paths": {"crop": str(self.weight), "disease": str(self.weight)},
                "missing": [], "warnings": [], "cancelled": False}):
            self.app.configure_model()
            dialog = self.app.model_dialog
            dialog.paths['crop'].set('manual-choice.pt')
            dialog.start_search()
            self.wait_for(lambda: dialog.paths['disease'].get() == str(self.weight))
            self.assertEqual(dialog.paths['crop'].get(), 'manual-choice.pt')
            self.assertIn('ready', dialog.search_status.cget('text'))

    def test_stale_search_result_is_ignored_after_model_switch(self):
        self.app.configure_model()
        dialog = self.app.model_dialog
        old_token = dialog._search_token
        dialog.choice.current(dialog.kinds.index('m6'))
        dialog.render_fields()
        dialog._search_events.put((old_token, 'wp1', {
            'paths': {'crop': 'stale.pt'}, 'missing': []}, None))
        self.wait_for(lambda: dialog._search_events.empty())
        self.assertEqual(dialog.paths['crop'].get(), '')

    def test_reference_scrolling_keeps_image_and_five_rankings_visible(self):
        self.root.deiconify()
        self.root.geometry('1280x840')
        self.root.update()
        before = (self.app.open_button.winfo_rootx(), self.app.open_button.winfo_rooty())
        preview_before = self.app.image_panel.winfo_rooty()
        self.assertIs(self.app.model_panel.winfo_toplevel(), self.root)
        self.assertFalse(any(child.winfo_class() == 'Toplevel' for child in self.root.winfo_children()))
        self.app.show_references(next(iter(self.catalog.candidates)), False)
        self.wait_for(lambda: not self.app.busy)
        self.assertEqual(len(self.app.photos), 4)
        for index, candidate in enumerate(list(self.catalog.candidates.values())[:5]):
            self.app.table.insert('', 'end', iid=str(index), values=(index+1,
                candidate['crop_name'], candidate['disease_name'], '0.1234'))
        self.root.update()
        self.assertTrue(self.app.gallery.winfo_ismapped())
        image_card = self.app.image_panel.master.master
        result_card = self.app.table.master.master
        image_share = image_card.winfo_height() / (image_card.winfo_height()+result_card.winfo_height())
        self.assertGreater(image_share, .57)
        self.assertLess(image_share, .62)
        last_row = self.app.table.rows['4']['canvas']
        self.assertLessEqual(last_row.winfo_y()+last_row.winfo_height(), self.app.table.winfo_height())
        body = self.app.table.master
        self.assertLessEqual(last_row.winfo_rooty()+last_row.winfo_height(),
                             body.winfo_rooty()+body.winfo_height()-22)
        self.app.page_canvas.event_generate('<MouseWheel>', delta=-120)
        self.root.update()
        self.assertGreater(self.app.page_canvas.yview()[0], 0)
        self.assertEqual(before, (self.app.open_button.winfo_rootx(), self.app.open_button.winfo_rooty()))
        self.assertEqual(preview_before, self.app.image_panel.winfo_rooty())
        self.assertTrue(self.app.export_button.winfo_ismapped())
        self.assertLess(self.app.page_canvas.canvasy(0), self.app.page_canvas.bbox('all')[3])
        self.assertEqual(int(self.app.page_canvas.cget('scrollregion').split()[-1]),
                         self.app.page_canvas.bbox('all')[3])

    def test_short_laptop_windows_keep_all_five_rankings_and_score_note_visible(self):
        # Maximised client areas of 1366x768, 1280x720 and 1280x640 screens. The app
        # bounds its own minimum size by the screen, so emulate that here.
        self.root.deiconify()
        for index, candidate in enumerate(list(self.catalog.candidates.values())[:5]):
            self.app.table.insert('', 'end', iid=str(index), values=(index+1,
                candidate['crop_name'], candidate['disease_name'], '0.1234'))
        body = self.app.table.master
        score_note = body.pack_slaves()[-1]
        for width, height in ((1366, 688), (1280, 640), (1280, 560)):
            with self.subTest(window=f'{width}x{height}'):
                self.root.minsize(min(1280, width), min(840, height))
                self.root.geometry(f'{width}x{height}')
                self.root.update()
                self.root.update()
                self.assertEqual(self.root.winfo_height(), height)
                visible_bottom = body.winfo_rooty() + body.winfo_height()
                for rank in range(5):
                    row = self.app.table.rows[str(rank)]['canvas']
                    self.assertEqual(row.winfo_height(), row.winfo_reqheight(), f'rank {rank+1} squeezed')
                    self.assertLessEqual(row.winfo_rooty()+row.winfo_height(), visible_bottom,
                                         f'rank {rank+1} clipped')
                self.assertEqual(score_note.winfo_height(), score_note.winfo_reqheight())
                self.assertLessEqual(score_note.winfo_rooty()+score_note.winfo_height(), visible_bottom)
                # The preview yields height first but must stay visible (55 px at 1280x560).
                self.assertGreater(self.app.image_panel.winfo_height(), 40)
                self.assertTrue(self.app.export_button.winfo_ismapped())

    def test_preview_fits_whole_image_without_changing_model_input(self):
        self.root.deiconify()
        self.root.geometry('1280x840')
        self.root.update()
        self.app.query = Image.new('RGB', (120, 90), (35, 130, 70))
        original = self.app.query.tobytes()
        self.app._draw_query()
        self.assertLessEqual(self.app.query_photo.width(), self.app.image_panel.winfo_width())
        self.assertLessEqual(self.app.query_photo.height(), self.app.image_panel.winfo_height())
        self.assertAlmostEqual(self.app.query_photo.width()/self.app.query_photo.height(), 120/90, delta=.01)
        self.assertFalse(hasattr(self.app, 'preview_mode'))
        self.assertEqual(self.app.query.size, (120, 90))
        self.assertEqual(self.app.query.tobytes(), original)

    def test_close_during_background_work_discards_callback_without_tk_access(self):
        started, finish, completed = threading.Event(), threading.Event(), threading.Event()
        callback = Mock()
        def action():
            started.set()
            finish.wait(3)
            completed.set()
            return 'worker-only data'
        self.app._work(action, callback, 'Synthetic slow operation')
        self.assertTrue(started.wait(2))
        self.app.close()
        finish.set()
        self.assertTrue(completed.wait(2))
        self.assertIsNone(self.app._on_done)
        callback.assert_not_called()
        self.app.close()  # Teardown must remain idempotent.


if __name__ == "__main__":
    unittest.main()
