"""Record a clean model-picker rehearsal using real local WP1/CLIP/M6 assets.

Only file dialogs and error reporting are automated. No predictor is injected.
Use a fitting/development query, never an official-test image.
"""
import argparse
import hashlib
import json
import platform
from pathlib import Path
import sys
import time
import tkinter as tk
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("wp1", "frozen_clip", "m6"), default="m6")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--clip", type=Path)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--crop", type=Path)
    parser.add_argument("--disease", type=Path)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--query", default="PWV2_peach_leaf_curl_6128.jpg")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--screenshots", action="store_true")
    args = parser.parse_args()
    from wp4.model_loader import MODEL_SPECS, validate_model_inputs
    supplied = {"clip_weights": args.clip, "adapter_checkpoint": args.adapter,
                "crop": args.crop, "disease": args.disease}
    paths = {key: supplied[key] for key, _ in MODEL_SPECS[args.model]["fields"]}
    validate_model_inputs(args.model, paths, args.device)
    args.output.mkdir(parents=True, exist_ok=False)
    from wp4.gui import PlantApp
    from wp4.service import present_prediction
    import torch
    torch.set_num_threads(2)
    if sys.platform == "win32":
        import ctypes
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except (AttributeError, OSError):
            pass
    root = tk.Tk()
    app = PlantApp(root, args.protocol, args.images, ROOT / "DPD_Small_train.csv")
    root.geometry("1240x900+20+20")

    def wait(condition, errors):
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            root.update()
            if errors.called:
                raise RuntimeError("GUI error: " + str(errors.call_args))
            if condition():
                return
            time.sleep(.03)
        raise TimeoutError("GUI action timed out")

    def screenshot(window, name):
        if args.screenshots:
            from PIL import ImageGrab
            window.lift()
            root.update()
            time.sleep(.3)
            if sys.platform == 'win32':
                # Target the application's own HWND; a desktop crop can capture
                # an unrelated foreground window when Windows refuses focus.
                frame = window.wm_frame()
                shot = ImageGrab.grab(window=int(frame, 0))
                if shot.getextrema() == ((0, 0), (0, 0), (0, 0)):
                    raise RuntimeError('Application window capture returned a black image')
            else:
                x, y = window.winfo_rootx(), window.winfo_rooty()
                shot = ImageGrab.grab(bbox=(x, y, x + window.winfo_width(), y + window.winfo_height()))
            shot.save(args.output / name)

    try:
        with patch("wp4.gui.messagebox.showerror") as errors:
            wait(lambda: app.catalog is not None and not app.busy, errors)
            assert app.predictor is None
            protocol = app.catalog.protocol
            scopes = {key: protocol[key] for key in
                      ("fitting_images", "seen_validation_images", "unseen_validation_images")}
            scope = next((key for key, names in scopes.items() if args.query in names), None)
            if scope is None or Path(args.query).name != args.query:
                raise ValueError("Query must be an original fitting/development filename")
            app.configure_button.invoke()
            dialog = app.model_dialog
            dialog.choice.current(dialog.kinds.index(args.model))
            dialog.render_fields()
            for key, path in paths.items():
                dialog.paths[key].set(str(path.resolve(strict=True)))
            dialog.device.set(args.device)
            screenshot(dialog, "model-picker.png")
            dialog.load()
            print(f"Loading actual {args.model} through normal GUI model picker", flush=True)
            wait(lambda: app.predictor is not None and not app.busy, errors)
            assert app.model_kind == args.model
            with patch("wp4.gui.filedialog.askopenfilename", return_value=str(args.images / args.query)):
                app.open_image()
            wait(lambda: app.query is not None and not app.busy, errors)
            app.identify_button.invoke()
            wait(lambda: app.export_data is not None and app.reference_note.cget('text').startswith('Selected match.')
                 and not app.busy, errors)
            canonical = app.predictor.predict([app.query], topk=5)
            assert present_prediction(canonical, app.catalog) == app.predictions
            assert tuple(canonical["scores"].shape) == (1, 323)
            expected_dtype = torch.float32 if args.model == 'wp1' else torch.float64
            assert canonical["scores"].dtype == expected_dtype
            (args.output / "canonical_scores.json").write_text(json.dumps({
                "candidate_ids": canonical["candidate_ids"].tolist(),
                "scores": canonical["scores"].tolist(),
                "topk_composition_ids": canonical["topk_composition_ids"].tolist(),
                "score_precision": str(expected_dtype), "candidate_sha256": canonical["candidate_sha256"]},
                allow_nan=False) + "\n", encoding="utf-8")
            reference = app.catalog.references(app.predictions[0]["composition_id"], app.query, app.query_path)
            fitting = {r["image_name"]: r for r in app.catalog.fitting}
            ref_names = [r["filename"] for r in reference["items"]]
            assert len(ref_names) == len(set(ref_names)) == len(app.photos) <= 4
            assert args.query not in ref_names and all(name in fitting for name in ref_names)
            exported = args.output / "prediction.json"
            with patch("wp4.gui.filedialog.asksaveasfilename", return_value=str(exported)):
                app.export()
            assert json.loads(exported.read_text(encoding="utf-8")) == app.export_data
            assert "Fitting images only" in app.reference_note.cget("text")
            screenshot(root, args.model + "-result.png")
            root.geometry("1100x800+20+20")
            root.update()
            screenshot(root, args.model + "-minimum-size.png")
            fitting_ids = {int(row['composition_id']) for row in app.catalog.fitting}
            fallback_id = next(cid for cid in app.class_ids if cid not in fitting_ids)
            app.class_choice.current(app.class_ids.index(fallback_id))
            app.browse_button.invoke()
            wait(lambda: app.reference_note.cget('text').startswith('Class browser (not a prediction).')
                 and not app.busy, errors)
            fallback = app.catalog.references(fallback_id)
            assert fallback['relation'] == 'Same disease; different crop'
            assert 'Same disease; different crop' in app.reference_note.cget('text')
            assert all(item['filename'] in fitting for item in fallback['items'])
            screenshot(root, "same-disease-fallback.png")
            from data_utils import load_metadata
            frame = load_metadata(ROOT / "DPD_Small_train.csv")
            row = frame[frame.image_name == args.query].iloc[0]
            evidence = {"clean_launch": True, "actual_model_picker": True,
                "predictor_injected": False, "real_m6_image_inference": args.model == 'm6',
                "real_model_image_inference": True, "model_kind": args.model,
                "query": args.query, "query_scope": scope,
                "truth": {"composition_id": int(row.composition_id), "crop": row.crop_name, "disease": row.disease_name},
                "predictions": app.predictions, "canonical_ranking_and_scores_match": True,
                "reference_names": ref_names, "reference_relation": reference["relation"],
                "reference_note": app.reference_note.cget("text"), "fitting_only": True,
                "export_matches": True, "model_provenance": app.export_data["model_provenance"],
                "runtime": {"python": platform.python_version(), "torch": str(torch.__version__),
                            "torchvision": str(__import__('torchvision').__version__), "device": args.device},
                "input_policy": app.export_data['input_policy'],
                "screenshot_method": 'application HWND capture' if sys.platform == 'win32' else 'window desktop crop',
                "fallback_browse": {"composition_id": fallback_id, "is_prediction": False,
                                    "reference_relation": fallback['relation'],
                                    "reference_names": [item['filename'] for item in fallback['items']]},
                "official_tests_used": False,
                "source_sha256": {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in [*sorted((ROOT / "wp4").glob("*.py")), Path(__file__)]}}
            (args.output / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(evidence, indent=2), flush=True)
    finally:
        app.close()


if __name__ == "__main__":
    main()
