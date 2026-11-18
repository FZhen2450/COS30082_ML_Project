"""Drive the REAL LeafLens window (wp4.gui.PlantApp) with J2 and capture screenshots (training images only).

  DISPLAY=:0 python scripts/j2_gui_smoke.py --images <DPD_Small_dataset> --query <training image name> --out <dir> [--device cuda]

Steps: start the app on the final-333 protocol -> select "J2 / final model" in the real model picker and let its
hash-based discovery find the J2 files -> load J2 -> open the query image -> Identify -> show the first match's
training references -> screenshot -> load J2 with a wrong asset file to confirm the error surfaces (no fallback).
Writes smoke.json and PNG screenshots to --out. The query must be a training-CSV image (official-test files refused).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True); ap.add_argument("--query", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda"); ap.add_argument("--wrong-asset")
    ap.add_argument("--hold", type=int, default=0, help="keep the window open N seconds after the result (for an external screenshot)")
    a = ap.parse_args()
    with open(ROOT / "DPD_Small_train.csv", newline="") as fh:
        train = {r[0]: r for r in csv.reader(fh) if r}
    if a.query not in train or "test" in a.query.lower():
        sys.exit("the smoke query must be a training-CSV image")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=False)
    import tkinter as tk
    from tkinter import messagebox
    from PIL import ImageGrab
    from wp4 import gui as G
    from wp4.service import open_rgb
    errors = []
    messagebox.showerror = lambda title, msg, **k: errors.append(msg)  # record instead of a blocking modal
    root = tk.Tk()
    app = G.PlantApp(root, ROOT / "artifacts/data/final_full8000_333_v1.json", Path(a.images), ROOT / "DPD_Small_train.csv")
    log = {"query": a.query, "query_true_composition_id": int(train[a.query][3]), "device": a.device, "steps": []}
    t0 = time.time()

    def shot(name):
        """Grab the whole X screen and crop to the visible part of the window (a bbox past the screen edge fails
        under WSLg). A capture failure is recorded, never fatal."""
        root.update(); time.sleep(0.4); root.update()
        try:
            screen = ImageGrab.grab(xdisplay=os.environ.get("DISPLAY"))
            x, y, w, h = root.winfo_rootx(), root.winfo_rooty(), root.winfo_width(), root.winfo_height()
            box = (max(0, x), max(0, y), min(screen.width, x + w), min(screen.height, y + h))
            screen.crop(box).save(out / f"{name}.png")
            log["steps"].append({"screenshot": f"{name}.png", "window": [x, y, w, h], "screen": list(screen.size),
                                 "t": round(time.time() - t0, 2)})
        except Exception as exc:  # noqa: BLE001
            log["steps"].append({"screenshot_failed": name, "error": f"{type(exc).__name__}: {exc}"})

    def wait(cond, then, timeout=240, what=""):
        start = time.time()
        def tick():
            if cond():
                then()
            elif time.time() - start > timeout:
                log["error"] = f"timeout waiting for {what}"; finish()
            else:
                root.after(100, tick)
        tick()

    def step_catalog():
        log["catalog"] = {"is_final": app.catalog.is_final, "candidates": len(app.catalog.candidates), "fitting": len(app.catalog.fitting)}
        panel = app.model_panel
        panel.choice.current(panel.kinds.index("j2")); panel.render_fields()
        wait(lambda: all(panel.paths[k].get() for k, _ in G.MODEL_SPECS["j2"]["fields"]), step_discovered, 600, "J2 discovery")

    def step_discovered():
        panel = app.model_panel
        log["discovered"] = {k: Path(panel.paths[k].get()).name for k, _ in G.MODEL_SPECS["j2"]["fields"]}
        shot("01_picker_j2_discovered")
        paths = {k: panel.paths[k].get() for k, _ in G.MODEL_SPECS["j2"]["fields"]}
        log["paths_keys"] = sorted(paths)
        t = time.time()
        app.connect_model("j2", paths, a.device)
        wait(lambda: app.predictor is not None and not app.busy, lambda: step_loaded(t, paths), 600, "J2 load")

    def step_loaded(t, paths):
        log["model_label"] = app.model_label; log["load_seconds"] = round(time.time() - t, 2)
        log["provenance"] = {k: app.predictor.provenance[k] for k in ("variant", "checkpoint_sha256", "inference_asset_sha256", "calibration")}
        p = Path(a.images) / a.query
        app.query, app.query_path = open_rgb(p), p
        app._draw_query(); app.clear_predictions(); app._controls()
        app.identify()
        wait(lambda: app.predictions and not app.busy, lambda: step_identified(paths), 300, "identify")

    def step_identified(paths):
        log["result_note"] = app.result_note.cget("text")
        log["top5"] = [{"rank": i + 1, "composition_id": r["composition_id"], "crop": r["crop_name"], "disease": r["disease_name"],
                        "score": round(r["score"], 4)} for i, r in enumerate(app.predictions)]
        app.show_references(app.predictions[0]["composition_id"], True)
        wait(lambda: not app.busy and app.reference_note.cget("text").startswith("Selected match"), lambda: step_refs(paths), 300, "references")

    def step_refs(paths):
        log["reference_note"] = app.reference_note.cget("text")
        shot("02_j2_prediction_and_references")
        if a.hold:
            (out / "READY_FOR_SCREENSHOT").write_text("prediction and references shown\n")
            root.after(a.hold * 1000, lambda: step_after_hold(paths))
            return
        step_after_hold(paths)

    def step_after_hold(paths):
        if a.wrong_asset:
            bad = dict(paths, j2_asset=a.wrong_asset)
            app.connect_model("j2", bad, a.device)
            wait(lambda: not app.busy and errors, step_error, 300, "error")
        else:
            finish()

    def step_error():
        log["wrong_asset_error"] = errors[-1]; log["predictor_after_error_unchanged"] = app.model_label
        finish()

    def finish():
        log["total_seconds"] = round(time.time() - t0, 2)
        (out / "smoke.json").write_text(json.dumps(log, indent=1))
        app.close()

    wait(lambda: app.catalog is not None and not app.busy, step_catalog, 300, "catalog")
    root.mainloop()
    print(json.dumps(log, indent=1))


if __name__ == "__main__":
    main()
