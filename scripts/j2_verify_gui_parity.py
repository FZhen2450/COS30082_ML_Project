"""Real-weight parity of the GUI's J2 predictor with the frozen Second Run J2 pipeline, on TRAINING images only.

  python scripts/j2_verify_gui_parity.py --clip <ViT-L-14.pt> --checkpoint <J2 checkpoint.pt> --asset <inference asset>
      --features <frozen features.pt> --rehearsal-scores <rehearsal seen_calibrated_scores.npy> --images <dataset dir>
      [--n 64] [--devices cpu,cuda] --out <report.json>

For training rows 0..n-1 (from DPD_Small_train.csv) it runs exactly the GUI path -- service.open_rgb (EXIF-corrected
RGB for display; J2 itself receives the stored pixels via J2Predictor.load_input), J2Predictor.predict on ONE image
at a time (zero-padded to the frozen batch shape of 32 on CUDA) -- and compares with
  (a) the frozen J2 features of the same rows (batch-32 extraction) scored by the same asset arithmetic, and
  (b) the saved official-size preflight rehearsal scores of the same rows (frozen pipeline, end to end).
Reports Top-1 / Top-5 agreement, the largest score gap and latency. No official-test image is opened.
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    for k in ("clip", "checkpoint", "asset", "features", "rehearsal-scores", "images", "out"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--n", type=int, default=64); ap.add_argument("--devices", default="cpu,cuda")
    a = ap.parse_args()
    import sys
    sys.path.insert(0, str(ROOT))
    from PIL import ImageOps, Image
    from wp4 import j2_predictor as jp
    from wp4.service import Catalog, open_rgb, present_prediction
    catalog = Catalog(ROOT / "artifacts/data/final_full8000_333_v1.json", ROOT / "DPD_Small_train.csv", a.images)
    with open(ROOT / "DPD_Small_train.csv", newline="") as fh:
        files = [r[0] for r in csv.reader(fh) if r][: a.n]
    for f in files:
        if "test" in f.lower():
            raise SystemExit("refusing a non-training file name")
    exif_rotated = 0
    for f in files:
        with Image.open(Path(a.images) / f) as im:
            exif_rotated += int(im.getexif().get(0x0112, 1) not in (1, None))
    cache = torch.load(a.features, map_location="cpu", weights_only=True)
    assert cache["rows"][: a.n] == list(range(a.n))
    asset = jp.load_asset(a.asset)
    ref_cache = jp.score_features(asset, cache["features"][: a.n, 0].float()).numpy()
    ref_reh = np.load(a.rehearsal_scores)[: a.n].astype(np.float64)
    top = lambda s: np.argsort(-s, 1, kind="stable")[:, :5]  # noqa: E731
    rep = {"n_training_images": a.n, "exif_rotated_images": exif_rotated, "official_test_data_read": False,
           "runtime": {"python": platform.python_version(), "torch": torch.__version__}, "devices": {}}
    for dev in a.devices.split(","):
        if dev == "cuda" and not torch.cuda.is_available():
            rep["devices"][dev] = "unavailable"; continue
        t0 = time.time()
        pred = jp.J2Predictor(catalog, a.clip, a.checkpoint, a.asset, dev)
        load_s = time.time() - t0
        rows, times = [], []
        for f in files:
            # exactly the GUI's branch (wp4/gui.py identify): J2 receives the file's stored pixels
            path = Path(a.images) / f
            img = pred.load_input(path) if getattr(pred, "input_orientation", "exif_corrected") == "stored_pixels" else open_rgb(path)
            t1 = time.time(); r = pred.predict([img], topk=5); times.append(time.time() - t1)
            shown = present_prediction(r, catalog)  # the GUI's own ranking and fail-closed checks
            assert [s["composition_id"] for s in shown] == r["topk_composition_ids"][0].tolist()
            rows.append(r["scores"][0].numpy())
        gui = np.stack(rows)
        out = {"model_load_seconds": round(load_s, 2), "predict_seconds_median": round(float(np.median(times)), 3),
               "predict_seconds_p95": round(float(np.percentile(times, 95)), 3)}
        for name, ref in (("vs_frozen_features_same_asset", ref_cache), ("vs_saved_rehearsal_scores", ref_reh)):
            tg, tr = top(gui), top(ref)
            out[name] = {"top1_agreement": float(np.mean(tg[:, 0] == tr[:, 0])), "top5_identical_rows": float(np.mean((tg == tr).all(1))),
                         "top5_set_overlap_mean": float(np.mean([len(set(x) & set(y)) / 5 for x, y in zip(tg, tr)])),
                         "max_abs_score_gap": float(np.abs(gui - ref).max()), "median_abs_score_gap": float(np.median(np.abs(gui - ref)))}
        out["top1_vs_training_label"] = float(np.mean(top(gui)[:, 0] == np.array([catalog_row["composition_id"] for catalog_row in
                                                                                    _labels(ROOT, files)])))
        rep["devices"][dev] = out
        del pred
        if dev == "cuda":
            torch.cuda.empty_cache()
    Path(a.out).write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


def _labels(root, files):
    with open(root / "DPD_Small_train.csv", newline="") as fh:
        by = {r[0]: int(r[3]) for r in csv.reader(fh) if r}
    return [{"composition_id": by[f]} for f in files]


if __name__ == "__main__":
    main()
