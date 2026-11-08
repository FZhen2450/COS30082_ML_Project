"""Report-ready confusion artifacts from an existing build_report matrix; no rescoring.

The matrix is the one build_report already computed: rows are the fold's truth
axis (323 seen classes for final), columns every allowed prediction (all 333).
This module only reorders, aggregates and draws it.
"""
from __future__ import annotations

import csv
import io

import numpy as np


def _csv(rows, fields):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def confusion_files(report, *, fold="seen", top_n=25, title=""):
    """Return {filename: bytes}: crop-grouped heatmap PNG, top confusions, crop-level counts."""
    matrix = report[fold]["confusion_matrix"]
    by_id = {c["composition_id"]: c for c in report["candidate_spec"]["candidates"]}
    counts = np.asarray(matrix["counts"], dtype=np.int64)
    truth, pred = list(matrix["truth_ids"]), list(matrix["prediction_ids"])
    if counts.shape != (len(truth), len(pred)) or int(counts.sum()) != matrix["count"]:
        raise ValueError("Confusion matrix shape or total differs from its declared axes")
    key = lambda i: (by_id[i]["crop_name"], i)  # group by crop, then original global ID
    rows = sorted(range(len(truth)), key=lambda r: key(truth[r]))
    cols = sorted(range(len(pred)), key=lambda c: key(pred[c]))
    ordered = counts[np.ix_(rows, cols)]
    support = ordered.sum(axis=1)

    cells = [(int(counts[r, c]), truth[r], pred[c], r) for r in range(len(truth)) for c in range(len(pred))
             if truth[r] != pred[c] and counts[r, c]]
    cells.sort(key=lambda x: (-x[0], x[1], x[2]))
    top = []
    for rank, (n, t, p, r) in enumerate(cells[:top_n], 1):
        a, b = by_id[t], by_id[p]
        top.append({"rank": rank, "truth_id": t, "truth_label": a["composition_name"],
                    "predicted_id": p, "predicted_label": b["composition_name"], "count": n,
                    "truth_support": int(counts[r].sum()), "share_of_truth_row": n / int(counts[r].sum()),
                    "same_crop": a["crop_id"] == b["crop_id"], "same_disease": a["disease_id"] == b["disease_id"],
                    "prediction_in_truth_axis": p in truth})

    crop = {}
    for r, t in enumerate(truth):
        for c, p in enumerate(pred):
            if counts[r, c]:
                k = (by_id[t]["crop_name"], by_id[p]["crop_name"])
                crop[k] = crop.get(k, 0) + int(counts[r, c])
    crop_rows = [{"truth_crop": a, "predicted_crop": b, "count": n} for (a, b), n in sorted(crop.items())]

    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    figure = Figure(figsize=(14, 13), layout="constrained")
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    with np.errstate(invalid="ignore", divide="ignore"):
        normalised = np.where(support[:, None] > 0, ordered / support[:, None], np.nan)
    image = axis.imshow(normalised, cmap="viridis", vmin=0, vmax=1, interpolation="nearest", aspect="auto")
    figure.colorbar(image, ax=axis, shrink=0.6, label="Row-normalised share of truth class")
    for axis_ids, order, draw in ((truth, rows, axis.axhline), (pred, cols, axis.axvline)):
        names = [by_id[axis_ids[i]]["crop_name"] for i in order]
        for i in range(1, len(names)):
            if names[i] != names[i - 1]:
                draw(i - 0.5, color="white", linewidth=0.3)
    axis.set(xlabel=f"Predicted composition ({len(pred)} allowed, grouped by crop)",
             ylabel=f"True composition ({len(truth)} classes, grouped by crop; blank rows = no images)",
             xticks=[], yticks=[])
    axis.set_title(title or f"{fold} confusion matrix (Top-1), n={matrix['count']}")
    png = io.BytesIO()
    figure.savefig(png, format="png", dpi=200, metadata={"Software": "WP5 confusion artifacts"})
    top_fields = ["rank", "truth_id", "truth_label", "predicted_id", "predicted_label", "count", "truth_support",
                  "share_of_truth_row", "same_crop", "same_disease", "prediction_in_truth_axis"]
    return {f"{fold}_confusion_heatmap.png": png.getvalue(),
            f"{fold}_confusion_top_pairs.csv": _csv(top, top_fields),
            f"{fold}_confusion_crop_level.csv": _csv(crop_rows, ["truth_crop", "predicted_crop", "count"])}


__all__ = ["confusion_files"]
