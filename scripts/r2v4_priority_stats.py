"""Descriptive data facts for prioritising J5 / J4b / J7 (no model, no training; post-H_RT, adaptive).

  python scripts/r2v4_priority_stats.py --out <E>/stage_hrt_20261116/priority_stats.json

Per inner fold (fitting rows only): how many sources each composition, crop and disease has; the share of fitting
images in multi-source compositions; the dominant-source share. Plus the seen-val source strata of the Phase 1/2
J rows (dominant / other fitting source / novel source) from the registered evaluation.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research_r2_v4.common import INNER, LABEL, V4, env  # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
    frame, bench, F = env()
    src, comp, crop, dis = (frame[c].to_numpy() for c in ("source", "comp_id", "crop_id", "disease_id"))
    per = {}
    for fn in INNER:
        fit = np.asarray(F[fn]["fitting_rows"])
        e = {}
        for name, key in (("composition", comp), ("crop", crop), ("disease", dis)):
            groups = {}
            for g, s in zip(key[fit], src[fit]):
                groups.setdefault(g, []).append(s)
            nsrc = {g: len(set(v)) for g, v in groups.items()}
            dom = {g: max(np.unique(v, return_counts=True)[1]) / len(v) for g, v in groups.items()}
            n = {g: len(v) for g, v in groups.items()}
            multi = [g for g in groups if nsrc[g] >= 2]
            e[name] = {"groups": len(groups), "multi_source_groups": len(multi),
                       "images_in_multi_source_groups_share": float(sum(n[g] for g in multi) / len(fit)),
                       "median_sources_per_group": float(np.median(list(nsrc.values()))),
                       "image_weighted_dominant_source_share": float(sum(dom[g] * n[g] for g in groups) / len(fit))}
        per[fn] = e
    agg = {name: {k: float(np.mean([per[f][name][k] for f in INNER])) for k in per[INNER[0]][name]} for name in ("composition", "crop", "disease")}
    ev2 = json.loads((V4 / "stage_j_phase2_20261115" / "evaluation" / "evaluation.json").read_text())
    strata = {r: e["seed0"]["source_strata_seen"] for r, e in ev2["rows"].items()}
    rep = {"label": LABEL + "; priority facts (descriptive)", "per_fold": per, "mean_over_folds": agg, "seen_val_source_strata_seed0": strata,
           "n_sources_total": int(len(np.unique(src)))}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps({"mean_over_folds": agg, "strata": strata, "n_sources": rep["n_sources_total"]}, indent=1))


if __name__ == "__main__":
    main()
