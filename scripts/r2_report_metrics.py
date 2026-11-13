"""Separate-metric report from saved fold scores (no refitting). Writes JSON + per-composition CSV.

  python scripts/r2_report_metrics.py --bench B.json --scores DIR/scores --rows R1 R2 --gammas 0 8 10.75 --out OUTPREFIX
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research_r2.data import HEALTHY, candidate_table, load_training_source  # noqa: E402
from research_r2.metrics import FoldScorer, harmonic  # noqa: E402


def scorer_for(frame, cands, fold, z):
    crops = {c: i for i, c in enumerate(sorted(frame.crop_id.unique()))}
    dis = {d: i for i, d in enumerate(sorted(frame.disease_id.unique()))}
    held = set(fold["heldout_composition_ids"])
    fm = np.array([int(c) not in held for c in cands.comp_id])
    return FoldScorer(z["seen"].astype(np.float64), z["seen_truth"], z["unseen"].astype(np.float64), z["unseen_truth"], fm,
                      np.array([crops[c] for c in cands.crop_id]), np.array([dis[d] for d in cands.disease_id]),
                      cands.disease.eq(HEALTHY).to_numpy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--rows", nargs="+", required=True)
    ap.add_argument("--folds", nargs="+", required=True)
    ap.add_argument("--gammas", nargs="+", type=float, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    frame = load_training_source("DPD_Small_train.csv")
    cands = candidate_table(frame)
    F = {f["name"]: f for f in json.loads(Path(a.bench).read_text())["folds"]}
    rep, comp_rows = {}, []
    for rid in a.rows:
        rep[rid] = {}
        for fn in a.folds:
            z = np.load(Path(a.scores) / rid / f"{fn}_scores.npz")
            sc = scorer_for(frame, cands, F[fn], z)
            entry = {"ausuc": sc.ausuc()}
            for g in a.gammas:
                r = sc.at(g, top5=True)
                entry[f"gamma_{g:g}"] = {k: r[k] for k in (
                    "seen_top1", "seen_top5", "unseen_top1_micro", "unseen_top1_macro", "unseen_top5_micro",
                    "H_bal", "H_micro", "H5_micro", "unseen_withincrop_macro", "seen_withincrop", "H_withincrop",
                    "unseen_crop_top1", "unseen_disease_top1", "unseen_healthy_shortcut", "unseen_pred_fitting_comp")}
                pu = sc.top1_pred("unseen", g)
                t = z["unseen_truth"]
                for col in np.unique(t):
                    m = t == col
                    comp_rows.append({"row": rid, "fold": fn, "gamma": g, "comp": cands.comp[col], "crop": cands.crop[col],
                                      "disease": cands.disease[col], "n": int(m.sum()), "top1": float(np.mean(pu[m] == col))})
            rep[rid][fn] = entry
        means = {}
        for g in a.gammas:
            key = f"gamma_{g:g}"
            means[key] = {k: float(np.mean([rep[rid][f][key][k] for f in a.folds])) for k in rep[rid][a.folds[0]][key]}
        means["ausuc"] = float(np.mean([rep[rid][f]["ausuc"] for f in a.folds]))
        rep[rid]["mean"] = means
    Path(a.out + ".json").write_text(json.dumps(rep, indent=1))
    pd.DataFrame(comp_rows).to_csv(a.out + "_per_composition.csv", index=False)
    for rid in a.rows:
        m = rep[rid]["mean"]
        print(f"== {rid} (mean over {','.join(a.folds)}) AUSUC={100*m['ausuc']:.1f}")
        for g in a.gammas:
            r = m[f"gamma_{g:g}"]
            print(f"  gamma={g:<5g} H_bal={100*r['H_bal']:5.1f} H_micro={100*r['H_micro']:5.1f} seen={100*r['seen_top1']:5.1f} "
                  f"unseen_micro={100*r['unseen_top1_micro']:5.1f} unseen_macro={100*r['unseen_top1_macro']:5.1f} "
                  f"H5={100*r['H5_micro']:5.1f} withincrop_unseen={100*r['unseen_withincrop_macro']:5.1f} "
                  f"shortcut={100*r['unseen_healthy_shortcut']:4.1f}")


if __name__ == "__main__":
    main()
