"""Calibration (gamma) robustness from training-derived fold scores only. Selects nothing.

Analyses
  A  per-fold oracle gamma and the regret of a fixed gamma
  B  composition bootstrap of the pooled gamma (resample held-out compositions + seen images per fold)
  C  number of unseen candidates: restrict each fold to k of its held-out pairs (k = 3, 5, 8, all)
  D  groups: unseen Top-1 at fixed gamma by held-out crop, disease and source dataset, including
     whether the image's source dataset occurs in that fold's fitting set; seen Top-1 by source
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
from research_r2.metrics import GAMMA_GRID, FoldScorer, harmonic  # noqa: E402


def load(frame, cands, fold, path, keep_cols=None, keep_unseen_comps=None):
    z = np.load(path)
    crops = {c: i for i, c in enumerate(sorted(frame.crop_id.unique()))}
    dis = {d: i for i, d in enumerate(sorted(frame.disease_id.unique()))}
    held = set(fold["heldout_composition_ids"])
    cid = cands.comp_id.to_numpy()
    s, u, ts, tu = z["seen"].astype(np.float64), z["unseen"].astype(np.float64), z["seen_truth"], z["unseen_truth"]
    urows = np.array(fold["unseen_val_rows"])
    if keep_unseen_comps is not None:
        keep_c = np.array([c not in held or c in keep_unseen_comps for c in cid])
        m = np.isin(cid[tu], list(keep_unseen_comps))
        u, tu, urows = u[m], tu[m], urows[m]
        remap = -np.ones(len(cid), int); remap[keep_c] = np.arange(keep_c.sum())
        s, u, ts, tu, cid2 = s[:, keep_c], u[:, keep_c], remap[ts], remap[tu], cid[keep_c]
    else:
        cid2 = cid
    sub = cands.set_index("comp_id").loc[cid2]
    fm = np.array([c not in held for c in cid2])
    sc = FoldScorer(s, ts, u, tu, fm, np.array([crops[c] for c in sub.crop_id]), np.array([dis[d] for d in sub.disease_id]),
                    sub.disease.eq(HEALTHY).to_numpy())
    return sc, urows, cid2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--scores", required=True, help="directory with <row>/<fold>_scores.npz")
    ap.add_argument("--row", required=True)
    ap.add_argument("--folds", nargs="+", required=True)
    ap.add_argument("--fixed-gamma", type=float, required=True)
    ap.add_argument("--boot", type=int, default=500)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    frame = load_training_source("DPD_Small_train.csv")
    cands = candidate_table(frame)
    F = {f["name"]: f for f in json.loads(Path(a.bench).read_text())["folds"]}
    g0 = a.fixed_gamma
    gi = int(np.argmin(np.abs(GAMMA_GRID - g0)))
    rep = {"row": a.row, "folds": a.folds, "fixed_gamma": g0}

    # A. per-fold oracle and regret
    curves, scorers = {}, {}
    for fn in a.folds:
        sc, urows, _ = load(frame, cands, F[fn], Path(a.scores) / a.row / f"{fn}_scores.npz")
        scorers[fn] = (sc, urows)
        curves[fn] = np.array([sc.at(g)["H_bal"] for g in GAMMA_GRID])
    A = {fn: {"oracle_gamma": float(GAMMA_GRID[c.argmax()]), "oracle_H_bal": float(c.max()), "fixed_H_bal": float(c[gi]),
              "regret_pp": float(100 * (c.max() - c[gi])),
              "gamma_range_within_2pp": [float(GAMMA_GRID[np.nonzero(c >= c.max() - 0.02)[0][0]]),
                                         float(GAMMA_GRID[np.nonzero(c >= c.max() - 0.02)[0][-1]])]} for fn, c in curves.items()}
    rep["A_per_fold"] = A

    # B. composition bootstrap of the pooled gamma
    rng = np.random.default_rng(0)
    boots = []
    pre = {}
    for fn, (sc, _) in scorers.items():
        ps = {g: sc.top1_pred("seen", g) == sc.parts["seen"]["truth"] for g in GAMMA_GRID}
        pu = {g: sc.top1_pred("unseen", g) == sc.parts["unseen"]["truth"] for g in GAMMA_GRID}
        pre[fn] = (ps, pu, sc.parts["unseen"]["truth"], len(sc.parts["seen"]["truth"]))
    for _ in range(a.boot):
        tot = np.zeros(len(GAMMA_GRID))
        for fn, (ps, pu, tu, ns) in pre.items():
            comps = np.unique(tu)
            pick = rng.choice(comps, len(comps), replace=True)
            srows = rng.integers(0, ns, ns)
            for j, g in enumerate(GAMMA_GRID):
                s_acc = ps[g][srows].mean()
                u_acc = np.mean([pu[g][tu == c].mean() for c in pick])
                tot[j] += harmonic(s_acc, u_acc)
        boots.append(float(GAMMA_GRID[tot.argmax()]))
    boots = np.array(boots)
    rep["B_bootstrap_pooled_gamma"] = {"median": float(np.median(boots)), "p2.5": float(np.percentile(boots, 2.5)),
                                       "p97.5": float(np.percentile(boots, 97.5)), "share_within_pm1_of_fixed": float(np.mean(np.abs(boots - g0) <= 1))}

    # C. number of unseen candidates
    C = {}
    for k in (3, 5, 8, None):
        og, fixed_h, oracle_h = [], [], []
        for fn in a.folds:
            held = sorted(F[fn]["heldout_composition_ids"])
            for rep_i in range(5 if k else 1):
                keep = set(held if k is None else np.random.default_rng(rep_i).choice(held, k, replace=False).tolist())
                sc, _, _ = load(frame, cands, F[fn], Path(a.scores) / a.row / f"{fn}_scores.npz", keep_unseen_comps=keep)
                c = np.array([sc.at(g)["H_bal"] for g in GAMMA_GRID])
                og.append(GAMMA_GRID[c.argmax()]); fixed_h.append(c[gi]); oracle_h.append(c.max())
        C["all" if k is None else str(k)] = {"median_oracle_gamma": float(np.median(og)), "mean_H_bal_fixed": float(np.mean(fixed_h)),
                                             "mean_H_bal_oracle": float(np.mean(oracle_h))}
    rep["C_unseen_candidate_count"] = C

    # D. groups
    rows = []
    for fn, (sc, urows) in scorers.items():
        fit_sources = set(frame.source.loc[F[fn]["fitting_rows"]])
        pu = sc.top1_pred("unseen", g0); tu = sc.parts["unseen"]["truth"]
        sub = frame.loc[urows]
        for r, p, t in zip(sub.itertuples(), pu, tu):
            rows.append({"fold": fn, "part": "unseen", "crop": r.crop, "disease": r.disease, "comp": r.comp, "source": r.source,
                         "source_in_fitting": r.source in fit_sources, "correct": bool(p == t)})
        ps = sc.top1_pred("seen", g0); ts = sc.parts["seen"]["truth"]; ps0 = sc.top1_pred("seen", 0.0)
        for r, p, p0, t in zip(frame.loc[F[fn]["seen_val_rows"]].itertuples(), ps, ps0, ts):
            rows.append({"fold": fn, "part": "seen", "crop": r.crop, "disease": r.disease, "comp": r.comp, "source": r.source,
                         "source_in_fitting": True, "correct": bool(p == t), "correct_gamma0": bool(p0 == t)})
    d = pd.DataFrame(rows)
    u = d[d.part == "unseen"]; s = d[d.part == "seen"]
    comp_acc = u.groupby(["fold", "comp", "crop", "disease", "source_in_fitting"]).correct.mean().reset_index()
    rep["D_unseen_by_source_novelty"] = {str(k): {"compositions": int(len(v)), "macro_top1": float(v.correct.mean())}
                                         for k, v in comp_acc.groupby("source_in_fitting")}
    per_crop = comp_acc.groupby("crop").correct.agg(["mean", "size"])
    per_dis = comp_acc.groupby("disease").correct.agg(["mean", "size"])
    rep["D_unseen_by_crop_macro"] = {"spread_p10_p90": [float(per_crop["mean"].quantile(0.1)), float(per_crop["mean"].quantile(0.9))],
                                     "n_crops": int(len(per_crop)), "worst": per_crop.sort_values("mean").head(5)["mean"].round(3).to_dict()}
    rep["D_unseen_by_disease_macro"] = {"spread_p10_p90": [float(per_dis["mean"].quantile(0.1)), float(per_dis["mean"].quantile(0.9))],
                                        "n_diseases": int(len(per_dis)), "worst": per_dis.sort_values("mean").head(5)["mean"].round(3).to_dict()}
    src = s.groupby("source").agg(n=("correct", "size"), top1_fixed=("correct", "mean"), top1_gamma0=("correct_gamma0", "mean"))
    src["drop_pp"] = 100 * (src.top1_gamma0 - src.top1_fixed)
    src = src[src.n >= 20]
    rep["D_seen_cost_by_source"] = {"n_sources_ge20": int(len(src)), "median_drop_pp": float(src.drop_pp.median()),
                                    "max_drop_pp": float(src.drop_pp.max()), "worst": src.sort_values("drop_pp", ascending=False).head(5).round(3).to_dict(orient="index")}
    Path(a.out).write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps({k: v for k, v in rep.items() if k != "A_per_fold"}, indent=1, default=float))
    for fn, v in A.items():
        print(f"  {fn}: oracle gamma {v['oracle_gamma']:5.2f}  H {100*v['oracle_H_bal']:5.1f}  fixed {100*v['fixed_H_bal']:5.1f}  "
              f"regret {v['regret_pp']:4.1f} pp  within-2pp range {v['gamma_range_within_2pp']}")


if __name__ == "__main__":
    main()
