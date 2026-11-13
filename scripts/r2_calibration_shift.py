"""Calibration under source shift and fitting-set size, on v2 INNER folds only (outer features poisoned).

Source shift: for each inner fold and each source dataset s that supplies held-out (unseen) images,
refit with every image from s removed from fitting. Compare, for the same images, decisions and
margins when s is in fitting (standard fold) versus source-novel (ablated fold):
  unseen images from s (held-out pairs)            -> unseen Top-1 at gamma, margin d = best_fit - best_heldout
  seen-val images from s whose pair keeps fitting images elsewhere -> seen Top-1 at gamma
Fitting size: composition-stratified subsamples (50%, 75%, 100%) of each inner fold's fitting rows;
report the pooled H_bal-optimal gamma per size (diagnostic; nothing is selected).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research_r2.benchmark import heldout_set_valid  # noqa: E402
from research_r2.data import HEALTHY, candidate_table, load_training_source  # noqa: E402
from research_r2.methods import RECIPES  # noqa: E402
from research_r2.metrics import GAMMA_GRID, FoldScorer  # noqa: E402
from research_r2.runner import load_features, make_fold  # noqa: E402


def scorer(frame, cands, fold, s_seen, s_uns, truth, healthy):
    crops = {c: i for i, c in enumerate(sorted(frame.crop_id.unique()))}
    dis = {d: i for i, d in enumerate(sorted(frame.disease_id.unique()))}
    held = set(fold["heldout_composition_ids"])
    fm = np.array([int(c) not in held for c in cands.comp_id])
    return FoldScorer(s_seen.numpy(), truth["seen"], s_uns.numpy(), truth["unseen"], fm,
                      np.array([crops[c] for c in cands.crop_id]), np.array([dis[d] for d in cands.disease_id]), healthy)


def run(frame, fold, feats, text, recipe, cfg):
    torch.manual_seed(0)
    fd, truth, healthy = make_fold(frame, fold, feats, text)
    s, u = recipe(fd, cfg)
    return s, u, truth, healthy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--matrix", required=True)
    ap.add_argument("--row", required=True)
    ap.add_argument("--gammas", nargs="+", type=float, required=True, help="gammas to report (e.g. inner gamma and 8)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    frame = load_training_source("DPD_Small_train.csv")
    cands = candidate_table(frame)
    bench = json.loads(Path(a.bench).read_text())
    row = next(r for r in json.loads(Path(a.matrix).read_text()) if r["id"] == a.row)
    recipe, cfg = RECIPES[row["recipe"]], row.get("cfg", {})
    feats, text, _, _ = load_features(a.features, row["backbones"])
    feats = feats.clone(); feats[bench["outer_exclusion_rows"]] = float("nan")  # inner-only analysis
    inner = [f for f in bench["folds"] if f["role"] == "inner"]
    rep = {"row": a.row, "gammas": a.gammas, "source_shift": [], "fitting_size": {}}

    for fold in inner:
        s, u, truth, healthy = run(frame, fold, feats, text, recipe, cfg)
        base = scorer(frame, cands, fold, s, u, truth, healthy)
        urows = np.array(fold["unseen_val_rows"]); srows = np.array(fold["seen_val_rows"])
        for src in sorted(set(frame.source.loc[urows])):
            fit2 = [r for r in fold["fitting_rows"] if frame.source[r] != src]
            held = set(fold["heldout_composition_ids"])
            sub = frame.loc[fit2]
            hsub = frame[frame.comp_id.isin(held)].drop_duplicates("comp_id")
            if not (set(hsub.crop) <= set(sub.crop) and set(hsub.disease) <= set(sub.disease)):
                rep["source_shift"].append({"fold": fold["name"], "source": src, "skipped": "concept coverage lost"})
                continue
            f2 = dict(fold, fitting_rows=fit2)
            s2, u2, _, _ = run(frame, f2, feats, text, recipe, cfg)
            abl = scorer(frame, cands, fold, s2, u2, truth, healthy)
            um = frame.source.loc[urows].to_numpy() == src
            fit_comps2 = set(sub.comp_id)
            sm = (frame.source.loc[srows].to_numpy() == src) & np.isin(frame.comp_id.loc[srows].to_numpy(), list(fit_comps2))
            entry = {"fold": fold["name"], "source": src, "n_unseen": int(um.sum()), "n_seen": int(sm.sum()),
                     "removed_fitting_images": len(fold["fitting_rows"]) - len(fit2),
                     "unseen_margin_shift_mean": float(np.mean(abl.parts["unseen"]["d"][um] - base.parts["unseen"]["d"][um])),
                     "seen_margin_shift_mean": float(np.mean(abl.parts["seen"]["d"][sm] - base.parts["seen"]["d"][sm])) if sm.any() else None}
            for g in a.gammas:
                pb, pa = base.top1_pred("unseen", g), abl.top1_pred("unseen", g)
                tu = base.parts["unseen"]["truth"]
                entry[f"unseen_top1_in_fitting_g{g:g}"] = float(np.mean(pb[um] == tu[um]))
                entry[f"unseen_top1_source_novel_g{g:g}"] = float(np.mean(pa[um] == tu[um]))
                if sm.any():
                    qb, qa = base.top1_pred("seen", g), abl.top1_pred("seen", g)
                    ts = base.parts["seen"]["truth"]
                    entry[f"seen_top1_in_fitting_g{g:g}"] = float(np.mean(qb[sm] == ts[sm]))
                    entry[f"seen_top1_source_novel_g{g:g}"] = float(np.mean(qa[sm] == ts[sm]))
                    entry[f"seen_to_heldout_source_novel_g{g:g}"] = float(np.mean(~base.fit_mask[qa[sm]]))
            rep["source_shift"].append(entry)
            print(json.dumps(entry), flush=True)

    rng = np.random.default_rng(0)
    for frac in (0.5, 0.75, 1.0):
        curves = []
        for fold in inner:
            fr = frame.loc[fold["fitting_rows"]]
            keep = []
            for _, idx in fr.groupby("comp_id").groups.items():
                idx = list(idx)
                k = max(1, math.ceil(frac * len(idx)))
                keep += list(rng.choice(idx, k, replace=False))
            s, u, truth, healthy = run(frame, dict(fold, fitting_rows=sorted(int(r) for r in keep)), feats, text, recipe, cfg)
            sc = scorer(frame, cands, fold, s, u, truth, healthy)
            curves.append([sc.at(g)["H_bal"] for g in GAMMA_GRID])
        m = np.mean(curves, 0)
        rep["fitting_size"][str(frac)] = {"pooled_oracle_gamma": float(GAMMA_GRID[m.argmax()]), "H_bal_at_oracle": float(m.max()),
                                          **{f"H_bal_g{g:g}": float(m[int(np.argmin(np.abs(GAMMA_GRID - g)))]) for g in a.gammas}}
        print(frac, rep["fitting_size"][str(frac)], flush=True)
    Path(a.out).write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
