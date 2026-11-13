"""Run frozen-feature experiment rows over benchmark folds and summarise them under the policy."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import HEALTHY, candidate_table, sha256_json
from .methods import RECIPES, FoldData, standardise
from .metrics import GAMMA_GRID, FoldScorer, lofo_calibrated, pooled_gamma


def load_features(feature_dir, backbones):
    recs = [torch.load(Path(feature_dir) / f"{b}.pt", map_location="cpu", weights_only=False) for b in backbones]
    files = recs[0]["files"]
    assert all(r["files"] == files for r in recs)
    x = torch.cat([r["features"].float() / r["features"].float().norm(dim=-1, keepdim=True) for r in recs], -1) \
        if len(recs) > 1 else recs[0]["features"].float()
    text = recs[0].get("text") if len(recs) == 1 else None  # text banks only align with a single CLIP backbone
    ident = {b: {"weights_sha256": r["weights_sha256"], "image_sha256_list_sha256": r["image_sha256_list_sha256"]}
             for b, r in zip(backbones, recs)}
    return x, text, files, ident


def make_fold(frame, fold, feats, text, text_backbone_only=None):
    cands = candidate_table(frame)
    col_of = {int(c): i for i, c in enumerate(cands.comp_id)}
    crops = sorted(frame.crop_id.unique()); dis = sorted(frame.disease_id.unique())
    ci = {c: i for i, c in enumerate(crops)}; di = {d: i for i, d in enumerate(dis)}
    srcs = sorted(frame.source.unique()); si = {s: i for i, s in enumerate(srcs)}
    fit = frame.loc[fold["fitting_rows"]]
    held = set(fold["heldout_composition_ids"])
    fd = FoldData(
        x_fit=feats[fold["fitting_rows"]], x_seen=feats[fold["seen_val_rows"], 0], x_uns=feats[fold["unseen_val_rows"], 0],
        fit_crop=torch.tensor([ci[c] for c in fit.crop_id]), fit_dis=torch.tensor([di[d] for d in fit.disease_id]),
        fit_comp_col=torch.tensor([col_of[c] for c in fit.comp_id]), fit_source=torch.tensor([si[s] for s in fit.source]),
        cand_crop=np.array([ci[c] for c in cands.crop_id]), cand_dis=np.array([di[d] for d in cands.disease_id]),
        fit_mask=np.array([int(c) not in held for c in cands.comp_id]),
        n_crop=len(crops), n_dis=len(dis), n_source=len(srcs))
    if text is not None:
        # text banks are ordered by ascending crop / disease / composition IDs, matching ci / di / col_of
        assert text["crop_ids"] == crops and text["disease_ids"] == dis and text["comp_ids"] == cands.comp_id.tolist()
        fd.text = text
    truth = {"seen": np.array([col_of[c] for c in frame.comp_id[fold["seen_val_rows"]]]),
             "unseen": np.array([col_of[c] for c in frame.comp_id[fold["unseen_val_rows"]]])}
    healthy = cands.disease.eq(HEALTHY).to_numpy()
    return standardise(fd), truth, healthy


def run_row(frame, bench, feature_dir, row, folds=None, out_dir=None, poison_rows=None):
    feats, text, files, ident = load_features(feature_dir, row["backbones"])
    assert files == frame.file.tolist()
    if poison_rows:
        # v2 inner stage: outer-population features become NaN, so any accidental use in fitting,
        # statistics or scoring raises (FoldScorer rejects non-finite scores) instead of leaking
        feats = feats.clone()
        feats[list(poison_rows)] = float("nan")
    recipe = RECIPES[row["recipe"]]
    results = {}
    t0 = time.time()
    for fold in bench["folds"]:
        if folds is not None and fold["name"] not in folds:
            continue
        torch.manual_seed(0)
        fd, truth, healthy = make_fold(frame, fold, feats, text)
        s_seen, s_uns = recipe(fd, row.get("cfg", {}))
        if row.get("tta_hflip"):
            # identical fitting, evaluation on the h-flip view; average composition log-scores
            # swap the two views: fitting multiset (hence statistics and heads) is unchanged up to
            # summation order, while evaluation now uses the h-flip view
            flipped = feats[:, [1, 0]]
            torch.manual_seed(0)
            fd2, _, _ = make_fold(frame, fold, flipped, text)
            f_seen, f_uns = recipe(fd2, row.get("cfg", {}))
            s_seen, s_uns = (s_seen + f_seen) / 2, (s_uns + f_uns) / 2
        sc = FoldScorer(s_seen.numpy(), truth["seen"], s_uns.numpy(), truth["unseen"], fd.fit_mask,
                        fd.cand_crop, fd.cand_dis, healthy)
        results[fold["name"]] = {"curve": sc.curve(), "ausuc": sc.ausuc(), "gamma0": sc.at(0.0, top5=True), "_scorer": sc}
        if out_dir is not None:
            p = Path(out_dir) / row["id"]
            p.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(p / f"{fold['name']}_scores.npz", seen=s_seen.numpy().astype(np.float32),
                                unseen=s_uns.numpy().astype(np.float32), seen_truth=truth["seen"], unseen_truth=truth["unseen"])
    return results, ident, time.time() - t0


def summarise(row, results, ident, seconds, screen=("S1", "S2", "S3", "S4", "S5")):
    sc = [f for f in screen if f in results]
    curves = {f: results[f]["curve"] for f in sc}
    cal = lofo_calibrated(curves) if len(sc) >= 2 else {}
    g_pool = pooled_gamma(curves) if sc else None
    per_fold = {}
    for f, r in results.items():
        entry = {"gamma0": r["gamma0"], "ausuc": r["ausuc"]}
        if f in cal:
            entry["lofo"] = cal[f]
        if g_pool is not None:
            entry["pooled_gamma"] = r["_scorer"].at(g_pool, top5=True)
        per_fold[f] = entry
    def mean(key, sub):
        vals = [per_fold[f][sub][key] for f in sc if sub in per_fold[f]]
        return float(np.mean(vals)) if vals else None
    summary = {
        "id": row["id"], "hypothesis": row.get("hypothesis", ""), "backbones": row["backbones"], "recipe": row["recipe"],
        "cfg": row.get("cfg", {}), "control": row.get("control"), "seconds": seconds, "feature_identity": ident,
        "screen_mean": {
            "primary_H_bal_lofo": mean("H_bal", "lofo"), "H_bal_gamma0": mean("H_bal", "gamma0"),
            "H_micro_lofo": mean("H_micro", "lofo"), "seen_top1_lofo": mean("seen_top1", "lofo"),
            "unseen_macro_lofo": mean("unseen_top1_macro", "lofo"), "unseen_micro_lofo": mean("unseen_top1_micro", "lofo"),
            "unseen_disease_lofo": mean("unseen_disease_top1", "lofo"), "unseen_crop_lofo": mean("unseen_crop_top1", "lofo"),
            "healthy_shortcut_lofo": mean("unseen_healthy_shortcut", "lofo"),
            "healthy_shortcut_gamma0": mean("unseen_healthy_shortcut", "gamma0"),
            "seen_top1_gamma0": mean("seen_top1", "gamma0"), "unseen_macro_gamma0": mean("unseen_top1_macro", "gamma0"),
            "H5_micro_gamma0": mean("H5_micro", "gamma0"),
            "unseen_withincrop_gamma0": mean("unseen_withincrop_macro", "gamma0"),
            "unseen_withincrop_lofo": mean("unseen_withincrop_macro", "lofo"),
            "H_withincrop_lofo": mean("H_withincrop", "lofo"),
            "ausuc": float(np.mean([per_fold[f]["ausuc"] for f in sc])) if sc else None,
            "pooled_gamma": g_pool,
        },
        "per_fold": per_fold,
    }
    return summary


def gate(cand: dict, ctrl: dict, screen=("S1", "S2", "S3", "S4", "S5")) -> dict:
    """Apply the predeclared improvement gate (research_r2/protocol.json)."""
    d = [cand["per_fold"][f]["lofo"]["H_bal"] - ctrl["per_fold"][f]["lofo"]["H_bal"] for f in screen]
    cm, km = cand["screen_mean"], ctrl["screen_mean"]
    checks = {
        "mean_gain_pp": 100 * (cm["primary_H_bal_lofo"] - km["primary_H_bal_lofo"]),
        "folds_improved": int(sum(x > 0 for x in d)),
        "ausuc_delta": cm["ausuc"] - km["ausuc"],
        "shortcut_delta_pp": 100 * (cm["healthy_shortcut_lofo"] - km["healthy_shortcut_lofo"]),
    }
    need = len(screen) - 1  # v1: 4 of 5 screening folds; v2: 5 of 6 inner folds (declared in protocol_v2)
    checks["folds_required"] = need
    checks["pass"] = bool(checks["mean_gain_pp"] >= 1.0 and checks["folds_improved"] >= need
                          and checks["ausuc_delta"] >= 0 and checks["shortcut_delta_pp"] <= 5.0)
    return checks
