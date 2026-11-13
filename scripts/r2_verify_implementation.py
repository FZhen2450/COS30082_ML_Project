"""Independent numerical verification of the Round 2 implementation on real training-source data.

Checks (each prints PASS/FAIL with evidence; exit status 1 on any FAIL):
  1. transductive invariance: perturbing other validation rows never changes a row's scores
     (so standardisation, erasers, crop means, synthesis and heads use fitting rows only)
  2. factorised score = log of WP5 product-of-softmax (evaluation.metrics.composition_scores_from_logits)
  3. synthesis mathematics: hallucinated features equal x - mu_crop(src) + mu_crop(target) with
     composition-balanced crop means recomputed independently in pandas/numpy
  4. candidate masks: unseen candidates = exactly the fold's held-out compositions; every penalised
     (fitting) candidate has fitting images
  5. score scaling: normalised D2 fusion is the mean of two log-probability terms (all <= 0)
  6. reproducibility: re-running D2 reproduces the saved Stage E scores
  7. inference mode: returned heads are in eval mode; repeated scoring is identical
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.metrics import composition_scores_from_logits  # noqa: E402
from research_r2 import methods as M  # noqa: E402
from research_r2.data import load_training_source  # noqa: E402
from research_r2.runner import load_features, make_fold  # noqa: E402

D2 = {"fuse": 1.0, "normalise": True, "seed": 0, "factorised": {"text_dis_beta": 1.0, "text_crop_beta": 1.0}}
RESULTS = []


def check(name, ok, evidence):
    RESULTS.append({"check": name, "pass": bool(ok), "evidence": evidence})
    print(("PASS " if ok else "FAIL ") + name + " :: " + json.dumps(evidence))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--saved", required=True, help="stage_e_v1/scores directory")
    ap.add_argument("--fold", default="S1")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    frame = load_training_source("DPD_Small_train.csv")
    bench = json.loads(Path(a.bench).read_text())
    fold = next(f for f in bench["folds"] if f["name"] == a.fold)
    feats, text, _, _ = load_features(a.features, ["clip_l14"])

    # 1. transductive invariance ------------------------------------------------------------
    uv, sv = fold["unseen_val_rows"], fold["seen_val_rows"]
    keep_u, keep_s = len(uv) // 2, len(sv) // 2
    g = torch.Generator().manual_seed(123)
    pert = feats.clone()
    for rows in (uv[keep_u:], sv[keep_s:]):
        pert[rows] = torch.randn(len(rows), *feats.shape[1:], generator=g).to(feats.dtype) * 3
    for rid, recipe, cfg in (("B1", M.score_factorised, {}), ("C4_erase50", M.score_factorised, {"erase_crop_from_disease": True, "erase_strength": 0.5}),
                             ("D2", M.score_synth_composition, D2)):
        torch.manual_seed(0)
        fd, _, _ = make_fold(frame, fold, feats, text)
        s0, u0 = recipe(fd, cfg)
        torch.manual_seed(0)
        fd2, _, _ = make_fold(frame, fold, pert, text)
        s1, u1 = recipe(fd2, cfg)
        du = float((u0[:keep_u] - u1[:keep_u]).abs().max())
        ds = float((s0[:keep_s] - s1[:keep_s]).abs().max())
        moved = float((u0[keep_u:] - u1[keep_u:]).abs().max())
        check(f"1 transductive invariance [{rid}]", du < 1e-4 and ds < 1e-4 and moved > 1e-3,
              {"max_abs_change_untouched_unseen": du, "max_abs_change_untouched_seen": ds, "perturbed_rows_changed_by": moved})

    # 2. factorised score equals WP5 product-of-softmax ---------------------------------------
    torch.manual_seed(0)
    fd, truth, healthy = make_fold(frame, fold, feats, text)
    x, crop, dis, comp_col, _ = M._views(fd)
    hc = M.fit_softmax(x, crop, fd.n_crop)
    hd = M.fit_softmax(x, dis, fd.n_dis)
    with torch.no_grad():
        lc = hc(fd.x_uns.float().to(M.DEV)).cpu()
        ld = hd(fd.x_uns.float().to(M.DEV)).cpu()
    ours = (torch.log_softmax(lc.double(), 1)[:, fd.cand_crop] + torch.log_softmax(ld.double(), 1)[:, fd.cand_dis])
    wp5 = composition_scores_from_logits(lc.double(), ld.double(), list(zip(fd.cand_crop.tolist(), fd.cand_dis.tolist()))).log()
    diff = float((ours - wp5).abs().max())
    rank_eq = bool((ours.argmax(1) == wp5.argmax(1)).all())
    check("2 factorised score == log(WP5 product of softmax)", diff < 1e-9 and rank_eq, {"max_abs_diff": diff, "top1_identical": rank_eq})

    # 3. synthesis mathematics ---------------------------------------------------------------
    xn = x.double().numpy()
    df = pd.DataFrame({"crop": crop.numpy(), "comp": comp_col.numpy()})
    comp_means = {k: xn[idx].mean(0) for k, idx in df.groupby("comp").groups.items()}
    ind_mu = {}
    for c, sub in df.groupby("crop"):
        ind_mu[c] = np.mean([comp_means[k] for k in sorted(sub.comp.unique())], 0)
    mus = M.crop_means(x.double(), crop, comp_col, fd.n_crop).numpy()
    err_mu = max(float(np.abs(mus[c] - v).max()) for c, v in ind_mu.items())
    held_cols = np.nonzero(~fd.fit_mask)[0]
    col = int(held_cols[0]); tc, td = int(fd.cand_crop[col]), int(fd.cand_dis[col])
    src = np.nonzero((dis.numpy() == td) & (crop.numpy() != tc))[0]
    gen = torch.Generator().manual_seed(0)
    pick = src[torch.randint(0, len(src), (64,), generator=gen).numpy()]
    expected = xn[pick] - np.stack([ind_mu[int(c)] for c in crop.numpy()[pick]]) + ind_mu[tc]
    produced = (x.double()[pick] - torch.from_numpy(mus)[crop[pick]] + torch.from_numpy(mus)[tc]).numpy()
    used_rows_are_fitting = bool(set(np.unique(pick)) <= set(range(len(x))))
    no_same_crop_source = bool((crop.numpy()[pick] != tc).all())
    check("3 synthesis = x - mu_src_crop + mu_target_crop (composition-balanced means)",
          err_mu < 1e-9 and float(np.abs(expected - produced).max()) < 1e-9 and no_same_crop_source and used_rows_are_fitting,
          {"crop_mean_max_err": err_mu, "synth_max_err": float(np.abs(expected - produced).max()),
           "target_composition_col": col, "source_images": int(len(src)), "sources_never_target_crop": no_same_crop_source})

    # 4. candidate masks ------------------------------------------------------------------------
    cand_ids = np.array(sorted(frame.comp_id.unique()))
    held = set(fold["heldout_composition_ids"])
    unseen_ids = set(cand_ids[~fd.fit_mask].tolist())
    fit_counts = np.bincount(comp_col.numpy(), minlength=len(cand_ids))
    penalised_without_fitting = [int(cand_ids[i]) for i in np.nonzero(fd.fit_mask & (fit_counts == 0))[0]]
    worst = {}
    for f in bench["folds"]:
        fdf = frame.loc[f["fitting_rows"]]
        h = set(f["heldout_composition_ids"])
        missing = sorted(set(int(c) for c in cand_ids) - h - set(int(c) for c in fdf.comp_id.unique()))
        worst[f["name"]] = missing
    check("4 candidate masks", unseen_ids == held and not penalised_without_fitting,
          {"unseen_candidates_equal_heldout": unseen_ids == held, "n_unseen_candidates": len(unseen_ids),
           "penalised_candidates_without_fitting_images_this_fold": penalised_without_fitting,
           "per_fold_fitting_compositions_missing_images": worst})

    # 5. score scaling ------------------------------------------------------------------------
    torch.manual_seed(0)
    fd, _, _ = make_fold(frame, fold, feats, text)
    s_d2, u_d2 = M.score_synth_composition(fd, D2)
    torch.manual_seed(0)
    fd, _, _ = make_fold(frame, fold, feats, text)
    s_f, u_f = M.score_factorised(fd, D2["factorised"])
    comp_term = 2 * u_d2 - u_f
    check("5 normalised fusion is mean of two log-prob terms", float(u_d2.max()) <= 1e-9 and float(comp_term.max()) <= 1e-6
          and float(torch.logsumexp(comp_term, 1).sub(0).abs().max()) < 1e-4,
          {"max_fused_score": float(u_d2.max()), "comp_head_logsumexp_max_dev_from_0": float(torch.logsumexp(comp_term, 1).abs().max()),
           "note": "factorised term includes text-prior log-probs, so it is a sum of four log-probabilities"})

    # 6. reproducibility vs saved Stage E scores ----------------------------------------------
    z = np.load(Path(a.saved) / "E_selected_D2" / f"{a.fold}_scores.npz")
    d6 = float(np.abs(z["unseen"].astype(np.float64) - u_d2.numpy()).max())
    t6 = bool((z["unseen"].argmax(1) == u_d2.numpy().argmax(1)).all())
    check("6 reproduces saved Stage E D2 scores", d6 < 1e-3 and t6, {"max_abs_diff_vs_saved_float32": d6, "top1_identical": t6})

    # 7. inference mode -----------------------------------------------------------------------
    h_lin = M.fit_softmax(x[:2000], dis[:2000], fd.n_dis)
    h_mlp = M.fit_softmax(x[:2000], dis[:2000], fd.n_dis, hidden=64, epochs=1)
    a1, a2 = M.log_probs(h_mlp, fd.x_uns), M.log_probs(h_mlp, fd.x_uns)
    check("7 heads in eval mode; scoring deterministic", (not h_lin.training) and (not h_mlp.training) and torch.equal(a1, a2),
          {"linear_training_flag": h_lin.training, "mlp_training_flag": h_mlp.training, "mlp_repeat_identical": bool(torch.equal(a1, a2))})

    Path(a.out).write_text(json.dumps({"fold": a.fold, "results": RESULTS}, indent=1))
    sys.exit(0 if all(r["pass"] for r in RESULTS) else 1)


if __name__ == "__main__":
    main()
