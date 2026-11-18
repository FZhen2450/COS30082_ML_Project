"""Second Run final sprint, Phase 1: assignment-aligned development metrics from EXISTING verified packets (no training).

Official H@1 (assignment brief 3.3; First Run evaluation/metrics.py) = harmonic mean of image-weighted seen Top-1 and
image-weighted unseen Top-1. On development folds that is FoldScorer's H_micro (seen_top1, unseen_top1_micro); H_bal
uses the unseen composition-macro accuracy instead. For each row and seed this reports every fold at:
  * gamma 0 (no calibration),
  * the v4 LOFO gamma chosen by H_bal (unchanged v4 rule),
  * a LOFO gamma chosen by H_micro on the OTHER five folds only (ties -> smaller gamma),
and the pooled H_micro-optimal gamma over all six folds (the candidate final calibration, development data only).

  python scripts/r2v5_phase1_eval.py --out <E>/v5/phase1
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research_r2.metrics import GAMMA_GRID  # noqa: E402
from research_r2_v4 import confirm, jrun  # noqa: E402
from research_r2_v4.common import INNER, V4, env  # noqa: E402


def _script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


SJ = _script("r2v4_stage_j")
SI = _script("r2v4_stage_i")
P1, P2 = V4 / "stage_j_phase1_20261114", V4 / "stage_j_phase2_20261115"
J4B = {0: V4 / "stage_j4b_20261117/runs", 1: V4 / "stage_j4b_confirm_20261117/runs", 2: V4 / "stage_j4b_confirm_20261117/runs"}
ROWS = {"J1_clipL_last2": {0: P1, 1: P2, 2: P2}, "J2_clipL_last4": {0: P1, 1: P2, 2: P2},
        "J3_clipL_lora8_mlp": {0: P1, 1: P2, 2: P2}, "J4b_best_photometric_aug": J4B}
KEYS = ["H_micro", "H_bal", "seen_top1", "seen_top5", "unseen_top1_micro", "unseen_top5_micro", "unseen_top1_macro",
        "unseen_crop_top1", "unseen_disease_top1", "unseen_withincrop_macro", "unseen_healthy_shortcut", "H5_micro"]


def grids(t):
    """Per fold: H_micro and H_bal over the gamma grid (single configuration)."""
    out = {}
    for f in INNER:
        s = t.sc[f][0]
        rows = [s.at(float(g)) for g in GAMMA_GRID]
        out[f] = {"H_micro": np.array([r["H_micro"] for r in rows]), "H_bal": np.array([r["H_bal"] for r in rows])}
    return out


def lofo(G, key):
    """gamma index chosen for each held fold on the other five folds (np.argmax -> smallest gamma on ties)."""
    return {k: int(np.argmax(np.mean([G[f][key] for f in INNER if f != k], 0))) for k in INNER}


def evaluate_table(t):
    G = grids(t)
    pol = {"gamma0": {k: 0 for k in INNER}, "lofo_H_bal": lofo(G, "H_bal"), "lofo_H_micro": lofo(G, "H_micro")}
    res = {}
    for name, gi in pol.items():
        per = {}
        for k in INNER:
            r = t.sc[k][0].at(float(GAMMA_GRID[gi[k]]), top5=True)
            per[k] = {x: r[x] for x in KEYS} | {"gamma": float(GAMMA_GRID[gi[k]])}
        res[name] = {"per_fold": per, "mean": {x: float(np.mean([per[k][x] for k in INNER])) for x in KEYS + ["gamma"]}}
    pooled_gi = int(np.argmax(np.mean([G[f]["H_micro"] for f in INNER], 0)))
    res["pooled_H_micro_curve"] = {"gamma_grid": GAMMA_GRID.tolist(),
                                   "mean_H_micro": np.mean([G[f]["H_micro"] for f in INNER], 0).tolist(),
                                   "argmax_gamma": float(GAMMA_GRID[pooled_gi])}
    sc0 = t.sc[INNER[0]][0]
    res["universe"] = {k: {"candidates": int(len(t.sc[k][0].fit_mask)), "fitting_F": int(t.sc[k][0].fit_mask.sum()),
                           "heldout_H": int((~t.sc[k][0].fit_mask).sum()), "seen_images": int(len(t.sc[k][0].parts["seen"]["truth"])),
                           "unseen_images": int(len(t.sc[k][0].parts["unseen"]["truth"]))} for k in INNER}
    del sc0
    return res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    frame, bench, F = env()
    out = {"label": "post-freeze exploratory development re-evaluation (v5 sprint); reused training-derived folds; not official",
           "metric_note": "H_micro = official harmonic formula on image-weighted seen/unseen Top-1 (development candidate universe)",
           "rows": {}}
    t = SI.table("D1_synthComp_norm", frame, F, Path("."))
    out["rows"]["D1_synthComp_norm"] = {"seeds": {"0": evaluate_table(t)}, "note": "frozen backbone; deterministic (single table)"}
    for row, bases in ROWS.items():
        cfg = jrun.j_rows(None if row != "J4b_best_photometric_aug" else "J2_clipL_last4")[row]
        seeds = {}
        for s, b in bases.items():
            v = confirm.verified_row_seed(b, row, s, INNER, cfg)
            seeds[str(s)] = evaluate_table(SJ.table_from_packets(frame, F, v, f"{row}/seed{s}"))
        out["rows"][row] = {"seeds": seeds}
    # three-seed summaries
    for row, d in out["rows"].items():
        S = d["seeds"]
        d["summary"] = {}
        for pol in ("gamma0", "lofo_H_bal", "lofo_H_micro"):
            d["summary"][pol] = {x: float(np.mean([S[s][pol]["mean"][x] for s in S])) for x in KEYS + ["gamma"]}
            d["summary"][pol]["per_fold_H_micro_seedmean"] = {k: float(np.mean([S[s][pol]["per_fold"][k]["H_micro"] for s in S])) for k in INNER}
            d["summary"][pol]["seed_H_micro"] = {s: S[s][pol]["mean"]["H_micro"] for s in S}
        curve = np.mean([S[s]["pooled_H_micro_curve"]["mean_H_micro"] for s in S], 0)
        d["summary"]["pooled_gamma_H_micro"] = {"gamma": float(GAMMA_GRID[int(np.argmax(curve))]), "H_micro_in_sample": float(curve.max()),
                                                "H_micro_at_lofo_Hbal_gamma_8": float(curve[int(np.where(GAMMA_GRID == 8.0)[0][0])])}
    o = Path(a.out); o.mkdir(parents=True, exist_ok=True)
    (o / "phase1_eval.json").write_text(json.dumps(out, indent=1))
    print(f"{'row':28s} {'pol':13s} {'H_micro':>8s} {'H_bal':>7s} {'seen1':>6s} {'seen5':>6s} {'un1':>6s} {'un5':>6s} {'unMac':>6s} "
          f"{'crop':>6s} {'dis':>6s} {'wcrop':>6s} {'hs':>5s} {'gamma':>5s}")
    for row, d in out["rows"].items():
        for pol in ("gamma0", "lofo_H_bal", "lofo_H_micro"):
            m = d["summary"][pol]
            print(f"{row:28s} {pol:13s} {100*m['H_micro']:8.2f} {100*m['H_bal']:7.2f} {100*m['seen_top1']:6.2f} {100*m['seen_top5']:6.2f} "
                  f"{100*m['unseen_top1_micro']:6.2f} {100*m['unseen_top5_micro']:6.2f} {100*m['unseen_top1_macro']:6.2f} "
                  f"{100*m['unseen_crop_top1']:6.2f} {100*m['unseen_disease_top1']:6.2f} {100*m['unseen_withincrop_macro']:6.2f} "
                  f"{100*m['unseen_healthy_shortcut']:5.2f} {m['gamma']:5.2f}")
        print(f"{'':28s} pooled H_micro gamma {d['summary']['pooled_gamma_H_micro']}  per-fold(lofo_H_micro):",
              {k: round(100 * v, 2) for k, v in d["summary"]["lofo_H_micro"]["per_fold_H_micro_seedmean"].items()},
              "seeds:", {s: round(100 * v, 2) for s, v in d["summary"]["lofo_H_micro"]["seed_H_micro"].items()})
    print("universe:", out["rows"]["J2_clipL_last4"]["seeds"]["0"]["universe"])


if __name__ == "__main__":
    main()
