"""H_RT supplement (v4, post-freeze exploratory, descriptive; written after the H_RT run, changes no registered output).

  python scripts/r2v4_hrt_supplement.py --runs <E>/stage_hrt_20261116/runs --eval <E>/stage_hrt_20261116/evaluation/evaluation.json \
      --out <E>/stage_hrt_20261116/supplement.json

1. Attempt audit: every attempt directory of every arm, with its status (ok / rejected / failed / interrupted) and reason.
2. D1 counterpart: the frozen-feature D1 recomputed for base / ablated / matched-random seed 0 of each pair (must reproduce the
   registered Stage H pair values), so J and D1 are compared ARM BY ARM on identical images and candidates:
   'does the J-over-D1 gain survive retraining without the source?'  = (J - D1) in the ablated arm vs (J - D1) in the base arm.
3. Composition-coverage breakdown (post hoc, descriptive): novel-source seen images binned by the share of their composition's
   fitting images that the ablated arm removed (the matched-random arm removes a different share); accuracy per arm and bin.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from research_r2.candidates import derive_fold  # noqa: E402
from research_r2.methods import RECIPES  # noqa: E402
from research_r2_v4 import hrt, jrun  # noqa: E402
from research_r2_v4.common import LABEL, V4, env, read_jsonl, restrict, score_fold  # noqa: E402
import r2v4_stage_j as sj  # noqa: E402

BINS = [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0001)]


def audit(runs):
    out = []
    for a in sorted(Path(runs).glob("*/*/*/attempt*")):
        row, pair, arm = a.parts[-4], a.parts[-3], a.parts[-2]
        e = {"row": row, "pair": pair, "arm": arm, "attempt": a.name}
        if (a / "hrt_receipt.json").exists() and json.loads((a / "hrt_receipt.json").read_text()).get("status") == "ok":
            e["status"] = "ok"
        elif (a / "failure.json").exists():
            f = json.loads((a / "failure.json").read_text())
            e["status"] = "rejected (trained, failed a post-training check)" if (a / "receipt.json").exists() else "failed"
            e["reason"] = f["error"]
            e["training_completed"] = (a / "receipt.json").exists()
            if (a / "receipt.json").exists():
                r = json.loads((a / "receipt.json").read_text())
                e["worktree_clean_at_start"] = r["code"]["worktree_clean"]; e["git_head"] = r["code"]["git_head"][:7]
        elif (a / "interrupted.json").exists():
            e["status"] = "interrupted"; e["reason"] = json.loads((a / "interrupted.json").read_text())["reason"]
        else:
            e["status"] = "unknown"
        out.append(e)
    return out


def d1_arms(frame, F, bench, fn, s, f_abl):
    feats = sj.frozen_clip(bench)
    rec = RECIPES["synth_composition"]
    fold = F[fn]
    f_rnd = derive_fold(frame, fold, hrt.arm_fold(frame, fold, s, "random")[0]["fitting_rows"])
    return {"base": restrict(frame, score_fold(frame, fold, feats, None, rec, jrun.D1_CFG), f_abl),
            "ablated": restrict(frame, score_fold(frame, f_abl, feats, None, rec, jrun.D1_CFG), f_abl),
            "random": restrict(frame, score_fold(frame, f_rnd, feats, None, rec, jrun.D1_CFG), f_abl)}


def j_arms(frame, F, runs, row, fn, s):
    fold = F[fn]
    bpk, _ = hrt.base_fold(row, fn)
    v = {a: hrt.verify_arm(runs, row, fold, s, a, frame) for a in hrt.ARMS}
    f_abl = v["ablated"][3]
    return f_abl, {"base": restrict(frame, hrt._arm_scores(frame, bpk, fold), f_abl),
                   "ablated": restrict(frame, hrt._arm_scores(frame, v["ablated"][0] / f"{fn}_scores.npz", f_abl), f_abl),
                   "random": restrict(frame, hrt._arm_scores(frame, v["random"][0] / f"{fn}_scores.npz", v["random"][3]), f_abl)}


def acc(sc, g, mask):
    return float(np.mean((sc.top1_pred("seen", g) == sc.parts["seen"]["truth"])[mask])) if mask.any() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True); ap.add_argument("--eval", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    frame, bench, F = env()
    ev_json = json.loads(Path(a.eval).read_text())
    pairs = hrt.declared_pairs()
    src, comp = frame.source.to_numpy(), frame.comp_id.to_numpy()
    rep = {"label": LABEL + "; H_RT supplement (post hoc, descriptive)", "attempts": audit(a.runs), "pairs": {}}
    stage_h = V4 / "stage_h_20261114" / "D1_synthComp_norm" / "pairs.jsonl"
    for fn, s in pairs:
        fold = F[fn]
        f_abl = hrt.arm_fold(frame, fold, s, "ablated")[0]
        D = d1_arms(frame, F, bench, fn, s, f_abl)
        ns = src[np.asarray(f_abl["seen_val_rows"])] == s
        # D1 reproduction of the registered Stage H pair (gamma 0 and D1's pooled gamma)
        reg = next(p for p in read_jsonl(stage_h) if p["fold"] == fn and p["source"] == s)
        gk_d1 = sorted((k for k in reg["base"] if k.startswith("g")), key=lambda k: float(k[1:]))
        g_d1 = float(gk_d1[-1][1:])
        repro = {}
        for arm, regv in (("base", reg["base"]), ("ablated", reg["ablated"]), ("random", reg["random"][0])):
            for gk in gk_d1:
                g = float(gk[1:])
                repro[f"{arm}/{gk}"] = max(abs(acc(D[arm], g, ns) - regv[gk]["novel_seen_top1"]), abs(D[arm].at(g)["H_bal"] - regv[gk]["H_bal"]))
        # composition coverage of each novel-source image
        fit = np.asarray(fold["fitting_rows"])
        f_rnd_rows = np.asarray(hrt.arm_fold(frame, fold, s, "random")[0]["fitting_rows"])
        seen_rows = np.asarray(f_abl["seen_val_rows"])
        nbase = {c: int((comp[fit] == c).sum()) for c in np.unique(comp[seen_rows])}
        frac_abl = np.array([1 - (comp[np.asarray(f_abl["fitting_rows"])] == comp[r]).sum() / nbase[comp[r]] for r in seen_rows])
        frac_rnd = np.array([1 - (comp[f_rnd_rows] == comp[r]).sum() / nbase[comp[r]] for r in seen_rows])
        e = {"d1_stage_h_reproduction_max_abs_diff": max(repro.values()), "d1_gamma": g_d1, "rows": {}}
        for row in hrt.ROWS:
            g = ev_json["rows"][row]["gamma_selected"]
            f_abl_j, J = j_arms(frame, F, a.runs, row, fn, s)
            assert f_abl_j["seen_val_rows"] == f_abl["seen_val_rows"]
            out = {"gamma": g}
            for arm in ("base", "ablated", "random"):
                out[arm] = {"J_novel_seen_top1": acc(J[arm], g, ns), "D1_novel_seen_top1_at_J_gamma": acc(D[arm], g, ns),
                            "D1_novel_seen_top1_at_D1_gamma": acc(D[arm], g_d1, ns),
                            "J_H_bal": J[arm].at(g)["H_bal"], "D1_H_bal_at_J_gamma": D[arm].at(g)["H_bal"], "D1_H_bal_at_D1_gamma": D[arm].at(g_d1)["H_bal"],
                            "J_unseen_disease_top1": J[arm].at(g)["unseen_disease_top1"], "D1_unseen_disease_top1": D[arm].at(g)["unseen_disease_top1"]}
                bs = hrt.paired_bootstrap_pair(J[arm], D[arm], g, ns)
                out[arm]["J_minus_D1_bootstrap"] = {k: {kk: vv for kk, vv in bs[k].items() if kk != "draws"} for k in ("H_bal", "novel_seen_top1")}
            out["gain_survives"] = {m: {"J_minus_D1_base_pp": 100 * (out["base"][f"J_{m}"] - out["base"][f"D1_{m}" + ("_at_J_gamma" if m != "unseen_disease_top1" else "")]),
                                        "J_minus_D1_ablated_pp": 100 * (out["ablated"][f"J_{m}"] - out["ablated"][f"D1_{m}" + ("_at_J_gamma" if m != "unseen_disease_top1" else "")])}
                                    for m in ("novel_seen_top1", "H_bal", "unseen_disease_top1")}
            strata = []
            for lo, hi in BINS:
                m = ns & (frac_abl >= lo) & (frac_abl < hi)
                strata.append({"ablated_removed_share": [lo, min(hi, 1.0)], "n_images": int(m.sum()),
                               "n_compositions": int(len(np.unique(comp[seen_rows][m]))),
                               "random_removed_share_mean": float(frac_rnd[m].mean()) if m.any() else None,
                               **{f"J_{arm}": acc(J[arm], g, m) for arm in ("base", "ablated", "random")},
                               **{f"D1_{arm}": acc(D[arm], g, m) for arm in ("base", "ablated", "random")}})
            out["coverage_strata_novel_seen"] = strata
            e["rows"][row] = out
        rep["pairs"][f"{fn}_{s}"] = e
    Path(a.out).write_text(json.dumps(rep, indent=1, default=float))
    for x in rep["attempts"]:
        if x["status"] != "ok":
            print("attempt", x["row"], x["pair"], x["arm"], x["attempt"], x["status"], x.get("reason", "")[:90])
    print("ok attempts:", sum(x["status"] == "ok" for x in rep["attempts"]), "of", len(rep["attempts"]))
    for k, e in rep["pairs"].items():
        print(f"\n{k}: D1 Stage H reproduction max |diff| {e['d1_stage_h_reproduction_max_abs_diff']:.2e}")
        for row, o in e["rows"].items():
            print(f" {row}: novel-seen J/D1 base {100*o['base']['J_novel_seen_top1']:.1f}/{100*o['base']['D1_novel_seen_top1_at_J_gamma']:.1f} "
                  f"ablated {100*o['ablated']['J_novel_seen_top1']:.1f}/{100*o['ablated']['D1_novel_seen_top1_at_J_gamma']:.1f} "
                  f"random {100*o['random']['J_novel_seen_top1']:.1f}/{100*o['random']['D1_novel_seen_top1_at_J_gamma']:.1f} | "
                  f"H_bal J/D1 base {100*o['base']['J_H_bal']:.1f}/{100*o['base']['D1_H_bal_at_J_gamma']:.1f} ablated {100*o['ablated']['J_H_bal']:.1f}/{100*o['ablated']['D1_H_bal_at_J_gamma']:.1f}")
            gs = o["gain_survives"]
            print("   J-D1 base->ablated:", {m: (round(v['J_minus_D1_base_pp'], 1), round(v['J_minus_D1_ablated_pp'], 1)) for m, v in gs.items()},
                  "| ablated J-D1 novel CI", [round(x, 1) for x in o["ablated"]["J_minus_D1_bootstrap"]["novel_seen_top1"]["ci95_pp"]],
                  "H_bal CI", [round(x, 2) for x in o["ablated"]["J_minus_D1_bootstrap"]["H_bal"]["ci95_pp"]])
            print("   strata", [(st["ablated_removed_share"], st["n_images"], round(st["random_removed_share_mean"] or 0, 2),
                                 *(None if st[f"J_{x}"] is None else round(100 * st[f"J_{x}"]) for x in ("base", "ablated", "random")),
                                 *(None if st[f"D1_{x}"] is None else round(100 * st[f"D1_{x}"]) for x in ("ablated",))) for st in o["coverage_strata_novel_seen"]])


if __name__ == "__main__":
    main()
