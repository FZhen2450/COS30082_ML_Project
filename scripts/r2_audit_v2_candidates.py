"""Audit v2 inner-fold candidate masks: 'phantom' candidates penalised as fitting without fitting images.

Post-hoc impact on saved v2 inner scores: compare metrics with phantom columns kept (as scored) vs
removed (column deletion; heads unchanged). Writes artifacts/r2/audit_v2_candidates.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research_r2.data import HEALTHY, candidate_table, load_training_source  # noqa: E402
from research_r2.metrics import GAMMA_GRID, FoldScorer  # noqa: E402

EXP = ROOT.parents[1] / "experiments/r2-compositional/v2"
ROWS = [("B1_clipL_factorised", "inner_stage_d_v1", 10.5), ("D1_synthComp_norm", "inner_stage_d_v1", 8.0),
        ("D2_synthComp_norm_text", "inner_stage_d_v1", 8.0), ("B0_clipL_zeroshot_comp", "inner_stage_bc_v1", 1.75)]


def main():
    fr = load_training_source(ROOT / "DPD_Small_train.csv"); c = candidate_table(fr); cid = c.comp_id.to_numpy()
    crops = {x: i for i, x in enumerate(sorted(fr.crop_id.unique()))}; dis = {x: i for i, x in enumerate(sorted(fr.disease_id.unique()))}
    cc = np.array([crops[x] for x in c.crop_id]); cd = np.array([dis[x] for x in c.disease_id]); hl = c.disease.eq(HEALTHY).to_numpy()
    b = json.loads((ROOT / "artifacts/r2/benchmark_v2.json").read_text())
    rep = {"folds": {}, "rows": {}}
    for f in b["folds"]:
        held = set(f["heldout_composition_ids"]); fitc = set(fr.comp_id.loc[f["fitting_rows"]])
        nodata = sorted(int(x) for x in cid if x not in held and x not in fitc)
        rep["folds"][f["name"]] = {"role": f["role"], "unseen_candidates": len(held), "penalised_without_fitting_images": len(nodata),
                                   "are_outer_compositions": sorted(nodata) == sorted(set(nodata) & set(b["outer_composition_ids"]))}
    ph = np.isin(cid, b["outer_composition_ids"])
    for rid, stage, g in ROWS:
        agg = {k: [] for k in ("old", "new")}; hits = {"seen": 0, "unseen": 0, "seen_top5_slots": 0, "n_seen": 0, "n_unseen": 0}
        for f in [x for x in b["folds"] if x["role"] == "inner"]:
            z = np.load(EXP / stage / "scores" / rid / f"{f['name']}_scores.npz"); held = set(f["heldout_composition_ids"])
            fm = np.array([x not in held for x in cid])
            old = FoldScorer(z["seen"], z["seen_truth"], z["unseen"], z["unseen_truth"], fm, cc, cd, hl)
            keep = ~ph; remap = -np.ones(len(cid), int); remap[keep] = np.arange(keep.sum())
            new = FoldScorer(z["seen"][:, keep], remap[z["seen_truth"]], z["unseen"][:, keep], remap[z["unseen_truth"]], fm[keep], cc[keep], cd[keep], hl[keep])
            for k, sc in (("old", old), ("new", new)):
                agg[k].append({"curve": [sc.at(x)["H_bal"] for x in GAMMA_GRID], "ausuc": sc.ausuc(), "g": sc.at(g, top5=True), "g0": sc.at(0.0, top5=True)})
            hits["seen"] += int(ph[old.top1_pred("seen", g)].sum()); hits["unseen"] += int(ph[old.top1_pred("unseen", g)].sum())
            s5 = np.argsort(-(z["seen"].astype(np.float64) - g * fm), 1, kind="stable")[:, :5]
            hits["seen_top5_slots"] += int(ph[s5].sum()); hits["n_seen"] += len(z["seen_truth"]); hits["n_unseen"] += len(z["unseen_truth"])
        def m(k, key, sub):
            return float(np.mean([e[sub][key] for e in agg[k]]))
        out = {"gamma": g, "phantom_top1": hits, "pooled_gamma_old": float(GAMMA_GRID[np.argmax(np.mean([e["curve"] for e in agg["old"]], 0))]),
               "pooled_gamma_new": float(GAMMA_GRID[np.argmax(np.mean([e["curve"] for e in agg["new"]], 0))])}
        for sub in ("g", "g0"):
            for key in ("H_bal", "H_micro", "seen_top1", "unseen_top1_macro", "seen_top5", "unseen_top5_micro", "H5_micro"):
                out[f"{sub}:{key}"] = [m("old", key, sub), m("new", key, sub)]
        out["ausuc"] = [float(np.mean([e["ausuc"] for e in agg["old"]])), float(np.mean([e["ausuc"] for e in agg["new"]]))]
        rep["rows"][rid] = out
        print(rid, json.dumps({k: (np.round(np.array(v) * 100, 2).tolist() if isinstance(v, list) else v) for k, v in out.items()}))
    (ROOT / "artifacts/r2/audit_v2_candidates.json").write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
