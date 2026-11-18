"""Paired composition bootstrap of outer differences (each row at its own registered inner gamma)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research_r2.data import candidate_table, load_training_source  # noqa: E402
from research_r2.metrics import harmonic  # noqa: E402
from r2_report_metrics import scorer_for  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
GAMMA = {"D1_synthComp_norm": 8.0, "D2_synthComp_norm_text": 8.0, "B1_clipL_factorised": 10.5}


def main(scores_dir, out, n_boot=5000):
    frame = load_training_source(ROOT / "DPD_Small_train.csv"); cands = candidate_table(frame)
    F = {f["name"]: f for f in json.loads((ROOT / "artifacts/r2/benchmark_v2.json").read_text())["folds"]}
    hits = {}
    for rid, g in GAMMA.items():
        for fn in ("O1", "O2"):
            z = np.load(Path(scores_dir) / rid / f"{fn}_scores.npz")
            sc = scorer_for(frame, cands, F[fn], z)
            hits[(rid, fn)] = (sc.top1_pred("seen", g) == z["seen_truth"], sc.top1_pred("unseen", g) == z["unseen_truth"], z["unseen_truth"])
    rng = np.random.default_rng(0)
    pairs = [("D1_synthComp_norm", "B1_clipL_factorised"), ("D2_synthComp_norm_text", "B1_clipL_factorised"),
             ("D2_synthComp_norm_text", "D1_synthComp_norm")]
    res = {}
    for a, b in pairs:
        db, dm = [], []
        for _ in range(n_boot):
            vb, vm = [], []
            for fn in ("O1", "O2"):
                sa, ua, tu = hits[(a, fn)]; sb, ub, _ = hits[(b, fn)]
                comps = np.unique(tu); pick = rng.choice(comps, len(comps)); srows = rng.integers(0, len(sa), len(sa))
                idx = np.concatenate([np.nonzero(tu == c)[0] for c in pick])
                ha = harmonic(sa[srows].mean(), np.mean([ua[tu == c].mean() for c in pick]))
                hb = harmonic(sb[srows].mean(), np.mean([ub[tu == c].mean() for c in pick]))
                ma = harmonic(sa[srows].mean(), ua[idx].mean()); mb = harmonic(sb[srows].mean(), ub[idx].mean())
                vb.append(ha - hb); vm.append(ma - mb)
            db.append(np.mean(vb)); dm.append(np.mean(vm))
        db, dm = 100 * np.array(db), 100 * np.array(dm)
        res[f"{a} - {b}"] = {"H_bal_diff_pp": {"median": float(np.median(db)), "ci95": [float(np.percentile(db, 2.5)), float(np.percentile(db, 97.5))],
                                               "P(diff>0)": float(np.mean(db > 0))},
                             "H_micro_diff_pp": {"median": float(np.median(dm)), "ci95": [float(np.percentile(dm, 2.5)), float(np.percentile(dm, 97.5))],
                                                 "P(diff>0)": float(np.mean(dm > 0))}}
    per = {}
    for rid in GAMMA:
        for fn in ("O1", "O2"):
            _, u, tu = hits[(rid, fn)]
            for c in np.unique(tu):
                per.setdefault(f"{fn}:{cands.comp[c]}", {"n": int((tu == c).sum())})[rid] = float(u[tu == c].mean())
    d1b = [v["D1_synthComp_norm"] - v["B1_clipL_factorised"] for v in per.values()]
    res["per_composition_D1_vs_B1"] = {"better": int(sum(x > 0 for x in d1b)), "worse": int(sum(x < 0 for x in d1b)), "tied": int(sum(x == 0 for x in d1b))}
    res["per_composition"] = per
    Path(out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "per_composition"}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
