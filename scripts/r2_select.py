"""Mechanical gate-chain selection with a nested leave-one-fold-out estimate (benchmark v3+).

Full selection: gate chain over all development folds (gate: >= +1 pp mean primary, >= n-1 of n folds
better, AUSUC not lower, healthy shortcut <= +5 pp), primary = mean H_bal with leave-one-fold-out gamma.

Nested estimate: for each development fold k, rerun the whole selection on the other folds only
(gamma LOFO within them; gate needs n-2 of n-1), then score the selected row AND the control on fold k
at the gamma pooled over the other folds. The mean over k estimates the selection procedure's output
without using fold k for any choice.

  python scripts/r2_select.py --bench B --protocol P --tree-key K --scores DIR1 [DIR2 ...] --out OUT.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research_r2.data import load_training_source  # noqa: E402
from research_r2.metrics import GAMMA_GRID  # noqa: E402
from research_r2.runner import scorer_from_npz  # noqa: E402


def fold_stats(frame, bench, dirs, rows, folds):
    F = {f["name"]: f for f in bench["folds"]}
    st = {}
    for r in rows:
        d = next(Path(x) / r for x in dirs if (Path(x) / r).exists())
        st[r] = {}
        for fn in folds:
            sc = scorer_from_npz(frame, F[fn], np.load(d / f"{fn}_scores.npz"))
            cur = [sc.at(g) for g in GAMMA_GRID]
            st[r][fn] = {"H": np.array([c["H_bal"] for c in cur]), "hs": np.array([c["unseen_healthy_shortcut"] for c in cur]),
                         "Hmic": np.array([c["H_micro"] for c in cur]), "ausuc": sc.ausuc()}
    return st


def summary(st, row, folds):
    """LOFO-calibrated per-fold H_bal within ``folds`` (gamma for fold j chosen on folds - {j})."""
    per = {}
    for j in folds:
        others = [f for f in folds if f != j]
        gi = int(np.argmax(np.mean([st[row][f]["H"] for f in others], 0)))
        per[j] = {"H": float(st[row][j]["H"][gi]), "hs": float(st[row][j]["hs"][gi]), "gamma": float(GAMMA_GRID[gi])}
    return {"per": per, "primary": float(np.mean([v["H"] for v in per.values()])),
            "ausuc": float(np.mean([st[row][f]["ausuc"] for f in folds])),
            "hs": float(np.mean([v["hs"] for v in per.values()]))}


def gate(c, k, folds):
    d = [c["per"][f]["H"] - k["per"][f]["H"] for f in folds]
    g = {"mean_gain_pp": 100 * (c["primary"] - k["primary"]), "folds_improved": int(sum(x > 0 for x in d)),
         "folds_required": len(folds) - 1, "ausuc_delta": c["ausuc"] - k["ausuc"], "shortcut_delta_pp": 100 * (c["hs"] - k["hs"])}
    g["pass"] = bool(g["mean_gain_pp"] >= 1.0 and g["folds_improved"] >= len(folds) - 1 and g["ausuc_delta"] >= 0
                     and g["shortcut_delta_pp"] <= 5.0)
    return g


def select(st, tree, start, folds):
    S = {r: summary(st, r, folds) for r in tree}
    cur, path, trace = start, [start], []
    while True:
        passing = []
        for ch in tree.get(cur, []):
            g = gate(S[ch], S[cur], folds)
            trace.append({"parent": cur, "child": ch, "child_primary": S[ch]["primary"], **g})
            if g["pass"]:
                passing.append(ch)
        if not passing:
            return cur, path, trace, S
        cur = max(passing, key=lambda r: (S[r]["primary"], -tree[path[-1]].index(r)))
        path.append(cur)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--tree-key", required=True)
    ap.add_argument("--scores", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    frame = load_training_source(ROOT / "DPD_Small_train.csv")
    bench = json.loads(Path(a.bench).read_text())
    proto = json.loads(Path(a.protocol).read_text())
    assert bench["semantic_sha256"] == proto["benchmark_semantic_sha256"]
    tree = proto[a.tree_key]["tree"]; start = proto[a.tree_key]["start"]
    folds = [f["name"] for f in bench["folds"] if f["role"] == "inner"]
    st = fold_stats(frame, bench, a.scores, list(tree), folds)
    sel, path, trace, S = select(st, tree, start, folds)
    nested = []
    for k in folds:
        inner = [f for f in folds if f != k]
        s_k, p_k, _, _ = select(st, tree, start, inner)
        def at_k(row):
            gi = int(np.argmax(np.mean([st[row][f]["H"] for f in inner], 0)))
            return float(st[row][k]["H"][gi]), float(st[row][k]["Hmic"][gi]), float(GAMMA_GRID[gi])
        hs, ms, gs = at_k(s_k); hc, mc, gc = at_k(start)
        nested.append({"held_fold": k, "selected": s_k, "path": p_k, "gamma": gs, "H_bal": hs, "H_micro": ms,
                       "control_H_bal": hc, "control_H_micro": mc, "control_gamma": gc})
    d = [100 * (n["H_bal"] - n["control_H_bal"]) for n in nested]
    rep = {"tree_key": a.tree_key, "folds": folds, "selected": sel, "path": path, "trace": trace,
           "primary": {r: S[r]["primary"] for r in S}, "ausuc": {r: S[r]["ausuc"] for r in S},
           "pooled_gamma": {r: float(GAMMA_GRID[int(np.argmax(np.mean([st[r][f]["H"] for f in folds], 0)))]) for r in tree},
           "nested": nested,
           "nested_summary": {"mean_H_bal_selected": float(np.mean([n["H_bal"] for n in nested])),
                              "mean_H_bal_control": float(np.mean([n["control_H_bal"] for n in nested])),
                              "mean_diff_pp": float(np.mean(d)), "folds_selected_better": int(sum(x > 0 for x in d)),
                              "selection_counts": {r: sum(n["selected"] == r for n in nested) for r in sorted(set(n["selected"] for n in nested))}}}
    Path(a.out).write_text(json.dumps(rep, indent=1))
    print(f"selected: {sel}  path: {' -> '.join(path)}")
    for t in trace:
        print(f"  {t['child']:<34} vs {t['parent']:<26} {t['mean_gain_pp']:+6.2f} pp {t['folds_improved']}/{len(folds)} "
              f"AUSUC {100*t['ausuc_delta']:+5.2f} pass={t['pass']}")
    print("nested:", json.dumps(rep["nested_summary"]))
    for n in nested:
        print(f"  hold {n['held_fold']}: selected {n['selected']:<30} H_bal {100*n['H_bal']:5.1f} vs control {100*n['control_H_bal']:5.1f}")


if __name__ == "__main__":
    main()
