"""Mechanical gate-chain selection with a nested leave-one-fold-out estimate (benchmark v3+).

Full selection: gate chain over all development folds (gate: >= +1 pp mean primary, >= n-1 of n folds
better, AUSUC not lower, healthy shortcut <= +5 pp), primary = mean H_bal with leave-one-fold-out gamma.

Nested estimate: for each development fold k, rerun the whole selection on the other folds only
(gamma LOFO within them; gate needs n-2 of n-1), then score the selected row AND the control on fold k
at the gamma pooled over the other folds. The mean over k estimates the selection procedure's output
without using fold k for any choice.

Input integrity (v3 hardening): every selected row must be declared, with its exact configuration, in a
protocol matrix whose sha256 the protocol fixes; its run-time record (summaries.jsonl) must name a committed
matrix with that configuration; every packet copy found in any --scores dir must be byte-identical (conflicts
are rejected, not shadowed by argument order); every packet must match the packet registry and pass strict
validation against the fold contract (ordered candidate ids, shapes, finite scores, truth = benchmark rows).
All scientific fields of the receipt are computed exactly as before; provenance is added under "inputs".

  python scripts/r2_select.py --bench B --protocol P --tree-key K --scores DIR1 [DIR2 ...] \
      --registry artifacts/r2/v3/packet_registry_v3.json --out OUT.json
  (--allow-unregistered: exploratory only; the receipt is stamped "provenance": "unregistered")
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
from research_r2.packets import (PacketError, declared_rows, committed_matrix_index, file_sha256,  # noqa: E402
                                 fold_contract, git_state, resolve_packets, runtime_summary, tree_rows,
                                 validate_packet)
from research_r2.runner import scorer_from_npz  # noqa: E402


def check_inputs(frame, bench, proto, tree_key, dirs, folds, registry):
    """Verify configuration identity and packet provenance of every selected row. Returns (paths, inputs)."""
    entry = proto[tree_key]
    rows = tree_rows(entry["tree"], entry["start"])
    decl = declared_rows(entry, rows)
    mindex = committed_matrix_index()
    F = {f["name"]: f for f in bench["folds"]}
    contracts = {fn: fold_contract(frame, F[fn]) for fn in folds}
    if registry is not None:
        if registry["benchmark_semantic_sha256"] != bench["semantic_sha256"]:
            raise PacketError("registry was made for a different benchmark")
        if registry["protocol_sha256"] != proto["_sha256"]:
            raise PacketError("registry was made for a different protocol file")
    paths, rec = {}, {}
    for rid in rows:
        res = resolve_packets(rid, folds, dirs)
        stages = sorted({str(Path(c).parent.parent.parent) for f in res.values() for c in f["copies"]})
        runtime = [runtime_summary(Path(sd), rid, decl[rid], mindex) for sd in stages]
        reg = None
        if registry is not None:
            reg = registry["rows"].get(rid)
            if reg is None:
                raise PacketError(f"row {rid!r} is not in the packet registry")
            if reg["config_sha256"] != decl[rid]["config_sha256"]:
                raise PacketError(f"row {rid!r}: registered configuration differs from the declared one")
        for fn in folds:
            validate_packet(np.load(res[fn]["path"]), contracts[fn])
            if reg is not None:
                rf = reg["folds"].get(fn)
                if rf is None or rf["sha256"] != res[fn]["sha256"]:
                    raise PacketError(f"{rid}/{fn}: packet sha256 {res[fn]['sha256']} is not the registered one")
                if rf["contract_sha256"] != contracts[fn]["sha256"]:
                    raise PacketError(f"{rid}/{fn}: fold contract changed since registration")
        paths[rid] = {fn: res[fn]["path"] for fn in folds}
        rec[rid] = {"config_sha256": decl[rid]["config_sha256"], "declared_in": decl[rid]["declared_in"],
                    "runtime_matrix_sha256": sorted({r["matrix_sha256"] for r in runtime}),
                    "packets": {fn: {"sha256": res[fn]["sha256"], "n_identical_copies": len(res[fn]["copies"])} for fn in folds}}
    inputs = {"provenance": "registered" if registry is not None else "unregistered",
              "benchmark_semantic_sha256": bench["semantic_sha256"], "protocol_sha256": proto["_sha256"],
              "matrices": {k: v for k, v in (entry.get("matrices") or entry.get("matrix")).items()},
              "registry_sha256": registry["_sha256"] if registry is not None else None,
              "contracts": {fn: contracts[fn]["sha256"] for fn in folds}, "rows": rec, "code": git_state(ROOT)}
    return paths, inputs


def fold_stats(frame, bench, paths, rows, folds):
    F = {f["name"]: f for f in bench["folds"]}
    st = {}
    for r in rows:
        st[r] = {}
        for fn in folds:
            sc = scorer_from_npz(frame, F[fn], np.load(paths[r][fn]))
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
    ap.add_argument("--registry", help="packet registry JSON (required unless --allow-unregistered)")
    ap.add_argument("--allow-unregistered", action="store_true", help="exploratory runs only; never cited as evidence")
    a = ap.parse_args()
    if not a.registry and not a.allow_unregistered:
        ap.error("--registry is required (or --allow-unregistered for exploratory runs)")
    frame = load_training_source(ROOT / "DPD_Small_train.csv")
    bench = json.loads(Path(a.bench).read_text())
    proto = json.loads(Path(a.protocol).read_text())
    proto["_sha256"] = file_sha256(a.protocol)
    if bench["semantic_sha256"] != proto["benchmark_semantic_sha256"]:
        raise PacketError("benchmark semantic hash differs from the protocol")
    registry = None
    if a.registry:
        registry = json.loads(Path(a.registry).read_text()); registry["_sha256"] = file_sha256(a.registry)
    tree = proto[a.tree_key]["tree"]; start = proto[a.tree_key]["start"]
    folds = [f["name"] for f in bench["folds"] if f["role"] == "inner"]
    paths, inputs = check_inputs(frame, bench, proto, a.tree_key, a.scores, folds, registry)
    st = fold_stats(frame, bench, paths, list(tree), folds)
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
    rep["inputs"] = inputs  # provenance only; every other field is the unchanged scientific receipt
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
