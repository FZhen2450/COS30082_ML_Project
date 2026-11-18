"""H_RT driver (v4, post-freeze exploratory; amendment_2 source_co_endpoint.H_RT). Diagnostic only.

  scope     python scripts/r2v4_hrt.py scope --out <E>/stage_hrt_20261116/scope.json
  run       python scripts/r2v4_hrt.py run --out <E>/stage_hrt_20261116 [--rows J1_clipL_last2 ...]
  evaluate  python scripts/r2v4_hrt.py evaluate --out <E>/stage_hrt_20261116 --eval-out <E>/stage_hrt_20261116/evaluation

Rows J1-J3 (three-seed passers), the three registered pairs, seed 0, arms ablated and random; strictly sequential.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from research_r2.metrics import GAMMA_GRID  # noqa: E402
from research_r2_v4 import confirm, hrt, jrun  # noqa: E402
from research_r2_v4.common import INNER, LABEL, V4, append_jsonl, env, now, read_jsonl  # noqa: E402

LINEAGE = ROOT / "artifacts/r2/lineage_groups_v1.json"


def cmd_scope(a):
    """Gate A record: registered pairs, index-set reproduction, arm support, lineage, base models."""
    frame, bench, F = env()
    pairs = hrt.declared_pairs()
    lin = np.asarray(json.loads(LINEAGE.read_text())["group"])
    src = frame.source.to_numpy()
    rep = {"label": LABEL + "; H_RT scope (Gate A)", "pairs": pairs, "rows": hrt.ROWS, "seed": hrt.SEED, "arms": list(hrt.ARMS), "per_pair": {}}
    for fn, s in pairs:
        fold = F[fn]
        e = {}
        for arm in hrt.ARMS:
            f_arm, removed, prov = hrt.arm_fold(frame, fold, s, arm)
            kept = np.asarray(f_arm["fitting_rows"])
            shared = sorted(set(lin[np.asarray(removed)].tolist()) & set(lin[kept].tolist()))
            ev_rows = np.asarray(list(f_arm["seen_val_rows"]) + list(f_arm["unseen_val_rows"]))
            e[arm] = {**prov, "lineage_groups_shared_removed_vs_kept": shared,
                      "kept_rows_in_shared_groups": [frame.file.iloc[i] for i in np.flatnonzero(np.isin(lin, shared)) if i in set(kept.tolist())],
                      "eval_rows_sharing_lineage_with_kept_fitting": int(np.isin(lin[ev_rows], lin[kept]).sum())}
        e["n_novel_source_seen"] = int((src[np.asarray(hrt.arm_fold(frame, fold, s, "ablated")[0]["seen_val_rows"])] == s).sum())
        e["n_source_unseen"] = int((src[np.asarray(fold["unseen_val_rows"])] == s).sum())
        rep["per_pair"][f"{fn}_{s}"] = e
    rep["base_models"] = {}
    for r in hrt.ROWS:
        v = confirm.verified_row_seed(hrt.PHASE1, r, hrt.SEED, INNER, jrun.j_rows().get(r))
        rep["base_models"][r] = {fn: {k: v[fn][1][k] for k in ("checkpoint_sha256", "packet_sha256", "torch_seed", "frozen_digest_before", "n_fitting")}
                                 for fn, _ in pairs}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=1))
    for k, e in rep["per_pair"].items():
        print(k, {arm: (e[arm]["support"]["n_fitting"], e[arm]["support"]["n_removed"], len(e[arm]["support"]["emptied_compositions"]),
                        len(e[arm]["lineage_groups_shared_removed_vs_kept"])) for arm in hrt.ARMS}, "novel", e["n_novel_source_seen"])


def cmd_run(a):
    frame, bench, F = env()
    pairs = hrt.declared_pairs()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    backend = jrun.ClipBackend()
    st = {}
    for row in a.rows:
        for fn, s in pairs:
            for arm in hrt.ARMS:
                append_jsonl(out / "run_log.jsonl", {"t": now(), "row": row, "pair": f"{fn}_{s}", "arm": arm, "event": "start"})
                r = hrt.run_arm(row, F[fn], s, arm, frame, bench, backend, out, image_root=a.images,
                                log=lambda m: (print(m, flush=True), append_jsonl(out / "run_log.jsonl", {"t": now(), "msg": m})))
                st[f"{row}/{fn}_{s}/{arm}"] = r
                append_jsonl(out / "run_log.jsonl", {"t": now(), "row": row, "pair": f"{fn}_{s}", "arm": arm, "status": r})
                print(row, fn, s, arm, r, flush=True)
    print(json.dumps(st, indent=1))
    if any(not v.startswith("ok") for v in st.values()):
        sys.exit(2)


def g_selected(row, frame, F, phase1=None):
    import r2v4_stage_j as sj
    v = confirm.verified_row_seed(phase1 or hrt.PHASE1, row, hrt.SEED, INNER, jrun.j_rows().get(row))
    return float(GAMMA_GRID[sj.table_from_packets(frame, F, v, row).choose(INNER)[1]])


def _reference(pairs_jsonl, fn, s):
    for p in read_jsonl(pairs_jsonl):
        if p["fold"] == fn and p["source"] == s and p.get("status", "ok") == "ok":
            gs = sorted((k for k in p["base"] if k.startswith("g")), key=lambda k: float(k[1:]))
            out = {}
            for g in gs:
                b, ab, r0 = p["base"][g], p["ablated"][g], p["random"][0][g]
                rm = {m: float(np.mean([x[g][m] for x in p["random"]])) for m in ("novel_seen_top1", "H_bal", "seen_top1", "unseen_top1_macro")}
                out[g] = {m: {"source_effect_vs_random_seed0": ab[m] - r0[m], "source_effect_vs_random_mean": ab[m] - rm[m],
                              "size_effect_random_seed0": r0[m] - b[m], "total_effect": ab[m] - b[m]}
                          for m in ("novel_seen_top1", "H_bal", "seen_top1", "unseen_top1_macro")}
            return out
    return None


def cmd_evaluate(a):
    frame, bench, F = env()
    pairs = hrt.declared_pairs()
    out = Path(a.out); eo = Path(a.eval_out); eo.mkdir(parents=True, exist_ok=True)
    res = {"label": LABEL + "; H_RT retrained source exclusion (diagnostic only; never a gate, never used to select)",
           "pairs": pairs, "seed": hrt.SEED, "rows": {}, "evaluated": now(),
           "comparison_note": "every arm restricted to the ablated fold's candidate universe and evaluation rows; "
                              "source = ablated - random(seed 0); size = random - base; total = ablated - base"}
    d1_ref = V4 / "stage_h_20261114" / "D1_synthComp_norm" / "pairs.jsonl"
    for row in hrt.ROWS:
        g = g_selected(row, frame, F)
        per = hrt.evaluate_row(row, pairs, out, frame, F, g, reps=a.reps)
        gk = f"g{g:g}"
        for k, v in per.items():
            v["reference_H_RF_seed0"] = _reference(hrt.PHASE1 / "hrf" / f"{row}__seed0" / "pairs.jsonl", v["fold"], v["source"])
            v["reference_D1_stage_H"] = _reference(d1_ref, v["fold"], v["source"])
        res["rows"][row] = {"gamma_selected": g, "per_pair": per, "aggregate_equal_weight": hrt.aggregate(per, gk)}
    all_pairs = {f"{r}/{k}": v for r, e in res["rows"].items() for k, v in e["per_pair"].items()}
    res["aggregate_all_rows"] = {k: {kk: vv for kk, vv in v.items() if kk != "per_pair_pp"} for k, v in hrt.aggregate(all_pairs, "selected").items()}
    (eo / "evaluation.json").write_text(json.dumps(hrt.strip_draws(res), indent=1, default=float))
    for row, e in res["rows"].items():
        gk = f"g{e['gamma_selected']:g}"
        for k, v in e["per_pair"].items():
            ef = v["effects"][gk]; b = v["bootstrap_ablated_minus"]["random"]
            print(f"{row:20s} {k:8s} novel-seen src {100*ef['novel_seen_top1']['source_effect']:+6.2f} "
                  f"[{b['novel_seen_top1']['ci95_pp'][0]:+.1f},{b['novel_seen_top1']['ci95_pp'][1]:+.1f}] size {100*ef['novel_seen_top1']['size_effect']:+6.2f} "
                  f"| H_bal src {100*ef['H_bal']['source_effect']:+5.2f} [{b['H_bal']['ci95_pp'][0]:+.2f},{b['H_bal']['ci95_pp'][1]:+.2f}] size {100*ef['H_bal']['size_effect']:+5.2f}")
        ag = e["aggregate_equal_weight"]
        print(f"{row:20s} aggregate novel-seen {ag['novel_seen_top1']['mean_source_effect_pp']:+.2f} {ag['novel_seen_top1']['ci95_pp']} "
              f"H_bal {ag['H_bal']['mean_source_effect_pp']:+.2f} {ag['H_bal']['ci95_pp']}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("scope"); p.add_argument("--out", required=True)
    p = sub.add_parser("run"); p.add_argument("--out", required=True); p.add_argument("--rows", nargs="+", default=hrt.ROWS, choices=hrt.ROWS)
    p.add_argument("--images", default=str(ROOT.parents[2] / "DPD_Small_dataset"))
    p = sub.add_parser("evaluate"); p.add_argument("--out", required=True); p.add_argument("--eval-out", required=True)
    p.add_argument("--reps", type=int, default=2000)
    a = ap.parse_args()
    {"scope": cmd_scope, "run": cmd_run, "evaluate": cmd_evaluate}[a.cmd](a)


if __name__ == "__main__":
    main()
