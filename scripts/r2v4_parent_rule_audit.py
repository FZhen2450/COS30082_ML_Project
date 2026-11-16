"""Audit of the registered J4a-J7 parent rule (amendment_2 nested_parent_rule) on the existing seed-0 tables.
Descriptive only: no J4-J7 model exists or is trained, and no rule is changed.

  python scripts/r2v4_parent_rule_audit.py --out <E>/stage_hrt_20261116/parent_rule_audit.json

Uses the verified Phase 1 seed-0 tables of D1, J1, J2, J3 and three HYPOTHETICAL child tables of parent J2, made
from existing score tables (no training): 'null' = J2's own table (a child that changes nothing), 'as_J3' = J3's
table relabelled as a J2 child, 'perfect' = synthetic scores with the truth boosted (an upper bound). They only
expose how the rule attributes held-fold results; they are not results.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from research_r2.packets import fold_contract  # noqa: E402
from research_r2_v4 import confirm, jrun  # noqa: E402
from research_r2_v4 import evaluate as ev  # noqa: E402
from research_r2_v4.common import INNER, LABEL, env, scorer  # noqa: E402
import r2v4_stage_j as sj  # noqa: E402

CHILD = "J5_best_source_balanced"  # any J4a-J7 id; the rule treats them identically


def perfect_table(frame, F):
    def arm(fn, seed):
        con = fold_contract(frame, F[fn]); rng = np.random.default_rng(seed); k = len(con["cand_ids"])
        s = rng.normal(size=(len(con["seen_truth"]), k)); u = rng.normal(size=(len(con["unseen_truth"]), k))
        s[np.arange(len(s)), con["seen_truth"]] += 50; u[np.arange(len(u)), con["unseen_truth"]] += 50
        return scorer(frame, {"seen": s, "unseen": u, "seen_truth": con["seen_truth"], "unseen_truth": con["unseen_truth"],
                              "cand_ids": con["cand_ids"], "fold": F[fn]})
    return ev.RowTable(CHILD, {fn: [arm(fn, 700 + j)] for j, fn in enumerate(INNER)}, ["single"])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
    frame, bench, F = env()
    si = sj._script("r2v4_stage_i")
    T = {sj.D1: si.table(sj.D1, frame, F, Path("."))}
    for r in sj.PARENTS:
        v = confirm.verified_row_seed(sj.ROOT.parents[1] / "experiments/r2-compositional/v4/stage_j_phase1_20261114", r, 0, INNER, jrun.j_rows().get(r))
        T[r] = sj.table_from_packets(frame, F, v, r)
    base = sj.nested_with_parent_rule(T, sj.D1, sj.PARENTS, INNER, {})
    rep = {"label": LABEL + "; parent-rule audit (descriptive; hypothetical child tables, no training)",
           "inner_parent_by_held_fold": {h["held_fold"]: h["inner_parent"] for h in base["held"]},
           "global_parent_seed0_rule": max(sj.PARENTS, key=lambda r: (T[r].summary(INNER)["primary"], -sj.PARENTS.index(r)))}
    elig = [k for k, p in rep["inner_parent_by_held_fold"].items() if p == "J2_clipL_last4"]
    rep["J2_child_eligible_held_folds"] = elig
    rep["max_child_attributable_folds"] = len(elig)
    rep["gate_requires_folds"] = len(INNER) - 1
    hyp = {"null_copy_of_J2": T["J2_clipL_last4"], "as_J3": T["J3_clipL_lora8_mlp"], "perfect": perfect_table(frame, F)}
    rep["scenarios"] = {}
    for name, ct in hyp.items():
        TT = dict(T, **{CHILD: ev.RowTable(CHILD, ct.sc, ct.configs)})
        cands = sj.PARENTS + [CHILD]
        st = sj.nested_with_parent_rule(TT, sj.D1, cands, INNER, {CHILD: "J2_clipL_last4"})
        single = sj.nested_with_parent_rule({sj.D1: T[sj.D1], CHILD: TT[CHILD]}, sj.D1, [CHILD], INNER, {})  # as cmd_evaluate does per row
        d = {h["held_fold"]: 100 * (h["H_bal"] - h["control_H_bal"]) for h in st["held"]}
        sel = {h["held_fold"]: h["selected"] for h in st["held"]}
        child_folds = [k for k, s in sel.items() if s == CHILD]
        rep["scenarios"][name] = {
            "stage_nested_with_child": {"mean_diff_pp": st["mean_diff_pp"], "folds_better": st["folds_better"],
                                        "meets_gate_thresholds": st["meets_gate_thresholds"], "selected": sel,
                                        "excluded_children": {h["held_fold"]: h["excluded_children"] for h in st["held"]}},
            "folds_where_child_selected": child_folds,
            "gain_pp_from_child_selected_folds": float(sum(d[k] for k in child_folds) / len(INNER)),
            "gain_pp_from_fallback_folds": float(sum(d[k] for k in d if k not in child_folds) / len(INNER)),
            "fallback_rows": {k: sel[k] for k in d if k not in child_folds},
            "single_row_nested_vs_D1_ignoring_parent_rule": {"mean_diff_pp": single["mean_diff_pp"], "folds_better": single["folds_better"],
                                                              "meets_gate_thresholds": single["meets_gate_thresholds"]},
            "child_vs_parent_J2_in_sample_pp": 100 * (TT[CHILD].summary(INNER)["primary"] - T["J2_clipL_last4"].summary(INNER)["primary"])}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps({k: v for k, v in rep.items() if k != "scenarios"}, indent=1))
    for n, s in rep["scenarios"].items():
        print(n, json.dumps({k: v for k, v in s.items() if k != "stage_nested_with_child"}), "| stage:",
              round(s["stage_nested_with_child"]["mean_diff_pp"], 2), s["stage_nested_with_child"]["folds_better"], s["stage_nested_with_child"]["selected"])


if __name__ == "__main__":
    main()
