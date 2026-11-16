"""Amendment-3 child workflow CLI (v4, post-freeze exploratory). Production paths only (no configuration overrides).

  preflight  python scripts/r2v4_child.py preflight --seed 0 --out <E>/stage_j4b_<date>/preflight_J2_seed0.json
             Re-verify the registered J2 runs of one seed: receipts and hashes, registered fold, contract, lineage,
             ordered image identity, and packet re-derivation from the feature cache (|diff| <= 1e-5). Writes a receipt.
  run        python scripts/r2v4_child.py run --child J4b_best_photometric_aug --seed 0 --out <E>/stage_j4b_<date>
             Requires the amendment-3 declaration and authorisation; dirty worktree refused; J2 evidence re-verified
             before and after.
  evaluate   python scripts/r2v4_child.py evaluate --child J4b_best_photometric_aug --mode screen --runs <E>/stage_j4b_<date>
             --out <E>/stage_j4b_<date>/evaluation
             Verified child-vs-J2 pair -> primary H_bal gate (sole promotion endpoint), decision recomputed from the
             verified tables, diagnostics (no pass/fail), bootstrap CI (exploratory).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402
import torch  # noqa: E402

from research_r2_v4 import child_gate as cg  # noqa: E402
from research_r2_v4 import child_inputs as ci  # noqa: E402
from research_r2_v4 import child_run as cr  # noqa: E402
from research_r2_v4 import confirm, jrun  # noqa: E402
from research_r2_v4 import evaluate as ev  # noqa: E402
from research_r2_v4.common import INNER, LABEL, env, now  # noqa: E402


def j2_fingerprint(seed):
    v = confirm.verified_row_seed(ci.PARENT_BASES[seed], ci.PARENT_ROW, seed, INNER, jrun.j_rows()[ci.PARENT_ROW])
    return {fn: {k: rec[k] for k in ("checkpoint_sha256", "features_sha256", "packet_sha256")} | {"receipt_sha256": jrun.sha256_file(p.parent / "receipt.json")}
            for fn, (p, rec) in v.items()}


def cmd_preflight(a):
    frame, bench, F = env()
    t0 = time.time()
    rep = {"label": LABEL + "; amendment-3 J2 parent preflight", "row": ci.PARENT_ROW, "seed": a.seed, "run_dir": ci.PARENT_BASES[a.seed].name,
           "tolerance": ci.REDERIVE_TOL, "device": "cuda" if torch.cuda.is_available() else "cpu", "started": now(), "folds": {}}
    try:
        t, v = ci.load_verified(ci.PARENT_BASES[a.seed], ci.PARENT_ROW, a.seed, frame, F, jrun.j_rows()[ci.PARENT_ROW], child=False)
        for fn in INNER:
            f = t.identity["folds"][fn]; rec = v[fn][1]
            rep["folds"][fn] = {"result": "pass", "rederive_max_abs_diff": f["rederive_max_abs_diff"], "identity_sha256": f["identity_sha256"],
                                "contract_sha256": f["contract_sha256"], "candidate_ids_sha256": f["candidate_ids_sha256"],
                                "seen_files_sha256": f["seen_files_sha256"], "unseen_files_sha256": f["unseen_files_sha256"],
                                "receipt_sha256": jrun.sha256_file(v[fn][0].parent / "receipt.json"), "checkpoint_sha256": rec["checkpoint_sha256"],
                                "features_sha256": rec["features_sha256"], "packet_sha256": rec["packet_sha256"], "attempt": v[fn][0].parent.name}
        rep["result"] = "pass"
    except Exception as e:  # noqa: BLE001
        rep["result"] = "FAIL"; rep["error"] = f"{type(e).__name__}: {e}"
    rep["seconds"] = time.time() - t0
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps({k: v for k, v in rep.items() if k != "folds"}, indent=1, default=float))
    for fn, f in rep["folds"].items():
        print(fn, f["result"], "max|diff|", f["rederive_max_abs_diff"])
    sys.exit(0 if rep["result"] == "pass" else 2)


def cmd_run(a):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    before = j2_fingerprint(a.seed)
    st = cr.run_child(a.child, a.seed, out, a.images, log=lambda m: (print(m, flush=True), (out / "run_log.txt").open("a").write(f"{now()} {m}\n")))
    after = j2_fingerprint(a.seed)
    st["_j2_evidence_unchanged"] = before == after
    (out / f"status_{a.child}_seed{a.seed}.json").write_text(json.dumps(st, indent=1))
    print(json.dumps(st, indent=1))
    if any(not str(v).startswith("ok") for k, v in st.items() if not k.startswith("_")) or not st["_j2_evidence_unchanged"] or "_worktree" in st:
        sys.exit(2)


def cmd_evaluate(a):
    frame, bench, F = env()
    eo = Path(a.out); eo.mkdir(parents=True, exist_ok=True)
    C, P, meta = ci.verified_pair(Path(a.runs), a.mode, frame, F, child_row=a.child)
    prim = cg.primary_gate(C, P, INNER, mode=a.mode, reps=2000)
    decision = cg.promotion_decision(C, P, a.mode)
    diag = cg.diagnostics(C, P, INNER, a.mode)
    seeds = meta["seeds"]
    strata = {s: {"child": ev.source_strata(C[s], frame, F, INNER), "parent": ev.source_strata(P[s], frame, F, INNER)} for s in seeds}
    fh = {s: {"child": ev.false_healthy(C[s], INNER), "parent": ev.false_healthy(P[s], INNER)} for s in seeds}
    lm = {s: {"child": {k: {key: C[s].at(k, *C[s].choose([f for f in INNER if f != k]))[key] for key in ev.KEYS} for k in INNER},
              "parent": {k: {key: P[s].at(k, *P[s].choose([f for f in INNER if f != k]))[key] for key in ev.KEYS} for k in INNER}} for s in seeds}
    units = json.loads((ROOT / "artifacts/r2/v4/stage_hrt_20261116_post_review_v3/strata_units.json").read_text())
    rec = {}
    for s in seeds:
        v = confirm.verified_row_seed(Path(a.runs), a.child, s, INNER, ci.declared_child_config(a.child))
        rec[s] = {fn: {k: r.get(k) for k in ("seconds", "peak_gpu_gib_training", "peak_gpu_gib_total", "checkpoint_sha256", "features_sha256",
                                             "packet_sha256", "trainable_tensors_changed", "reload_max_abs_diff", "torch_seed", "config_sha256")}
                   | {"attempt": p.parent.name, "grad_checks": r.get("grad_checks"), "epoch_loss": [e["train_loss_mean"] for e in r.get("epochs", [])],
                      "git_head": r["code"]["git_head"], "jrun_sha256": r["code"]["jrun_sha256"],
                      "rederive_max_abs_diff": C[s].identity["folds"][fn]["rederive_max_abs_diff"]}
                   for fn, (p, r) in v.items()}
    res = {"label": LABEL + "; amendment 3 child-vs-J2 screen (exploratory)", "child_row": a.child, "parent_row": ci.PARENT_ROW,
           "mode": a.mode, "modification": meta["modification"], "evaluated": now(),
           "primary": {"endpoint": prim["endpoint"], "gate": prim["gate"], "bootstrap_exploratory": prim["bootstrap"],
                       "per_fold_gamma": {s: {k: {"child": prim["detail"][s][k]["child"]["gamma"], "parent": prim["detail"][s][k]["parent"]["gamma"]}
                                              for k in INNER} for s in seeds},
                       "per_fold": {s: {k: {"child_H_bal": prim["detail"][s][k]["child"]["E"], "parent_H_bal": prim["detail"][s][k]["parent"]["E"],
                                            "child_ausuc": prim["detail"][s][k]["child"]["ausuc"], "parent_ausuc": prim["detail"][s][k]["parent"]["ausuc"],
                                            "child_shortcut": prim["detail"][s][k]["child"]["hs"], "parent_shortcut": prim["detail"][s][k]["parent"]["hs"]}
                                        for k in INNER} for s in seeds}},
           "decision": decision, "diagnostics": diag,
           "diagnostic_source_strata_seen": {s: {"child": strata[s]["child"], "parent": strata[s]["parent"],
                                                 "delta_pp": {k: (100 * (strata[s]["child"][k]["top1"] - strata[s]["parent"][k]["top1"])
                                                                  if strata[s]["child"][k]["top1"] is not None else None) for k in strata[s]["child"]}}
                                             for s in seeds},
           "diagnostic_source_strata_units": units,
           "diagnostic_false_healthy": fh, "lofo_metrics": lm, "child_receipts": rec,
           "parent_identity": {s: {fn: {k: P[s].identity["folds"][fn][k] for k in ("identity_sha256", "checkpoint_sha256", "packet_sha256", "rederive_max_abs_diff")}
                                   for fn in INNER} for s in seeds}}
    (eo / "evaluation.json").write_text(json.dumps(res, indent=1, default=float))
    g = prim["gate"]
    print(f"{a.child} vs J2 ({a.mode}): H_bal {g['mean_gain_pp']:+.2f} pp, {g['folds_improved']}/6, AUSUC {100 * g['ausuc_delta']:+.2f}, "
          f"shortcut {g['shortcut_delta_pp']:+.2f} pp -> pass={decision['pass']}; bootstrap CI {prim['bootstrap']['ci95_pp']}")
    print("diagnostics:", {e: round(v["mean_delta_pp"], 2) for e, v in diag["endpoints"].items()},
          "calibration-matched", round(diag["calibration_matched_H_bal"]["mean_delta_pp"], 2))
    print("source strata delta:", res["diagnostic_source_strata_seen"][seeds[0]]["delta_pp"])


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("preflight"); p.add_argument("--seed", type=int, required=True); p.add_argument("--out", required=True)
    p = sub.add_parser("run"); p.add_argument("--child", required=True); p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out", required=True); p.add_argument("--images", default=str(ROOT.parents[2] / "DPD_Small_dataset"))
    p = sub.add_parser("evaluate"); p.add_argument("--child", required=True); p.add_argument("--mode", choices=["screen", "confirm"], required=True)
    p.add_argument("--runs", required=True); p.add_argument("--out", required=True)
    a = ap.parse_args()
    {"preflight": cmd_preflight, "run": cmd_run, "evaluate": cmd_evaluate}[a.cmd](a)


if __name__ == "__main__":
    main()
