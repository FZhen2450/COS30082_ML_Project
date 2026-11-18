"""Amendment-4 CLI (v4, post-freeze exploratory): J4b seeds 1-2 and the three-seed confirmation vs J2. Production only.

  build     python scripts/r2v4_a4.py build       writes research_r2_v4/amendment_4.json ONCE from committed code
                                                  (refused if it exists or the worktree is dirty)
  declare   python scripts/r2v4_a4.py declare     writes DECLARATION_AMENDMENT_4.json ONCE (then commit both)
  verify    python scripts/r2v4_a4.py verify      amendment 3 (pinned tool + recorded hashes) and amendment 4 -> OK / problems
  run       python scripts/r2v4_a4.py run --seed 1
            one authorised seed into <E>/stage_j4b_confirm_20261117/runs (fixed); J2 evidence fingerprinted before and after
  evaluate  python scripts/r2v4_a4.py evaluate
            three-seed J4b vs J2 confirmation -> <E>/stage_j4b_confirm_20261117/evaluation/evaluation.json
The J2 parent preflight is the existing pinned command: python scripts/r2v4_child.py preflight --seed N --out ...
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research_r2.data import sha256_file  # noqa: E402
from research_r2_v4 import amendment4 as a4  # noqa: E402
from research_r2_v4 import child_gate as cg  # noqa: E402
from research_r2_v4 import child_inputs as ci  # noqa: E402
from research_r2_v4 import confirm, jrun  # noqa: E402
from research_r2_v4.common import INNER, env, now  # noqa: E402

NEW_FILES = ["research_r2_v4/amendment4.py", "scripts/r2v4_a4.py", "tests/test_r2v4_amendment4.py", "docs/R2_v4_Amendment4_Plan.md"]


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def cmd_build(a):
    out = ROOT / a4.AMENDMENT_REL
    if out.exists():
        sys.exit(f"{out.name} exists; amendments are never rewritten")
    if git("status", "--porcelain"):
        sys.exit("worktree not clean; commit the implementation first")
    p = a4.a3_problems()
    if p:
        sys.exit(f"amendment 3 does not verify: {p}")
    frame, bench, F = env()
    cfg, par = ci.declared_child_config(a4.CHILD_ROW), jrun.j_rows()[ci.PARENT_ROW]
    a3 = json.loads((ROOT / "research_r2_v4/amendment_3.json").read_text())
    pins = {f: sha256_file(ROOT / f) for f in sorted(set(NEW_FILES) | set(a3["code_pins"]))}
    am = {
        "schema": "r2v4-amendment-v1", "amendment": 4, "declared": a4.DECLARED_DATE,
        "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "built_from_commit": git("rev-parse", "HEAD"),
        "status": ("post-freeze EXPLORATORY development cross-validation on the reused v3 inner folds; ADAPTIVE (J4b prioritised after "
                   "H_RT; seeds 1-2 authorised after the reviewed seed-0 screen pass); not blind, independent or confirmatory in the "
                   "statistical sense; 'confirmation' means the declared amendment-3 three-seed child-vs-J2 gate"),
        "extends": {"amendment": 3, "amendment_3_sha256": a4.A3_SHA256["research_r2_v4/amendment_3.json"],
                    "declaration_3_sha256": a4.A3_SHA256["research_r2_v4/DECLARATION_AMENDMENT_3.json"],
                    "relation": ("append-only: amendment 3 and its 24 code-pinned files are unchanged and must still verify with the pinned "
                                 "tool; amendment 4 adds the authorisation for J4b seeds 1-2 and the three-seed confirmation only")},
        "unchanged": ["protocol_v4.json and amendments 1-3 with their declarations", "the amendment-3 primary gate, thresholds, endpoint (H_bal), "
                      "diagnostics, LOFO calibration, gamma grid and tie rules", "jrun.py training code and the J4b / J2 recipes",
                      "all v1-v3 evidence; Stage H, I, J Phase 1/2, H_RT and the J4b seed-0 screen evidence"],
        "child": {"row": a4.CHILD_ROW, "config": asdict(cfg), "config_sha256": cfg.sha(),
                  "modification_vs_parent": dict([ci.modification(asdict(cfg), asdict(par))])},
        "parent": {"row": ci.PARENT_ROW, "config": asdict(par), "config_sha256": par.sha(), "runs": {str(s): p.name for s, p in ci.PARENT_BASES.items()}},
        "child_runs": {str(s): str(p.relative_to(a4.V4)) for s, p in a4.CHILD_BASES.items()},
        "authorised_now": {"rows": [a4.CHILD_ROW], "seeds": list(a4.RUN_SEEDS),
                           "evaluation": {"mode": "confirm", "seeds": list(confirm.CONFIRM_SEEDS), "folds": list(INNER)},
                           "not_authorised": ["J4a", "J5", "J6", "J7", "J8", "J4b seeds other than 1-2", "J4b or any other H_RT arms",
                                              "any other model, row or experiment", "official-test images, labels, scores or inference"]},
        "training": ("child_run._run unchanged (jrun.run_fold: declared 5 epochs, AdamW lr 1e-5 / 1e-3, wd 0.05, cosine with 1 warm-up epoch, "
                     "AMP, batch 32, composition-balanced sampler, registered photometric augmentation, initialisation from the original "
                     "pretrained weights, torch seed 1000*seed + fold index); a new attempt directory per fold; failures kept"),
        "fold_pins": a4.fold_pins(frame, F),
        "evidence_pins": a4.evidence_pins(),
        "confirmation": {"decision_function": "amendment4.confirmation_decision -> child_gate.promotion_decision(C, P, 'confirm') on I1-I6",
                         "inputs": "amendment4.verified_confirmation_pair: child_inputs.load_verified (mandatory re-derivation) per seed and base",
                         "thresholds": cg.THRESHOLDS | {"folds_required": 5, "improvement": "strict (> 0)"}, "endpoint": cg.PRIMARY_ENDPOINT,
                         "calibration": "each row's own LOFO (config, gamma) on the other five folds; ties -> lower config, smaller gamma",
                         "per_fold": "per-fold means over seeds 0, 1, 2 of child - J2",
                         "meaning": ("pass -> 'reproducible exploratory improvement over J2 on the development folds' (not independent confirmation, "
                                     "not official-test superiority, no source-robustness claim); fail -> negative result kept, thresholds unchanged")},
        "diagnostics_no_pass_fail": list(cg.DIAGNOSTIC_ENDPOINTS) + ["seen / unseen Top-5", "calibration-matched H_bal", "per-fold healthy shortcut",
                                                                     "false-healthy", "source strata with unique-image and composition-source-cluster counts",
                                                                     "paired composition-cluster bootstrap CI (exploratory)", "per-seed deltas"],
        "stop": "after the three-seed evaluation, regardless of outcome; final research decision report; no further experiments",
        "code_pins": pins,
    }
    out.write_text(json.dumps(am, indent=1) + "\n")
    print(json.dumps({k: am[k] for k in ("built_from_commit", "authorised_now")} | {"n_code_pins": len(pins)}, indent=1))


def cmd_declare(a):
    print(json.dumps(a4.declare(), indent=1))


def cmd_verify(a):
    p = a4.a3_problems() + a4.declaration_problems()
    print("OK" if not p else f"PROBLEMS: {p}")
    sys.exit(1 if p else 0)


def j2_fingerprint(seed):
    v = confirm.verified_row_seed(ci.PARENT_BASES[seed], ci.PARENT_ROW, seed, INNER, jrun.j_rows()[ci.PARENT_ROW])
    return {fn: {k: rec[k] for k in ("checkpoint_sha256", "features_sha256", "packet_sha256")} | {"receipt_sha256": jrun.sha256_file(p.parent / "receipt.json")}
            for fn, (p, rec) in v.items()}


def cmd_run(a):
    stage = a4.CONFIRM_STAGE; stage.mkdir(parents=True, exist_ok=True)
    logf = stage / "run_log.txt"
    before = j2_fingerprint(a.seed)
    st = a4.run_child(a4.CHILD_ROW, a.seed, a.images, log=lambda m: (print(m, flush=True), logf.open("a").write(f"{now()} {m}\n")))
    st["_j2_evidence_unchanged"] = before == j2_fingerprint(a.seed)
    (stage / f"status_{a4.CHILD_ROW}_seed{a.seed}.json").write_text(json.dumps(st, indent=1))
    print(json.dumps(st, indent=1))
    if any(not str(v).startswith("ok") for k, v in st.items() if not k.startswith("_")) or not st["_j2_evidence_unchanged"] or "_worktree" in st:
        sys.exit(2)


def cmd_evaluate(a):
    frame, bench, F = env()
    eo = a4.CONFIRM_STAGE / "evaluation"
    if eo.exists():
        sys.exit(f"{eo} exists; evaluations are never overwritten")
    C, P, meta = a4.verified_confirmation_pair(frame, F)
    res = a4.confirmation_report(C, P, meta, frame, F, ci.declared_child_config(a4.CHILD_ROW), reps=2000)
    res["git_head"] = git("rev-parse", "HEAD")
    eo.mkdir(parents=True)
    (eo / "evaluation.json").write_text(json.dumps(res, indent=1, default=float))
    g, b = res["primary"]["gate"], res["primary"]["bootstrap_exploratory"]
    diag = res["diagnostics"]
    print(f"{a4.CHILD_ROW} vs J2 (confirm, seeds {res['seeds']}): H_bal {g['mean_gain_pp']:+.2f} pp, {g['folds_improved']}/6, "
          f"AUSUC {100 * g['ausuc_delta']:+.2f}, shortcut {g['shortcut_delta_pp']:+.2f} pp -> pass={res['decision']['pass']}; bootstrap CI {b['ci95_pp']}")
    print("per-fold:", {k: round(v, 2) for k, v in g["per_fold_delta_pp"].items()})
    print("per-seed:", {s: round(v["mean_delta_pp"], 2) for s, v in res["primary"]["per_seed"].items()})
    print("diagnostics:", {e: round(v["mean_delta_pp"], 2) for e, v in diag["endpoints"].items()},
          "calibration-matched", round(diag["calibration_matched_H_bal"]["mean_delta_pp"], 2))
    print("source strata (seed mean):", {k: (round(v["delta_pp"], 2) if v["delta_pp"] is not None else None)
                                         for k, v in res["diagnostic_source_strata_seen"]["seed_mean"].items()})


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build"); sub.add_parser("declare"); sub.add_parser("verify")
    p = sub.add_parser("run"); p.add_argument("--seed", type=int, required=True)
    p.add_argument("--images", default=str(ROOT.parents[2] / "DPD_Small_dataset"))
    sub.add_parser("evaluate")
    a = ap.parse_args()
    {"build": cmd_build, "declare": cmd_declare, "verify": cmd_verify, "run": cmd_run, "evaluate": cmd_evaluate}[a.cmd](a)


if __name__ == "__main__":
    main()
