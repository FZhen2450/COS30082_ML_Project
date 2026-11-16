"""Write research_r2_v4/amendment_3.json (once) from the COMMITTED implementation: exact configurations of J2 and
J4b, the primary gate, diagnostics, verification requirements, disclosures and sha256 code pins. Then declare with
``python scripts/r2v4_declare.py --amendment 3`` and verify with ``--verify --amendment 3``.

Refuses if amendment_3.json already exists or the worktree is not clean (the pins must describe committed code).
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research_r2.data import sha256_file  # noqa: E402
from research_r2_v4 import child_gate as cg  # noqa: E402
from research_r2_v4 import child_inputs as ci  # noqa: E402
from research_r2_v4 import confirm, jrun  # noqa: E402

OUT = ROOT / "research_r2_v4" / "amendment_3.json"
PINNED = ["research_r2_v4/child_inputs.py", "research_r2_v4/child_gate.py", "research_r2_v4/child_run.py", "research_r2_v4/jrun.py",
          "research_r2_v4/confirm.py", "research_r2_v4/evaluate.py", "research_r2_v4/common.py", "research_r2/methods.py", "research_r2/runner.py",
          "research_r2/packets.py", "research_r2/metrics.py", "research_r2/candidates.py", "research_r2/data.py", "scripts/r2v4_child.py",
          "scripts/r2v4_declare.py", "scripts/r2v4_amendment3_build.py", "tests/test_r2v4_child_gate.py", "tests/test_r2v4_child_inputs.py",
          "tests/test_r2v4_child_run.py", "docs/R2_v4_Amendment3_PROPOSAL_v3.md", "docs/R2_v4_Amendment3_v3_Repair.md", "artifacts/r2/lineage_groups_v1.json",
          "research_r2_v4/matrices/stage_j_v4.json", "research_r2_v4/amendment_2.json"]


def main():
    if OUT.exists():
        sys.exit(f"{OUT.name} exists; amendments are never rewritten")
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    if dirty:
        sys.exit(f"worktree not clean; commit first:\n{dirty}")
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    j2 = jrun.j_rows()[ci.PARENT_ROW]
    j4b = ci.declared_child_config("J4b_best_photometric_aug", ci.PARENT_ROW)
    a2 = json.loads((ROOT / "research_r2_v4/amendment_2.json").read_text())
    am = {
        "schema": "r2v4-amendment-v1", "amendment": 3, "declared": "2026-11-17",
        "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "built_from_commit": head,
        "status": ("post-freeze EXPLORATORY development cross-validation on the reused v3 inner folds; ADAPTIVE (child rows prioritised "
                   "after the H_RT result); nothing here is blind, independent or confirmatory"),
        "pins": {"research_r2_v4/DECLARATION_AMENDMENT_2.json": sha256_file(ROOT / "research_r2_v4/DECLARATION_AMENDMENT_2.json")},
        "unchanged": ["protocol_v4.json, amendment_1.json, amendment_2.json and every earlier declaration",
                      "all v1-v3 evidence; Stage H, Stage I, Stage J Phase 1 and Phase 2, H_RT results and receipts",
                      "J1-J3 gates, metric definitions, gamma grid, candidate universe, thresholds and tie rules"],
        "scope": ("replaces, prospectively and for J4a-J7 evaluation only, the use of amendment_2.nested_parent_rule as the child gate; "
                  "the registered J4a-J7 row definitions and the parent J2 are unchanged"),
        "parent": {"row": ci.PARENT_ROW, "config": asdict(j2), "config_sha256": j2.sha(),
                   "runs": {str(s): p.name for s, p in ci.PARENT_BASES.items()},
                   "selection": ("best of J1-J3 by seed-0 screen H_bal (stage_j_v4 order / amendment 2), fixed before any child experiment; "
                                 "J2 was chosen after examining all six development folds, so the comparator is not independent of held-fold outcomes")},
        "authorised_now": {"rows": ["J4b_best_photometric_aug"], "seeds": [0], "mode": "screen",
                           "not_authorised": ["J4a", "J5", "J6", "J7", "J8", "J4b seeds 1-2", "J4b H_RT arms", "any other model or experiment"]},
        "child": {"row": "J4b_best_photometric_aug", "config": asdict(j4b), "config_sha256": j4b.sha(),
                  "modification_vs_parent": dict([ci.modification(asdict(j4b), asdict(j2))]),
                  "augmentation": a2["execution"]["row_details"]["J4b"],
                  "training": "jrun.run_fold unchanged: declared 5 epochs, AdamW lr 1e-5 / 1e-3, wd 0.05, cosine with 1 warm-up epoch, AMP, batch 32, "
                              "composition-balanced sampler, seed-derived initialisation from the original pretrained weights (torch seed 1000*seed + fold index)",
                  "folds": "the six registered inner folds I1-I6 (benchmark v3), ordered as registered"},
        "primary_gate": {"endpoint": cg.PRIMARY_ENDPOINT, "comparison": "child vs J2 on every fold (same parent for every fold)",
                         "calibration": "each row's own LOFO choice (config, gamma) on the other five folds by mean H_bal; ties -> lower config, smaller gamma",
                         "per_fold": "d_s(k) = H_bal child - H_bal J2; AUSUC and healthy-shortcut differences likewise; means over seeds",
                         "thresholds": cg.THRESHOLDS | {"folds_required": "n_folds - 1 = 5", "improvement": "strict (> 0)"},
                         "screen_seeds": list(confirm.SCREEN_SEEDS), "confirmation_seeds": list(confirm.CONFIRM_SEEDS),
                         "decision_function": "child_gate.promotion_decision(child tables, parent tables, mode): recomputed from verified tables only",
                         "null_child": "a child identical to J2 fails (0 pp, 0/6)",
                         "decision_meaning": "screen pass -> seeds 1-2 may be considered only after independent review; confirmation pass -> "
                                             "'confirmed vs J2' (exploratory); never promotion above D1 without separate review"},
        "diagnostics_no_pass_fail": list(cg.DIAGNOSTIC_ENDPOINTS) + ["calibration-matched H_bal (child at J2's gamma)",
                                                                     "source-stratified seen Top-1 (dominant / other fitting source / novel)",
                                                                     "false-healthy", "paired composition-cluster bootstrap CI (exploratory)"],
        "verification": ["child_inputs.verified_pair (production): registered J2 runs and the authorised child only, no overrides",
                         "receipts via confirm.verified_row_seed (single attempt, file hashes, declared config, torch seed, gradient/AMP/frozen/reload "
                         "checks, code identity, clean worktree)",
                         "registered fold, receipt contract = fold_contract(registered fold), feature-cache row set, no fitting/eval lineage overlap",
                         "ordered evaluation image identity; child_identity.json sidecar for every child attempt",
                         f"mandatory packet re-derivation from the feature cache in registered order, |diff| <= {ci.REDERIVE_TOL}",
                         "exactly one registered modification vs J2; same training-code hash; distinct child and parent checkpoints",
                         "sealed tables (HMAC over identity and score arrays); unsealed or non-re-derived tables are refused by the gate",
                         "child_run: declaration verified, (row, seed) authorised, clean worktree, protected evidence directories, new attempt dirs, "
                         "failures and rejections kept",
                         "J2 parent preflight (re-derivation) per seed before any child scoring with that seed"],
        "uncertainty": ("bootstrap CIs are exploratory; bootstrap_sign_proportion_le_0 is descriptive, not a p-value, never multiplicity-adjusted; "
                        "an optional cluster-randomisation test is proposed in the v3 proposal and is not part of this declaration"),
        "disclosures": ["reused development folds (informed v1-v3, Stage I, J1-J3)", "adaptive selection of J4b after H_RT",
                        "calibration dependence (LOFO gamma ~8; H_bal at gamma 0 is 5-9%)",
                        "fixed parent J2 selected on all six folds; primary gate is an exploratory paired comparison, not nested",
                        "image-fold counts are not independent observations (e.g. other-source seen: 628 image-folds = 111 unique images)"],
        "stop": "after the J4b seed-0 evaluation, regardless of outcome; report and wait for independent review",
        "code_pins": {f: sha256_file(ROOT / f) for f in PINNED},
    }
    OUT.write_text(json.dumps(am, indent=1) + "\n")
    print(json.dumps({k: am[k] for k in ("built_from_commit", "authorised_now", "code_pins")}, indent=1))


if __name__ == "__main__":
    main()
