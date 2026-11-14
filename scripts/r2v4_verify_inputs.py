"""v4 pre-execution input verification (read-only; writes one JSON receipt).

Checks, each recorded with its outcome:
  1. git: HEAD descends from the declaration commit; declared files unchanged since it (DECLARATION.json);
     the declaration commit contains no v4 results (only feasibility evidence).
  2. benchmark v3: semantic sha256 recomputed; full v3 validator (v2 partition + lineage checks + candidate
     policy) re-run against benchmark v2 and the v1 lineage groups.
  3. candidate policy: per fold, F/H/E re-derived and equal to the stored universe; inner E = 22 outer comps.
  4. lineage boundaries (explicit): per inner fold no lineage group spans fitting rows and seen/unseen-val rows;
     no inner fitting row shares a group with an outer-exclusion row; all rows index the 8,000-row training CSV.
  5. D1 scores: the registered v3 D1 packets (registry sha256 + strict contract validation) and a fresh
     in-memory D1 recomputation agree within 1e-5 on every score.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research_r2.benchmark_v3 import validate_benchmark_v3  # noqa: E402
from research_r2.candidates import POLICY, candidate_universe  # noqa: E402
from research_r2.data import TRAIN_CSV_SHA256, load_training_source, sha256_file, sha256_json  # noqa: E402
from research_r2.methods import RECIPES  # noqa: E402
from research_r2.packets import file_sha256, fold_contract, resolve_packets, validate_packet  # noqa: E402
from research_r2.runner import load_features, make_fold  # noqa: E402

EXP = ROOT.parents[1] / "experiments" / "r2-compositional"
DECL_COMMIT = "3660975"
TOL = 1e-5


def git(*a):
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rep, ok = {"schema": "r2v4-input-verification-v1"}, True

    # 1. git / declaration
    head = git("rev-parse", "HEAD")
    anc = subprocess.run(["git", "merge-base", "--is-ancestor", DECL_COMMIT, "HEAD"], cwd=ROOT).returncode == 0
    decl = json.loads((ROOT / "research_r2_v4/DECLARATION.json").read_text())["sha256"]
    now = {f: sha256_file(ROOT / f) for f in decl}
    at_decl = {f: subprocess.run(["git", "show", f"{DECL_COMMIT}:{f}"], cwd=ROOT, capture_output=True).stdout for f in decl}
    import hashlib
    at_decl = {f: hashlib.sha256(b).hexdigest() for f, b in at_decl.items()}
    files_in_decl = git("show", "--name-only", "--format=", DECL_COMMIT).splitlines()
    v4_results_in_decl = [f for f in files_in_decl if f.startswith("artifacts/r2/v4/") and "feasibility_20261114" not in f]
    rep["git"] = {"head": head, "declaration_commit": git("rev-parse", DECL_COMMIT), "declaration_is_ancestor": anc,
                  "declared_sha256": decl, "unchanged_since_declaration": now == decl == at_decl,
                  "declaration_commit_files": files_in_decl, "v4_results_in_declaration_commit": v4_results_in_decl,
                  "declaration_commit_time": git("show", "-s", "--format=%cI", DECL_COMMIT)}
    ok &= anc and now == decl == at_decl and not v4_results_in_decl

    # 2-4. benchmark, policy, lineage
    frame = load_training_source(ROOT / "DPD_Small_train.csv")
    b3 = json.loads((ROOT / "artifacts/r2/benchmark_v3.json").read_text())
    b2 = json.loads((ROOT / "artifacts/r2/benchmark_v2.json").read_text())
    group = np.array(json.loads((ROOT / "artifacts/r2/lineage_groups_v1.json").read_text())["group"])
    sem = sha256_json({k: v for k, v in b3.items() if k not in ("semantic_sha256", "provenance")})
    checks = validate_benchmark_v3(frame, group, b3, b2)
    rep["benchmark"] = {"train_csv_sha256": sha256_file(ROOT / "DPD_Small_train.csv"), "expected_train_csv": TRAIN_CSV_SHA256,
                        "semantic_sha256_stored": b3["semantic_sha256"], "semantic_sha256_recomputed": sem,
                        "validator_checks": checks}
    ok &= sem == b3["semantic_sha256"] == "0e85819cbd845eb6794cd8414950aac794e351636ae1f1202306b4ed8b1a7e42"
    outer_rows = set(b3["outer_exclusion_rows"]) | set(b3["outer_seen_rows"])
    outer_groups = set(group[list(outer_rows)])
    per_fold = {}
    for f in [f for f in b3["folds"] if f["role"] == "inner"]:
        cu = candidate_universe(frame, f)
        fit_g = set(group[f["fitting_rows"]])
        val = f["seen_val_rows"] + f["unseen_val_rows"]
        all_rows = f["fitting_rows"] + val
        pf = {"policy_ok": cu["policy"] == POLICY == f["candidate_policy"] and cu["universe"] == f["candidate_composition_ids"],
              "F": len(cu["fitting"]), "H": len(cu["heldout"]), "E": len(cu["excluded"]),
              "E_is_outer_compositions": set(cu["excluded"]) == set(b3["outer_composition_ids"]),
              "lineage_groups_shared_fit_val": len(fit_g & set(group[val])),
              "lineage_groups_shared_fit_outer": len(fit_g & outer_groups),
              "rows_overlap_fit_val": len(set(f["fitting_rows"]) & set(val)),
              "rows_overlap_with_outer": len(set(all_rows) & outer_rows),
              "rows_in_training_csv": bool(min(all_rows) >= 0 and max(all_rows) < len(frame) == 8000)}
        pf["ok"] = (pf["policy_ok"] and pf["E_is_outer_compositions"] and pf["lineage_groups_shared_fit_val"] == 0
                    and pf["lineage_groups_shared_fit_outer"] == 0 and pf["rows_overlap_fit_val"] == 0
                    and pf["rows_overlap_with_outer"] == 0 and pf["rows_in_training_csv"])
        ok &= pf["ok"]
        per_fold[f["name"]] = pf
    rep["folds"] = per_fold

    # 5. D1 scores: registered packets vs fresh recomputation
    reg = json.loads((ROOT / "artifacts/r2/v3/packet_registry_v3.json").read_text())["rows"]["D1_synthComp_norm"]
    # every v3 copy of the D1 packets; resolve_packets rejects the set unless all copies are byte-identical
    dirs = [EXP / "v3" / d / "scores" for d in ("inner_stage_d_v1", "inner_stage_f_v3", "repro_hardening_20261113")]
    res = resolve_packets("D1_synthComp_norm", list(per_fold), [d for d in dirs if d.is_dir()])
    feats, text, files, _ = load_features(EXP / "features", ["clip_l14"])
    feats = feats.clone(); feats[b3["outer_exclusion_rows"]] = float("nan")
    F = {f["name"]: f for f in b3["folds"]}
    d1 = {}
    for fn in per_fold:
        z = np.load(res[fn]["path"])
        validate_packet(z, fold_contract(frame, F[fn]))
        torch.manual_seed(0)
        fd, truth, _ = make_fold(frame, F[fn], feats, text)
        s, u = RECIPES["synth_composition"](fd, {"fuse": 1.0, "normalise": True})
        diff = max(float(np.abs(z["seen"] - s.numpy()).max()), float(np.abs(z["unseen"] - u.numpy()).max()))
        d1[fn] = {"packet_sha256": res[fn]["sha256"], "registered_sha256": reg["folds"][fn]["sha256"],
                  "registry_match": res[fn]["sha256"] == reg["folds"][fn]["sha256"],
                  "copies": len(res[fn]["copies"]), "fresh_vs_packet_max_abs_diff": diff,
                  "truth_match": bool(np.array_equal(z["seen_truth"], truth["seen"]) and np.array_equal(z["unseen_truth"], truth["unseen"]))}
        ok &= d1[fn]["registry_match"] and diff <= TOL and d1[fn]["truth_match"]
    rep["d1_scores"] = d1
    rep["pass"] = bool(ok)
    Path(a.out).write_text(json.dumps(rep, indent=1, default=str))
    print(json.dumps({k: v for k, v in rep.items() if k in ("pass",)}), json.dumps(rep["git"]["unchanged_since_declaration"]))
    for fn, v in per_fold.items():
        print(fn, v)
    for fn, v in d1.items():
        print(fn, v["registry_match"], f"{v['fresh_vs_packet_max_abs_diff']:.2e}", v["truth_match"])
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
