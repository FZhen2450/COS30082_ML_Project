"""Benchmark v3 = v2 image partitions + an explicit per-fold candidate universe (research_r2.candidates).

Rows, folds, lineage handling and outer isolation are copied unchanged from v2 (whose semantic hash is
recorded). Each fold gains ``candidate_policy`` and ``candidate_composition_ids`` (F u H) plus the
counts of F, H and E. Outer folds are kept protected and marked previously examined: they were scored
once under v2 and may only be re-scored as labelled diagnostics.
"""
from __future__ import annotations

import copy

from .benchmark_v2 import validate_benchmark_v2
from .candidates import POLICY, candidate_universe
from .data import sha256_json

SCHEMA = "r2-compositional-benchmark-v3-candidate-policy"


def build_benchmark_v3(frame, v2: dict) -> dict:
    b = copy.deepcopy({k: v for k, v in v2.items() if k not in ("semantic_sha256", "provenance")})
    b["schema"] = SCHEMA
    b["derived_from"] = {"schema": v2["schema"], "semantic_sha256": v2["semantic_sha256"]}
    b["candidate_policy"] = POLICY
    for f in b["folds"]:
        cu = candidate_universe(frame, f)
        f["candidate_policy"] = POLICY
        f["candidate_composition_ids"] = cu["universe"]
        f["candidate_counts"] = {"fitting": len(cu["fitting"]), "heldout": len(cu["heldout"]), "excluded": len(cu["excluded"])}
        f["excluded_composition_ids"] = cu["excluded"]
        if f["role"] == "outer":
            f["previously_examined"] = "scored once under v2 (commit 74766e5); diagnostic re-scoring only"
    b["semantic_sha256"] = sha256_json(b)
    return b


def validate_benchmark_v3(frame, group, b: dict, v2: dict) -> list[str]:
    checks = validate_benchmark_v2(frame, group, dict(b, folds=b["folds"]))
    for k in ("outer_composition_ids", "outer_seen_rows", "outer_exclusion_rows", "inner_population_rows"):
        assert b[k] == v2[k], k
    for f, g in zip(b["folds"], v2["folds"]):
        for k in ("name", "role", "heldout_composition_ids", "fitting_rows", "seen_val_rows", "unseen_val_rows"):
            assert f[k] == g[k], (f["name"], k)
    checks.append("v3 image partitions identical to v2")
    outer = set(b["outer_composition_ids"])
    for f in b["folds"]:
        cu = candidate_universe(frame, f)
        assert cu["universe"] == f["candidate_composition_ids"]
        if f["role"] == "inner":
            assert set(f["excluded_composition_ids"]) == outer, f["name"]
        else:
            assert f["excluded_composition_ids"] == [], f["name"]
        fit = set(frame.comp_id.loc[f["fitting_rows"]])
        assert all(c in fit for c in cu["fitting"]) and not (set(cu["heldout"]) & fit)
    checks.append("every fold: universe = F u H; every penalised candidate has fitting images; inner E = the 22 outer compositions; outer E empty")
    return checks
