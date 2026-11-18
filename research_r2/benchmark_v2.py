"""Benchmark v2: an outer confirmation population isolated from all inner development.

Construction (seeded, deterministic, training-source images only):
  1. Outer compositions O = O1 ∪ O2 (each ~11 eligible diseased pairs). O is chosen so that, with
     ALL of O removed, every outer crop and disease keeps support and every outer disease is
     still observed on another crop (concept coverage for the compositional task).
  2. Outer seen holdout G_O: lineage-aware ~10% per remaining composition.
  3. Outer exclusion set X = images of O ∪ G_O, closed under lineage (any image linked to X joins X).
  4. Inner population P = all images minus X. Inner eligibility, inner folds I1..I6 and the inner
     seen holdout G_I are built from P alone (same rules as v1). Inner folds never contain any
     row of X, so no fitting, statistic, calibration or selection step of the inner stage can
     see outer images or their derived views.
  5. Outer folds O1, O2 (role "outer", protected): fit on everything except the evaluated group
     and G_O; unseen = that group's images; seen = G_O. Scored only via a registered frozen recipe.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .benchmark import (_gser, _materialise, assign_folds, eligible_compositions, heldout_set_valid,
                        seen_holdout, validate_benchmark)
from .data import sha256_json

SCHEMA = "r2-compositional-benchmark-v2-outer-isolated"


def choose_outer(frame, eligible, n_groups, group_size, seed, min_inner_eligible):
    rng = np.random.default_rng(seed)
    order = [int(c) for c in rng.permutation(eligible)]
    chosen: list[int] = []
    for c in order:
        if len(chosen) == n_groups * group_size:
            break
        trial = set(chosen) | {c}
        if not heldout_set_valid(frame, trial):
            continue
        rest = frame[~frame.comp_id.isin(trial)]
        if len(eligible_compositions(rest)) < min_inner_eligible:
            continue
        chosen.append(c)
    if len(chosen) < n_groups * group_size:
        raise RuntimeError("could not choose a valid outer composition set")
    groups = assign_folds(frame, chosen, n_groups, seed + 7)  # size-balanced split, validity vs full population
    for gk in groups:  # each group must also be valid on its own against the full population
        assert heldout_set_valid(frame, set(gk))
    return groups


def build_benchmark_v2(frame: pd.DataFrame, group, *, seed: int = 3031, n_outer_groups: int = 2,
                       outer_group_size: int = 11, outer_seen_fraction: float = 0.10, n_inner: int = 6,
                       inner_seen_fraction: float = 0.15, min_images: int = 5) -> dict:
    gser = _gser(frame, group)
    eligible_all = eligible_compositions(frame, min_images)
    outer_groups = choose_outer(frame, eligible_all, n_outer_groups, outer_group_size, seed,
                                min_inner_eligible=n_inner * 10)
    O = set(c for g in outer_groups for c in g)
    non_outer = frame[~frame.comp_id.isin(O)]
    G_O = set(seen_holdout(non_outer, gser, outer_seen_fraction, seed + 2))
    x0 = set(frame.index[frame.comp_id.isin(O)]) | G_O
    x_groups = set(gser.loc[sorted(x0)])
    X = set(frame.index[gser.isin(x_groups)])  # lineage closure
    P = frame.drop(index=sorted(X))
    inner_eligible = eligible_compositions(P, min_images)
    raw = assign_folds(P, inner_eligible, n_inner, seed + 3)
    G_I = set(seen_holdout(P, gser, inner_seen_fraction, seed + 4))
    gP = gser.loc[P.index]
    folds = []
    for k, held in enumerate(raw):
        folds.append(_materialise(P, gP, G_I, set(held), name=f"I{k + 1}", role="inner"))
    for k, held in enumerate(outer_groups):
        folds.append(_materialise(frame, gser, G_O, set(held), name=f"O{k + 1}", role="outer"))
    bench = {
        "schema": SCHEMA, "seed": seed, "n_inner": n_inner, "n_outer_groups": n_outer_groups,
        "outer_group_size": outer_group_size, "outer_seen_fraction": outer_seen_fraction,
        "inner_seen_fraction": inner_seen_fraction, "min_images": min_images,
        "outer_composition_ids": sorted(O), "outer_seen_rows": sorted(int(r) for r in G_O),
        "outer_exclusion_rows": sorted(int(r) for r in X), "inner_population_rows": sorted(int(r) for r in P.index),
        "inner_eligible_composition_ids": inner_eligible, "inner_seen_holdout_rows": sorted(int(r) for r in G_I),
        "candidate_composition_ids": sorted(int(c) for c in frame.comp_id.unique()),
        "folds": folds,
    }
    bench["semantic_sha256"] = sha256_json({k: v for k, v in bench.items() if k != "semantic_sha256"})
    return bench


def validate_benchmark_v2(frame: pd.DataFrame, group, bench: dict) -> list[str]:
    gser = _gser(frame, group)
    X = set(bench["outer_exclusion_rows"])
    P_rows = set(bench["inner_population_rows"])
    checks = []
    assert not (X & P_rows) and len(X | P_rows) == len(frame), "X and P must partition the population"
    assert not (set(gser.loc[sorted(X)]) & set(gser.loc[sorted(P_rows)])), "lineage group spans X and P"
    checks.append("outer exclusion set and inner population partition all images; no lineage group spans both")
    inner = [f for f in bench["folds"] if f["role"] == "inner"]
    outer = [f for f in bench["folds"] if f["role"] == "outer"]
    for f in inner:
        used = set(f["fitting_rows"]) | set(f["seen_val_rows"]) | set(f["unseen_val_rows"]) | \
            set(f["lineage_dropped_rows"]) | set(f["excluded_fitting_rows"])
        assert used == P_rows, f"{f['name']} must cover exactly the inner population"
        assert not (used & X)
    checks.append(f"{len(inner)} inner folds use only inner-population rows (zero outer rows)")
    P = frame.loc[sorted(P_rows)]
    sub = {"folds": inner, "eligible_composition_ids": bench["inner_eligible_composition_ids"]}
    checks += ["inner " + c for c in validate_benchmark(P, gser, sub)]
    O = set(bench["outer_composition_ids"])
    assert not (O & set(bench["inner_eligible_composition_ids"]))
    assert not (O & set(P.comp_id)), "outer compositions must be absent from the inner population"
    assert set(frame[frame.comp_id.isin(O)].crop) <= set(P.crop)
    assert set(frame[frame.comp_id.isin(O)].disease) <= set(P.disease)
    checks.append("outer compositions absent from inner population; outer crops and diseases all covered by inner data")
    full = {"folds": outer, "eligible_composition_ids": bench["outer_composition_ids"]}
    for f in outer:
        assert set(f["seen_val_rows"]) == set(bench["outer_seen_rows"]) - set(frame.index[frame.comp_id.isin(f["heldout_composition_ids"])])
    checks += ["outer " + c for c in validate_benchmark(frame, gser, {**full, "folds": outer})]
    return checks
