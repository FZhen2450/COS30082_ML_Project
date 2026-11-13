"""Round 2 training-source benchmark: composition-disjoint folds that mirror the official task.

Official unseen structure (from the assignment brief and the 10 published class names,
which the lecturer permits as semantic information): every unseen class is DISEASED, its
crop is known from other training compositions and its disease is known on other crops.
Eligibility below encodes exactly that structure using training data only.

Per fold f with held-out composition set H_f:
    unseen_val_f = all images of H_f (minus cross-composition lineage-linked images)
    seen_val_f   = G  minus images of H_f          (G: one global lineage-aware holdout)
    fitting_f    = everything else
Candidates are always all 323 training-source compositions (ascending global IDs), so
each fold has |H_f| ~ 11 "unseen" candidates among 323, mirroring 10 among 333.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import HEALTHY, sha256_json

SCHEMA = "r2-compositional-benchmark-v1"


def eligible_compositions(frame: pd.DataFrame, min_images: int = 5) -> list[int]:
    n = frame.groupby("comp_id").size()
    crops_per_disease = frame.groupby("disease").crop.nunique()
    comps_per_crop = frame.groupby("crop").comp_id.nunique()
    cand = frame.drop_duplicates("comp_id")
    keep = cand[(cand.disease != HEALTHY)
                & cand.disease.map(lambda d: crops_per_disease[d] >= 2)
                & cand.crop.map(lambda c: comps_per_crop[c] >= 2)
                & cand.comp_id.map(lambda c: n[c] >= min_images)]
    return sorted(int(c) for c in keep.comp_id)


def heldout_set_valid(frame: pd.DataFrame, held: set[int]) -> bool:
    """Every held-out crop and disease must keep fitting support from other compositions,
    and every held-out disease must remain observed on at least one other crop."""
    rest = frame[~frame.comp_id.isin(held)]
    crops, diseases = set(rest.crop), set(rest.disease)
    for _, r in frame[frame.comp_id.isin(held)].drop_duplicates("comp_id").iterrows():
        if r.crop not in crops or r.disease not in diseases:
            return False
        if not (rest[rest.disease == r.disease].crop != r.crop).any():
            return False
    return True


def assign_folds(frame: pd.DataFrame, eligible: list[int], n_folds: int, seed: int) -> list[list[int]]:
    """Seeded balanced assignment: largest compositions first, each to the fold with the
    fewest compositions (ties: fewest images, then lowest index) that stays valid."""
    rng = np.random.default_rng(seed)
    size = frame.groupby("comp_id").size()
    order = list(rng.permutation(eligible))
    order.sort(key=lambda c: -int(size[c]))  # stable: random order within equal size
    folds: list[list[int]] = [[] for _ in range(n_folds)]
    for comp in order:
        ranked = sorted(range(n_folds), key=lambda f: (len(folds[f]), int(size[folds[f]].sum()) if folds[f] else 0, f))
        for f in ranked:
            if heldout_set_valid(frame, set(folds[f]) | {int(comp)}):
                folds[f].append(int(comp))
                break
        else:
            raise RuntimeError(f"composition {comp} cannot be placed in any fold")
    return [sorted(f) for f in folds]


def seen_holdout(frame: pd.DataFrame, group: np.ndarray, fraction: float, seed: int,
                 min_comp_images: int = 4) -> list[int]:
    """Lineage-aware per-composition image holdout G (row indices).

    Only lineage groups wholly inside one composition are eligible; compositions with
    fewer than ``min_comp_images`` images contribute none (keeps rare concepts in fitting).
    """
    rng = np.random.default_rng(seed)
    g = pd.Series(group, index=frame.index)
    comps_per_group = frame.groupby(g).comp_id.nunique()
    chosen: list[int] = []
    for comp, rows in frame.groupby("comp_id").groups.items():
        rows = list(rows)
        if len(rows) < min_comp_images:
            continue
        target = int(round(fraction * len(rows)))
        groups = sorted({int(group[r]) for r in rows if comps_per_group[group[r]] == 1})
        rng.shuffle(groups)
        taken = 0
        for gid in groups:
            if taken >= target:
                break
            members = [r for r in rows if group[r] == gid]
            if taken + len(members) > max(target, 1) + 1 and taken > 0:
                continue
            chosen.extend(members)
            taken += len(members)
        if taken >= len(rows):  # never remove a composition entirely from fitting
            raise RuntimeError(f"seen holdout consumed composition {comp}")
    return sorted(int(r) for r in chosen)


def build_benchmark(frame: pd.DataFrame, group: np.ndarray, *, seed: int = 2026, n_folds: int = 8,
                    n_screen: int = 5, min_images: int = 5, seen_fraction: float = 0.15,
                    legacy_heldout_ids: list[int] | None = None) -> dict:
    eligible = eligible_compositions(frame, min_images)
    raw_folds = assign_folds(frame, eligible, n_folds, seed)
    role_order = list(np.random.default_rng(seed + 1).permutation(n_folds))
    G = set(seen_holdout(frame, group, seen_fraction, seed + 2))
    gser = pd.Series(group, index=frame.index)
    folds = []
    for rank, f in enumerate(role_order):
        held = set(raw_folds[f])
        folds.append(_materialise(frame, gser, G, held,
                                  name=(f"S{rank + 1}" if rank < n_screen else f"L{rank - n_screen + 1}"),
                                  role=("screen" if rank < n_screen else "locked")))
    if legacy_heldout_ids is not None:
        folds.append(_materialise(frame, gser, G, set(int(c) for c in legacy_heldout_ids), name="D0",
                                  role="legacy_diagnostic_previously_examined", require_eligible=False))
    bench = {
        "schema": SCHEMA, "seed": seed, "n_folds": n_folds, "n_screen": n_screen,
        "min_images": min_images, "seen_fraction": seen_fraction,
        "eligible_composition_ids": eligible,
        "candidate_composition_ids": sorted(int(c) for c in frame.comp_id.unique()),
        "seen_holdout_rows": sorted(G),
        "folds": folds,
    }
    bench["semantic_sha256"] = sha256_json({k: v for k, v in bench.items() if k != "semantic_sha256"})
    return bench


def _materialise(frame, gser, G, held, *, name, role, require_eligible=True):
    if require_eligible and not heldout_set_valid(frame, held):
        raise RuntimeError(f"fold {name} violates concept coverage")
    is_held = frame.comp_id.isin(held)
    held_groups = set(gser[is_held])
    # cross-composition lineage leak: a held-out image linked to any non-held-out image
    leaked_groups = {gid for gid in held_groups if (~is_held[gser == gid]).any()}
    unseen = frame.index[is_held & ~gser.isin(leaked_groups)]
    dropped = frame.index[is_held & gser.isin(leaked_groups)]
    seen = sorted(r for r in G if not is_held[r])
    # remove from fitting any non-held image linked to a held-out image (both sides clean)
    fit_mask = ~is_held & ~frame.index.isin(G) & ~gser.isin(leaked_groups)
    fitting = frame.index[fit_mask]
    sub = frame[is_held].drop_duplicates("comp_id")
    return {
        "name": name, "role": role,
        "heldout_composition_ids": sorted(int(c) for c in held),
        "fitting_rows": [int(r) for r in fitting],
        "seen_val_rows": [int(r) for r in seen],
        "unseen_val_rows": [int(r) for r in unseen],
        "lineage_dropped_rows": [int(r) for r in dropped],
        "excluded_fitting_rows": [int(r) for r in frame.index[~is_held & ~frame.index.isin(G) & ~fit_mask]],
        "heldout_crops": sorted(set(sub.crop)), "heldout_diseases": sorted(set(sub.disease)),
        "heldout_healthy_compositions": int((sub.disease == HEALTHY).sum()),
    }


def validate_benchmark(frame: pd.DataFrame, group: np.ndarray, bench: dict) -> list[str]:
    """Return a list of human-readable check names that passed; raise on any violation."""
    checks = []
    n = len(frame)
    all_comps = set(int(c) for c in frame.comp_id.unique())
    for fold in bench["folds"]:
        fit, sv, uv = (set(fold[k]) for k in ("fitting_rows", "seen_val_rows", "unseen_val_rows"))
        drop, excl = set(fold["lineage_dropped_rows"]), set(fold["excluded_fitting_rows"])
        assert not (fit & sv) and not (fit & uv) and not (sv & uv), fold["name"]
        assert len(fit | sv | uv | drop | excl) == n == len(fit) + len(sv) + len(uv) + len(drop) + len(excl), fold["name"]
        held = set(fold["heldout_composition_ids"])
        fit_comps = set(frame.comp_id[list(fit)])
        assert not (fit_comps & held), f"{fold['name']}: held-out composition in fitting"
        assert set(frame.comp_id[list(uv)]) <= held
        assert set(frame.comp_id[list(sv)]) <= all_comps - held
        assert set(frame.crop[list(uv)]) <= set(frame.crop[list(fit)])
        assert set(frame.disease[list(uv)]) <= set(frame.disease[list(fit)])
        fit_groups = set(group[list(fit)])
        assert not (fit_groups & set(group[list(uv)])), f"{fold['name']}: lineage leak fitting/unseen"
        assert not (fit_groups & set(group[list(sv)])), f"{fold['name']}: lineage leak fitting/seen"
        if fold["role"] != "legacy_diagnostic_previously_examined":
            assert fold["heldout_healthy_compositions"] == 0
        checks.append(f"{fold['name']}: disjoint, complete, composition-disjoint, concept-covered, lineage-clean")
    names = [f["heldout_composition_ids"] for f in bench["folds"] if f["role"] in ("screen", "locked")]
    flat = [c for f in names for c in f]
    assert len(flat) == len(set(flat)) == len(bench["eligible_composition_ids"]), "folds must partition eligible set"
    checks.append("new folds partition the eligible composition set exactly once")
    return checks


def fold_summary(frame: pd.DataFrame, bench: dict) -> pd.DataFrame:
    rows = []
    for f in bench["folds"]:
        uv = frame.loc[f["unseen_val_rows"]]
        per = uv.groupby("comp_id").size()
        rows.append({"fold": f["name"], "role": f["role"], "heldout_comps": len(f["heldout_composition_ids"]),
                     "fitting": len(f["fitting_rows"]), "seen_val": len(f["seen_val_rows"]),
                     "unseen_val": len(f["unseen_val_rows"]), "lineage_dropped": len(f["lineage_dropped_rows"]),
                     "largest_comp_share": round(float(per.max() / per.sum()), 3) if len(per) else 0.0,
                     "heldout_crops": len(f["heldout_crops"]), "heldout_diseases": len(f["heldout_diseases"]),
                     "healthy_heldout": f["heldout_healthy_compositions"]})
    return pd.DataFrame(rows)
