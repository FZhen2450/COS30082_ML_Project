"""Explicit candidate-universe policy (introduced with benchmark v3).

Every composition is in exactly one class per fold:

  F  fitting      >= 1 fitting image in this fold. Scored, penalised by calibrated stacking ("seen").
  H  held-out     evaluated as unseen in this fold. Scored, NOT penalised. Its *name* may be used for
                  text prompts and for synthesising features from other compositions' fitting images.
  E  excluded     no fitting image and not evaluated in this fold (e.g. outer compositions during inner
                  development, or a composition emptied by lineage exclusion). NOT scored: removed from
                  the candidate universe, its name is not used, and no features are synthesised for it.

Rationale: the official task has F = 323 trained compositions, H = 10 named unseen ones, E = empty.
Treating E as F (the v2 inner behaviour) adds penalised candidates the model never saw; treating E as
H would put ~33 unseen candidates among 323 (official: 10 of 333) and use outer names during inner
development. Removing E is the only choice that keeps both the official structure and the isolation.

Invariants (checked by ``candidate_universe``): F and H are disjoint; every F has fitting images;
no H has fitting images; every validation truth lies in F or H (seen-val in F, unseen-val in H).
"""
from __future__ import annotations

import pandas as pd

POLICY = "r2-candidate-policy-v1: universe = F u H; E removed"


def candidate_universe(frame: pd.DataFrame, fold: dict) -> dict:
    all_ids = sorted(int(c) for c in frame.comp_id.unique())
    H = sorted(int(c) for c in fold["heldout_composition_ids"])
    fit_ids = set(int(c) for c in frame.comp_id.loc[fold["fitting_rows"]])
    F = sorted(c for c in all_ids if c in fit_ids)
    E = sorted(c for c in all_ids if c not in fit_ids and c not in set(H))
    if set(F) & set(H):
        raise ValueError(f"{fold['name']}: held-out composition has fitting images")
    seen_truth = set(int(c) for c in frame.comp_id.loc[fold["seen_val_rows"]])
    unseen_truth = set(int(c) for c in frame.comp_id.loc[fold["unseen_val_rows"]])
    if not seen_truth <= set(F):
        raise ValueError(f"{fold['name']}: seen-val truth outside F: {sorted(seen_truth - set(F))}")
    if not unseen_truth <= set(H):
        raise ValueError(f"{fold['name']}: unseen-val truth outside H")
    return {"policy": POLICY, "fitting": F, "heldout": H, "excluded": E, "universe": sorted(F + H)}


def derive_fold(frame: pd.DataFrame, fold: dict, fitting_rows) -> dict:
    """A fold with modified fitting rows (subsample / source ablation) that keeps the v3 policy:
    compositions left without fitting images move from F to E, and their seen-val rows are dropped
    (recorded under ``dropped_seen_val_rows``). Legacy folds are returned with only fitting rows changed."""
    f2 = dict(fold, fitting_rows=sorted(int(r) for r in fitting_rows))
    if "candidate_policy" not in fold:
        return f2
    fit_ids = set(int(c) for c in frame.comp_id.loc[f2["fitting_rows"]])
    keep = [r for r in fold["seen_val_rows"] if int(frame.comp_id[r]) in fit_ids]
    f2["dropped_seen_val_rows"] = sorted(set(fold["seen_val_rows"]) - set(keep))
    f2["seen_val_rows"] = keep
    f2["candidate_composition_ids"] = candidate_universe(frame, f2)["universe"]
    return f2
