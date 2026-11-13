"""Shared v4 helpers: environment loading, scoring a fold, and common-universe restriction.

Everything reuses research_r2 (make_fold, fold_contract, FoldScorer); nothing there is modified.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from research_r2.data import load_training_source
from research_r2.packets import fold_contract, validate_packet
from research_r2.runner import load_features, make_fold, scorer_from_scores

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT.parents[1] / "experiments" / "r2-compositional"
V4 = EXP / "v4"
FEATURES = EXP / "features"
BENCH = ROOT / "artifacts/r2/benchmark_v3.json"
INNER = ["I1", "I2", "I3", "I4", "I5", "I6"]
LABEL = "post-freeze exploratory (v4); development cross-validation on v3 inner folds; not blind"


def env():
    frame = load_training_source(ROOT / "DPD_Small_train.csv")
    bench = json.loads(BENCH.read_text())
    folds = {f["name"]: f for f in bench["folds"]}
    return frame, bench, folds


def poisoned_features(bench, backbones, feature_dir=FEATURES):
    """v3 inner-stage convention: outer-exclusion rows become NaN so any accidental use raises."""
    feats, text, files, ident = load_features(feature_dir, backbones)
    feats = feats.clone()
    feats[bench["outer_exclusion_rows"]] = float("nan")
    return feats, text, files, ident


def score_fold(frame, fold, feats, text, recipe, cfg, prep=None):
    """``prep(fd, fold, feats)`` (optional) adds fold-derived inputs, e.g. I6's raw fitting statistics."""
    torch.manual_seed(0)
    fd, truth, healthy = make_fold(frame, fold, feats, text)
    if prep is not None:
        prep(fd, fold, feats)
    s, u = recipe(fd, cfg)
    return {"seen": s.numpy().astype(np.float64), "unseen": u.numpy().astype(np.float64),
            "seen_truth": truth["seen"], "unseen_truth": truth["unseen"], "cand_ids": np.asarray(fd.extra["cand_ids"]),
            "fold": fold}


def restrict(frame, arm: dict, target_fold: dict):
    """Restrict an arm's scores to ``target_fold``'s candidate universe and evaluation rows.

    Columns are matched by global composition id, rows by benchmark row id; truth is re-indexed and the
    result is validated strictly against the target fold's contract, so arms are only ever compared on an
    identical universe, identical rows and identical truth columns. Raises on any misalignment.
    """
    src = arm["fold"]
    con = fold_contract(frame, target_fold)
    ids = list(arm["cand_ids"])
    col = [ids.index(int(c)) for c in con["cand_ids"]]
    out = {"cand_ids": np.asarray(con["cand_ids"])}
    for part, key in (("seen", "seen_val_rows"), ("unseen", "unseen_val_rows")):
        pos = {r: i for i, r in enumerate(src[key])}
        rows = [pos[r] for r in target_fold[key]]
        sc = arm[part][np.ix_(rows, col)]
        old_truth_ids = np.asarray(ids)[arm[f"{part}_truth"][rows]]
        new_col = {int(c): i for i, c in enumerate(con["cand_ids"])}
        out[part] = sc
        out[f"{part}_truth"] = np.array([new_col[int(c)] for c in old_truth_ids])
    validate_packet(out, con)
    return scorer_from_scores(frame, target_fold, out["seen"], out["seen_truth"], out["unseen"], out["unseen_truth"],
                              out["cand_ids"])


def scorer(frame, arm):
    return scorer_from_scores(frame, arm["fold"], arm["seen"], arm["seen_truth"], arm["unseen"], arm["unseen_truth"],
                              arm["cand_ids"])


def append_jsonl(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as fh:
        fh.write(json.dumps(obj, default=float) + "\n")


def read_jsonl(path):
    p = Path(path)
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
