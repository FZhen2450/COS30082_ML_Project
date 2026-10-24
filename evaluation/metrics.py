"""Model-independent composition evaluation for DPD-Small.

Candidate composition IDs must be GLOBAL IDs and preserve an explicit score-column order.
This module never reads official test data or chooses a candidate universe.
"""
from __future__ import annotations
from typing import Mapping, Sequence
import numpy as np
import torch


def _matrix(scores):
    if isinstance(scores, torch.Tensor):
        scores = scores.detach().cpu().numpy()
    arr = np.asarray(scores, dtype=np.float64)
    if arr.ndim != 2 or 0 in arr.shape:
        raise ValueError("scores must have shape [N, K] with N,K > 0")
    if not np.isfinite(arr).all():
        raise ValueError("scores contain NaN or infinity")
    return arr


def topk_accuracy(scores, ground_truth_ids: Sequence[int], candidate_ids: Sequence[int], k: int = 1) -> float:
    """Return accuracy on a 0--1 scale, using global composition IDs.

    Ties are resolved by their original candidate column order (stable sort).
    """
    scores = _matrix(scores)
    candidates = [int(v) for v in candidate_ids]
    truth = [int(v) for v in ground_truth_ids]
    if len(candidates) != scores.shape[1] or len(set(candidates)) != len(candidates):
        raise ValueError("candidate_ids must be unique and match score columns")
    if len(truth) != scores.shape[0]:
        raise ValueError("ground_truth_ids length must match number of rows")
    if not set(truth).issubset(set(candidates)):
        raise ValueError("ground-truth composition absent from candidate universe")
    if not (1 <= k <= len(candidates)):
        raise ValueError("k must be between 1 and the number of candidates")
    order = np.argsort(-scores, axis=1, kind="stable")[:, :k]
    ranked = np.asarray(candidates, dtype=np.int64)[order]
    return float(np.mean((ranked == np.asarray(truth, dtype=np.int64)[:, None]).any(axis=1)))


def evaluate_compositions(scores, ground_truth_ids, candidate_ids) -> dict:
    scores = _matrix(scores)
    return {
        "count": int(scores.shape[0]),
        "top1": topk_accuracy(scores, ground_truth_ids, candidate_ids, k=1),
        "top5": topk_accuracy(scores, ground_truth_ids, candidate_ids, k=min(5, len(candidate_ids))),
        "candidate_count": int(len(candidate_ids)),
    }


def harmonic_mean(seen_accuracy: float, unseen_accuracy: float) -> float:
    s, u = float(seen_accuracy), float(unseen_accuracy)
    if not (0 <= s <= 1 and 0 <= u <= 1):
        raise ValueError("accuracies must be fractions in [0, 1]")
    return 0.0 if s + u == 0 else 2 * s * u / (s + u)


def summarize_seen_unseen(seen: Mapping, unseen: Mapping) -> dict:
    """Accept evaluate_compositions outputs, and compute Top-1 harmonic mean."""
    for name, part in (("seen", seen), ("unseen", unseen)):
        if not all(key in part for key in ("top1", "top5", "count")):
            raise ValueError(f"{name} results lack top1, top5 or count")
    return {"seen": dict(seen), "unseen": dict(unseen),
            "harmonic_mean_top1": harmonic_mean(seen["top1"], unseen["top1"])}


def composition_scores_from_logits(crop_logits, disease_logits, candidate_pairs: Sequence[tuple[int, int]]):
    """P(crop|x) * P(disease|x), one column per supplied (crop,disease) pair.

    These are unnormalised candidate ranking scores, NOT calibrated composition
    probabilities. The candidate universe must be agreed independently.
    """
    if not isinstance(crop_logits, torch.Tensor) or not isinstance(disease_logits, torch.Tensor):
        raise TypeError("crop_logits and disease_logits must be torch tensors")
    if crop_logits.ndim != 2 or disease_logits.ndim != 2 or crop_logits.shape[0] != disease_logits.shape[0]:
        raise ValueError("logits require matching batch sizes and rank 2")
    if not candidate_pairs or len(set(tuple(p) for p in candidate_pairs)) != len(candidate_pairs):
        raise ValueError("candidate_pairs must be nonempty and unique")
    crop_ids = [int(p[0]) for p in candidate_pairs]
    disease_ids = [int(p[1]) for p in candidate_pairs]
    if min(crop_ids) < 0 or max(crop_ids) >= crop_logits.shape[1]:
        raise ValueError("crop index out of range")
    if min(disease_ids) < 0 or max(disease_ids) >= disease_logits.shape[1]:
        raise ValueError("disease index out of range")
    pc = torch.softmax(crop_logits, dim=1)
    pd = torch.softmax(disease_logits, dim=1)
    return pc[:, crop_ids] * pd[:, disease_ids]
