"""Frozen finalist predictors for the v2 robustness runner, without new loaders.

Every finalist already exposes, or can be composed into, the shared WP4/WP5
predictor contract ``predict(list[PIL RGB]) -> {"candidate_ids", "scores", ...}``
that applies the model's own frozen preprocessing and canonical scorer:

* WP1 pair: ``independent_baseline.IndependentPredictor`` (this repository).
* Frozen CLIP and M6: ``wp4.clip_predictor.CLIPPredictor`` (merged in ``gui``).
* WP2 ConvNeXt-Base: no predictor class exists, so ``WP2NotebookPredictor``
  composes the owner's own notebook objects (model, ``checkpoint_inference_
  transform`` result, ``amp_context`` and ``composition_scores``) unchanged.

This module constructs no model, downloads nothing and reads no files.
``PredictorAdapter`` only validates outputs against the declared contract.
"""
from __future__ import annotations

from types import MappingProxyType

import numpy as np

from evaluation.model_adapters import (CLIP_ADAPTER, CLIP_FROZEN, SCORING, SELECTION, WP1_INDEPENDENT,
                                       WP2_CONVNEXT)

_CONTRACTS = {
    WP1_INDEPENDENT: {
        "loader": "independent_baseline.IndependentPredictor(crop.pt, disease.pt, protocol, source_csv)",
        "preprocessing": "each head's torchvision DEFAULT weight transforms (resize + centre crop)",
        "predictor_score_semantics": "sum_component_log_probabilities",
        "score_dtype": "float32", "historical_batch_size": 2,
        # Declared in the identified development packet. load_checkpoint rejects a
        # different installed torchvision, so run in the recorded environment.
        "packet_environment": {"torch": "2.7.1+cu118"},
        "owner_assets": "both head checkpoints (Fa Zhen / Andy)"},
    WP2_CONVNEXT: {
        "loader": "notebook model factory + checkpoint_inference_transform (legacy 224 policy)",
        "preprocessing": "get_transforms(224): square bilinear antialiased resize, ImageNet normalization",
        "predictor_score_semantics": "sum_log_softmax_v1",
        "score_dtype": "float64", "historical_batch_size": 16,
        "packet_environment": {"torch": "2.5.1+cu121", "precision": "bf16", "channels_last": True,
                               "deterministic_algorithms": False, "cudnn_tf32": True},
        "owner_assets": "selected epoch-1 checkpoint and notebook objects (Yuen Kai)"},
    CLIP_FROZEN: {
        "loader": "wp4.clip_predictor.CLIPPredictor(catalog, clip_weights)",
        "preprocessing": "clip _transform(224): bicubic resize, centre crop, CLIP normalization",
        "predictor_score_semantics": "direct_float64_clip_cosine_similarity", "variant": "frozen_clip",
        "score_dtype": "float64", "historical_batch_size": None,
        "packet_environment": {"torch": "2.5.1+cu121"},
        "owner_assets": "pinned clip package install approval; public ViT-L/14 weights"},
    CLIP_ADAPTER: {
        "loader": "wp4.clip_predictor.CLIPPredictor(catalog, clip_weights, adapter_checkpoint)",
        "preprocessing": "clip _transform(224): bicubic resize, centre crop, CLIP normalization",
        "predictor_score_semantics": "direct_float64_clip_cosine_similarity", "variant": "m6",
        "score_dtype": "float64", "historical_batch_size": None,
        "packet_environment": {"torch": "2.5.1+cu121"},
        "owner_assets": "M6 adapter checkpoint (Yuen Kai) plus the frozen CLIP assets"},
}
FINALIST_CONTRACTS = MappingProxyType({
    kind: MappingProxyType({**spec, "packet_score_semantics": SCORING[kind],
                            # Copies: never expose model_adapters.SELECTION or nested dicts mutably.
                            "packet_environment": MappingProxyType(dict(spec["packet_environment"])),
                            "selection": MappingProxyType(dict(SELECTION.get(kind, {})))})
    for kind, spec in _CONTRACTS.items()})


def _numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


class PredictorAdapter:
    """Validate a frozen finalist predictor's outputs against its declared contract."""

    def __init__(self, predictor, *, model_kind, candidate_ids):
        if model_kind not in FINALIST_CONTRACTS:
            raise ValueError(f"unknown finalist model kind: {model_kind}")
        ids = [int(v) for v in candidate_ids]
        if not ids or ids != sorted(set(ids)):
            raise ValueError("candidate_ids must be unique and ascending global IDs")
        if not callable(getattr(predictor, "predict", None)):
            raise TypeError("predictor must expose predict(images)")
        self.predictor, self.kind, self.ids = predictor, model_kind, np.asarray(ids, dtype=np.int64)
        self.contract = FINALIST_CONTRACTS[model_kind]

    def predict(self, images):
        images = list(images)
        # Explicit top-k: the shared contract's default (5) is invalid for universes below 5 columns.
        output = self.predictor.predict(images, topk=min(5, int(self.ids.size)))
        returned_ids = _numpy(output.get("candidate_ids"))
        if returned_ids.shape != self.ids.shape or not np.array_equal(returned_ids.astype(np.int64), self.ids):
            raise ValueError("predictor candidate columns differ from the declared candidate universe")
        semantics = output.get("score_semantics")
        if semantics != self.contract["predictor_score_semantics"]:
            raise ValueError(f"unexpected score semantics {semantics!r} for {self.kind}")
        variant = self.contract.get("variant")
        if variant is not None and (output.get("model_provenance") or {}).get("variant") != variant:
            raise ValueError(f"predictor variant does not match {self.kind}")
        scores = _numpy(output["scores"])
        if scores.dtype != np.dtype(self.contract["score_dtype"]):
            raise ValueError(f"{self.kind} scores must be {self.contract['score_dtype']}, got {scores.dtype}")
        if scores.shape != (len(images), self.ids.size) or not np.isfinite(scores).all():
            raise ValueError("scores must be finite with shape [batch, candidates]")
        result = {"scores": scores.copy()}
        if "topk_composition_ids" in output:
            topk = _numpy(output["topk_composition_ids"]).astype(np.int64)
            expected = self.ids[np.argsort(-scores, axis=1, kind="stable")[:, :topk.shape[1]]]
            if not np.array_equal(topk, expected):
                raise ValueError("predictor top-k disagrees with its own canonical scores (stable ties)")
            result["topk_composition_ids"] = topk
        for extra in ("crop_logits", "disease_logits"):
            if extra in output:
                result[extra] = _numpy(output[extra]).copy()
        return result


class WP2NotebookPredictor:
    """Compose the WP2 owner's own notebook objects into the shared predictor contract.

    ``transform``   = ``checkpoint_inference_transform(...)`` result (PIL -> normalized tensor).
    ``amp_context`` = the notebook's ``amp_context(device, precision)``.
    ``scorer``      = the notebook's ``composition_scores(crop, disease, candidate_spec)``.
    The forward mirrors ``d_collect_logits`` (channels_last, declared precision),
    with the independent CPU snapshot ``.detach().to("cpu", float32, copy=True)``
    recommended in docs/WP2_Review_2026-11-07.md. No scorer arithmetic is re-implemented here.
    """

    def __init__(self, model, transform, scorer, candidate_spec, *, amp_context, precision,
                 channels_last, device="cpu"):
        import torch
        if precision not in ("fp32", "fp16", "bf16"):
            raise ValueError("precision must be declared by the owner: fp32, fp16 or bf16")
        if not isinstance(channels_last, bool):
            raise TypeError("channels_last must be declared explicitly")
        ids = [int(v) for v in candidate_spec["ids"]]
        if not ids or ids != sorted(set(ids)):
            raise ValueError("candidate_spec ids must be unique and ascending")
        self.model, self.transform, self.scorer = model, transform, scorer
        self.spec, self.amp_context, self.precision = candidate_spec, amp_context, precision
        self.channels_last, self.device = channels_last, torch.device(device)
        self.ids = torch.tensor(ids, dtype=torch.int64)

    def predict(self, images, topk=5):
        import torch
        if not images or not 1 <= int(topk) <= len(self.ids):
            raise ValueError("Supply images and a valid top-k")
        if self.model.training:
            raise ValueError("model must be in evaluation mode")
        batch = torch.stack([self.transform(image.convert("RGB")) for image in images]).to(self.device)
        if self.channels_last and batch.ndim == 4:
            batch = batch.contiguous(memory_format=torch.channels_last)
        with torch.inference_mode(), self.amp_context(self.device, self.precision):
            output = self.model(batch)
        crop = output["crop_logits"].detach().to(device="cpu", dtype=torch.float32, copy=True)
        disease = output["disease_logits"].detach().to(device="cpu", dtype=torch.float32, copy=True)
        scores = self.scorer(crop, disease, self.spec)
        order = scores.argsort(dim=1, descending=True, stable=True)[:, :int(topk)]
        return {"candidate_ids": self.ids, "scores": scores, "topk_composition_ids": self.ids[order],
                "crop_logits": crop, "disease_logits": disease,
                "score_semantics": "sum_log_softmax_v1"}


__all__ = ["FINALIST_CONTRACTS", "PredictorAdapter", "WP2NotebookPredictor"]
