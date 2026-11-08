"""Final-333 views over the frozen selected predictors. No training, no new scorer.

Each view wraps an already constructed and verified development predictor and
changes only its candidate universe. Weights, preprocessing and score formulas are
unchanged, and the historical 323 columns stay bit-identical:

* WP1: ``IndependentPredictor.predict`` itself, on a shallow copy holding the 333
  Issue #8 rows. The scorer is the unchanged ``candidate_log_scores``.
* Frozen CLIP and M6: the ``CLIPPredictor.predict`` image path and cosine, evaluated
  as two column blocks (see ``Final333CLIP``):
  - the predictor's own 323 historical text rows, hash-gated against the bank that
    the GUI clean-parity run reproduced;
  - the 10 extra Issue #8 prompts, encoded once by the same frozen CLIP text
    encoder and shared by both variants.
  M6's adapter acts only on image features (no class-specific parameters), and its
  training used only the 258 fitting text vectors. The 10 new rows are therefore
  inference-only text rows, like the 65 development-unseen ones.
* WP2: nothing new. ``robustness_adapters.WP2NotebookPredictor`` already accepts any
  candidate spec; ``wp2_final_spec`` supplies the 333 spec for the owner's notebook
  ``score_components``.

``export_blind_packet`` validates each batch through the existing
``PredictorAdapter`` and emits a truth-free ``wp5-model-output-v1`` packet with full
scores, following an ``inference_order``.
"""
from __future__ import annotations

import copy

import numpy as np
import torch

from evaluation.final_manifest import INFERENCE_ORDER_SCHEMA
from evaluation.model_adapters import CLIP_ADAPTER, CLIP_FROZEN, SCORING
from evaluation.reporting import FINAL_NAMESPACE, FINAL_PROMPT_SHA256, fingerprint, verify_final_payloads
from evaluation.robustness_adapters import PredictorAdapter

# Bank reproduced by both real predictors in notebooks/gui_native_clean_parity.ipynb (DIRECT_BANK_SHA256).
HISTORICAL_DIRECT_BANK_SHA256 = "7f402c8dbebe821b1e1f2d799c9a0ce0e0d56a10a54aad563c6cd79d510df732"


def final_tables(candidate_payload, prompt_payload=None, *, synthetic=False):
    """Return (candidate rows, {global_id: prompt}); real payloads must match Issue #8."""
    if not synthetic:
        if prompt_payload is None:
            raise ValueError("Real final scoring verifies both Issue #8 payloads")
        verify_final_payloads(candidate_payload, prompt_payload)
    if candidate_payload.get("schema") != FINAL_NAMESPACE:
        raise ValueError("Unexpected final candidate payload")
    rows = [dict(r) for r in candidate_payload["rows"]]
    ids = [r["composition_id"] for r in rows]
    if not ids or ids != sorted(set(ids)):
        raise ValueError("Final candidates require unique ascending global IDs")
    prompts = ({} if prompt_payload is None
               else {r["global_composition_id"]: r["prompt"] for r in prompt_payload["rows"]})
    if prompt_payload is not None and sorted(prompts) != ids:
        raise ValueError("Prompt rows must cover exactly the final candidates")
    return rows, prompts


class Final333Independent:
    """WP1 pair over the 333 rows via the unchanged ``IndependentPredictor.predict``."""

    def __init__(self, predictor, candidate_payload, prompt_payload=None, *, synthetic=False):
        rows, _ = final_tables(candidate_payload, prompt_payload, synthetic=synthetic)
        by_id = {r["composition_id"]: r for r in rows}
        for row in predictor.protocol["candidates"]:
            if {key: by_id.get(row["composition_id"], {}).get(key) for key in row} != row:
                raise ValueError("Development candidates differ from their rows in the final table")
        self.base = predictor
        self.view = copy.copy(predictor)
        self.view.protocol = {**predictor.protocol, "candidates": rows,
                              "candidate_sha256": fingerprint({"schema": FINAL_NAMESPACE, "rows": rows})}

    def predict(self, images, topk=5):
        from independent_baseline import IndependentPredictor
        return IndependentPredictor.predict(self.view, images, topk)


def encode_prompts(predictor, prompts, tokenize):
    """Frozen CLIP text features for new prompts, exactly as CLIPPredictor encodes its bank."""
    from wp4.clip_predictor import unit_features
    tokens = tokenize(list(prompts), context_length=77, truncate=False)
    if tuple(tokens.shape) != (len(prompts), 77):
        raise ValueError("Native tokenizer rows changed")
    with torch.inference_mode():
        features = predictor.model.encode_text(tokens.to(predictor.device)).cpu()
    return unit_features(features)


class Final333CLIP:
    """Frozen CLIP or M6 over 333 prompts; ``CLIPPredictor.predict`` arithmetic in two column blocks.

    Image path, adapter and cosine formula are those of ``CLIPPredictor.predict``. The
    historical 323 columns are computed with the historical [B,323] product, so they stay
    bit-identical to the clean-parity predictor. A single [B,333] float64 product differs
    in the last ULP for about 2% of entries, because BLAS blocking depends on width. The
    10 extra columns use the same formula, and the blocks are placed by global ID.
    """

    def __init__(self, predictor, candidate_payload, prompt_payload, *, tokenize=None, extra_bank=None,
                 base_bank_sha256=HISTORICAL_DIRECT_BANK_SHA256, synthetic=False):
        from wp4.clip_predictor import tensor_hash
        rows, prompts = final_tables(candidate_payload, prompt_payload, synthetic=synthetic)
        self.ids = [r["composition_id"] for r in rows]
        base_ids = list(predictor.expected["ids"])
        base_prompts = {r["global_composition_id"]: r["prompt"] for r in predictor.expected["records"]}
        if not set(base_ids) <= set(self.ids) or any(prompts[i] != base_prompts.get(i) for i in base_ids):
            raise ValueError("Historical prompts differ from their rows in the final prompt table")
        bank = predictor.direct
        if (not isinstance(bank, torch.Tensor) or bank.dtype != torch.float64 or bank.ndim != 2
                or bank.shape[0] != len(base_ids) or tensor_hash(bank) != base_bank_sha256):
            raise ValueError("Historical direct text bank differs from the clean-parity bank")
        known = set(base_ids)
        self.extra_ids = [i for i in self.ids if i not in known]
        if extra_bank is None:
            if tokenize is None:
                raise ValueError("Supply the pinned CLIP tokenizer or a shared precomputed extra bank")
            extra_bank = encode_prompts(predictor, [prompts[i] for i in self.extra_ids], tokenize)
        if (not isinstance(extra_bank, torch.Tensor) or extra_bank.dtype != torch.float64
                or tuple(extra_bank.shape) != (len(self.extra_ids), bank.shape[1])
                or not torch.isfinite(extra_bank).all()):
            raise ValueError("Extra text bank must be finite float64 [extra, dim]")
        position = {gid: n for n, gid in enumerate(self.ids)}
        self.base_columns = [position[i] for i in base_ids]
        self.extra_columns = [position[i] for i in self.extra_ids]
        self.base, self.base_bank = predictor, bank.detach().cpu()
        self.extra_bank = extra_bank.detach().cpu().clone()
        self.bank = torch.empty((len(self.ids), bank.shape[1]), dtype=torch.float64)
        self.bank[self.base_columns], self.bank[self.extra_columns] = self.base_bank, self.extra_bank
        self.prompt_sha256 = fingerprint(dict(prompt_payload)) if synthetic else FINAL_PROMPT_SHA256
        self.provenance = {**predictor.provenance, "final333": {
            "historical_bank_sha256": base_bank_sha256, "extra_ids": self.extra_ids,
            "extra_bank_sha256": tensor_hash(self.extra_bank), "final_bank_sha256": tensor_hash(self.bank),
            "prompt_sha256": self.prompt_sha256,
            "scoring": "CLIPPredictor cosine; historical [B,323] block + extra [B,10] block, placed by global ID"}}

    @torch.inference_mode()
    def predict(self, images, topk=5):
        from wp4.clip_predictor import unit_features
        base = self.base
        if not images or type(topk) is not int or not 1 <= topk <= len(self.ids):
            raise ValueError("Supply images and a valid top-k")
        if base.model.training or any(p.requires_grad for p in base.model.parameters()):
            raise ValueError("CLIP must remain frozen and in evaluation mode")
        batch = torch.stack([base.preprocess(image.convert("RGB")) for image in images])
        if batch.ndim != 4 or batch.shape[1:] != (3, 224, 224) or not torch.isfinite(batch).all():
            raise ValueError("Invalid native-preprocessed image batch")
        features = base.model.encode_image(batch.to(base.device)).float()
        if features.shape != (len(images), 768):
            raise ValueError("Expected native CLIP image features [B,768]")
        if base.adapter is not None:
            features = base.adapter(features)
        unit = unit_features(features)
        scores = torch.empty((len(images), len(self.ids)), dtype=torch.float64)
        scores[:, self.base_columns] = unit @ unit_features(self.base_bank).T
        scores[:, self.extra_columns] = unit @ unit_features(self.extra_bank).T
        if not torch.isfinite(scores).all():
            raise ValueError("Non-finite cosine scores")
        ids = torch.tensor(self.ids, dtype=torch.int64)
        order = scores.argsort(dim=1, descending=True, stable=True)[:, :topk]
        return {"candidate_ids": ids, "scores": scores, "topk_composition_ids": ids[order],
                "candidate_hash_namespace": FINAL_NAMESPACE, "prompt_sha256": self.prompt_sha256,
                "score_semantics": "direct_float64_clip_cosine_similarity", "model_provenance": self.provenance}


class Restricted323:
    """A final-333 view cut back to the historical 323 columns, for the development rehearsal.

    It feeds the existing gates unchanged: the GUI clean-parity ``run_clean_gate``, and
    the historical full-score or Top-5 packets. It proves that the 333 view kept every
    historical column before any official image is scored.
    """

    def __init__(self, view, historical_ids):
        self.view, self.ids = view, [int(i) for i in historical_ids]
        everything = [int(i) for i in view.ids] if hasattr(view, "ids") else None
        self.columns = None if everything is None else [everything.index(i) for i in self.ids]

    def predict(self, images, topk=5):
        output = self.view.predict(images, topk=min(len(self.ids), max(5, topk)))
        returned = output["candidate_ids"].tolist()
        columns = self.columns or [returned.index(i) for i in self.ids]
        scores = output["scores"][:, columns]
        ids = torch.tensor(self.ids, dtype=torch.int64)
        order = scores.argsort(dim=1, descending=True, stable=True)[:, :topk]
        return {**output, "candidate_ids": ids, "scores": scores, "topk_composition_ids": ids[order]}


def wp2_final_spec(candidate_payload, prompt_payload=None, *, synthetic=False):
    """333 spec for ``WP2NotebookPredictor``; scorer = notebook ``score_components(c, d, ids, pairs)``."""
    rows, _ = final_tables(candidate_payload, prompt_payload, synthetic=synthetic)
    return {"ids": [r["composition_id"] for r in rows], "pairs": [[r["crop_id"], r["disease_id"]] for r in rows]}


def export_blind_packet(predictor, order, fold, load, *, model_kind, provenance, selection, batch_size):
    """Truth-free wp5-model-output-v1 packet for one fold, in inference-order sequence."""
    if not isinstance(order, dict) or order.get("schema") != INFERENCE_ORDER_SCHEMA or fold not in order["folds"]:
        raise ValueError("Export follows a WP5 final inference order")
    if provenance.score_semantics != SCORING[model_kind]:
        raise ValueError("Packet score semantics must be the selected model's WP5 canonical name")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("Declare the recorded inference batch size")
    adapter = PredictorAdapter(predictor, model_kind=model_kind, candidate_ids=order["candidate_ids"])
    names = list(order["folds"][fold])
    matrices = [adapter.predict([load(name) for name in names[start:start + batch_size]])["scores"]
                for start in range(0, len(names), batch_size)]
    scores = np.concatenate(matrices)
    if scores.dtype.name != provenance.score_precision:
        raise ValueError("Score dtype differs from the declared canonical precision")
    packet = {"schema": "wp5-model-output-v1", "model_kind": model_kind,
              **{key: getattr(provenance, key) for key in (
                  "model_id", "run_id", "checkpoint_id", "checkpoint_sha256", "preprocessing", "code_version",
                  "environment", "score_precision", "score_semantics")},
              "manifest_sha256": order["manifest_sha256"], "source_sha256": order["source_sha256"],
              "candidate_ids": list(order["candidate_ids"]),
              "candidate_hash_namespace": order["candidate_hash_namespace"],
              "candidate_sha256": order["candidate_sha256"],
              "split_hash_namespace": order["split_hash_namespace"], "split_sha256": order["split_sha256"],
              "selection": dict(selection), "fold": fold, "image_names": names, "scores": scores.tolist()}
    if model_kind in (CLIP_FROZEN, CLIP_ADAPTER):
        packet["prompt_sha256"] = provenance.prompt_sha256
    return packet


__all__ = ["HISTORICAL_DIRECT_BANK_SHA256", "final_tables", "Final333Independent", "encode_prompts",
           "Final333CLIP", "Restricted323", "wp2_final_spec", "export_blind_packet"]
