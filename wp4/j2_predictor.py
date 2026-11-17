"""Second Run final model J2 for the final-333 GUI (local-only; imported lazily by the model loader).

J2 = CLIP ViT-L/14 with its last four visual blocks (plus ln_post and the projection) fine-tuned on all 8,000
training images (Second Run v5, frozen checkpoint sha256 e9dbbdf3...), scored by the unchanged D1 synthetic
crop-disease composition downstream and calibrated with the fixed gamma = 8.75 on the 323 seen candidates.

The D1 downstream (feature standardisation, crop / disease / composition linear heads) is loaded from a private,
hash-pinned inference asset captured from ONE run of the frozen J2 scoring pipeline on the frozen training
features (scripts/j2_build_inference_asset.py); nothing is refitted at prediction time and the arithmetic below is
the frozen scorer's. Every identity is checked before use; any failure raises -- there is no fallback model.
"""
from __future__ import annotations

from pathlib import Path
import platform

import torch
import torch.nn.functional as F

from .clip_predictor import (CLIP_REVISION, CLIP_SHA256, NATIVE_PREPROCESSING, _sha256_stream, file_hash,
                             load_native_clip)

J2_CHECKPOINT_SHA256 = "e9dbbdf3fc95ba791e8e8129ef38bbd85352a0974c3091d6b7667bd633abb1c9"
J2_ASSET_SHA256 = "dd6495bb863c6f2e9b9d473f99ad06d7c1efa349cc72b3602b444eef3c6d2783"  # j2_d1_inference_asset_v1.pt
J2_CONFIG_SHA256 = "b42fa6e750c4ff3f23e0165edeadf32bc71e6290e630ad54ecc9669081670053"
J2_FEATURES_SHA256 = "eee9759f48ec5988e6c58e8ffc2c3d64ba6bd1c99ef6b2bafb07e96a38c8ba25"
FINAL_CANDIDATE_SHA256 = "ba17e94119e8b2b7a4b18603e25405d2f52bf4a93396973f80491800697eb62f"
FINAL_UNSEEN_IDS = (39, 84, 107, 124, 133, 136, 144, 166, 276, 303)
GAMMA = 8.75
ASSET_SCHEMA = "j2-d1-inference-asset-v1"
TRAINABLE_BLOCKS = (20, 21, 22, 23)  # ViT-L/14 has 24 visual blocks; J2 trains the last four
SCORE_SEMANTICS = "j2_d1_fused_log_probability_minus_gamma_8.75_on_seen_candidates"


def _check_asset(asset):
    """Structure, identities and shapes of the inference asset (fail closed)."""
    need = ("schema", "checkpoint_sha256", "features_sha256", "config_sha256", "candidate_sha256", "gamma",
            "candidate_ids", "cand_crop", "cand_dis", "seen_mask", "mu", "sd", "heads", "fuse", "normalise")
    if not isinstance(asset, dict) or any(k not in asset for k in need) or asset["schema"] != ASSET_SCHEMA:
        raise ValueError("Unsupported J2 inference asset")
    if (asset["checkpoint_sha256"] != J2_CHECKPOINT_SHA256 or asset["features_sha256"] != J2_FEATURES_SHA256
            or asset["config_sha256"] != J2_CONFIG_SHA256 or asset["candidate_sha256"] != FINAL_CANDIDATE_SHA256):
        raise ValueError("J2 inference asset was not built from the frozen J2 model and candidate contract")
    if asset["gamma"] != GAMMA or asset["fuse"] != 1.0 or asset["normalise"] is not True:
        raise ValueError("J2 calibration or fusion differs from the frozen configuration")
    ids = asset["candidate_ids"]
    if not isinstance(ids, torch.Tensor) or ids.tolist() != list(range(333)):
        raise ValueError("J2 candidates must be the 333 global ids in ascending order")
    seen = asset["seen_mask"]
    if seen.dtype != torch.bool or seen.tolist() != [i not in FINAL_UNSEEN_IDS for i in range(333)]:
        raise ValueError("J2 seen/unseen candidate mask differs from the declared unseen ids")
    heads = asset["heads"]
    if set(heads) != {"composition", "crop", "disease"}:
        raise ValueError("J2 asset must hold the composition, crop and disease heads")
    d = 768
    for name, t in [("mu", asset["mu"]), ("sd", asset["sd"])] + [(f"{h}.{k}", v) for h in heads for k, v in heads[h].items()]:
        if not isinstance(t, torch.Tensor) or t.dtype != torch.float32 or not torch.isfinite(t).all():
            raise ValueError(f"J2 asset tensor {name} must be finite float32")
    if asset["mu"].shape != (d,) or asset["sd"].shape != (d,) or (asset["sd"] <= 0).any():
        raise ValueError("J2 standardisation must be [768] with positive scale")
    nc, nd = heads["crop"]["weight"].shape[0], heads["disease"]["weight"].shape[0]
    shapes = {"composition": (333, d), "crop": (nc, d), "disease": (nd, d)}
    for h, (n, dd) in shapes.items():
        if heads[h]["weight"].shape != (n, dd) or heads[h]["bias"].shape != (n,):
            raise ValueError(f"J2 {h} head has an unexpected shape")
    cc, cd = asset["cand_crop"], asset["cand_dis"]
    if cc.shape != (333,) or cd.shape != (333,) or int(cc.min()) < 0 or int(cc.max()) >= nc or int(cd.min()) < 0 or int(cd.max()) >= nd:
        raise ValueError("J2 candidate crop / disease index maps are invalid")
    return asset


def load_asset(path, expected_sha256=None):
    """Read the pinned asset with weights_only (no code execution) and verify it before and after reading."""
    expected = expected_sha256 if expected_sha256 is not None else J2_ASSET_SHA256
    if expected is None:
        raise ValueError("J2 inference asset hash is not registered")
    with Path(path).open("rb") as stream:
        if _sha256_stream(stream) != expected:
            raise ValueError("Select the registered J2 inference asset (SHA-256 mismatch)")
        stream.seek(0)
        asset = torch.load(stream, map_location="cpu", weights_only=True)
        stream.seek(0)
        if _sha256_stream(stream) != expected:
            raise ValueError("J2 inference asset changed while loading")
    return _check_asset(asset)


def score_features(asset, features):
    """The frozen J2/D1 score for raw centre-view CLIP image features [B, 768] -> calibrated [B, 333] float64.

    standardise (L2, then z-score with the fitting statistics) -> crop / disease / composition log-softmax ->
    factorised score log p(crop) + log p(disease) fused with log p(composition), normalised by (1 + fuse) ->
    minus gamma on the 323 seen candidates."""
    x = features.detach().float().cpu()
    if x.ndim != 2 or x.shape[1] != 768 or not torch.isfinite(x).all():
        raise ValueError("Expected finite CLIP image features [B,768]")
    x = x / x.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    x = (x - asset["mu"]) / asset["sd"]
    h = asset["heads"]
    lp = {k: F.log_softmax(F.linear(x, h[k]["weight"], h[k]["bias"]), 1).double() for k in ("crop", "disease", "composition")}
    factorised = lp["crop"][:, asset["cand_crop"]] + lp["disease"][:, asset["cand_dis"]]
    fused = (factorised + asset["fuse"] * lp["composition"]) / (1.0 + asset["fuse"])
    calibrated = fused - asset["gamma"] * asset["seen_mask"].double()[None]
    if calibrated.shape != (len(x), 333) or not torch.isfinite(calibrated).all():
        raise ValueError("Non-finite or malformed J2 scores")
    return calibrated


def load_j2_backbone(clip_weights, checkpoint, device):
    """Pinned native CLIP + the J2 trainable tensors from the frozen checkpoint (verified keys and metadata)."""
    with Path(checkpoint).open("rb") as stream:
        if _sha256_stream(stream) != J2_CHECKPOINT_SHA256:
            raise ValueError("Select the frozen J2 checkpoint (SHA-256 mismatch)")
        stream.seek(0)
        payload = torch.load(stream, map_location="cpu", weights_only=True)
        stream.seek(0)
        if _sha256_stream(stream) != J2_CHECKPOINT_SHA256:
            raise ValueError("J2 checkpoint changed while loading")
    meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
    if (meta.get("row") != "J2_clipL_last4" or meta.get("seed") != 0 or meta.get("fold") != "FULL8000"
            or meta.get("config_sha256") != J2_CONFIG_SHA256):
        raise ValueError("J2 checkpoint metadata differs from the frozen full-8,000 J2 model")
    state = payload.get("backbone")
    want = sorted([f"visual.transformer.resblocks.{b}.{k}" for b in TRAINABLE_BLOCKS for k in (
        "attn.in_proj_weight", "attn.in_proj_bias", "attn.out_proj.weight", "attn.out_proj.bias", "ln_1.weight", "ln_1.bias",
        "mlp.c_fc.weight", "mlp.c_fc.bias", "mlp.c_proj.weight", "mlp.c_proj.bias", "ln_2.weight", "ln_2.bias")]
                  + ["visual.ln_post.weight", "visual.ln_post.bias", "visual.proj"])
    if not isinstance(state, dict) or sorted(state) != want or sorted(meta.get("backbone_keys", [])) != want:
        raise ValueError("J2 checkpoint must hold exactly the 51 trainable last-4-block tensors")
    if any(not isinstance(v, torch.Tensor) or not torch.isfinite(v).all() for v in state.values()):
        raise ValueError("J2 checkpoint tensors must be finite")
    model, preprocess, _ = load_native_clip(clip_weights, device)
    params = dict(model.named_parameters())
    with torch.no_grad():
        for k, v in state.items():
            if params[k].shape != v.shape:
                raise ValueError(f"J2 tensor {k} has an unexpected shape")
            params[k].copy_(v.to(params[k].dtype))  # CUDA: CLIP's fp16 subset, exactly as the frozen extraction
    model.eval().requires_grad_(False)
    return model, preprocess


class J2Predictor:
    def __init__(self, catalog, clip_weights, j2_checkpoint, j2_asset, device="cpu", asset_sha256=None):
        if not getattr(catalog, "is_final", False):
            raise ValueError("J2 scores the final 333-candidate protocol only")
        ids = [row["composition_id"] for row in catalog.protocol["candidates"]]
        if ids != list(range(333)) or catalog.protocol["candidate_sha256"] != FINAL_CANDIDATE_SHA256:
            raise ValueError("Loaded protocol is not the final 333-candidate contract J2 was evaluated on")
        self.device = torch.device(device)
        if self.device.type not in ("cpu", "cuda") or (self.device.type == "cuda" and not torch.cuda.is_available()):
            raise ValueError("Select an available CPU or CUDA device")
        self.asset = load_asset(j2_asset, asset_sha256)
        self.model, self.preprocess = load_j2_backbone(clip_weights, j2_checkpoint, self.device)
        self.candidate_ids = torch.arange(333, dtype=torch.int64)
        self.candidate_sha256 = catalog.protocol["candidate_sha256"]
        import torchvision
        self.provenance = {
            "variant": "j2", "model": "CLIP ViT-L/14, last 4 visual blocks fine-tuned (Second Run J2)",
            "training_scope": "all 8,000 DPD-Small training images, 5 epochs, seed 0",
            "checkpoint_sha256": J2_CHECKPOINT_SHA256, "inference_asset_sha256": asset_sha256 or J2_ASSET_SHA256,
            "clip_revision": CLIP_REVISION, "clip_sha256": CLIP_SHA256, "preprocessing": NATIVE_PREPROCESSING,
            "downstream": "D1 synthetic crop-disease composition scoring (fitted heads from the frozen pipeline)",
            "calibration": {"gamma": GAMMA, "applied_to": "323 seen candidates"},
            "catalog_candidate_sha256": self.candidate_sha256,
            "runtime": {"python": platform.python_version(), "torch": str(torch.__version__),
                        "torchvision": str(torchvision.__version__), "device": str(self.device)}}

    # J2 was trained and officially evaluated on the files' STORED pixel orientation (PIL open + RGB, no EXIF
    # rotation), in fp16 batches of 32 on CUDA. The GUI therefore hands J2 the stored pixels (load_input) while it
    # keeps displaying the EXIF-corrected picture; on CUDA a single query is zero-padded to the same batch shape.
    input_orientation = "stored_pixels"
    CUDA_BATCH = 32

    @staticmethod
    def load_input(path):
        from PIL import Image
        with Image.open(path) as image:
            image.load()
            return image.convert("RGB")

    @torch.inference_mode()
    def features(self, images):
        batch = torch.stack([self.preprocess(image.convert("RGB")) for image in images])
        if batch.ndim != 4 or batch.shape[1:] != (3, 224, 224) or not torch.isfinite(batch).all():
            raise ValueError("Invalid CLIP-preprocessed image batch")
        n = len(images)
        if self.device.type == "cuda" and n % self.CUDA_BATCH:
            batch = torch.cat([batch, torch.zeros((self.CUDA_BATCH - n % self.CUDA_BATCH, 3, 224, 224))])
        batch = batch.to(self.device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.device.type == "cuda"):
            feats = self.model.encode_image(batch).float()[:n]
        if feats.shape != (n, 768):
            raise ValueError("Expected CLIP image features [B,768]")
        return feats

    @torch.inference_mode()
    def predict(self, images, topk=5):
        if not images or type(topk) is not int or not 1 <= topk <= 333:
            raise ValueError("Supply images and a valid top-k")
        scores = score_features(self.asset, self.features(images))
        order = scores.argsort(dim=1, descending=True, stable=True)[:, :topk]
        return {"candidate_ids": self.candidate_ids, "scores": scores, "topk_composition_ids": self.candidate_ids[order],
                "candidate_sha256": self.candidate_sha256, "model_candidate_sha256": FINAL_CANDIDATE_SHA256,
                "score_semantics": SCORE_SEMANTICS, "model_provenance": self.provenance}


__all__ = ["J2Predictor", "load_asset", "score_features", "load_j2_backbone", "file_hash"]
