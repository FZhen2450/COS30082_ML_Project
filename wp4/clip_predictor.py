"""Local-only CLIP/M6 inference against the verified development catalog.

Imported lazily by the model loader, never by GUI startup. The checkpoint
interface is the published WP3B M4 adapter-only envelope. The private selected
M6 artifact still has to pass this interface and its separately pinned hash.
"""
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import unicodedata

import torch
from torch import nn

CLIP_REVISION = "d05afc436d78f1c48dc0dbf8e5980a9d471f35f6"
CLIP_SHA256 = "b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836"
M6_SHA256 = "8835a18b9eb933ae224bd17866f18bc56406e24d8612743f4e7c88848a8135cc"
SOURCE_SHA256 = "9f1b729ab08f120c9e8d3a5681aa1ccbdf1d9210f9aaa6143959363dcb106a7d"
CATALOG_SHA256 = "8217be059e3a4a144f89081ee43b7cd042bb5a62c370197bf0e3fca1c6d28bf3"
CANDIDATE_SHA256 = "4cb0a24a96cf647f32311817fa7b082a46d3cf00c20ae61f4287e83cc898dfaf"
SPLIT_SHA256 = "f0c7ba212449fd68f1d19c0f84cc91bf4cc1af29ae405128e5198bce89f25f85"
PROMPT_SHA256 = "753672c7964bcb20a1346145f21a72b40a19f55590a006b5783823f23c13f537"
PROMPT_POLICY = {
    "version": "label_only_single_template_v2",
    "normalization": "Unicode NFKC; underscores to spaces; collapse whitespace; lowercase; no aliases/symptoms",
    "healthy": "a photo of a healthy {crop} plant",
    "diseased": "a photo of {article} {crop} plant affected by {disease}",
    "article_rule": "an for normalized crop labels starting with a/e/i/o/u; otherwise a; diseased prompts only; verified for the current 55 crop labels",
}
NATIVE_PREPROCESSING = {
    "resize_shorter_edge": 224, "interpolation": "bicubic", "center_crop": 224,
    "convert": "RGB", "mean": [.48145466, .4578275, .40821073],
    "std": [.26862954, .26130258, .27577711], "augmentation": None,
}
ADAPTER_SPEC = {
    "id": "bounded_residual_visual_768_64_v1", "feature_dim": 768,
    "bottleneck": 64, "residual_scale": .1, "activation": "ReLU", "bias": True,
    "up_initialization": "zeros", "parameters": 99136, "residual_cap": .2,
}


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def tensor_hash(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def unit_features(features):
    """Preserve WP3's CPU float64 scaled-norm arithmetic, including near ties."""
    if not isinstance(features, torch.Tensor) or features.ndim != 2 or not features.is_floating_point():
        raise ValueError("Expected floating [N,D] features")
    value = features.detach().to(device="cpu", dtype=torch.float64)
    if not value.numel() or not torch.isfinite(value).all():
        raise ValueError("Features must be nonempty and finite")
    scale = value.abs().amax(dim=1, keepdim=True)
    if (scale == 0).any():
        raise ValueError("Features must be nonzero")
    scaled = value / scale
    return scaled / torch.linalg.vector_norm(scaled, dim=1, keepdim=True)


def normalized_label(value):
    return " ".join(unicodedata.normalize("NFKC", value).replace("_", " ").split()).lower()


def catalog_contract(catalog):
    """Prove both namespaces describe the same complete ordered development task."""
    from shared_protocol import FOLDS, digest
    p = catalog.protocol
    candidates = list(catalog.candidates.values())
    ids = list(catalog.candidates)
    if (p.get("historical_reconciliation") != "ordered_filenames_match_supplied_reference"
            or p.get("counts") != [4868, 1238, 1894] or len(ids) != 323
            or ids != sorted(set(ids)) or p.get("source_sha256") != SOURCE_SHA256
            or file_hash(catalog.source_csv) != SOURCE_SHA256):
        raise ValueError("CLIP requires the reconciled 323-candidate development protocol")
    if (p.get("candidate_sha256") != CATALOG_SHA256 or digest(candidates) != CATALOG_SHA256
            or candidates != p["candidates"]
            or p["split_sha256"] != digest({key: p[key] for key in FOLDS})):
        raise ValueError("Catalog candidate labels or split fingerprint changed")
    pairs = [[row["crop_id"], row["disease_id"]] for row in candidates]
    if json_hash([[key, *pair] for key, pair in zip(ids, pairs)]) != CANDIDATE_SHA256:
        raise ValueError("Historical candidate columns do not match the catalog")
    membership = {name: p[key] for name, key in zip(("fit", "val_seen", "val_unseen"), FOLDS)}
    if json_hash(membership) != SPLIT_SHA256:
        raise ValueError("Historical ordered development folds do not match")
    records = []
    names = {kind: {} for kind in ("crop", "disease")}
    for column, row in enumerate(candidates):
        crop, disease = (normalized_label(row[k + "_name"]) for k in ("crop", "disease"))
        prompt = (PROMPT_POLICY["healthy"].format(crop=crop) if disease == "healthy" else
                  PROMPT_POLICY["diseased"].format(article="an" if crop[0] in "aeiou" else "a",
                                                   crop=crop, disease=disease))
        records.append({"score_column": column, "global_composition_id": row["composition_id"],
                        "crop_id": row["crop_id"], "disease_id": row["disease_id"],
                        "crop": crop, "disease": disease, "prompt": prompt})
        for kind in names:
            names[kind][row[kind + "_id"]] = row[kind + "_name"]
    if json_hash({"policy": PROMPT_POLICY, "rows": records}) != PROMPT_SHA256:
        raise ValueError("Development prompt policy or labels changed")
    return {"ids": ids, "pairs": pairs, "records": records,
            "names": {kind: [[key, name] for key, name in sorted(values.items())]
                      for kind, values in names.items()},
            "bridge": {
                "catalog_candidate": {"namespace": "wp1-canonical-json-candidate-records-v1", "sha256": CATALOG_SHA256},
                "model_candidate": {"namespace": "wp3-json-global-id-crop-disease-tuples", "sha256": CANDIDATE_SHA256},
                "model_split_sha256": SPLIT_SHA256, "prompt_sha256": PROMPT_SHA256,
                "mapping_verification": "exact_ordered_ids_pairs_names_and_fold_filenames",
            }}


class ResidualVisualAdapter(nn.Module):
    """The published bounded 768→64→768 inference arithmetic; no training API."""
    def __init__(self):
        super().__init__()
        self.down = nn.Linear(768, 64)
        self.up = nn.Linear(64, 768)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, features):
        if features.ndim != 2 or features.shape[1] != 768:
            raise ValueError("Expected frozen CLIP image features [B,768]")
        base = features.detach().float()
        norms = torch.linalg.vector_norm(base, dim=1, keepdim=True)
        if not len(base) or not torch.isfinite(base).all() or not torch.isfinite(norms).all() or (norms <= 0).any():
            raise ValueError("Image features must be finite and nonzero")
        residual = self.up(torch.relu(self.down(base / norms)))
        if not torch.isfinite(residual).all():
            raise ValueError("Non-finite adapter residual")
        delta = .1 * residual
        size = torch.linalg.vector_norm(delta, dim=1, keepdim=True)
        if not torch.isfinite(size).all():
            raise ValueError("Non-finite adapter norm")
        factor = (.2 / size.clamp_min(torch.finfo(delta.dtype).eps)).clamp(max=1.)
        output = base + norms * (delta * factor)
        if not torch.isfinite(output).all() or not torch.isfinite(torch.linalg.vector_norm(output, dim=1)).all():
            raise ValueError("Non-finite adapted features")
        return output


def _load_m6_checkpoint(path, expected):
    """Read only the selected, hash-pinned adapter; never execute notebook/pickle code."""
    path = Path(path)
    with path.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != M6_SHA256:
            raise ValueError("Select the pinned M6 seed-45 epoch-1 checkpoint (SHA-256 mismatch)")
        stream.seek(0)
        payload = torch.load(stream, map_location="cpu", weights_only=True)
        stream.seek(0)
        if hashlib.file_digest(stream, "sha256").hexdigest() != M6_SHA256:
            raise ValueError("Adapter file changed while loading")
    if not isinstance(payload, dict) or payload.get("schema") != "wp3b-visual-adapter-m4-v1":
        raise ValueError("Unsupported adapter envelope; obtain the selected M6 checkpoint/schema handoff")
    contract = payload["contract"]
    checksum = json_hash(contract)
    if payload.get("contract_sha256") != checksum or payload.get("adapter_spec") != ADAPTER_SPEC:
        raise ValueError("Adapter contract hash or architecture mismatch")
    if (contract["adapter"] != ADAPTER_SPEC or contract["preprocessing"] != NATIVE_PREPROCESSING
            or contract["candidate_ids"] != expected["ids"] or contract["candidate_pairs"] != expected["pairs"]
            or contract["component_names"] != expected["names"]
            or contract["direct_prompt_records"] != expected["records"]
            or contract["direct_prompt_policy"] != PROMPT_POLICY
            or contract["direct_prompt_sha256"] != PROMPT_SHA256):
        raise ValueError("Adapter labels, candidate columns, prompts or preprocessing mismatch")
    if any(contract["split"].get(key) != value for key, value in {
            "source_sha256": SOURCE_SHA256, "candidate_sha256": CANDIDATE_SHA256,
            "split_sha256": SPLIT_SHA256}.items()):
        raise ValueError("Adapter source or ordered split mismatch")
    model = contract["model"]
    if (model.get("name") != "ViT-L/14" or model.get("revision") != CLIP_REVISION
            or model.get("sha256") != CLIP_SHA256 or model.get("embedding_dimension") != 768
            or model.get("input_resolution") != 224):
        raise ValueError("Adapter frozen-CLIP identity mismatch")
    hp, selection = contract["hyperparameters"], payload["selection"]
    if (type(selection.get("epoch")) is not int or selection["epoch"] != 1
            or type(hp.get("training_seed", hp.get("seed"))) is not int
            or hp.get("training_seed", hp.get("seed")) != 45
            or hp.get("residual_cap") != .2):
        raise ValueError("Expected selected M6 seed 45, epoch 1, cap 0.20")
    state = payload["text_state"]
    direct = state["direct"]
    if (state.get("contract_sha256") != checksum or state.get("candidate_ids") != expected["ids"]
            or not isinstance(direct, torch.Tensor) or direct.shape != (323, 768)
            or direct.dtype != torch.float64 or direct.requires_grad or not torch.isfinite(direct).all()
            or tensor_hash(direct) != state["hashes"].get("direct")):
        raise ValueError("Direct text bank or its mapping/hash changed")
    unit_features(direct)  # Check nonzero rows, without replacing the saved bank.
    adapter = ResidualVisualAdapter()
    state_dict = payload["adapter_state"]
    if not isinstance(state_dict, dict) or any(not isinstance(v, torch.Tensor) or v.dtype != torch.float32
                                              or not torch.isfinite(v).all() for v in state_dict.values()):
        raise ValueError("Adapter state must contain finite float32 tensors")
    adapter.load_state_dict(state_dict, strict=True)
    adapter.eval().requires_grad_(False)
    return adapter, direct.detach().cpu().clone(), {"adapter_sha256": M6_SHA256,
        "adapter_contract_sha256": checksum, "adapter_seed": 45, "adapter_epoch": 1,
        "residual_cap": .2, "direct_text_sha256": tensor_hash(direct)}


def load_m6_checkpoint(path, expected):
    try:
        return _load_m6_checkpoint(path, expected)
    except (KeyError, TypeError, AttributeError, RuntimeError) as exc:
        raise ValueError("Invalid M6 checkpoint interface, state shape or metadata") from exc


def load_native_clip(path, device):
    """Verified official JIT → pinned native build/preprocess, with no pickle fallback."""
    try:
        distribution = importlib.metadata.distribution("clip")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError("Install the pinned official CLIP package; see docs/WP4_GUI.md") from exc
    direct = json.loads(distribution.read_text("direct_url.json") or "{}")
    if distribution.version != "1.0" or direct.get("vcs_info", {}).get("commit_id") != CLIP_REVISION:
        raise RuntimeError("The installed CLIP package must match the pinned official revision")
    import clip
    from clip.model import build_model
    from clip.clip import _transform
    with Path(path).open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != CLIP_SHA256:
            raise ValueError("Select the pinned official ViT-L/14 weights (SHA-256 mismatch)")
        stream.seek(0)
        scripted = torch.jit.load(stream, map_location="cpu").eval()
        stream.seek(0)
        if hashlib.file_digest(stream, "sha256").hexdigest() != CLIP_SHA256:
            raise ValueError("CLIP file changed while loading")
    model = build_model(scripted.state_dict()).to(device)
    del scripted
    if device.type == "cpu":
        model.float()
    model.eval().requires_grad_(False)
    if int(model.visual.input_resolution) != 224 or int(model.context_length) != 77:
        raise ValueError("Unexpected native CLIP image/token contract")
    return model, _transform(224), clip.tokenize


class CLIPPredictor:
    def __init__(self, catalog, clip_weights, adapter_checkpoint=None, device="cpu"):
        self.expected = catalog_contract(catalog)
        self.device = torch.device(device)
        if self.device.type not in ("cpu", "cuda") or (self.device.type == "cuda" and not torch.cuda.is_available()):
            raise ValueError("Select an available CPU or CUDA device")
        self.adapter = None
        self.provenance = {"model": "CLIP ViT-L/14", "clip_revision": CLIP_REVISION,
                           "clip_sha256": CLIP_SHA256, "preprocessing": NATIVE_PREPROCESSING,
                           "protocol_bridge": self.expected["bridge"]}
        if adapter_checkpoint is not None:
            self.adapter, self.direct, provenance = load_m6_checkpoint(adapter_checkpoint, self.expected)
            self.adapter.to(self.device)
            self.provenance.update(provenance)
        self.model, self.preprocess, tokenize = load_native_clip(clip_weights, self.device)
        import torchvision
        self.provenance["runtime"] = {"python": platform.python_version(), "torch": str(torch.__version__),
                                      "torchvision": str(torchvision.__version__), "device": str(self.device)}
        if self.adapter is None:
            prompts = [row["prompt"] for row in self.expected["records"]]
            tokens = tokenize(prompts, context_length=77, truncate=False)
            if tokens.shape != (323, 77):
                raise ValueError("Native tokenizer candidate rows changed")
            with torch.inference_mode():
                parts = [self.model.encode_text(tokens[i:i + 32].to(self.device)).cpu()
                         for i in range(0, len(tokens), 32)]
            self.direct = unit_features(torch.cat(parts))
            if self.direct.shape != (323, 768):
                raise ValueError("Unexpected CLIP text feature dimensions")
            self.provenance["direct_text_sha256"] = tensor_hash(self.direct)
        self.provenance["variant"] = "m6" if self.adapter is not None else "frozen_clip"

    @torch.inference_mode()
    def predict(self, images, topk=5):
        if not images or type(topk) is not int or not 1 <= topk <= len(self.expected["ids"]):
            raise ValueError("Supply images and a valid top-k")
        if self.model.training or any(p.requires_grad for p in self.model.parameters()):
            raise ValueError("CLIP must remain frozen and in evaluation mode")
        batch = torch.stack([self.preprocess(image.convert("RGB")) for image in images])
        if batch.ndim != 4 or batch.shape[1:] != (3, 224, 224) or not torch.isfinite(batch).all():
            raise ValueError("Invalid native-preprocessed image batch")
        features = self.model.encode_image(batch.to(self.device)).float()
        if features.shape != (len(images), 768):
            raise ValueError("Expected native CLIP image features [B,768]")
        if self.adapter is not None:
            features = self.adapter(features)
        scores = unit_features(features) @ unit_features(self.direct).T
        if not torch.isfinite(scores).all():
            raise ValueError("Non-finite cosine scores")
        ids = torch.tensor(self.expected["ids"], dtype=torch.int64)
        order = scores.argsort(dim=1, descending=True, stable=True)[:, :topk]
        return {"candidate_ids": ids, "scores": scores, "topk_composition_ids": ids[order],
                "candidate_sha256": CATALOG_SHA256, "model_candidate_sha256": CANDIDATE_SHA256,
                "score_semantics": "direct_float64_clip_cosine_similarity",
                "model_provenance": self.provenance}
