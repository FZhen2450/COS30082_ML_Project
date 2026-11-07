"""Lightweight model choices; importing this module never imports CLIP/Torch."""
from pathlib import Path

MODEL_SPECS = {
    "wp1": {"label": "WP1 independent crop / disease CNNs", "fields": (
        ("crop", "Crop checkpoint"), ("disease", "Disease checkpoint"))},
    "frozen_clip": {"label": "Frozen CLIP ViT-L/14", "fields": (("clip_weights", "CLIP ViT-L/14 weights"),)},
    "m6": {"label": "M6 capped adapter (seed 45, epoch 1)", "fields": (
        ("clip_weights", "CLIP ViT-L/14 weights"), ("adapter_checkpoint", "Selected M6 checkpoint"))},
}


def validate_model_inputs(kind, paths, device):
    if kind not in MODEL_SPECS or device not in ("cpu", "cuda"):
        raise ValueError("Select a supported model and device")
    for key, label in MODEL_SPECS[kind]["fields"]:
        if not paths.get(key) or not Path(paths[key]).is_file():
            raise ValueError(f"Select an existing {label.lower()} file")


def load_model(catalog, kind, paths, device="cpu"):
    validate_model_inputs(kind, paths, device)
    if kind == "wp1":
        from .service import load_independent
        return load_independent(catalog, paths["crop"], paths["disease"], device)
    from .clip_predictor import CLIPPredictor
    return CLIPPredictor(catalog, paths["clip_weights"],
                         paths.get("adapter_checkpoint") if kind == "m6" else None, device)
