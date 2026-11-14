"""Stage J building blocks (v4): CLIP-L/14 partial fine-tuning configurations and the shared extraction path.

``extract_with_model`` is THE extraction used by J0 and every J row: model cast to fp16 (as clip.load does on
CUDA), encode_image under fp16 autocast, batch 32, deterministic order, centre + h-flip views, stored fp16.
With an untouched model it must reproduce the v3 clip_l14 features (J0).
"""
from __future__ import annotations

import hashlib

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader

from research_r2.features import _Images, load_backbone


class LoRA(nn.Module):
    """y = W x + (alpha/r) B A x with W frozen; A ~ N(0, 1/r^2) scale, B = 0 (output unchanged at init)."""

    def __init__(self, base: nn.Linear, r=8, alpha=16, seed=0):
        super().__init__()
        self.base = base
        for p in base.parameters():
            p.requires_grad_(False)
        g = torch.Generator(device="cpu").manual_seed(seed)
        self.a = nn.Parameter((torch.randn(r, base.in_features, generator=g) / r).to(base.weight.device, base.weight.dtype))
        self.b = nn.Parameter(torch.zeros(base.out_features, r, device=base.weight.device, dtype=base.weight.dtype))
        self.s = alpha / r

    def forward(self, x):
        return self.base(x) + self.s * (x @ self.a.T.to(x.dtype)) @ self.b.T.to(x.dtype)


class Ckpt(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, x):
        return checkpoint(self.block, x, use_reentrant=False) if self.training and torch.is_grad_enabled() else self.block(x)


def build(trainable: str, device="cuda", lora_seed=0):
    """Fresh CLIP-L/14 from the local weights, fp32 for training, with the declared trainable set."""
    model, tf, _, wfile, _ = load_backbone("clip_l14", device)
    model.float()
    for p in model.parameters():
        p.requires_grad_(False)
    v = model.visual
    blocks = v.transformer.resblocks
    if trainable.startswith("last"):
        k = int(trainable[4:])
        for b in list(blocks)[-k:]:
            for p in b.parameters():
                p.requires_grad_(True)
        for p in list(v.ln_post.parameters()) + [v.proj]:
            p.requires_grad_(True)
    elif trainable == "lora8_mlp":
        for i, b in enumerate(blocks):
            b.mlp.c_fc = LoRA(b.mlp.c_fc, 8, 16, seed=lora_seed * 1000 + 2 * i)
            b.mlp.c_proj = LoRA(b.mlp.c_proj, 8, 16, seed=lora_seed * 1000 + 2 * i + 1)
            blocks[i] = Ckpt(b)
    elif trainable != "none":
        raise KeyError(trainable)
    return model, tf, wfile


def param_digest(params) -> str:
    h = hashlib.sha256()
    for p in params:
        h.update(p.detach().float().cpu().numpy().tobytes())
    return h.hexdigest()


@torch.no_grad()
def extract_with_model(model, tf, paths, batch_size=32, workers=8):
    """Identical to research_r2.features.extract_images, but for a given (possibly fine-tuned) model.
    Precision: CLIP's own convert_weights (the subset clip.load halves on CUDA: conv/linear/attention/proj), NOT
    model.half(), so LayerNorms and embeddings stay fp32 exactly as in the v3 extraction."""
    from clip.model import convert_weights
    model.eval()
    convert_weights(model)
    loader = DataLoader(_Images(paths, tf), batch_size=batch_size, num_workers=workers, shuffle=False)
    out = []
    for x in loader:
        x = x.to("cuda", non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16):
            f0 = model.encode_image(x).float()
            f1 = model.encode_image(torch.flip(x, dims=[3])).float()
        out.append(torch.stack([f0, f1], 1).cpu())
    return torch.cat(out).half()
