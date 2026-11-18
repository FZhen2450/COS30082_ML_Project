"""R2 v4 feasibility: GPU memory and throughput probes for partial fine-tuning (resource numbers only).

Nothing here is a performance result. Real DPD-Small *training* images (rows of the training CSV only)
feed a few optimisation steps per configuration with random-initialised crop/disease heads; no weights,
features or scores are written. Output: one JSON with peak memory, step time and images/second.

  python scripts/r2v4_probe_gpu.py --images ../../../DPD_Small_dataset --out OUT.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import traceback
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research_r2.data import load_training_source  # noqa: E402
from research_r2.features import _Images, load_backbone  # noqa: E402

N_CROP, N_DIS = 55, 175


class LoRA(nn.Module):
    """y = W x + (alpha/r) B A x, W frozen; A ~ N(0, 1/r), B = 0 (standard LoRA init)."""

    def __init__(self, base: nn.Linear, r=8, alpha=16):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.a = nn.Parameter(torch.randn(r, base.in_features, device=base.weight.device) / r)
        self.b = nn.Parameter(torch.zeros(base.out_features, r, device=base.weight.device))
        self.s = alpha / r

    def forward(self, x):
        return self.base(x) + self.s * (x @ self.a.T.to(x.dtype)) @ self.b.T.to(x.dtype)


def clip_parts(model):
    v = model.visual

    def stem(x):
        x = v.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        cls = v.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
        x = torch.cat([cls, x], 1) + v.positional_embedding.to(x.dtype)
        return v.ln_pre(x).permute(1, 0, 2)  # LND

    def head(x):
        return v.ln_post(x.permute(1, 0, 2)[:, 0, :]) @ v.proj

    return v, stem, list(v.transformer.resblocks), head


def probe_clip(model, mode, k, x, steps, ckpt=False, lora_r=0):
    v, stem, blocks, head = clip_parts(model)
    model.float()
    for p in model.parameters():
        p.requires_grad_(False)
    n_blocks = len(blocks)
    trainable_blocks = blocks[n_blocks - k:] if mode == "lastk" else blocks
    frozen_blocks = blocks[:n_blocks - k] if mode == "lastk" else []
    if mode == "lastk":
        for b in trainable_blocks:
            for p in b.parameters():
                p.requires_grad_(True)
        for p in list(v.ln_post.parameters()) + [v.proj]:
            p.requires_grad_(True)
    elif mode == "lora":
        for b in blocks:
            b.mlp.c_fc = LoRA(b.mlp.c_fc, lora_r)
            b.mlp.c_proj = LoRA(b.mlp.c_proj, lora_r)
            # attn.out_proj is read as .weight inside nn.MultiheadAttention, so it cannot be wrapped here
    elif mode == "full":
        for p in v.parameters():
            p.requires_grad_(True)
    dim = v.proj.shape[1]
    heads = nn.Linear(dim, N_CROP + N_DIS).cuda()
    params = [p for p in model.parameters() if p.requires_grad] + list(heads.parameters())
    opt = torch.optim.AdamW(params, lr=1e-5, weight_decay=0.05)
    scaler = torch.cuda.amp.GradScaler()
    yc = torch.randint(0, N_CROP, (len(x),), device="cuda")
    yd = torch.randint(0, N_DIS, (len(x),), device="cuda")

    def step():
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            if mode == "lastk":
                with torch.no_grad():
                    h = stem(x)
                    for b in frozen_blocks:
                        h = b(h)
                h = h.detach()
            else:
                h = stem(x)
            for b in trainable_blocks:
                h = checkpoint(b, h, use_reentrant=False) if ckpt else b(h)
            z = heads(head(h).float())
            loss = F.cross_entropy(z[:, :N_CROP], yc) + F.cross_entropy(z[:, N_CROP:], yd)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        return loss

    return _time(step, steps), sum(p.numel() for p in params)


def probe_convnext(model, mode, x, steps):
    model.float()
    for p in model.parameters():
        p.requires_grad_(False)
    if mode == "stage4":
        for p in list(model.features[7].parameters()) + list(model.classifier.parameters()):
            p.requires_grad_(True)
    else:
        for p in model.parameters():
            p.requires_grad_(True)
    heads = nn.Linear(1024, N_CROP + N_DIS).cuda()
    params = [p for p in model.parameters() if p.requires_grad] + list(heads.parameters())
    opt = torch.optim.AdamW(params, lr=1e-5, weight_decay=0.05)
    scaler = torch.cuda.amp.GradScaler()
    yc = torch.randint(0, N_CROP, (len(x),), device="cuda")
    yd = torch.randint(0, N_DIS, (len(x),), device="cuda")

    def step():
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            if mode == "stage4":
                with torch.no_grad():
                    h = model.features[:7](x)
                h = model.features[7](h.detach())
                h = model.classifier(model.avgpool(h))
            else:
                h = model(x)
            z = heads(h.float())
            loss = F.cross_entropy(z[:, :N_CROP], yc) + F.cross_entropy(z[:, N_CROP:], yd)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

    return _time(step, steps), sum(p.numel() for p in params)


def _time(fn, steps, warmup=3):
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(steps):
        t = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t)
    return statistics.median(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="DPD_Small_train.csv")
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--steps", type=int, default=8)
    a = ap.parse_args()
    frame = load_training_source(a.csv)  # training CSV only; refuses official-test metadata
    paths = [Path(a.images) / f for f in frame.file.iloc[:512]]
    rep = {"schema": "r2v4-gpu-probe-v1", "device": torch.cuda.get_device_name(0), "torch": torch.__version__,
           "batch": a.batch, "steps": a.steps, "free_total_bytes_at_start": list(torch.cuda.mem_get_info()),
           "note": "resource probe only: random heads, a few steps, nothing saved", "probes": []}

    def record(name, fn):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        entry = {"name": name}
        try:
            out = fn()
            entry.update(out)
            entry["peak_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
        except torch.cuda.OutOfMemoryError as e:
            entry["error"] = "OOM: " + str(e).splitlines()[0][:200]
        except Exception as e:  # recorded, never hidden
            entry["error"] = f"{type(e).__name__}: {e}"[:300]
            entry["trace"] = traceback.format_exc()[-800:]
        rep["probes"].append(entry)
        print(json.dumps(entry), flush=True)
        torch.cuda.empty_cache()

    # 1. JPEG decode + transform throughput (CPU side) for the CLIP-L/14 transform
    _, tf, _, _, _ = load_backbone("clip_b32", "cpu")  # same 224 bicubic transform family; small model
    for workers in (0, 4, 8):
        def dl(workers=workers):
            t = time.perf_counter()
            n = 0
            for xb in DataLoader(_Images(paths, tf), batch_size=64, num_workers=workers):
                n += len(xb)
            return {"images": n, "images_per_s": round(n / (time.perf_counter() - t), 1)}
        record(f"decode_workers{workers}", dl)

    ds = _Images(paths[:a.batch], tf)
    xb = torch.stack([ds[i] for i in range(a.batch)]).cuda()

    # 2. CLIP-L/14 frozen inference
    def infer():
        model, _, enc, _, _ = load_backbone("clip_l14", "cuda")
        x64 = torch.cat([xb, xb])
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            s = _time(lambda: enc(x64), a.steps)
        del model
        return {"step_s": round(s, 4), "images_per_s": round(64 / s, 1)}
    record("clip_l14_frozen_infer_bs64", infer)

    # 3. CLIP-L/14 partial fine-tuning variants
    for name, mode, k, ckpt, r in (("clip_l14_last2", "lastk", 2, False, 0), ("clip_l14_last4", "lastk", 4, False, 0),
                                   ("clip_l14_last8", "lastk", 8, False, 0), ("clip_l14_last8_ckpt", "lastk", 8, True, 0),
                                   ("clip_l14_lora8_mlp_all_ckpt", "lora", 24, True, 8),("clip_l14_full_ckpt", "full", 24, True, 0)):
        def ft(mode=mode, k=k, ckpt=ckpt, r=r):
            model, _, _, _, _ = load_backbone("clip_l14", "cuda")
            s, n = probe_clip(model, mode, k, xb, a.steps, ckpt=ckpt, lora_r=r)
            del model
            return {"step_s": round(s, 4), "images_per_s": round(a.batch / s, 1), "trainable_params": n}
        record(name, ft)

    # 4. ConvNeXt-B
    for name, mode in (("convnext_b_stage4", "stage4"), ("convnext_b_full", "full")):
        def cn(mode=mode):
            model, _, _, _, _ = load_backbone("tv_convnext_base", "cuda")
            s, n = probe_convnext(model, mode, xb, a.steps)
            del model
            return {"step_s": round(s, 4), "images_per_s": round(a.batch / s, 1), "trainable_params": n}
        record(name, cn)

    # 5. token-cache size for CLIP-L/14 (tokens after block 24-k, per image per view, fp16)
    rep["token_cache_bytes_per_image_view_fp16"] = 257 * 1024 * 2
    rep["token_cache_gib_8000x2views"] = round(257 * 1024 * 2 * 8000 * 2 / 2**30, 2)
    Path(a.out).write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
