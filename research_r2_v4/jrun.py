"""Stage J six-fold workflow (v4, post-freeze exploratory). Declared in research_r2_v4/matrices/stage_j_v4.json,
amendment_1.json ("J0", "J_pilot") and amendment_2.json ("execution", "source_co_endpoint").

One (row, seed, fold) = train on fitting rows only -> save checkpoint -> reload into a fresh model and verify ->
extract fold features FROM THE RELOADED MODEL -> cache with metadata -> unchanged D1 downstream -> validated packet
-> fold receipt. Every check raises (fail closed); a fold receipt is written only after all checks pass.

Backbones and image sources are injected so a synthetic backend can dry-run the whole workflow without images.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from research_r2.data import sha256_file, sha256_json
from research_r2.methods import RECIPES, within_concept_source_weight
from research_r2.packets import fold_contract, validate_packet
from research_r2.runner import make_fold

D1_CFG = {"fuse": 1.0, "normalise": True}


class JCheckError(RuntimeError):
    """A fail-closed training / provenance check failed."""


# ----------------------------------------------------------------------------- configuration
@dataclass
class JConfig:
    row: str
    backbone: str = "clip_l14"          # clip_l14 | convnext | synthetic
    trainable: str = "last2"            # none | last2 | last4 | lora8_mlp | stage4 | synthetic
    augment: str = "flip"               # flip | geometric | photometric
    source_balanced: bool = False
    extra: str | None = None            # None | supcon | adversary
    epochs: int = 5
    batch: int = 32
    lr_backbone: float = 1e-5
    lr_head: float = 1e-3
    weight_decay: float = 0.05
    warmup_epochs: int = 1
    max_steps: int | None = None        # metric-free smoke runs only
    concat_frozen: str | None = None    # J8: v3 frozen feature backbone to concatenate (clip_l14)
    workers: int = 8

    def sha(self):
        return sha256_json({k: v for k, v in asdict(self).items() if k != "workers"})


def j_rows(parent: str | None = None) -> dict:
    p = {"J1_clipL_last2": "last2", "J2_clipL_last4": "last4", "J3_clipL_lora8_mlp": "lora8_mlp"}
    rows = {"J0_pipeline_zero_epochs": JConfig("J0_pipeline_zero_epochs", trainable="none", epochs=0)}
    rows.update({r: JConfig(r, trainable=t) for r, t in p.items()})
    rows["J8_convnext_stage4_concat"] = JConfig("J8_convnext_stage4_concat", backbone="convnext", trainable="stage4",
                                                concat_frozen="clip_l14")
    rows["J8_zero_epochs"] = JConfig("J8_zero_epochs", backbone="convnext", trainable="none", epochs=0, concat_frozen="clip_l14")
    if parent is not None:
        t = p[parent]
        rows.update({"J4a_best_geometric_aug": JConfig("J4a_best_geometric_aug", trainable=t, augment="geometric"),
                     "J4b_best_photometric_aug": JConfig("J4b_best_photometric_aug", trainable=t, augment="photometric"),
                     "J5_best_source_balanced": JConfig("J5_best_source_balanced", trainable=t, source_balanced=True),
                     "J6_best_crosscrop_supcon": JConfig("J6_best_crosscrop_supcon", trainable=t, extra="supcon"),
                     "J7_best_crop_adversary": JConfig("J7_best_crop_adversary", trainable=t, extra="adversary")})
    return rows


# ----------------------------------------------------------------------------- guarded image access
class ImageAccess:
    """Opens only training-CSV images of rows allowed in the current phase; checks happen BEFORE any open."""

    def __init__(self, frame, image_root, loader=None):
        self.frame, self.root = frame, Path(image_root) if image_root else None
        self.train_files = set(frame.file.tolist())
        self.allowed: set[int] = set()
        self.phase = None
        self.opened = 0
        self._loader = loader  # synthetic: callable(row) -> PIL/tensor

    def allow(self, phase, rows):
        self.phase, self.allowed = phase, set(int(r) for r in rows)

    def check(self, row):
        row = int(row)
        if row not in self.allowed:
            raise JCheckError(f"image access refused: row {row} is not allowed in phase {self.phase!r}")
        name = self.frame.file.iloc[row]
        if name not in self.train_files or "test" in name.lower():
            raise JCheckError(f"image access refused: {name!r} is not a training-CSV image")
        return name

    def get(self, row):
        name = self.check(row)
        self.opened += 1
        if self._loader is not None:
            return self._loader(int(row))
        from PIL import Image
        with Image.open(self.root / name) as im:
            return im.convert("RGB")


class RowsDataset(Dataset):
    def __init__(self, access, rows, transform, labels=None):
        self.access, self.rows, self.tf, self.labels = access, list(rows), transform, labels

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        x = self.tf(self.access.get(self.rows[i]))
        if self.labels is None:
            return x
        return (x,) + tuple(l[i] for l in self.labels)


# ----------------------------------------------------------------------------- backends
class LayoutChunks(Dataset):
    """v3 batch layout: consecutive chunks of ``batch`` rows over ALL training rows (as research_r2.features.extract_images).
    Only allowed rows are loaded; every other slot is a zero tensor, so batch shapes (and hence fp16 kernel choice
    and rounding) match the v3 extraction while no non-fold image is ever opened."""

    def __init__(self, access, rows, transform, n_total, batch=32, shape=(3, 224, 224)):
        self.access, self.tf, self.batch, self.shape = access, transform, batch, shape
        self.allowed = set(int(r) for r in rows)
        self.chunks = [c for c in range(0, n_total, batch) if any(r in self.allowed for r in range(c, min(c + batch, n_total)))]
        self.n_total = n_total

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, i):
        c = self.chunks[i]
        rows = list(range(c, min(c + self.batch, self.n_total)))
        x = torch.zeros((len(rows),) + self.shape)
        keep = torch.zeros(len(rows), dtype=torch.bool)
        for j, r in enumerate(rows):
            if r in self.allowed:
                x[j] = self.tf(self.access.get(r)); keep[j] = True
        return x, keep, torch.tensor(rows)


def extract_layout(forward, tf, access, rows, n_total, workers):
    """Run ``forward`` (centre and h-flip) in the v3 batch layout; return features for ``rows`` in ascending order."""
    dl = DataLoader(LayoutChunks(access, rows, tf, n_total), batch_size=None, num_workers=workers, shuffle=False)
    got = {}
    with torch.no_grad():
        for x, keep, rr in dl:
            x = x.to("cuda", non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                f = torch.stack([forward(x).float(), forward(torch.flip(x, dims=[3])).float()], 1).cpu()
            for j in torch.nonzero(keep).flatten().tolist():
                got[int(rr[j])] = f[j]
    want = sorted(int(r) for r in rows)
    if sorted(got) != want:
        raise JCheckError("layout extraction did not return exactly the requested rows")
    return torch.stack([got[r] for r in want]).half()


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


class Heads(nn.Module):
    def __init__(self, d, nc, nd, extra=None):
        super().__init__()
        self.extra = extra
        self.crop = nn.Linear(d, nc)
        if extra == "adversary":
            self.proj_d = nn.Linear(d, d)
            self.dis = nn.Linear(d, nd)
            self.adv = nn.Linear(d, nc)
        else:
            self.dis = nn.Linear(d, nd)

    def forward(self, z):
        if self.extra == "adversary":
            pd_ = self.proj_d(z)
            return self.crop(z), self.dis(pd_), self.adv(GradReverse.apply(pd_, 0.1))
        return self.crop(z), self.dis(z), None


class ClipBackend:
    name, dim = "clip_l14", 768

    def build(self, cfg, device="cuda", lora_seed=0):
        from research_r2_v4.finetune import build
        return build(cfg.trainable, device, lora_seed=lora_seed)

    @staticmethod
    def embed(model, x):
        return model.encode_image(x).float()

    @staticmethod
    def extract(model, tf, access, rows, workers):
        from clip.model import convert_weights
        model.eval()
        convert_weights(model)  # the subset clip.load halves on CUDA (v3 precision)
        return extract_layout(model.encode_image, tf, access, rows, len(access.frame), workers)


class ConvNextBackend:
    name, dim = "convnext", 1024

    def build(self, cfg, device="cuda", lora_seed=0):
        from research_r2.features import load_backbone
        model, tf, _, wfile, _ = load_backbone("tv_convnext_base", device)
        model.float()
        for p in model.parameters():
            p.requires_grad_(False)
        if cfg.trainable == "stage4":
            for p in list(model.features[7].parameters()) + list(model.classifier.parameters()):
                p.requires_grad_(True)
        elif cfg.trainable != "none":
            raise KeyError(cfg.trainable)
        return model, tf, wfile

    @staticmethod
    def embed(model, x):
        return model(x).float()

    @staticmethod
    def extract(model, tf, access, rows, workers):
        model.eval()
        return extract_layout(model, tf, access, rows, len(access.frame), workers)


class SyntheticBackend:
    """Dry-run backend: tiny MLP over 3x8x8 synthetic tensors (no image files). Layer 1 frozen, layer 2 trainable."""
    name, dim = "synthetic", 16

    def build(self, cfg, device="cpu", lora_seed=0):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(12345)  # fixed "pretrained" weights, independent of the training seed
            m = nn.Sequential(nn.Flatten(), nn.Linear(192, 32), nn.GELU(), nn.Linear(32, 16)).to(device)
        for p in m.parameters():
            p.requires_grad_(False)
        if cfg.trainable == "synthetic":
            for p in m[3].parameters():
                p.requires_grad_(True)
        elif cfg.trainable != "none":
            raise KeyError(cfg.trainable)
        return m, (lambda t: t), "synthetic-weights"

    @staticmethod
    def embed(model, x):
        return model(x.float()).float()

    @staticmethod
    def extract(model, tf, access, rows, workers):
        model.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(rows), 256):
                x = torch.stack([tf(access.get(r)) for r in rows[i:i + 256]])
                out.append(torch.stack([model(x), model(torch.flip(x, dims=[2]))], 1))
        return torch.cat(out).half()


def synthetic_loader(frame, seed=0):
    """Deterministic 3x8x8 'images': crop and disease patterns plus row-seeded noise (no file access)."""
    g = torch.Generator().manual_seed(seed)
    ec = torch.randn(int(frame.crop_id.max()) + 1, 192, generator=g)
    ed = torch.randn(int(frame.disease_id.max()) + 1, 192, generator=g)
    crop, dis = frame.crop_id.to_numpy(), frame.disease_id.to_numpy()

    def load(row):
        n = torch.randn(192, generator=torch.Generator().manual_seed(10_000 + row))
        return (ec[crop[row]] + 0.7 * ed[dis[row]] + 0.5 * n).reshape(3, 8, 8)
    return load


# ----------------------------------------------------------------------------- augmentation
def train_transform(cfg, base_tf):
    """base_tf is the deterministic evaluation transform; the declared augmentation wraps it."""
    import torchvision.transforms as T
    if cfg.backbone == "synthetic":
        return base_tf
    norm = [t for t in base_tf.transforms if isinstance(t, (T.ToTensor, T.Normalize))]
    resize_crop = [t for t in base_tf.transforms if isinstance(t, (T.Resize, T.CenterCrop))]
    if cfg.augment == "flip":
        return T.Compose(resize_crop + [T.RandomHorizontalFlip()] + norm)
    if cfg.augment == "geometric":
        return T.Compose([T.RandomResizedCrop(224, scale=(0.6, 1.0), interpolation=T.InterpolationMode.BICUBIC),
                          T.RandomHorizontalFlip()] + norm)
    if cfg.augment == "photometric":
        return T.Compose(resize_crop + [T.ColorJitter(0.3, 0.3, 0.3, 0.05), T.RandomGrayscale(0.1),
                                        T.RandomApply([T.GaussianBlur(9, (0.1, 2.0))], p=0.2), T.RandomHorizontalFlip()] + norm)
    raise KeyError(cfg.augment)


# ----------------------------------------------------------------------------- checks
def param_digest(params) -> str:
    h = hashlib.sha256()
    for p in params:
        h.update(p.detach().float().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class GradMonitor:
    """Fail-closed gradient bookkeeping over checked steps (amendment_2 execution.fail_closed)."""

    def __init__(self, named_trainable):
        self.names = [n for n, _ in named_trainable]
        self.params = [p for _, p in named_trainable]
        self.ever_nonzero = [False] * len(self.params)
        self.checks, self.nonfinite_run, self.max_nonfinite_run, self.finite_checks = 0, 0, 0, 0

    def check(self):
        self.checks += 1
        missing = [n for n, p in zip(self.names, self.params) if p.grad is None]
        if missing:
            raise JCheckError(f"trainable tensors without gradient: {missing[:5]} (+{max(0, len(missing) - 5)})")
        norms = torch.stack([p.grad.detach().float().norm() for p in self.params])
        finite = bool(torch.isfinite(norms).all())
        if finite:
            self.finite_checks += 1
            self.nonfinite_run = 0
            for i, v in enumerate(norms.tolist()):
                self.ever_nonzero[i] |= v > 0
        else:
            self.nonfinite_run += 1
            self.max_nonfinite_run = max(self.max_nonfinite_run, self.nonfinite_run)
            if self.nonfinite_run >= 3:
                raise JCheckError("non-finite gradients at 3 consecutive checked steps")
        return finite

    def finish(self, skipped, steps):
        if self.finite_checks == 0:
            raise JCheckError("no checked step had finite gradients")
        dead = [n for n, ok in zip(self.names, self.ever_nonzero) if not ok]
        if dead:
            raise JCheckError(f"trainable tensors never received a non-zero gradient: {dead[:5]}")
        if steps and skipped / steps > 0.05:
            raise JCheckError(f"AMP skipped {skipped}/{steps} steps (> 5%)")
        return {"checks": self.checks, "finite_checks": self.finite_checks, "max_nonfinite_run": self.max_nonfinite_run,
                "skipped_steps": skipped, "steps": steps, "tensors": len(self.params)}


def supcon_crosscrop(z, crop, dis, tau=0.1):
    z = F.normalize(z, dim=-1)
    sim = z @ z.T / tau
    eye = torch.eye(len(z), dtype=torch.bool, device=z.device)
    pos = (dis[:, None] == dis[None]) & (crop[:, None] != crop[None]) & ~eye
    has = pos.any(1)
    if not has.any():
        return z.sum() * 0.0
    logp = sim.masked_fill(eye, -1e9).log_softmax(1)
    return (-(logp * pos).sum(1)[has] / pos.sum(1)[has]).mean()


# ----------------------------------------------------------------------------- one fold
def seed_all(seed, fold_index):
    s = 1000 * seed + fold_index
    torch.manual_seed(s); np.random.seed(s)
    torch.backends.cudnn.benchmark = False
    return s


def save_atomic_json(path, obj):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=float))
    os.replace(tmp, path)


def checkpoint_keys(model, heads):
    return sorted(n for n, p in model.named_parameters() if p.requires_grad), sorted(heads.state_dict())


def save_checkpoint(path, model, heads, meta):
    bk, hk = checkpoint_keys(model, heads)
    params = dict(model.named_parameters())
    torch.save({"meta": dict(meta, backbone_keys=bk, head_keys=hk),
                "backbone": {k: params[k].detach().cpu().clone() for k in bk}, "heads": heads.state_dict()}, path)
    return sha256_file(path)


def load_verified(path, backend, cfg, frame_dims, probe, expected_meta, device):
    """Fresh model + checkpoint -> verify keys, metadata and outputs on ``probe`` (fail closed)."""
    try:
        st = torch.load(path, map_location=device, weights_only=False)
    except Exception as e:
        raise JCheckError(f"checkpoint unreadable: {type(e).__name__}: {e}") from e
    meta = st.get("meta", {})
    for k, v in expected_meta.items():
        if meta.get(k) != v:
            raise JCheckError(f"checkpoint metadata mismatch on {k!r}")
    model, tf, _ = backend.build(cfg, device)
    heads = Heads(backend.dim, *frame_dims, extra=cfg.extra).to(device)
    bk, hk = checkpoint_keys(model, heads)
    if sorted(st["backbone"]) != bk or sorted(meta.get("backbone_keys", [])) != bk:
        raise JCheckError(f"checkpoint backbone keys differ from the trainable set ({len(st['backbone'])} vs {len(bk)})")
    if sorted(st["heads"]) != hk:
        raise JCheckError("checkpoint head keys differ")
    res = model.load_state_dict(st["backbone"], strict=False)
    if res.unexpected_keys or (set(bk) & set(res.missing_keys)):
        raise JCheckError("checkpoint did not load every trainable key")
    heads.load_state_dict(st["heads"], strict=True)
    model.eval(); heads.eval()
    with torch.no_grad():
        out = backend.embed(model, probe["x"])
    diff = float((out - probe["z"]).abs().max())
    if diff != 0.0:
        raise JCheckError(f"reloaded checkpoint output differs (max abs diff {diff:.3e})")
    return model, tf, heads, diff


def run_fold(cfg: JConfig, seed: int, fold: dict, fold_index: int, frame, bench, backend, access: ImageAccess,
             out_dir: Path, device="cuda", frozen_feats=None, recipe=None):
    """Execute one (row, seed, fold). Returns the fold receipt; raises JCheckError on any failed check."""
    t0 = time.time()
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    fit = [int(r) for r in fold["fitting_rows"]]
    seen, uns = [int(r) for r in fold["seen_val_rows"]], [int(r) for r in fold["unseen_val_rows"]]
    forbidden = set(seen) | set(uns) | set(int(r) for r in bench["outer_exclusion_rows"]) | set(int(r) for r in bench.get("outer_seen_rows", []))
    if set(fit) & forbidden:
        raise JCheckError("fitting rows overlap evaluation or outer rows")
    rs = seed_all(seed, fold_index)
    crops = sorted(frame.crop_id.unique()); diss = sorted(frame.disease_id.unique())
    ci = {c: i for i, c in enumerate(crops)}; di = {d: i for i, d in enumerate(diss)}
    yc = torch.tensor([ci[c] for c in frame.crop_id.to_numpy()[fit]])
    yd = torch.tensor([di[d] for d in frame.disease_id.to_numpy()[fit]])
    model, tf, wfile = backend.build(cfg, device, lora_seed=seed)
    heads = Heads(backend.dim, len(crops), len(diss), extra=cfg.extra).to(device)
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    frozen = [p for p in model.parameters() if not p.requires_grad]
    frozen_before = param_digest(frozen)
    start = [p.detach().clone() for _, p in named]
    rec = {"row": cfg.row, "seed": seed, "fold": fold["name"], "config": asdict(cfg), "config_sha256": cfg.sha(),
           "torch_seed": rs, "weights_file": wfile, "trainable_backbone_params": int(sum(p.numel() for _, p in named)),
           "frozen_params": int(sum(p.numel() for p in frozen)), "n_fitting": len(fit), "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    steps = skipped = 0
    if cfg.epochs > 0:
        if not named:
            raise JCheckError("training requested but no trainable backbone tensor")
        access.allow("train", fit)
        comp = frame.comp_id.to_numpy()[fit]
        _, inv, cnt = np.unique(comp, return_inverse=True, return_counts=True)
        g = torch.Generator().manual_seed(rs)
        sampler = WeightedRandomSampler(torch.tensor(1.0 / cnt[inv]), num_samples=len(fit), replacement=True, generator=g)
        idx = torch.arange(len(fit))
        if cfg.source_balanced:
            src = frame.source.to_numpy()[fit]
            _, src_i = np.unique(src, return_inverse=True)
            wc = within_concept_source_weight(yc, torch.tensor(src_i)).float(); wc = wc / wc.mean()
            wd = within_concept_source_weight(yd, torch.tensor(src_i)).float(); wd = wd / wd.mean()
        else:
            wc = wd = torch.ones(len(fit))
        ds = RowsDataset(access, fit, train_transform(cfg, tf), labels=(yc, yd, idx))
        dl = DataLoader(ds, batch_size=cfg.batch, sampler=sampler, num_workers=cfg.workers, drop_last=True,
                        worker_init_fn=lambda w, s=rs: torch.manual_seed(s + 100 * w), persistent_workers=False)
        params = [{"params": [p for _, p in named], "lr": cfg.lr_backbone}, {"params": heads.parameters(), "lr": cfg.lr_head}]
        opt = torch.optim.AdamW(params, weight_decay=cfg.weight_decay)
        per_epoch = len(dl); total = cfg.epochs * per_epoch; warm = cfg.warmup_epochs * per_epoch
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / max(1, warm) if s < warm
                                                  else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
        use_amp = device == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        mon = GradMonitor(named)
        check_every = 50
        losses = []
        model.train(); heads.train()
        done = False
        for ep in range(cfg.epochs):
            for x, c, d, ix in dl:
                x, c, d = x.to(device, non_blocking=True), c.to(device), d.to(device)
                with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                    z = backend.embed(model, x)
                    zc, zd, za = heads(z)
                    lc = F.cross_entropy(zc.float(), c, reduction="none") * wc[ix].to(device)
                    ld = F.cross_entropy(zd.float(), d, reduction="none") * wd[ix].to(device)
                    loss = lc.mean() + ld.mean()
                    if cfg.extra == "adversary":
                        loss = loss + F.cross_entropy(za.float(), c)
                    if cfg.extra == "supcon":
                        loss = loss + 0.5 * supcon_crosscrop(z.float(), c, d)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                steps += 1
                is_check = steps in (1, 5) or steps % check_every == 0 or steps == total or (cfg.max_steps and steps == cfg.max_steps)
                if is_check:
                    scaler.unscale_(opt)
                    mon.check()
                scale_before = scaler.get_scale() if use_amp else 1.0
                scaler.step(opt); scaler.update(); sched.step()
                if use_amp and scaler.get_scale() < scale_before:
                    skipped += 1
                losses.append(float(loss))
                if cfg.max_steps and steps >= cfg.max_steps:
                    done = True
                    break
            if done:
                break
        rec["grad_checks"] = mon.finish(skipped, steps)
        rec["train_loss"] = {"first50": float(np.mean(losses[:50])), "last50": float(np.mean(losses[-50:])), "steps": steps}
        if not all(math.isfinite(v) for v in losses[-50:]):
            raise JCheckError("non-finite training loss in the last 50 steps")
    if param_digest(frozen) != frozen_before:
        raise JCheckError("frozen parameters changed during training")
    changed = [float((p.detach() - s).norm()) > 0 for (_, p), s in zip(named, start)]
    if cfg.epochs > 0 and not all(changed):
        raise JCheckError(f"{changed.count(False)} trainable tensors did not change")
    rec["trainable_tensors_changed"] = f"{sum(changed)}/{len(changed)}"
    # --- checkpoint, reload, verify
    access.allow("probe", fit[:8])
    probe_tf = tf
    probe_x = torch.stack([probe_tf(access.get(r)) for r in fit[:8]]).to(device)
    model.eval(); heads.eval()
    with torch.no_grad():
        probe = {"x": probe_x, "z": backend.embed(model, probe_x)}
    meta = {"row": cfg.row, "seed": seed, "fold": fold["name"], "config_sha256": cfg.sha()}
    ck = out_dir / "checkpoint.pt"
    rec["checkpoint_sha256"] = save_checkpoint(ck, model, heads, meta)
    del model
    model, tf, heads, diff = load_verified(ck, backend, cfg, (len(crops), len(diss)), probe, meta, device)
    rec["reload_max_abs_diff"] = diff
    # --- extraction from the reloaded model (fold rows only)
    rows = sorted(set(fit) | set(seen) | set(uns))
    access.allow("extract", rows)
    feats = backend.extract(model, tf, access, rows, cfg.workers)
    cache_meta = {"row": cfg.row, "seed": seed, "fold": fold["name"], "config_sha256": cfg.sha(),
                  "checkpoint_sha256": rec["checkpoint_sha256"], "rows_sha256": sha256_json(rows),
                  "backbone": backend.name, "schema": "r2v4-j-fold-features-v1"}
    cache = out_dir / "features.pt"
    torch.save({"meta": cache_meta, "rows": rows, "features": feats}, cache)
    rec["features_sha256"] = sha256_file(cache)
    feats_loaded, rows_loaded = load_feature_cache(cache, cache_meta)
    # --- downstream D1 on this fold only
    full = torch.full((len(frame), 2, feats_loaded.shape[-1]), float("nan"))
    full[rows_loaded] = feats_loaded.float()
    if cfg.concat_frozen:
        fz = frozen_feats.float()
        fz = fz / fz.norm(dim=-1, keepdim=True)
        full = torch.cat([fz, full / full.norm(dim=-1, keepdim=True)], -1)
    torch.manual_seed(0)
    fd, truth, _ = make_fold(frame, fold, full, None)
    s, u = (recipe or RECIPES["synth_composition"])(fd, D1_CFG)
    pk = {"cand_ids": np.asarray(fd.extra["cand_ids"]).astype(np.int64), "seen": s.numpy().astype(np.float32),
          "unseen": u.numpy().astype(np.float32), "seen_truth": truth["seen"].astype(np.int64),
          "unseen_truth": truth["unseen"].astype(np.int64)}
    con = fold_contract(frame, fold)
    validate_packet(pk, con)
    pkt = out_dir / f"{fold['name']}_scores.npz"
    np.savez_compressed(pkt, **pk)
    validate_packet(np.load(pkt), con)
    rec.update({"packet_sha256": sha256_file(pkt), "contract_sha256": con["sha256"], "images_opened_main_process": access.opened,
                "seconds": time.time() - t0, "status": "ok", "finished": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    save_atomic_json(out_dir / "receipt.json", rec)
    return rec


def load_feature_cache(path, expected_meta):
    """Refuse a feature cache whose metadata does not match exactly (stale or foreign cache)."""
    try:
        st = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        raise JCheckError(f"feature cache unreadable: {e}") from e
    if st.get("meta") != expected_meta:
        bad = [k for k in set(expected_meta) | set(st.get("meta", {})) if st.get("meta", {}).get(k) != expected_meta.get(k)]
        raise JCheckError(f"feature cache metadata mismatch: {sorted(bad)}")
    if sha256_json(st["rows"]) != expected_meta["rows_sha256"] or len(st["rows"]) != len(st["features"]):
        raise JCheckError("feature cache rows do not match")
    return st["features"], st["rows"]


# ----------------------------------------------------------------------------- row driver
def fold_dir(base, row, seed, fold_name):
    d = Path(base) / row / f"seed{seed}" / fold_name
    d.mkdir(parents=True, exist_ok=True)
    n = 1 + len([p for p in d.iterdir() if p.name.startswith("attempt")])
    return d, d / f"attempt{n}"


def verify_receipt(adir: Path):
    """A fold counts only if its receipt says ok AND every referenced file still has the recorded sha256."""
    r = adir / "receipt.json"
    if not r.exists():
        return None
    rec = json.loads(r.read_text())
    if rec.get("status") != "ok":
        return None
    for key, fn in (("checkpoint_sha256", "checkpoint.pt"), ("features_sha256", "features.pt"),
                    ("packet_sha256", f"{rec['fold']}_scores.npz")):
        if not (adir / fn).exists() or sha256_file(adir / fn) != rec[key]:
            raise JCheckError(f"{adir}: {fn} missing or changed since its receipt")
    return rec


def completed_attempt(base, row, seed, fold_name):
    d = Path(base) / row / f"seed{seed}" / fold_name
    if not d.exists():
        return None
    for a in sorted(d.glob("attempt*"), key=lambda p: int(p.name[7:]), reverse=True):
        rec = verify_receipt(a)
        if rec is not None:
            return a, rec
    return None


def run_row(cfg, seed, folds, frame, bench, backend, access, base, device="cuda", frozen_feats=None, log=print, recipe=None):
    """Run every fold not yet completed. A failure is recorded in the attempt dir and the row is reported incomplete."""
    status = {}
    for i, fn in enumerate(folds):
        if completed_attempt(base, cfg.row, seed, fn["name"]):
            status[fn["name"]] = "ok (existing)"
            continue
        _, adir = fold_dir(base, cfg.row, seed, fn["name"])
        try:
            rec = run_fold(cfg, seed, fn, i, frame, bench, backend, access, adir, device, frozen_feats, recipe)
            status[fn["name"]] = "ok"
            log(json.dumps({k: rec[k] for k in ("row", "seed", "fold", "seconds", "reload_max_abs_diff", "trainable_tensors_changed")}))
        except Exception as e:
            adir.mkdir(parents=True, exist_ok=True)
            save_atomic_json(adir / "failure.json", {"row": cfg.row, "seed": seed, "fold": fn["name"], "status": "failed",
                                                      "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-3000:]})
            status[fn["name"]] = f"failed: {type(e).__name__}: {e}"
            log(f"FAILED {cfg.row} seed{seed} {fn['name']}: {type(e).__name__}: {e}")
    return status


def row_packets(base, row, seed, fold_names):
    """All six validated fold packets of a row, or JCheckError naming the missing folds (evaluation refuses)."""
    out, missing = {}, []
    for fn in fold_names:
        got = completed_attempt(base, row, seed, fn)
        if got is None:
            missing.append(fn)
            continue
        adir, rec = got
        out[fn] = (adir / f"{fn}_scores.npz", rec)
    if missing:
        raise JCheckError(f"{row} seed{seed}: incomplete, missing folds {missing}")
    return out
