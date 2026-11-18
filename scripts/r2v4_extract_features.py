"""v4 versioned feature extraction from local CLIP ViT-L/14 weights (training CSV images only; no downloads).

  --kind ml   I7: CLS token after resblocks 16, 20, 24, each -> ln_post -> proj (768-d), centre + h-flip views,
              stored as features [8000, 2, 3*768] (layer order 16, 20, 24). Block 24 is compared per image with
              the v3 clip_l14 features (cosine) and the agreement is recorded.
  --kind mc   I8: views 0-1 copied unchanged from the v3 clip_l14 features (centre, h-flip); views 2-11 =
              Resize(256, bicubic) + FiveCrop(224) x {no flip, h-flip} from images -> features [8000, 12, 768].
              Only fitting rows ever use views 2-11 (make_fold reads evaluation rows from view 0).
Same precision as v3: model in fp16 (clip.load on CUDA), forward under fp16 autocast, stored fp16.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research_r2.data import load_training_source, sha256_file, sha256_json  # noqa: E402
from research_r2.features import WEIGHTS, load_backbone  # noqa: E402
from research_r2_v4.common import EXP, FEATURES  # noqa: E402

LAYERS = (16, 20, 24)


class FiveCropDS(Dataset):
    def __init__(self, paths, norm_tf):
        import torchvision.transforms as T
        self.paths = list(paths)
        self.resize = T.Resize(256, interpolation=T.InterpolationMode.BICUBIC)
        self.five = T.FiveCrop(224)
        self.norm_tf = norm_tf  # ToTensor + Normalize of the CLIP transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            crops = self.five(self.resize(im.convert("RGB")))
        t = torch.stack([self.norm_tf(c) for c in crops])  # [5, 3, 224, 224]
        return torch.cat([t, torch.flip(t, dims=[3])])  # [10, ...]: 5 crops, then their h-flips


class PlainDS(Dataset):
    def __init__(self, paths, tf):
        self.paths, self.tf = list(paths), tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            return self.tf(im.convert("RGB"))


def multilayer(model, x):
    v = model.visual
    x = x.type(model.dtype)
    x = v.conv1(x)
    x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
    x = torch.cat([v.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
    x = v.ln_pre(x + v.positional_embedding.to(x.dtype)).permute(1, 0, 2)
    outs = []
    for i, blk in enumerate(v.transformer.resblocks, start=1):
        x = blk(x)
        if i in LAYERS:
            outs.append(v.ln_post(x.permute(1, 0, 2)[:, 0, :]) @ v.proj)
    return torch.cat(outs, -1)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["ml", "mc"], required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    frame = load_training_source(ROOT / "DPD_Small_train.csv")  # training rows only
    digests = json.loads((EXP / "benchmark" / "image_digests.json").read_text())["images"]
    assert [d["file"] for d in digests] == frame.file.tolist()
    paths = [Path(a.images) / f for f in frame.file]
    base = torch.load(FEATURES / "clip_l14.pt", map_location="cpu", weights_only=False)
    assert base["files"] == frame.file.tolist()
    model, tf, enc, wfile, _ = load_backbone("clip_l14", "cuda")
    t0 = time.time(); torch.cuda.reset_peak_memory_stats()
    amp = torch.autocast("cuda", dtype=torch.float16)
    rec = {"backbone": f"clip_l14_{a.kind}", "files": frame.file.tolist(), "weights_file": wfile,
           "weights_sha256": sha256_file(WEIGHTS / wfile), "image_sha256_list_sha256": sha256_json([d["sha256"] for d in digests]),
           "train_csv_sha256": sha256_file(ROOT / "DPD_Small_train.csv"), "torch": torch.__version__,
           "precision": "fp16 weights (clip.load on CUDA), fp16 autocast forward, stored fp16",
           "source_v3_features_sha256": sha256_file(FEATURES / "clip_l14.pt")}
    if a.kind == "ml":
        out = []
        for x in DataLoader(PlainDS(paths, tf), batch_size=32, num_workers=8, shuffle=False):
            x = x.cuda(non_blocking=True)
            with amp:
                out.append(torch.stack([multilayer(model, x).float(), multilayer(model, torch.flip(x, dims=[3])).float()], 1).cpu())
        feats = torch.cat(out).half()
        last = feats[:, :, 2 * 768:].float()
        ref = base["features"].float()
        cos = torch.nn.functional.cosine_similarity(last, ref, dim=-1)
        rec.update({"schema": "r2v4-frozen-features-multilayer-v1", "layers": list(LAYERS), "views": ["centre", "hflip"],
                    "transform": "v3 CLIP-L/14 transform (Resize 224 bicubic, CenterCrop 224, CLIP normalisation)",
                    "per_layer": "CLS after resblock L -> ln_post -> proj",
                    "block24_vs_v3": {"cos_min": float(cos.min()), "cos_mean": float(cos.mean()),
                                      "bit_identical_fraction": float((feats[:, :, 2 * 768:] == base["features"]).all(-1).float().mean())}})
    else:
        norm_tf = torch.nn.Sequential()  # placeholder replaced below
        import torchvision.transforms as T
        norm_tf = T.Compose([t for t in tf.transforms if isinstance(t, (T.ToTensor, T.Normalize))])
        assert len(norm_tf.transforms) == 2, tf
        out = []
        for x in DataLoader(FiveCropDS(paths, norm_tf), batch_size=8, num_workers=8, shuffle=False):
            b = x.shape[0]
            x = x.reshape(b * 10, *x.shape[2:]).cuda(non_blocking=True)
            with amp:
                out.append(enc(x).float().reshape(b, 10, -1).cpu())
        extra = torch.cat(out).half()
        feats = torch.cat([base["features"], extra], 1)
        rec.update({"schema": "r2v4-frozen-features-multicrop-v1",
                    "views": ["centre (v3)", "hflip (v3)"] + [f"fivecrop{k}" for k in range(5)] + [f"fivecrop{k}_hflip" for k in range(5)],
                    "transform_extra_views": "Resize(256, bicubic) + FiveCrop(224) [tl, tr, bl, br, centre], then h-flip of each; CLIP normalisation",
                    "views_0_1": "copied unchanged from v3 clip_l14 features (evaluation view identical to D1)",
                    "use_rule": "only fitting rows use views 2-11; evaluation rows use view 0 (research_r2.runner.make_fold)"})
    rec["features"] = feats
    rec["features_sha256"] = __import__("hashlib").sha256(feats.numpy().tobytes()).hexdigest()
    rec["shape"] = list(feats.shape)
    rec["seconds"] = time.time() - t0
    rec["peak_gpu_bytes"] = torch.cuda.max_memory_allocated()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(rec, a.out)
    meta = {k: v for k, v in rec.items() if k not in ("features", "files")}
    Path(str(a.out) + ".json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
