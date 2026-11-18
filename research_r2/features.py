"""Frozen-backbone feature extraction from locally verified weights (no downloads).

Each backbone emits a pooled embedding for two deterministic views (centre, h-flip of
centre). CLIP backbones also emit text embeddings for the versioned prompt set below.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

WEIGHTS = Path(__file__).resolve().parents[3] / "pretrained_weights"

PROMPTS_V1 = {
    "version": "r2-prompts-v1",
    "crop": ["a photo of a {crop} plant leaf.", "a close-up photo of a {crop} leaf."],
    "disease": ["a photo of a plant leaf with {disease}.", "a close-up photo of leaf symptoms of {disease}."],
    "healthy_disease": ["a photo of a healthy plant leaf.", "a close-up photo of a healthy leaf."],
    "comp": ["a photo of a {crop} leaf with {disease}.", "a close-up photo of {disease} on a {crop} leaf."],
    "healthy_comp": ["a photo of a healthy {crop} leaf.", "a close-up photo of a healthy {crop} leaf."],
}


def readable(name: str) -> str:
    return name.replace("_", " ")


class _Images(Dataset):
    def __init__(self, paths, transform):
        self.paths, self.transform = list(paths), transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            return self.transform(im.convert("RGB"))


def _torchvision(name, device):
    import torchvision.models as tvm
    import torchvision.transforms as T
    norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    if name == "tv_vit_b16":
        m = tvm.vit_b_16()
        m.load_state_dict(torch.load(WEIGHTS / "torch_cache/hub/checkpoints/vit_b_16-c867db91.pth", map_location="cpu"))
        m.heads = torch.nn.Identity()
        tf = T.Compose([T.Resize(256, interpolation=T.InterpolationMode.BILINEAR), T.CenterCrop(224), T.ToTensor(), norm])
        file = "torch_cache/hub/checkpoints/vit_b_16-c867db91.pth"
    elif name == "tv_convnext_base":
        m = tvm.convnext_base()
        m.load_state_dict(torch.load(WEIGHTS / "convnext_base-6075fbad.pth", map_location="cpu"))
        m.classifier[2] = torch.nn.Identity()
        tf = T.Compose([T.Resize(232, interpolation=T.InterpolationMode.BILINEAR), T.CenterCrop(224), T.ToTensor(), norm])
        file = "convnext_base-6075fbad.pth"
    elif name == "tv_effnet_b3":
        m = tvm.efficientnet_b3()
        m.load_state_dict(torch.load(WEIGHTS / "efficientnet_b3_rwightman-b3899882.pth", map_location="cpu"))
        m.classifier = torch.nn.Identity()
        tf = T.Compose([T.Resize(320, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(300), T.ToTensor(), norm])
        file = "efficientnet_b3_rwightman-b3899882.pth"
    else:
        raise KeyError(name)
    return m.eval().to(device), tf, (lambda x: m(x)), file


def load_backbone(name: str, device: str):
    """Return (module, transform, encode_fn, weight_file_relative, clip_model_or_None)."""
    if name.startswith("clip_"):
        import clip
        file = {"clip_l14": "clip/ViT-L-14.pt", "clip_b32": "clip/ViT-B-32.pt"}[name]
        model, tf = clip.load(str(WEIGHTS / file), device=device, jit=False)
        model.eval()
        return model, tf, (lambda x: model.encode_image(x)), file, model
    m, tf, enc, file = _torchvision(name, device)
    return m, tf, enc, file, None


@torch.no_grad()
def extract_images(name: str, paths, device="cuda", batch_size=32, workers=8):
    model, tf, enc, file, _ = load_backbone(name, device)
    loader = DataLoader(_Images(paths, tf), batch_size=batch_size, num_workers=workers, shuffle=False)
    out = []
    amp = torch.autocast("cuda", dtype=torch.float16) if device.startswith("cuda") else torch.autocast("cpu", enabled=False)
    for x in loader:
        x = x.to(device, non_blocking=True)
        with amp:
            f0 = enc(x).float()
            f1 = enc(torch.flip(x, dims=[3])).float()
        out.append(torch.stack([f0, f1], 1).cpu())
    feats = torch.cat(out)
    return feats.half(), file


@torch.no_grad()
def clip_text_bank(name: str, candidates, device="cuda") -> dict:
    """Averaged, L2-normalised prompt embeddings for crops, diseases and compositions."""
    import clip
    _, _, _, _, model = load_backbone(name, device)

    def embed(templates_list):
        vecs = []
        for templates in templates_list:
            tok = clip.tokenize(templates).to(device)
            e = model.encode_text(tok).float()
            e = e / e.norm(dim=-1, keepdim=True)
            e = e.mean(0)
            vecs.append(e / e.norm())
        return torch.stack(vecs).cpu()

    crops = sorted(set(zip(candidates.crop_id, candidates.crop)))
    diseases = sorted(set(zip(candidates.disease_id, candidates.disease)))
    P = PROMPTS_V1
    crop_t = [[t.format(crop=readable(c)) for t in P["crop"]] for _, c in crops]
    dis_t = [P["healthy_disease"] if d == "healthy" else [t.format(disease=readable(d)) for t in P["disease"]]
             for _, d in diseases]
    comp_t = [[t.format(crop=readable(r.crop)) for t in P["healthy_comp"]] if r.disease == "healthy"
              else [t.format(crop=readable(r.crop), disease=readable(r.disease)) for t in P["comp"]]
              for r in candidates.itertuples()]
    return {"prompt_version": P["version"], "crop_ids": [c for c, _ in crops], "disease_ids": [d for d, _ in diseases],
            "comp_ids": candidates.comp_id.tolist(), "crop": embed(crop_t), "disease": embed(dis_t),
            "comp": embed(comp_t), "logit_scale": float(model.logit_scale.exp().item())}


os.environ.setdefault("TORCH_HOME", str(WEIGHTS / "torch_cache"))
