"""Extract frozen features for the 8,000 training-source images (no official-test access)."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research_r2.data import candidate_table, load_training_source, sha256_file, sha256_json  # noqa: E402
from research_r2.features import WEIGHTS, clip_text_bank, extract_images  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--digests", required=True, help="benchmark/image_digests.json")
    ap.add_argument("--backbones", nargs="+", default=["clip_l14", "clip_b32", "tv_vit_b16", "tv_convnext_base", "tv_effnet_b3"])
    a = ap.parse_args()
    frame = load_training_source(a.csv)
    digests = json.loads(Path(a.digests).read_text())["images"]
    assert [d["file"] for d in digests] == frame.file.tolist()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cands = candidate_table(frame)
    for name in a.backbones:
        dest = out / f"{name}.pt"
        if dest.exists():
            print("exists", dest)
            continue
        t = time.time()
        torch.cuda.reset_peak_memory_stats()
        feats, wfile = extract_images(name, [Path(a.images) / f for f in frame.file])
        rec = {"schema": "r2-frozen-features-v1", "backbone": name, "views": ["centre", "hflip"],
               "features": feats, "files": frame.file.tolist(),
               "image_sha256_list_sha256": sha256_json([d["sha256"] for d in digests]),
               "weights_file": wfile, "weights_sha256": sha256_file(WEIGHTS / wfile),
               "train_csv_sha256": sha256_file(a.csv), "seconds": time.time() - t,
               "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
               "torch": torch.__version__, "precision": "fp16 autocast forward, stored fp16"}
        if name.startswith("clip_"):
            rec["text"] = clip_text_bank(name, cands)
        torch.save(rec, dest)
        print(f"{name}: {tuple(feats.shape)} in {rec['seconds']:.1f}s peak {rec['peak_gpu_bytes']/2**30:.2f} GiB")


if __name__ == "__main__":
    main()
