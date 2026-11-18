"""Aggregate source-ablation stress JSON files (image-weighted over ablations)."""
import json
import sys

import numpy as np

for path in sys.argv[1:]:
    d = json.load(open(path))
    E = [e for e in d["source_shift"] if "skipped" not in e]
    ES = [e for e in E if e["n_seen"] > 0]
    wu = np.array([e["n_unseen"] for e in E]); ws = np.array([e["n_seen"] for e in ES])
    out = {"row": d["row"], "ablations": len(E), "skipped": len(d["source_shift"]) - len(E), "unseen_images": int(wu.sum()), "seen_images": int(ws.sum())}
    for g in d["gammas"]:
        k = f"{g:g}"
        out[f"g{k}"] = {
            "seen_in_fitting": float(np.sum([e[f"seen_top1_in_fitting_g{k}"] * e["n_seen"] for e in ES]) / ws.sum()),
            "seen_source_novel": float(np.sum([e[f"seen_top1_source_novel_g{k}"] * e["n_seen"] for e in ES]) / ws.sum()),
            "seen_novel_pulled_to_heldout": float(np.sum([e[f"seen_to_heldout_source_novel_g{k}"] * e["n_seen"] for e in ES]) / ws.sum()),
            "unseen_in_fitting": float(np.sum([e[f"unseen_top1_in_fitting_g{k}"] * e["n_unseen"] for e in E]) / wu.sum()),
            "unseen_source_novel": float(np.sum([e[f"unseen_top1_source_novel_g{k}"] * e["n_unseen"] for e in E]) / wu.sum())}
    out["fitting_size"] = d["fitting_size"]
    print(json.dumps(out))
