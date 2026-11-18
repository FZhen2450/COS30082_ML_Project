"""Check that the current code reproduces saved v2 inner scores on the legacy (v2) benchmark path.

Guards against Phase 2 edits changing default behaviour. Exit status 1 on any mismatch.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research_r2 import methods as M  # noqa: E402
from research_r2.data import load_training_source  # noqa: E402
from research_r2.runner import load_features, make_fold  # noqa: E402

EXP = ROOT.parents[1] / "experiments/r2-compositional"
ROWS = [("B1_clipL_factorised", M.score_factorised, {}),
        ("D1_synthComp_norm", M.score_synth_composition, {"fuse": 1.0, "normalise": True}),
        ("D2_synthComp_norm_text", M.score_synth_composition, {"fuse": 1.0, "normalise": True,
                                                                "factorised": {"text_dis_beta": 1.0, "text_crop_beta": 1.0}})]


def main():
    fr = load_training_source(ROOT / "DPD_Small_train.csv")
    b = json.loads((ROOT / "artifacts/r2/benchmark_v2.json").read_text())
    feats, text, _, _ = load_features(EXP / "features", ["clip_l14"])
    feats = feats.clone(); feats[b["outer_exclusion_rows"]] = float("nan")
    ok = True
    for fold in [f for f in b["folds"] if f["role"] == "inner"]:
        for rid, rec, cfg in ROWS:
            torch.manual_seed(0)
            fd, _, _ = make_fold(fr, fold, feats, text)
            s, u = rec(fd, cfg)
            z = np.load(EXP / "v2/inner_stage_d_v1/scores" / rid / f"{fold['name']}_scores.npz")
            d = max(float(np.abs(z["unseen"] - u.numpy()).max()), float(np.abs(z["seen"] - s.numpy()).max()))
            same = bool((z["unseen"].argmax(1) == u.numpy().argmax(1)).all() and (z["seen"].argmax(1) == s.numpy().argmax(1)).all())
            good = d < 1e-4 and same
            ok &= good
            print(f"{'PASS' if good else 'FAIL'} {fold['name']} {rid:<24} max|diff| {d:.2e} top1_identical={same}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
