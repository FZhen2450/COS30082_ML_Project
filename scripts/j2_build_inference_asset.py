"""Build the private J2 inference asset for the GUI by CAPTURING the D1 downstream heads that the frozen Second Run
J2 scoring pipeline fits on the frozen training features. Nothing is retrained, recalibrated or re-selected, and no
official-test data is read.

  python scripts/j2_build_inference_asset.py --r2-root <Second Run v5 worktree> --run <final_J2_full8000 dir> \
      --rehearsal <rehearsal_07e05ef dir> --out <new private asset file>

How: research_r2.methods.fit_softmax is wrapped (recording only) during ONE call of the frozen
research_r2_v5.final.raw_scores, which fits the composition head, then the crop and disease heads. The fitting-feature
standardisation is recomputed with the same arithmetic. The asset is accepted only if wp4/j2_predictor.score_features
-- the exact arithmetic the GUI runs -- reproduces (a) the frozen pipeline's calibrated scores on the same features and
(b) the saved official-size preflight rehearsal scores (2,000 + 1,448 training images), with identical Top-5 rankings.
The Second Run code is read from --r2-root and is not copied into this repository.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r2-root", required=True); ap.add_argument("--run", required=True)
    ap.add_argument("--rehearsal", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        sys.exit(f"{out} exists; assets are never overwritten")
    sys.path.insert(0, str(ROOT)); sys.path.insert(1, str(Path(a.r2_root).resolve()))
    import numpy as np
    import torch
    import research_r2.methods as M
    from research_r2_v4 import jrun
    from research_r2_v4.common import env
    from research_r2_v5 import final as fm
    from wp4 import j2_predictor as jp

    run = Path(a.run) / "run"
    rec = jrun.verify_receipt(run)
    if rec is None or rec["checkpoint_sha256"] != jp.J2_CHECKPOINT_SHA256 or rec["features_sha256"] != jp.J2_FEATURES_SHA256:
        sys.exit("frozen J2 run does not verify")
    c = fm.load_config()
    if c["gamma"] != jp.GAMMA or c["config_sha256"] != jp.J2_CONFIG_SHA256 or fm.FINAL_CANDIDATE_SHA256 != jp.FINAL_CANDIDATE_SHA256:
        sys.exit("frozen configuration differs")
    frame, _, _ = env()
    rows = fm.load_payload(frame)
    feats, frows = jrun.load_feature_cache(run / "features.pt", rec["feature_cache_meta"])
    assert frows == list(range(len(frame)))

    # 1. capture the heads fitted by ONE call of the frozen pipeline (query: the rehearsal "seen" part, rows 0..1999)
    captured, original = [], M.fit_softmax
    def recording(*args, **kw):
        head = original(*args, **kw)
        captured.append({"n_classes": int(args[2]), "state": {k: v.detach().float().cpu().clone() for k, v in head.state_dict().items()}})
        return head
    M.fit_softmax = recording
    try:
        t0 = time.time()
        q_seen = feats[:2000, 0].float()
        ref_seen = fm.calibrate(fm.raw_scores(frame, feats, rows, q_seen), rows, c["gamma"])
        fit_seconds = time.time() - t0
    finally:
        M.fit_softmax = original
    if [h["n_classes"] for h in captured] != [333, frame.crop_id.nunique(), frame.disease_id.nunique()]:
        sys.exit(f"unexpected fitted heads {[h['n_classes'] for h in captured]}")
    comp, crop, dis = (h["state"] for h in captured)

    # 2. standardisation and candidate index maps, exactly as research_r2.methods.standardise / final.final_fold_data
    xf = feats.float()
    xf = xf / xf.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    flat = xf.reshape(-1, xf.shape[-1])
    mu, sd = flat.mean(0), flat.std(0).clamp_min(1e-6)
    crops = sorted(frame.crop_id.unique()); diss = sorted(frame.disease_id.unique())
    ci = {cc: i for i, cc in enumerate(crops)}; di = {d: i for i, d in enumerate(diss)}
    asset = {"schema": jp.ASSET_SCHEMA, "checkpoint_sha256": jp.J2_CHECKPOINT_SHA256, "features_sha256": jp.J2_FEATURES_SHA256,
             "config_sha256": jp.J2_CONFIG_SHA256, "candidate_sha256": jp.FINAL_CANDIDATE_SHA256, "gamma": float(c["gamma"]),
             "fuse": 1.0, "normalise": True, "candidate_ids": torch.arange(333, dtype=torch.int64),
             "cand_crop": torch.tensor([ci[r["crop_id"]] for r in rows], dtype=torch.int64),
             "cand_dis": torch.tensor([di[r["disease_id"]] for r in rows], dtype=torch.int64),
             "seen_mask": torch.tensor([r["composition_id"] not in jp.FINAL_UNSEEN_IDS for r in rows], dtype=torch.bool),
             "crop_ids": torch.tensor([int(x) for x in crops], dtype=torch.int64), "disease_ids": torch.tensor([int(x) for x in diss], dtype=torch.int64),
             "mu": mu.contiguous(), "sd": sd.contiguous(),
             "heads": {"composition": {"weight": comp["weight"], "bias": comp["bias"]}, "crop": {"weight": crop["weight"], "bias": crop["bias"]},
                       "disease": {"weight": dis["weight"], "bias": dis["bias"]}}}
    jp._check_asset(asset)

    # 3. parity: adapter arithmetic vs the frozen pipeline and vs the saved preflight rehearsal (same features)
    def compare(name, mine, ref):
        mine, ref = np.asarray(mine, np.float64), np.asarray(ref, np.float64)
        t_m, t_r = np.argsort(-mine, 1, kind="stable")[:, :5], np.argsort(-ref, 1, kind="stable")[:, :5]
        return {"set": name, "n": int(len(mine)), "max_abs_diff": float(np.abs(mine - ref).max()),
                "top1_equal": float(np.mean(t_m[:, 0] == t_r[:, 0])), "top5_equal_rows": float(np.mean((t_m == t_r).all(1)))}
    parity = [compare("frozen pipeline, training rows 0-1999", jp.score_features(asset, q_seen), ref_seen)]
    q_uns = feats[2000:3448, 0].float()
    ref_uns = fm.calibrate(fm.raw_scores(frame, feats, rows, q_uns), rows, c["gamma"])
    parity.append(compare("frozen pipeline, training rows 2000-3447", jp.score_features(asset, q_uns), ref_uns))
    reh = Path(a.rehearsal)
    for part, q in (("seen", q_seen), ("unseen", q_uns)):
        parity.append(compare(f"saved preflight rehearsal '{part}' (float32 file)", jp.score_features(asset, q),
                              np.load(reh / f"{part}_calibrated_scores.npy")))
    # Arithmetic parity (identical input features): tight score tolerance. End-to-end parity against the saved
    # rehearsal: its query features were re-extracted in a different fp16 batch layout, so scores move slightly;
    # identical Top-1/Top-5 rankings are required and the score gap is reported, not hidden.
    for p in parity:
        p["criterion"] = ("arithmetic: same features, max_abs_diff < 1e-4, identical Top-5" if p["set"].startswith("frozen")
                          else "end-to-end: rehearsal re-extracted features, identical Top-1 and Top-5 required; gap reported")
    ok = all(p["top1_equal"] == 1.0 and p["top5_equal_rows"] == 1.0 and (p["max_abs_diff"] < 1e-4 or not p["set"].startswith("frozen"))
             for p in parity)
    if not ok:
        print(json.dumps(parity, indent=1)); sys.exit("parity FAILED: asset not written")
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(asset, out)
    receipt = {"asset": out.name, "asset_sha256": sha(out), "schema": jp.ASSET_SCHEMA, "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "from": {"checkpoint_sha256": jp.J2_CHECKPOINT_SHA256, "features_sha256": jp.J2_FEATURES_SHA256,
                        "receipt_sha256": sha(run / "receipt.json"), "config_sha256": jp.J2_CONFIG_SHA256, "gamma": c["gamma"],
                        "candidate_sha256": jp.FINAL_CANDIDATE_SHA256},
               "method": "heads captured from one call of the frozen research_r2_v5.final.raw_scores (no refit by this script); standardisation recomputed with the same arithmetic",
               "fit_seconds_frozen_pipeline": fit_seconds, "fit_device": str(M.DEV), "parity": parity,
               "official_test_data_read": False}
    out.with_suffix(".receipt.json").write_text(json.dumps(receipt, indent=1))
    print(json.dumps(receipt, indent=1))


if __name__ == "__main__":
    main()
