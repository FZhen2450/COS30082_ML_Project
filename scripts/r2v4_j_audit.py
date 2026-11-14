"""Real-data audit of the Stage J workflow (no training): compare zero-epoch rows with the registered v3 evidence.

  J0_pipeline_zero_epochs  fold packets vs the registered v3 D1 packets (max abs diff <= 1e-5) and fold features vs
                           v3 clip_l14 features (bit-identical expected)
  J8_zero_epochs           fold ConvNeXt features vs v3 tv_convnext_base features, and fold packets vs the registered
                           v3 F10 packets (D1 on [L2 CLIP-L, L2 ConvNeXt])
  smoke                    the metric-free J1 smoke receipt: checks passed, packet validated; no metric is computed
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research_r2_v4 import jrun  # noqa: E402
from research_r2_v4.common import EXP, INNER, env  # noqa: E402

TOL = 1e-5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit-dir", required=True)
    ap.add_argument("--smoke-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    frame, bench, F = env()
    sys.path.insert(0, str(ROOT / "scripts"))
    import r2v4_stage_i as si
    rep, ok = {"tolerance": TOL}, True
    for row, ref_row, feat in (("J0_pipeline_zero_epochs", "D1_synthComp_norm", "clip_l14"),
                               ("J8_zero_epochs", "F10_concat_convnext", "tv_convnext_base")):
        ref, _ = si.v3_packets(ref_row, frame, F)
        v3 = torch.load(EXP / "features" / f"{feat}.pt", map_location="cpu", weights_only=False)["features"]
        pk = jrun.row_packets(a.audit_dir, row, 0, INNER)
        rr = {}
        for fn, (p, rec) in pk.items():
            z = np.load(p)
            st = torch.load(p.parent / "features.pt", map_location="cpu", weights_only=False)
            same_ids = bool(np.array_equal(z["cand_ids"], ref[fn]["cand_ids"]))
            d = max(float(np.abs(z["seen"] - ref[fn]["seen"]).max()), float(np.abs(z["unseen"] - ref[fn]["unseen"]).max()))
            fbit = bool(torch.equal(st["features"], v3[st["rows"]]))
            fdiff = float((st["features"].float() - v3[st["rows"]].float()).abs().max())
            rr[fn] = {"cand_ids_identical": same_ids, "truth_identical": bool(np.array_equal(z["seen_truth"], ref[fn]["seen_truth"]) and np.array_equal(z["unseen_truth"], ref[fn]["unseen_truth"])),
                      "packet_max_abs_diff_vs_registered": d, "features_bit_identical_vs_v3": fbit, "features_max_abs_diff_vs_v3": fdiff,
                      "rows_extracted": len(st["rows"]), "reload_max_abs_diff": rec["reload_max_abs_diff"]}
            ok &= same_ids and rr[fn]["truth_identical"] and d <= TOL
        rep[row] = {"reference": ref_row, "folds": rr}
        print(row, {fn: (round(v["packet_max_abs_diff_vs_registered"], 9), v["features_bit_identical_vs_v3"]) for fn, v in rr.items()})
    smoke = {}
    for rdir in Path(a.smoke_dir).glob("J*/seed0/*/attempt*"):
        r = rdir / "receipt.json"
        f = rdir / "failure.json"
        smoke[str(rdir.relative_to(a.smoke_dir))] = json.loads(r.read_text()) if r.exists() else json.loads(f.read_text())
    rep["smoke"] = {k: {x: v.get(x) for x in ("status", "error", "grad_checks", "train_loss", "trainable_tensors_changed",
                                              "reload_max_abs_diff", "packet_sha256", "seconds", "trainable_backbone_params")} for k, v in smoke.items()}
    # every attempt is reported; the audit needs the LATEST attempt of each smoke fold to be ok (earlier failed
    # attempts are kept as evidence, e.g. the 20-step run that the AMP-skip rule correctly failed)
    latest = {}
    for k in smoke:
        fold_key, att = k.rsplit("/attempt", 1)
        if fold_key not in latest or int(att) > int(latest[fold_key].rsplit("/attempt", 1)[1]):
            latest[fold_key] = k
    rep["smoke_latest_attempts"] = sorted(latest.values())
    ok &= bool(latest) and all(smoke[k].get("status") == "ok" for k in latest.values())
    rep["pass"] = bool(ok)
    Path(a.out).write_text(json.dumps(rep, indent=1))
    print("smoke:", json.dumps(rep["smoke"], indent=1)[:1500])
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
