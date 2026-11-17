"""Second Run final model CLI (J2 full-data + D1 downstream + frozen gamma over the 333-candidate universe).

  train          python scripts/r2v5_final.py train --out <E>/v5/final_J2_full8000
                 trains the frozen configuration (research_r2_v5/final_config.json) on all 8,000 training images
                 (one GPU job); clean worktree required; refuses an existing output directory
  verify         python scripts/r2v5_final.py verify --run <E>/v5/final_J2_full8000
                 receipt + file hashes, configuration, torch seed, gradient/AMP/frozen/reload checks, feature cache
                 (8,000 rows, finite), candidate payload, independent checkpoint reload (re-extracted training features
                 must equal the cache), query-path consistency, deterministic finite 333-column scores, Top-1/Top-5
                 consistency, stable tie order; writes verification.json
  official-eval  ONE-SHOT official DPD-Small evaluation. NOT AUTHORISED IN THIS TASK: refuses unless
                 R2V5_OFFICIAL_EVAL_AUTHORIZED=1 AND --owner-authorization is given; refuses if the output exists.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402
import torch  # noqa: E402

from research_r2_v4 import confirm, jrun  # noqa: E402
from research_r2_v4.common import env, now  # noqa: E402
from research_r2_v5 import final as fm  # noqa: E402

IMAGES = ROOT.parents[2] / "DPD_Small_dataset"
FIRST_RUN_OFFICIAL_H1 = {"WP1 independent ResNet50": 0.09945951, "WP2 ConvNeXt-Base": 0.01757603,
                         "Frozen CLIP ViT-L/14": 0.06617868, "M6 CLIP adapter": 0.05958218}


def cfg_of(c):
    cfg = jrun.JConfig(**c["config"])
    if cfg.sha() != c["config_sha256"] or asdict(cfg) != asdict(jrun.j_rows()[c["row"]]):
        raise fm.FinalError("frozen configuration differs from the validated J2 row")
    return cfg


def git(*a):
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def cmd_train(a):
    c = fm.load_config(); cfg = cfg_of(c)
    out = Path(a.out)
    if out.exists():
        sys.exit(f"{out} exists; final runs are never overwritten (use a new directory)")
    if git("status", "--porcelain"):
        sys.exit("worktree not clean; commit first")
    frame, _, _ = env()
    fm.load_payload(frame)
    access = jrun.ImageAccess(frame, a.images)
    rec = fm.train_final(cfg, c["seed"], frame, jrun.ClipBackend(), access, out / "run", device="cuda")
    print(json.dumps({k: rec[k] for k in ("status", "n_fitting", "seconds", "checkpoint_sha256", "features_sha256", "reload_max_abs_diff",
                                          "trainable_tensors_changed", "peak_gpu_gib_training", "peak_gpu_gib_total", "torch_seed")}, indent=1))


def cmd_verify(a):
    c = fm.load_config(); cfg = cfg_of(c)
    frame, _, _ = env()
    rows = fm.load_payload(frame)
    run = Path(a.run) / "run"
    rec = jrun.verify_receipt(run)
    if rec is None:
        raise fm.FinalError("no ok receipt")
    rep = {"verified": now(), "git_head": git("rev-parse", "HEAD"), "config_sha256": c["config_sha256"], "gamma": c["gamma"]}
    probs = confirm.receipt_problems(rec, c["row"], c["seed"], fm.FOLD_NAME, 0, cfg)
    if rec.get("n_fitting") != len(frame):
        probs.append("not trained on all training rows")
    if probs:
        raise fm.FinalError("; ".join(probs))
    rep["receipt"] = {k: rec[k] for k in ("checkpoint_sha256", "features_sha256", "packet_sha256", "torch_seed", "n_fitting", "seconds",
                                          "reload_max_abs_diff", "trainable_tensors_changed", "grad_checks", "peak_gpu_gib_training",
                                          "peak_gpu_gib_total", "epochs", "train_loss", "code")}
    rep["receipt_sha256"] = jrun.sha256_file(run / "receipt.json")
    feats, frows = jrun.load_feature_cache(run / "features.pt", rec["feature_cache_meta"])
    if frows != list(range(len(frame))) or not torch.isfinite(feats.float()).all():
        raise fm.FinalError("feature cache must hold finite features for every training row")
    # independent reload: fresh model + checkpoint, re-extract training rows 0..63 (same batch layout) == cache
    meta = {"row": c["row"], "seed": c["seed"], "fold": fm.FOLD_NAME, "config_sha256": c["config_sha256"]}
    model, tf = fm.load_model(run / "checkpoint.pt", cfg, frame, meta)
    acc = jrun.ImageAccess(frame, a.images); probe = list(range(64)); acc.allow("verify", probe)
    re_tr = jrun.ClipBackend.extract(model, tf, acc, probe, 8)
    d_reload = float((re_tr.float() - feats[:64].float()).abs().max())
    model, tf = fm.load_model(run / "checkpoint.pt", cfg, frame, meta)
    q = fm.extract_query(model, tf, frame.file.iloc[:64].tolist(), a.images)
    d_query = float((q - feats[:64, 0].float()).abs().max())
    rep["reload_reextract_max_abs_diff"], rep["query_path_max_abs_diff"] = d_reload, d_query
    if d_reload > 1e-2 or d_query > 1e-2:
        raise fm.FinalError(f"reloaded / query features differ from the cache ({d_reload}, {d_query})")
    # deterministic, finite scores on a training-image smoke set (sanity only: these images were trained on)
    smoke = list(range(0, len(frame), 25))
    s1 = fm.raw_scores(frame, feats, rows, feats[smoke, 0])
    s2 = fm.raw_scores(frame, feats, rows, feats[smoke, 0])
    if not np.array_equal(s1, s2):
        raise fm.FinalError("scores are not deterministic")
    cal = fm.calibrate(s1, rows, c["gamma"])
    t5 = fm.topk(cal, 5)
    full = np.argsort(-cal, axis=1, kind="stable")
    if not (np.array_equal(t5, full[:, :5]) and np.array_equal(t5[:, 0], cal.argmax(1))):
        raise fm.FinalError("Top-1 / Top-5 ordering inconsistent")
    truth = frame.comp_id.to_numpy()[smoke]
    m_g, m_0 = fm.metrics(cal, truth), fm.metrics(fm.calibrate(s1, rows, 0.0), truth)
    unseen = np.isin(t5[:, 0], fm.FINAL_UNSEEN_IDS)
    rep["smoke_training_images"] = {"n": len(smoke), "note": "training images (fitted); sanity of the pipeline, NOT a performance estimate",
                                    "at_gamma": m_g, "at_gamma0": m_0, "fraction_top1_unseen_id_at_gamma": float(unseen.mean()),
                                    "deterministic": True, "finite": True, "columns": int(cal.shape[1])}
    rep["candidate_payload_sha256"] = fm.FINAL_CANDIDATE_SHA256
    rep["frozen_hashes"] = {p: jrun.sha256_file(ROOT / p) for p in ("research_r2_v5/final.py", "research_r2_v5/final_config.json",
                                                                   "scripts/r2v5_final.py", "artifacts/r2/v5/final_candidates_cos30082_v1.json",
                                                                   "research_r2_v4/jrun.py", "research_r2/methods.py", "research_r2/runner.py")}
    rep["checkpoint_sha256_now"] = jrun.sha256_file(run / "checkpoint.pt")
    (Path(a.run) / "verification.json").write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps({k: rep[k] for k in ("reload_reextract_max_abs_diff", "query_path_max_abs_diff", "smoke_training_images", "checkpoint_sha256_now")},
                     indent=1, default=float))


def read_official_csv(path):
    """Assignment CSV: file, crop label, disease label, composition label, crop name, disease name, composition name."""
    with open(path, newline="") as fh:
        rows = [r for r in csv.reader(fh) if r]
    return [r[0] for r in rows], np.array([int(r[3]) for r in rows]), np.array([int(r[1]) for r in rows]), np.array([int(r[2]) for r in rows])


def cmd_official_eval(a):
    if os.environ.get("R2V5_OFFICIAL_EVAL_AUTHORIZED") != "1" or not (a.owner_authorization or "").strip():
        sys.exit("REFUSED: official evaluation needs explicit owner authorisation (R2V5_OFFICIAL_EVAL_AUTHORIZED=1 and "
                 "--owner-authorization). It was not authorised in the final sprint.")
    out = Path(a.out)
    if out.exists():
        sys.exit(f"{out} exists: the official evaluation is one-shot and never repeated or overwritten")
    c = fm.load_config(); cfg = cfg_of(c)
    frame, _, _ = env()
    rows = fm.load_payload(frame)
    run = Path(a.run) / "run"
    ver = json.loads((Path(a.run) / "verification.json").read_text())
    if jrun.sha256_file(run / "checkpoint.pt") != ver["checkpoint_sha256_now"]:
        raise fm.FinalError("checkpoint changed since verification")
    for p, h in ver["frozen_hashes"].items():
        if jrun.sha256_file(ROOT / p) != h:
            raise fm.FinalError(f"frozen file changed since verification: {p}")
    out.mkdir(parents=True)
    feats, _ = jrun.load_feature_cache(run / "features.pt", jrun.verify_receipt(run)["feature_cache_meta"])
    meta = {"row": c["row"], "seed": c["seed"], "fold": fm.FOLD_NAME, "config_sha256": c["config_sha256"]}
    res = {"label": "ONE-SHOT official DPD-Small evaluation of the frozen Second Run final model", "started": now(),
           "owner_authorization": a.owner_authorization, "gamma": c["gamma"], "checkpoint_sha256": ver["checkpoint_sha256_now"],
           "disclosure": "First Run official scores were already known to the project; this is a single final assessment, no tuning follows"}
    cal = {}
    for part, path in (("seen", a.seen_csv), ("unseen", a.unseen_csv)):
        files, comp, crop, dis = read_official_csv(path)
        model, tf = fm.load_model(run / "checkpoint.pt", cfg, frame, meta)
        q = fm.extract_query(model, tf, files, a.images)
        cal[part] = fm.calibrate(fm.raw_scores(frame, feats, rows, q), rows, c["gamma"])
        t = fm.topk(cal[part], 5)
        pc = np.array([rows[i]["crop_id"] for i in t[:, 0]]); pd_ = np.array([rows[i]["disease_id"] for i in t[:, 0]])
        res[part] = fm.metrics(cal[part], comp) | {"crop_top1": float(np.mean(pc == crop)), "disease_top1": float(np.mean(pd_ == dis)),
                                                   "per_crop_top1": {int(k): float(np.mean(t[crop == k, 0] == comp[crop == k])) for k in np.unique(crop)},
                                                   "per_disease_top1": {int(k): float(np.mean(t[dis == k, 0] == comp[dis == k])) for k in np.unique(dis)}}
        if part == "seen":
            cm = np.zeros((333, 333), dtype=np.int64)
            np.add.at(cm, (comp, t[:, 0]), 1)
            np.save(out / "seen_confusion_333x333.npy", cm)
        np.save(out / f"{part}_calibrated_scores.npy", cal[part].astype(np.float32))
    res["H@1"] = fm.harmonic(res["seen"]["top1"], res["unseen"]["top1"])
    res["H@5"] = fm.harmonic(res["seen"]["top5"], res["unseen"]["top5"])
    res["first_run_official_H@1_historical"] = FIRST_RUN_OFFICIAL_H1
    res["finished"] = now()
    (out / "official_evaluation.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("H@1", "H@5")} | {p: {k: res[p][k] for k in ("top1", "top5", "count")} for p in ("seen", "unseen")}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("train"); p.add_argument("--out", required=True); p.add_argument("--images", default=str(IMAGES))
    p = sub.add_parser("verify"); p.add_argument("--run", required=True); p.add_argument("--images", default=str(IMAGES))
    p = sub.add_parser("official-eval"); p.add_argument("--run", required=True); p.add_argument("--seen-csv", required=True)
    p.add_argument("--unseen-csv", required=True); p.add_argument("--out", required=True); p.add_argument("--owner-authorization")
    p.add_argument("--images", default=str(IMAGES))
    a = ap.parse_args()
    {"train": cmd_train, "verify": cmd_verify, "official-eval": cmd_official_eval}[a.cmd](a)


if __name__ == "__main__":
    main()
