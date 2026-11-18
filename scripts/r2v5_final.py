"""Second Run final model CLI (J2 full-data + D1 downstream + frozen gamma over the 333-candidate universe).

  train          python scripts/r2v5_final.py train --out <E>/v5/final_J2_full8000
                 trains the frozen configuration (research_r2_v5/final_config.json) on all 8,000 training images
                 (one GPU job); clean worktree required; refuses an existing output directory
  verify         python scripts/r2v5_final.py verify --run <E>/v5/final_J2_full8000
                 receipt + file hashes, configuration, torch seed, gradient/AMP/frozen/reload checks, feature cache
                 (8,000 rows, finite), candidate payload, independent checkpoint reload (re-extracted training features
                 must equal the cache), query-path consistency, deterministic finite 333-column scores, Top-1/Top-5
                 consistency, stable tie order; writes verification_<head>.json (never overwritten)
  rehearse       official-size pipeline rehearsal on training images (2,000 + 1,448); not a performance estimate
  official-eval  ONE-SHOT official DPD-Small evaluation (--run, --verification, --seen-csv, --unseen-csv, --out). Refuses unless
                 R2V5_OFFICIAL_EVAL_AUTHORIZED=1 AND --owner-authorization is given; refuses if the output exists.
"""
from __future__ import annotations

import argparse
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
    rep["frozen_hashes"] = frozen_hashes()
    rep["checkpoint_sha256_now"] = jrun.sha256_file(run / "checkpoint.pt")
    rep["features_sha256_now"] = jrun.sha256_file(run / "features.pt")
    if rep["checkpoint_sha256_now"] != fm.FROZEN_CHECKPOINT_SHA256 or rep["features_sha256_now"] != rec["features_sha256"]:
        raise fm.FinalError("checkpoint or feature cache differs from the frozen candidate / its receipt")
    out = Path(a.run) / f"verification_{rep['git_head'][:7]}.json"
    if out.exists():
        sys.exit(f"{out.name} exists; verification records are never overwritten")
    out.write_text(json.dumps(rep, indent=1, default=float))
    print(out.name)
    print(json.dumps({k: rep[k] for k in ("reload_reextract_max_abs_diff", "query_path_max_abs_diff", "smoke_training_images", "checkpoint_sha256_now",
                                          "features_sha256_now")}, indent=1, default=float))


FROZEN_FILES = ("research_r2_v5/final.py", "research_r2_v5/final_config.json", "scripts/r2v5_final.py",
                "artifacts/r2/v5/final_candidates_cos30082_v1.json", "research_r2_v4/jrun.py", "research_r2/methods.py",
                "research_r2/runner.py", "research_r2/data.py", "research_r2/candidates.py", "DPD_Small_train.csv")


def frozen_hashes():
    return {p: jrun.sha256_file(ROOT / p) for p in FROZEN_FILES}


def check_frozen(run_dir, verification):
    """Every frozen identity, before anything else is read: checkpoint (pinned), feature cache (= receipt), receipt,
    configuration, code and payload (= the verification record)."""
    run = Path(run_dir) / "run"
    ver = json.loads(Path(verification).read_text())
    rec = jrun.verify_receipt(run)  # raises if checkpoint / features / packet differ from the receipt
    if rec is None:
        raise fm.FinalError("no ok receipt")
    if jrun.sha256_file(run / "checkpoint.pt") != fm.FROZEN_CHECKPOINT_SHA256 or ver.get("checkpoint_sha256_now") != fm.FROZEN_CHECKPOINT_SHA256:
        raise fm.FinalError("checkpoint is not the frozen candidate")
    if jrun.sha256_file(run / "features.pt") != rec["features_sha256"] or ver.get("features_sha256_now") != rec["features_sha256"]:
        raise fm.FinalError("feature cache is not the verified one")
    if jrun.sha256_file(run / "receipt.json") != ver.get("receipt_sha256"):
        raise fm.FinalError("receipt changed since verification")
    now_h = frozen_hashes()
    if ver.get("frozen_hashes") != now_h:
        raise fm.FinalError(f"frozen files changed since verification: {sorted(k for k in now_h if ver.get('frozen_hashes', {}).get(k) != now_h[k])}")
    return rec


def evaluate_parts(parts, frame, rows, run, cfg, c, images, out):
    """Inference + scoring for already-validated parts {name: records}. Same code for the rehearsal and the official run."""
    feats, _ = jrun.load_feature_cache(run / "features.pt", jrun.verify_receipt(run)["feature_cache_meta"])
    meta = {"row": c["row"], "seed": c["seed"], "fold": fm.FOLD_NAME, "config_sha256": c["config_sha256"]}
    res = {}
    for part, recs in parts.items():
        files = [r["file"] for r in recs]
        comp = np.array([r["comp_id"] for r in recs]); crop = np.array([r["crop_id"] for r in recs]); dis = np.array([r["disease_id"] for r in recs])
        t0 = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        model, tf = fm.load_model(run / "checkpoint.pt", cfg, frame, meta)
        q = fm.extract_query(model, tf, files, images)
        del model
        raw = fm.raw_scores(frame, feats, rows, q)
        cal = fm.calibrate(raw, rows, c["gamma"])
        t = fm.topk(cal, 5)
        if cal.shape != (len(files), 333) or not np.isfinite(cal).all() or not np.array_equal(t[:, 0], cal.argmax(1)):
            raise fm.FinalError(f"{part}: invalid score matrix or ranking")
        pc = np.array([rows[i]["crop_id"] for i in t[:, 0]]); pd_ = np.array([rows[i]["disease_id"] for i in t[:, 0]])
        res[part] = fm.metrics(cal, comp) | {
            "crop_top1": float(np.mean(pc == crop)), "disease_top1": float(np.mean(pd_ == dis)),
            "fraction_top1_unseen_id": float(np.mean(np.isin(t[:, 0], fm.FINAL_UNSEEN_IDS))),
            "per_crop_top1": {int(k): float(np.mean(t[crop == k, 0] == comp[crop == k])) for k in np.unique(crop)},
            "per_disease_top1": {int(k): float(np.mean(t[dis == k, 0] == comp[dis == k])) for k in np.unique(dis)},
            "seconds": time.time() - t0, "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None}
        if part == "seen":
            np.save(out / "seen_confusion_333x333.npy", fm.confusion(comp, t[:, 0]))
        np.save(out / f"{part}_calibrated_scores.npy", cal.astype(np.float32))
        np.save(out / f"{part}_top5_ids.npy", t.astype(np.int16))
    res["H@1"] = fm.harmonic(res["seen"]["top1"], res["unseen"]["top1"])
    res["H@5"] = fm.harmonic(res["seen"]["top5"], res["unseen"]["top5"])
    return res


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
    rec = check_frozen(a.run, a.verification)                       # 1. frozen identities
    parts = {"seen": fm.read_assignment_csv(a.seen_csv), "unseen": fm.read_assignment_csv(a.unseen_csv)}
    summary = fm.validate_official_inputs(parts, frame, rows, image_root=a.images)  # 2. both parts, before any inference
    out.mkdir(parents=True)
    res = {"label": "ONE-SHOT official DPD-Small evaluation of the frozen Second Run final model", "started": now(),
           "owner_authorization": a.owner_authorization, "gamma": c["gamma"], "checkpoint_sha256": fm.FROZEN_CHECKPOINT_SHA256,
           "features_sha256": rec["features_sha256"], "verification": Path(a.verification).name, "git_head": git("rev-parse", "HEAD"),
           "input_summary": summary, "csv_sha256": {"seen": jrun.sha256_file(a.seen_csv), "unseen": jrun.sha256_file(a.unseen_csv)},
           "disclosure": "First Run official scores were already known to the project; this is a single final assessment, no tuning follows"}
    res |= evaluate_parts(parts, frame, rows, Path(a.run) / "run", cfg, c, a.images, out)  # 3. inference
    res["first_run_official_H@1_historical"] = FIRST_RUN_OFFICIAL_H1
    res["finished"] = now()
    (out / "official_evaluation.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("H@1", "H@5")} | {p: {k: res[p][k] for k in ("top1", "top5", "count")} for p in ("seen", "unseen")}, indent=1))


def cmd_rehearse(a):
    """Official-size pipeline rehearsal on TRAINING images only (2,000 + 1,448 rows): same frozen checks, same inference
    and output code, relaxed label rules (training images carry seen labels). Accuracy is NOT a performance estimate."""
    out = Path(a.out)
    if out.exists():
        sys.exit(f"{out} exists")
    c = fm.load_config(); cfg = cfg_of(c)
    frame, _, _ = env()
    rows = fm.load_payload(frame)
    check_frozen(a.run, a.verification)
    recs = [{"file": f, "crop_id": int(cr), "disease_id": int(d), "comp_id": int(cp)}
            for f, cr, d, cp in zip(frame.file, frame.crop_id, frame.disease_id, frame.comp_id)]
    parts = {"seen": recs[:fm.OFFICIAL_COUNTS["seen"]], "unseen": recs[fm.OFFICIAL_COUNTS["seen"]:fm.OFFICIAL_COUNTS["seen"] + fm.OFFICIAL_COUNTS["unseen"]]}
    out.mkdir(parents=True)
    t0 = time.time()
    res = evaluate_parts(parts, frame, rows, Path(a.run) / "run", cfg, c, a.images, out)
    first = {p: (np.load(out / f"{p}_top5_ids.npy"), np.load(out / f"{p}_calibrated_scores.npy")) for p in parts}
    res2 = evaluate_parts(parts, frame, rows, Path(a.run) / "run", cfg, c, a.images, out)  # determinism at full size
    same = all(np.array_equal(first[p][0], np.load(out / f"{p}_top5_ids.npy")) and np.array_equal(first[p][1], np.load(out / f"{p}_calibrated_scores.npy"))
               for p in parts)
    strip = lambda r: {p: {k: v for k, v in r[p].items() if k not in ("seconds", "peak_gpu_gib")} for p in parts} | {"H@1": r["H@1"], "H@5": r["H@5"]}  # noqa: E731
    rep = {"label": "REHEARSAL on training images (already fitted): pipeline, sizes, memory and determinism only; NOT a performance estimate",
           "sizes": {p: len(v) for p, v in parts.items()}, "deterministic_repeat": bool(same and strip(res) == strip(res2)),
           "total_seconds": time.time() - t0,
           "result": {p: {k: res[p][k] for k in ("count", "top1", "top5", "fraction_top1_unseen_id", "seconds", "peak_gpu_gib")} for p in parts},
           "confusion_shape": list(np.load(out / "seen_confusion_333x333.npy").shape), "H@1_pipeline_only": res["H@1"]}
    (out / "rehearsal.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    if not rep["deterministic_repeat"]:
        sys.exit("rehearsal not deterministic")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("train"); p.add_argument("--out", required=True); p.add_argument("--images", default=str(IMAGES))
    p = sub.add_parser("verify"); p.add_argument("--run", required=True); p.add_argument("--images", default=str(IMAGES))
    p = sub.add_parser("rehearse"); p.add_argument("--run", required=True); p.add_argument("--verification", required=True)
    p.add_argument("--out", required=True); p.add_argument("--images", default=str(IMAGES))
    p = sub.add_parser("official-eval"); p.add_argument("--run", required=True); p.add_argument("--verification", required=True)
    p.add_argument("--seen-csv", required=True); p.add_argument("--unseen-csv", required=True); p.add_argument("--out", required=True)
    p.add_argument("--owner-authorization"); p.add_argument("--images", default=str(IMAGES))
    a = ap.parse_args()
    {"train": cmd_train, "verify": cmd_verify, "rehearse": cmd_rehearse, "official-eval": cmd_official_eval}[a.cmd](a)


if __name__ == "__main__":
    main()
