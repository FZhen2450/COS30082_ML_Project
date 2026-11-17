"""Second Run final model (sprint, 17-18 Nov 2026): J2 (CLIP ViT-L/14, last 4 blocks fine-tuned) trained on all 8,000
authorised DPD-Small training images, with the unchanged D1 synthetic-composition downstream, scored over the
assignment's complete 333-composition candidate universe and calibrated with a FIXED development-derived gamma.

Pieces (all reuse v4 code unchanged):
  * training: research_r2_v4.jrun.run_fold on a full-data "fold" (every training row fits; no evaluation rows);
    checkpoint saved, reloaded and verified, training-image features extracted from the reloaded model;
  * candidates: the pinned First Run final candidate payload (cos30082-final-candidates-v1, sha256 ba17e941...,
    333 class rows: id, crop, disease, names; no per-image data), checked against the training CSV;
  * scorer: research_r2.methods.RECIPES["synth_composition"] with D1_CFG on a FoldData whose fitting set is all 8,000
    training images (both views) and whose candidates are the 333 payload rows; fit_mask = seen (323), the 10 unseen
    ids are unpenalised and get crop-shifted synthetic prototypes from their crop and disease (names only);
  * calibration: calibrated score = score - gamma * [candidate is seen], gamma frozen in final_config.json from the
    development folds only (J2 three seeds, pooled H_micro-optimal and the LOFO H_micro mean both = 8.75);
  * ranking: descending calibrated score, stable ties in ascending global id (First Run TIE_RULE).
Official-test images are never read here; the official evaluation entry refuses to run without explicit authorisation.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from research_r2.data import HEALTHY, sha256_file, sha256_json
from research_r2.candidates import POLICY
from research_r2.methods import RECIPES, FoldData, standardise
from research_r2_v4 import jrun

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "research_r2_v5" / "final_config.json"
PAYLOAD = ROOT / "artifacts" / "r2" / "v5" / "final_candidates_cos30082_v1.json"
FINAL_CANDIDATE_SHA256 = "ba17e94119e8b2b7a4b18603e25405d2f52bf4a93396973f80491800697eb62f"  # = evaluation/reporting.py
FINAL_UNSEEN_IDS = (39, 84, 107, 124, 133, 136, 144, 166, 276, 303)  # = evaluation/reporting.py FINAL_UNSEEN_IDS
FOLD_NAME = "FULL8000"


class FinalError(RuntimeError):
    """A fail-closed check of the final model failed."""


# ----------------------------------------------------------------------------- candidates
def load_payload(frame, path=PAYLOAD):
    """The 333-row candidate payload, verified: file sha256, ids 0..332 ascending, the 10 unseen ids are exactly the
    ids absent from training, every seen row equals the training CSV, every unseen crop and disease occurs in training."""
    if sha256_file(path) != FINAL_CANDIDATE_SHA256:
        raise FinalError("candidate payload sha256 differs from the pinned First Run final candidate hash")
    p = json.loads(Path(path).read_text())
    rows = p["rows"]
    if p.get("schema") != "cos30082-final-candidates-v1" or [r["composition_id"] for r in rows] != list(range(333)):
        raise FinalError("payload must hold 333 rows with ascending global ids 0..332")
    train = frame.drop_duplicates("comp_id").set_index("comp_id")
    unseen = tuple(sorted(set(range(333)) - set(int(c) for c in train.index)))
    if unseen != FINAL_UNSEEN_IDS or len(train) != 323:
        raise FinalError(f"ids absent from training {unseen} != declared unseen ids {FINAL_UNSEEN_IDS}")
    crops, dis = set(frame.crop_id), set(frame.disease_id)
    for r in rows:
        c = r["composition_id"]
        if c in train.index:
            t = train.loc[c]
            if (int(t.crop_id), int(t.disease_id)) != (r["crop_id"], r["disease_id"]):
                raise FinalError(f"payload row {c} disagrees with the training CSV")
        elif r["crop_id"] not in crops or r["disease_id"] not in dis:
            raise FinalError(f"unseen composition {c}: crop or disease never observed in training")
    return rows


# ----------------------------------------------------------------------------- training
def full_fold(frame):
    """Every training row fits; no evaluation rows; candidates = the 323 training compositions (run_fold contract)."""
    return {"name": FOLD_NAME, "fitting_rows": list(range(len(frame))), "seen_val_rows": [], "unseen_val_rows": [],
            "heldout_composition_ids": [], "candidate_policy": POLICY,
            "candidate_composition_ids": sorted(int(c) for c in frame.comp_id.unique())}


NO_OUTER = {"outer_exclusion_rows": [], "outer_seen_rows": []}  # the final model uses every training image


def train_final(cfg, seed, frame, backend, access, out_dir, device="cuda"):
    return jrun.run_fold(cfg, seed, full_fold(frame), 0, frame, NO_OUTER, backend, access, Path(out_dir), device)


# ----------------------------------------------------------------------------- scoring
def final_fold_data(frame, train_feats, rows, query):
    """FoldData: fitting = all training images (both views), candidates = 333 payload rows, query = [n, d] centre-view
    features of the images to score."""
    if train_feats.shape[0] != len(frame) or train_feats.ndim != 3:
        raise FinalError("training features must cover every training row, [8000, 2, d]")
    if not torch.isfinite(train_feats.float()).all() or not torch.isfinite(query.float()).all():
        raise FinalError("non-finite features")
    crops = sorted(frame.crop_id.unique()); dis = sorted(frame.disease_id.unique()); srcs = sorted(frame.source.unique())
    ci = {c: i for i, c in enumerate(crops)}; di = {d: i for i, d in enumerate(dis)}; si = {s: i for i, s in enumerate(srcs)}
    fd = FoldData(
        x_fit=train_feats, x_seen=query, x_uns=query[:1],
        fit_crop=torch.tensor([ci[c] for c in frame.crop_id]), fit_dis=torch.tensor([di[d] for d in frame.disease_id]),
        fit_comp_col=torch.tensor([int(c) for c in frame.comp_id]), fit_source=torch.tensor([si[s] for s in frame.source]),
        cand_crop=np.array([ci[r["crop_id"]] for r in rows]), cand_dis=np.array([di[r["disease_id"]] for r in rows]),
        fit_mask=np.array([r["composition_id"] not in FINAL_UNSEEN_IDS for r in rows]),
        n_crop=len(crops), n_dis=len(dis), n_source=len(srcs))
    fd.extra["cand_ids"] = np.arange(333)
    return standardise(fd)


def raw_scores(frame, train_feats, rows, query):
    """Uncalibrated D1 composition scores [n, 333] (columns = global ids 0..332). Deterministic: seed 0."""
    torch.manual_seed(0)
    fd = final_fold_data(frame, train_feats.float(), rows, query.float())
    s, _ = RECIPES["synth_composition"](fd, jrun.D1_CFG)
    s = s.double().numpy()
    if s.shape != (len(query), 333) or not np.isfinite(s).all():
        raise FinalError("scores must be finite with 333 columns")
    return s


def seen_mask(rows):
    return np.array([r["composition_id"] not in FINAL_UNSEEN_IDS for r in rows])


def calibrate(scores, rows, gamma):
    return np.asarray(scores, np.float64) - float(gamma) * seen_mask(rows)[None]


def topk(cal, k=5):
    """Global ids of the top-k candidates per row; stable ties -> ascending global id."""
    return np.argsort(-cal, axis=1, kind="stable")[:, :k]


def load_config(path=CONFIG):
    c = json.loads(Path(path).read_text())
    for k in ("row", "config_sha256", "seed", "gamma", "candidate_payload_sha256", "unseen_ids", "recipe", "d1_cfg"):
        if k not in c:
            raise FinalError(f"final_config.json lacks {k}")
    if c["candidate_payload_sha256"] != FINAL_CANDIDATE_SHA256 or tuple(c["unseen_ids"]) != FINAL_UNSEEN_IDS:
        raise FinalError("final_config.json candidate identity differs from the pinned payload")
    if c["recipe"] != "synth_composition" or c["d1_cfg"] != jrun.D1_CFG:
        raise FinalError("final_config.json downstream differs from the validated D1 recipe")
    return c


def load_model(ckpt, cfg, frame, expected_meta, backend=None, device="cuda"):
    """Fresh CLIP + the final checkpoint's trainable tensors and heads; keys and metadata verified (fail closed)."""
    backend = backend or jrun.ClipBackend()
    st = torch.load(ckpt, map_location=device, weights_only=False)
    meta = st.get("meta", {})
    for k, v in expected_meta.items():
        if meta.get(k) != v:
            raise FinalError(f"checkpoint metadata mismatch on {k!r}")
    model, tf, _ = backend.build(cfg, device)
    heads = jrun.Heads(backend.dim, frame.crop_id.nunique(), frame.disease_id.nunique(), extra=cfg.extra).to(device)
    bk, hk = jrun.checkpoint_keys(model, heads)
    if sorted(st["backbone"]) != bk or sorted(st["heads"]) != hk:
        raise FinalError("checkpoint keys differ from the declared trainable set")
    res = model.load_state_dict(st["backbone"], strict=False)
    if res.unexpected_keys or (set(bk) & set(res.missing_keys)):
        raise FinalError("checkpoint did not load every trainable key")
    heads.load_state_dict(st["heads"], strict=True)
    model.eval(); heads.eval()
    return model, tf


# ----------------------------------------------------------------------------- query images (evaluation only)
class QueryAccess:
    """Opens an explicit list of image files for SCORING ONLY (never training). Used by the guarded official
    evaluation and by the smoke test on training images. Duck-types jrun.ImageAccess for extract_layout."""

    def __init__(self, files, image_root):
        import pandas as pd
        self.frame = pd.DataFrame({"file": list(files)})
        self.root = Path(image_root)
        self.opened = 0

    def get(self, row):
        from PIL import Image
        self.opened += 1
        with Image.open(self.root / self.frame.file.iloc[int(row)]) as im:
            return im.convert("RGB")


def extract_query(model, tf, files, image_root, workers=8):
    """Centre-view features [n, d] of the listed images with the reloaded final checkpoint (v3 batch layout)."""
    from clip.model import convert_weights
    acc = QueryAccess(files, image_root)
    model.eval(); convert_weights(model)
    f = jrun.extract_layout(model.encode_image, tf, acc, list(range(len(files))), len(files), workers)
    return f[:, 0].float()


# ----------------------------------------------------------------------------- official input integrity (before inference)
OFFICIAL_COUNTS = {"seen": 2000, "unseen": 1448}  # assignment brief section 4.2
FROZEN_CHECKPOINT_SHA256 = "e9dbbdf3fc95ba791e8e8129ef38bbd85352a0974c3091d6b7667bd633abb1c9"


def read_assignment_csv(path):
    """Assignment metadata CSV, no header, exactly 7 columns: file, crop id, disease id, composition id, crop name,
    disease name, composition name. Any malformed row (including a header line) raises."""
    import csv
    out = []
    with open(path, newline="") as fh:
        for i, r in enumerate(csv.reader(fh)):
            if not r:
                continue
            if len(r) != 7:
                raise FinalError(f"{Path(path).name}: row {i} has {len(r)} columns, expected 7")
            try:
                crop, dis, comp = int(r[1]), int(r[2]), int(r[3])
            except ValueError:
                raise FinalError(f"{Path(path).name}: row {i} has non-integer labels") from None
            if not r[0].strip() or "/" in r[0] or "\\" in r[0] or r[0].startswith("."):
                raise FinalError(f"{Path(path).name}: row {i} has an invalid image file name")
            out.append({"file": r[0], "crop_id": crop, "disease_id": dis, "comp_id": comp})
    return out


def validate_official_inputs(parts, frame, rows, counts=OFFICIAL_COUNTS, image_root=None):
    """Fail-closed integrity checks on BOTH official parts before any model is loaded or any image is scored.
    parts: {"seen": [records], "unseen": [records]} from read_assignment_csv."""
    by_id = {r["composition_id"]: r for r in rows}
    allowed = {"seen": {c for c in by_id if c not in FINAL_UNSEEN_IDS}, "unseen": set(FINAL_UNSEEN_IDS)}
    if set(parts) != {"seen", "unseen"}:
        raise FinalError("exactly the seen and unseen parts are required")
    train_files = set(frame.file)
    summary = {}
    for part, recs in parts.items():
        if len(recs) != counts[part]:
            raise FinalError(f"{part}: {len(recs)} rows, expected {counts[part]}")
        files = [r["file"] for r in recs]
        if len(set(files)) != len(files):
            raise FinalError(f"{part}: duplicate image file names")
        if set(files) & train_files:
            raise FinalError(f"{part}: {len(set(files) & train_files)} file(s) also in the training CSV")
        for r in recs:
            c = r["comp_id"]
            if c not in by_id:
                raise FinalError(f"{part}: composition {c} outside the 333-candidate universe")
            if c not in allowed[part]:
                raise FinalError(f"{part}: composition {c} is not a {part} composition")
            if (r["crop_id"], r["disease_id"]) != (by_id[c]["crop_id"], by_id[c]["disease_id"]):
                raise FinalError(f"{part}: composition {c} crop/disease labels disagree with the candidate payload")
        if image_root is not None:
            missing = [f for f in files if not (Path(image_root) / f).is_file()]
            if missing:
                raise FinalError(f"{part}: {len(missing)} image file(s) missing, e.g. {missing[0]}")
        summary[part] = {"rows": len(recs), "compositions": len({r['comp_id'] for r in recs})}
    if {r["file"] for r in parts["seen"]} & {r["file"] for r in parts["unseen"]}:
        raise FinalError("seen and unseen parts share image files")
    return summary


def confusion(truth_ids, pred_ids, n=333):
    cm = np.zeros((n, n), dtype=np.int64)
    np.add.at(cm, (np.asarray(truth_ids), np.asarray(pred_ids)), 1)
    if cm.shape != (333, 333) or int(cm.sum()) != len(truth_ids):
        raise FinalError("confusion matrix must be 333 x 333 and count every image once")
    return cm


def metrics(cal, truth_ids, k=5):
    """Image-weighted Top-1 / Top-k on global ids (First Run evaluation/metrics.topk_accuracy contract)."""
    t = np.asarray(truth_ids, np.int64)
    order = topk(cal, k)
    return {"count": int(len(t)), "top1": float(np.mean(order[:, 0] == t)), f"top{k}": float(np.mean((order == t[:, None]).any(1)))}


def harmonic(s, u):
    return 0.0 if s + u == 0 else 2 * s * u / (s + u)


def provenance(path):
    return {"path_name": Path(path).name, "sha256": sha256_file(path)}


__all__ = ["load_payload", "full_fold", "train_final", "raw_scores", "calibrate", "topk", "metrics", "harmonic",
           "load_config", "extract_query", "QueryAccess", "FinalError", "FINAL_UNSEEN_IDS", "FINAL_CANDIDATE_SHA256",
           "HEALTHY", "sha256_json"]
