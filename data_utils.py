"""Framework-independent metadata audits and versioned train-only splits."""

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import pandas as pd
from PIL import Image

COLUMN_NAMES = ["image_name", "crop_id", "disease_id", "composition_id",
                "crop_name", "disease_name", "composition_name"]
DEFAULT_CSVS = ("DPD_Small_train.csv", "DPD_Small_test_seen.csv", "DPD_Small_test_unseen.csv")
SPLIT_VERSION = "pytorch-v1"


def load_metadata(path):
    frame = pd.read_csv(path, header=None)
    if frame.shape[1] != len(COLUMN_NAMES) or frame.empty:
        raise ValueError(f"Expected non-empty seven-column metadata: {path}")
    frame.columns = COLUMN_NAMES
    if frame.isna().any().any():
        raise ValueError(f"Missing metadata values: {path}")
    for col in ("crop_id", "disease_id", "composition_id"):
        numeric = pd.to_numeric(frame[col], errors="raise")
        if ((numeric < 0) | (numeric % 1 != 0)).any():
            raise ValueError(f"Invalid {col}: {path}")
        frame[col] = numeric.astype("int64")
    for name in frame.image_name:
        if not isinstance(name, str) or name in (".", "..") or any(c in name for c in ("/", "\\", ":")):
            raise ValueError(f"Expected flat image filename, got {name!r}")
    return frame


def validate_metadata(train, seen, unseen):
    frames = [train, seen, unseen]
    for frame in frames:
        if frame.image_name.duplicated().any():
            raise ValueError("Duplicate image names within a split")
    for i in range(3):
        for j in range(i + 1, 3):
            if set(frames[i].image_name) & set(frames[j].image_name):
                raise ValueError("Image filename overlap across official splits")
    all_rows = pd.concat(frames)
    for kind in ("crop", "disease", "composition"):
        pairs = all_rows[[f"{kind}_id", f"{kind}_name"]].drop_duplicates()
        if pairs[f"{kind}_id"].duplicated().any() or pairs[f"{kind}_name"].duplicated().any():
            raise ValueError(f"Inconsistent {kind} ID/name mapping")
    pairs = all_rows[["composition_id", "crop_id", "disease_id"]].drop_duplicates()
    if pairs.composition_id.duplicated().any() or pairs.duplicated(["crop_id", "disease_id"]).any():
        raise ValueError("Inconsistent composition/component mapping")
    if set(train.composition_id) & set(unseen.composition_id):
        raise ValueError("Official unseen compositions appear in training")
    if not set(seen.composition_id) <= set(train.composition_id):
        raise ValueError("Seen-test contains compositions absent from training")
    for col in ("crop_id", "disease_id"):
        if set(train[col]) != set(range(train[col].nunique())):
            raise ValueError(f"Training {col} must be contiguous and start at zero")
        if not set(all_rows[col]) <= set(train[col]):
            raise ValueError(f"Test {col} absent from training")
    return {"split_sizes": [len(f) for f in frames],
            "num_crops": int(train.crop_id.nunique()),
            "num_diseases": int(train.disease_id.nunique()),
            "num_seen_compositions": int(train.composition_id.nunique()),
            "num_unseen_compositions": int(unseen.composition_id.nunique()),
            "singleton_compositions": int((train.composition_id.value_counts() == 1).sum()),
            "filename_overlap": False, "seen_unseen_composition_overlap": False}


def make_validation_split(frame, val_split=0.2, seed=42, mode="stratified"):
    """Classwise split keeps singletons in train; heldout removes whole classes.

    Classwise rounding makes the realized image fraction approximate.
    heldout mode interprets val_split as a fraction of composition classes.
    """
    if not 0 < val_split < 1:
        raise ValueError("val_split must be between 0 and 1")
    frame = frame.reset_index(drop=True)
    rng = random.Random(seed)
    groups = {int(key): list(group.index) for key, group in frame.groupby("composition_id", sort=True)}
    val_indices = []
    if mode == "stratified":
        for indices in groups.values():
            rng.shuffle(indices)
            n = min(len(indices) - 1, max(1, round(len(indices) * val_split))) if len(indices) > 1 else 0
            val_indices.extend(indices[:n])
    elif mode == "heldout_composition":
        candidates = list(groups)
        rng.shuffle(candidates)
        remaining_crop = Counter(frame.crop_id)
        remaining_disease = Counter(frame.disease_id)
        target = max(1, round(len(groups) * val_split))
        heldout = []
        for comp_id in candidates:
            group = frame.loc[groups[comp_id]]
            crops, diseases = Counter(group.crop_id), Counter(group.disease_id)
            if all(remaining_crop[k] > n for k, n in crops.items()) and all(remaining_disease[k] > n for k, n in diseases.items()):
                heldout.append(comp_id)
                val_indices.extend(groups[comp_id])
                remaining_crop.subtract(crops)
                remaining_disease.subtract(diseases)
                if len(heldout) == target:
                    break
        if len(heldout) < target:
            raise ValueError(f"Only {len(heldout)} valid heldout classes; requested {target}. Reduce val_split.")
    else:
        raise ValueError(f"Unknown split mode: {mode}")
    validation = frame.loc[sorted(val_indices)].reset_index(drop=True)
    training = frame.drop(index=val_indices).reset_index(drop=True)
    if training.empty or validation.empty:
        raise ValueError("Split produced an empty training or validation set")
    for col in ("crop_id", "disease_id"):
        if set(frame[col]) != set(training[col]):
            raise ValueError(f"Split removed a training component: {col}")
    return training, validation


def save_split_manifest(train, val, source_csv, output_dir, mode, seed, fraction):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "split_version": SPLIT_VERSION, "source_csv": Path(source_csv).name,
        "source_sha256": hashlib.sha256(Path(source_csv).read_bytes()).hexdigest(),
        "mode": mode, "seed": seed, "requested_validation_fraction": fraction,
        "fraction_unit": "compositions" if mode == "heldout_composition" else "images_per_class",
        "train_images": train.image_name.tolist(), "validation_images": val.image_name.tolist(),
        "train_composition_ids": sorted(map(int, train.composition_id.unique())),
        "validation_composition_ids": sorted(map(int, val.composition_id.unique())),
        "train_size": len(train), "validation_size": len(val),
    }
    path = output / f"{mode}_seed{seed}.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


def build_class_mappings(train_csv=DEFAULT_CSVS[0], unseen_csv=DEFAULT_CSVS[2]):
    train, unseen = load_metadata(train_csv), load_metadata(unseen_csv)
    mappings = {}
    for kind in ("crop", "disease"):
        pairs = train[[f"{kind}_id", f"{kind}_name"]].drop_duplicates().sort_values(f"{kind}_id")
        mappings[f"{kind}_id_to_name"] = dict(zip(pairs.iloc[:, 0], pairs.iloc[:, 1]))
        mappings[f"{kind}_name_to_id"] = {v: k for k, v in mappings[f"{kind}_id_to_name"].items()}
        mappings[f"num_{kind}s"] = len(pairs)
    for prefix, frame in (("seen", train), ("unseen", unseen)):
        pairs = frame[["composition_id", "composition_name"]].drop_duplicates().sort_values("composition_id")
        ids = list(map(int, pairs.composition_id))
        mappings[f"{prefix}_comp_id_to_name"] = dict(zip(ids, pairs.composition_name))
        mappings[f"{prefix}_comp_id_to_index"] = {value: index for index, value in enumerate(ids)}
        mappings[f"{prefix}_comp_index_to_id"] = dict(enumerate(ids))
        mappings[f"num_{prefix}_compositions"] = len(ids)
    # Evaluation metadata only. Never use unseen images for training/model selection.
    pairs = pd.concat([train, unseen])[["composition_id", "crop_id", "disease_id"]].drop_duplicates().sort_values("composition_id")
    ids = list(map(int, pairs.composition_id))
    mappings["composition_id_to_pair"] = {int(r.composition_id): (int(r.crop_id), int(r.disease_id)) for r in pairs.itertuples()}
    mappings["composition_id_to_index"] = {value: index for index, value in enumerate(ids)}
    mappings["composition_index_to_id"] = dict(enumerate(ids))
    mappings["num_compositions"] = len(ids)
    return mappings


def audit_images(frames, img_dir):
    """Fully decode every image, including images after position 50 in each split."""
    failures, sizes = [], []
    for split, frame in zip(("train", "test_seen", "test_unseen"), frames):
        for name in frame.image_name:
            try:
                with Image.open(Path(img_dir) / name) as image:
                    image.load()
                    sizes.append(image.size)
            except (OSError, ValueError) as exc:
                failures.append({"split": split, "image_name": name, "error": str(exc)})
    return failures, sizes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=".")
    parser.add_argument("--output-dir", default="artifacts/data")
    parser.add_argument("--check-images", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    root = Path(args.data_dir)
    frames = [load_metadata(root / name) for name in DEFAULT_CSVS]
    report = validate_metadata(*frames)
    report["split_version"] = SPLIT_VERSION
    report["validation_splits"] = {}
    for mode in ("stratified", "heldout_composition"):
        train, val = make_validation_split(frames[0], seed=args.seed, mode=mode)
        path = save_split_manifest(train, val, root / DEFAULT_CSVS[0], args.output_dir, mode, args.seed, 0.2)
        report["validation_splits"][mode] = {"train": len(train), "validation": len(val), "manifest": str(path)}
    report["image_audit"] = {"status": "not_run"}
    failed = False
    if args.check_images:
        failures, sizes = audit_images(frames, root / "DPD_Small_dataset")
        report["image_audit"] = {"status": "failed" if failures else "passed", "decoded": len(sizes), "failures": failures}
        failed = bool(failures)
    output = Path(args.output_dir) / "data_audit.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
