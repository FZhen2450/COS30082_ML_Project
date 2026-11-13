"""Training-source loading with an explicit official-test refusal guard."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

TRAIN_CSV_SHA256 = "9f1b729ab08f120c9e8d3a5681aa1ccbdf1d9210f9aaa6143959363dcb106a7d"
COLUMNS = ["file", "crop_id", "disease_id", "comp_id", "crop", "disease", "comp"]
FORBIDDEN_CSV_NAMES = ("DPD_Small_test_seen.csv", "DPD_Small_test_unseen.csv")
HEALTHY = "healthy"


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha256_json(value) -> str:
    """Canonical JSON digest (sorted keys, compact separators, UTF-8)."""
    blob = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def assert_not_official_test(path) -> None:
    name = Path(path).name
    if name in FORBIDDEN_CSV_NAMES or "test" in name.lower():
        raise PermissionError(f"Round 2 development code must not read official test metadata: {name}")


def load_training_source(csv_path, expected_sha256: str | None = TRAIN_CSV_SHA256) -> pd.DataFrame:
    """Load the 8,000-row training CSV only; verify its historical byte hash."""
    assert_not_official_test(csv_path)
    if expected_sha256 is not None:
        actual = sha256_file(csv_path)
        if actual != expected_sha256:
            raise ValueError(f"training CSV hash mismatch: {actual} != {expected_sha256}")
    frame = pd.read_csv(csv_path, header=None, names=COLUMNS)
    for col in ("crop_id", "disease_id", "comp_id"):
        frame[col] = frame[col].astype(int)
    if frame.file.duplicated().any():
        raise ValueError("duplicate training filenames")
    frame["source"] = frame.file.str.split("_").str[0]
    frame["healthy"] = frame.disease.eq(HEALTHY)
    frame = frame.reset_index(drop=True)
    frame["row"] = frame.index.astype(int)
    return frame


def candidate_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Ordered training-source candidates: one row per composition, ascending global ID."""
    table = (frame[["comp_id", "crop_id", "disease_id", "crop", "disease", "comp"]]
             .drop_duplicates().sort_values("comp_id").reset_index(drop=True))
    if table.comp_id.duplicated().any():
        raise ValueError("composition ID maps to more than one crop/disease pair")
    return table
