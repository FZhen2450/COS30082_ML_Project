"""Train-only three-way development data; no test loading, scoring or training."""

import hashlib
import json
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader

from data_utils import (
    DEFAULT_CSVS, SPLIT_VERSION, load_metadata, make_validation_split, validate_metadata,
)
from dataset import DPDataset, seed_everything, seed_worker


@dataclass(frozen=True)
class DevelopmentSplit:
    """Original labels and one candidate universe for both validation folds.

    Treat the contained DataFrames/mappings as read-only. Frozen field bindings
    do not make pandas objects immutable; DPDataset copies each frame.
    """

    fit: pd.DataFrame
    val_seen: pd.DataFrame
    val_unseen: pd.DataFrame
    mappings: dict[str, Any]
    candidate_ids: tuple[int, ...]
    candidate_pairs: tuple[tuple[int, int], ...]
    source_csv: str
    source_sha256: str
    candidate_sha256: str
    seed: int
    heldout_fraction: float
    seen_fraction: float
    split_version: str = f"{SPLIT_VERSION}/three-way-v1"


def _fraction(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not 0 < value < 1:
        raise ValueError(f"{name} must be a finite fraction between zero and one")
    return float(value)


def _training_mappings(frame: pd.DataFrame) -> dict[str, Any]:
    """Build only training-derived maps; the combined train/test wrapper is unsuitable."""
    mappings = {}
    for kind in ("crop", "disease"):
        pairs = frame[[f"{kind}_id", f"{kind}_name"]].drop_duplicates()
        pairs = pairs.sort_values(f"{kind}_id")
        id_to_name = {int(key): str(name) for key, name in pairs.itertuples(index=False, name=None)}
        mappings[f"{kind}_id_to_name"] = id_to_name
        mappings[f"{kind}_name_to_id"] = {name: key for key, name in id_to_name.items()}
        mappings[f"num_{kind}s"] = len(id_to_name)
    pairs = frame[["composition_id", "crop_id", "disease_id", "composition_name"]]
    pairs = pairs.drop_duplicates().sort_values("composition_id")
    mappings["composition_id_to_pair"] = {
        int(row.composition_id): (int(row.crop_id), int(row.disease_id))
        for row in pairs.itertuples()
    }
    mappings["composition_id_to_name"] = {
        int(row.composition_id): str(row.composition_name) for row in pairs.itertuples()
    }
    ids = tuple(mappings["composition_id_to_pair"])
    mappings["composition_id_to_index"] = {comp: index for index, comp in enumerate(ids)}
    mappings["composition_index_to_id"] = dict(enumerate(ids))
    mappings["num_compositions"] = len(ids)
    return mappings


def _check_partitions(
    source: pd.DataFrame, fit: pd.DataFrame, seen: pd.DataFrame, unseen: pd.DataFrame,
) -> None:
    combined = pd.concat([fit, seen, unseen], ignore_index=True)
    if (combined.image_name.duplicated().any()
            or set(combined.image_name) != set(source.image_name)):
        raise ValueError("Development partitions must be disjoint and cover training filenames")
    if not source.sort_values("image_name").reset_index(drop=True).equals(
        combined.sort_values("image_name").reset_index(drop=True)
    ):
        raise ValueError("Development partitions changed original metadata rows or label types")
    fit_ids, seen_ids, unseen_ids = (set(frame.composition_id) for frame in (fit, seen, unseen))
    if not seen_ids <= fit_ids or unseen_ids & (fit_ids | seen_ids):
        raise ValueError("Seen coverage or held-out composition isolation failed")
    for column in ("crop_id", "disease_id"):
        if set(fit[column]) != set(source[column]):
            raise ValueError(f"Fitting partition lost training components: {column}")


def build_development_split(
    train_csv: str | Path = DEFAULT_CSVS[0],
    *,
    seed: int = 42,
    heldout_fraction: float = 0.2,
    seen_fraction: float = 0.2,
) -> DevelopmentSplit:
    """Compose existing train-only splits without reading images or writing files.

    The default DPD-Small source yields 4868 fit / 1238 seen / 1894 unseen
    images. heldout_fraction counts compositions; seen_fraction is classwise
    image sampling within the retained portion. Synthetic training CSVs are
    supported for tests. Known official test CSV paths are rejected.
    """
    if isinstance(seed, bool) or not isinstance(seed, Integral) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    seed = int(seed)
    heldout_fraction = _fraction(heldout_fraction, "heldout_fraction")
    seen_fraction = _fraction(seen_fraction, "seen_fraction")
    path = Path(train_csv)
    if path.name in DEFAULT_CSVS[1:] or path.resolve().name in DEFAULT_CSVS[1:]:
        raise ValueError("Official test metadata cannot be used as development training data")
    before_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    source = load_metadata(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != before_hash:
        raise ValueError("Training CSV changed while loading")
    # Empty control frames reuse WP1 validation without loading either test CSV.
    empty = source.iloc[:0]
    validate_metadata(source, empty, empty)
    retained, unseen = make_validation_split(
        source, val_split=heldout_fraction, seed=seed, mode="heldout_composition"
    )
    fit, seen = make_validation_split(
        retained, val_split=seen_fraction, seed=seed, mode="stratified"
    )
    _check_partitions(source, fit, seen, unseen)
    mappings = _training_mappings(source)
    ids = tuple(mappings["composition_id_to_pair"])
    pairs = tuple(mappings["composition_id_to_pair"][comp] for comp in ids)
    records = [[comp, *pair] for comp, pair in zip(ids, pairs)]
    candidate_hash = hashlib.sha256(
        json.dumps(records, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return DevelopmentSplit(
        fit=fit, val_seen=seen, val_unseen=unseen, mappings=mappings,
        candidate_ids=ids, candidate_pairs=pairs, source_csv=str(path.resolve()),
        source_sha256=before_hash, candidate_sha256=candidate_hash, seed=seed,
        heldout_fraction=heldout_fraction, seen_fraction=seen_fraction,
    )


def get_development_dataloaders(
    split: DevelopmentSplit,
    img_dir: str | Path = "DPD_Small_dataset",
    *,
    batch_size: int = 16,
    image_size: int = 224,
    augment_fit: bool = True,
    as_dict_targets: bool = True,
    normalize_imagenet: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Return fit/seen-validation/unseen-validation loaders, with no iteration.

    Reuses DPDataset and its transforms. Only fitting shuffles/augments; both
    validation loaders use the same split candidate_ids/candidate_pairs.
    Construct once per run: seeding follows the existing helper's global RNG
    behavior and does not promise deterministic CUDA kernels.
    """
    if not isinstance(split, DevelopmentSplit):
        raise TypeError("split must be a DevelopmentSplit")
    for name, value, minimum in (
        ("batch_size", batch_size, 1), ("image_size", image_size, 1),
        ("num_workers", num_workers, 0),
    ):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    image_dir = Path(img_dir)
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    seed_everything(split.seed)
    loaders = []
    for index, frame in enumerate((split.fit, split.val_seen, split.val_unseen)):
        dataset = DPDataset(
            frame, image_dir, image_size=int(image_size), as_dict_targets=as_dict_targets,
            augment=augment_fit and index == 0, normalize_imagenet=normalize_imagenet,
        )
        loaders.append(DataLoader(
            dataset, batch_size=int(batch_size), shuffle=index == 0,
            num_workers=int(num_workers), pin_memory=pin_memory,
            worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(split.seed + index),
        ))
    return tuple(loaders)
