"""DPD-Small PyTorch loaders. WP1 co-owners: Andy and Fa Zhen.

Targets retain official global IDs, including non-contiguous composition IDs.
Only training images receive augmentation. Official tests are never split.
"""

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

from data_utils import (
    COLUMN_NAMES, build_class_mappings, load_metadata, make_validation_split,
    save_split_manifest, validate_metadata,
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def seed_everything(seed=42):
    """Seed Python, NumPy and Torch; CUDA kernels may still be nondeterministic."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def get_transforms(image_size=224, augment=False, normalize_imagenet=True):
    if image_size <= 0:
        raise ValueError("image_size must be positive")
    steps = [v2.ToImage(), v2.Resize((image_size, image_size), antialias=True)]
    if augment:
        steps += [
            v2.RandomHorizontalFlip(), v2.RandomVerticalFlip(),
            v2.RandomAffine(degrees=18, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        ]
    steps += [v2.ToDtype(torch.float32, scale=True)]
    if normalize_imagenet:
        steps += [v2.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return v2.Compose(steps)


class DPDataset(Dataset):
    """Return float32 CHW RGB images and int64 crop/disease/composition targets."""

    def __init__(self, frame, img_dir, image_size=224, as_dict_targets=True,
                 augment=False, normalize_imagenet=True, transform=None):
        self.frame = frame.reset_index(drop=True).copy()
        self.img_dir = Path(img_dir)
        self.as_dict_targets = as_dict_targets
        self.transform = transform if transform is not None else get_transforms(image_size, augment, normalize_imagenet)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        path = self.img_dir / row.image_name
        try:
            with Image.open(path) as image:
                image = self.transform(image.convert("RGB"))
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"Cannot decode image: {path}") from exc
        values = tuple(torch.tensor(int(row[name]), dtype=torch.long)
                       for name in ("crop_id", "disease_id", "composition_id"))
        targets = dict(zip(("crop", "disease", "composition"), values))
        return image, targets if self.as_dict_targets else values


def get_dataloaders(
    data_dir=".", img_dir="DPD_Small_dataset",
    train_csv="DPD_Small_train.csv", test_seen_csv="DPD_Small_test_seen.csv",
    test_unseen_csv="DPD_Small_test_unseen.csv", val_split=0.2,
    batch_size=32, image_size=224, seed=42, as_dict_targets=True,
    augment_train=True, num_workers=0, pin_memory=False,
    split_mode="stratified", split_dir=None, normalize_imagenet=True,
):
    """Return train, validation, seen-test, unseen-test DataLoaders and mappings.

    heldout_composition selects whole training-only compositions for validation
    while preserving each crop/disease component in training. Default workers=0
    works in Windows notebooks; guard scripts with __main__ to enable workers.
    Iterating official test loaders is reserved for final evaluation.
    """
    if batch_size <= 0 or num_workers < 0:
        raise ValueError("batch_size must be positive; num_workers must be nonnegative")
    seed_everything(seed)
    root = Path(data_dir)
    paths = [root / name for name in (train_csv, test_seen_csv, test_unseen_csv)]
    frames = [load_metadata(path) for path in paths]
    validate_metadata(*frames)
    train_frame, val_frame = make_validation_split(frames[0], val_split, seed, split_mode)
    if split_dir is not None:
        save_split_manifest(train_frame, val_frame, paths[0], split_dir,
                            split_mode, seed, val_split)
    mappings = build_class_mappings(paths[0], paths[2])
    full_img_dir = root / img_dir
    if not full_img_dir.is_dir():
        raise FileNotFoundError(f"Extract the Canvas images into {full_img_dir}")
    loaders = []
    for i, frame in enumerate([train_frame, val_frame, frames[1], frames[2]]):
        dataset = DPDataset(frame, full_img_dir, image_size, as_dict_targets,
                            augment_train and i == 0, normalize_imagenet)
        loaders.append(DataLoader(
            dataset, batch_size=batch_size, shuffle=i == 0,
            num_workers=num_workers, pin_memory=pin_memory,
            worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(seed + i),
        ))
    return (*loaders, mappings)


if __name__ == "__main__":
    train, val, seen, unseen, mappings = get_dataloaders(batch_size=16)
    images, targets = next(iter(train))
    print("PyTorch batch:", images.shape, images.dtype)
    print("Targets:", {key: (value.shape, value.dtype) for key, value in targets.items()})
    print("Split sizes:", [len(loader.dataset) for loader in (train, val, seen, unseen)])
    print("Classes:", {key: value for key, value in mappings.items() if key.startswith("num_")})
