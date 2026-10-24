# COS30082 Applied Machine Learning Project

## Out-of-Distribution Plant Disease Identification

Swinburne University of Technology. This repository contains the DPD-Small
metadata, PyTorch data pipeline, EDA and team execution records. Independent
crop/disease model training, multi-task training and the proposed CLIP approach
are still pending; no trained-model performance is claimed.

## Team work packages

| Work package | Co-owners | Deliverables |
| --- | --- | --- |
| WP1 Data + independent baseline | Andy Wei Jian CHU / Fa Zhen NG | Data, EDA, splits, augmentation; two independent classifiers |
| WP2 Multi-task learning | Yuen Kai CHIA / Jonathan Zheng Li CHAI | Shared backbone, dual heads, loss tuning and diagnostics |
| WP3 Zero-shot + integration | Yuen Kai CHIA / Andy Wei Jian CHU | CLIP reference, trainable improvement and inference contract |
| WP4 UI + image retrieval | Fa Zhen NG / Frederick | GUI, inference adapter and training-image retrieval |
| WP5 Evaluation + robustness | Jonathan Zheng Li CHAI / Frederick | Metrics, ablations and field robustness |

WP1 division: Andy handles data/splits/EDA/augmentation checks; Fa Zhen handles
independent models and run configurations. Both compare backbones and review
baseline results. Contributions will be attributed from evidence.

The [24 October framework decision and WP1 checklist](docs/WP1_Status_and_Framework_2026-10-24.md)
supplements the historical 16 October Word plan. PyTorch is now the primary
framework. The original Word plan is ignored by Git; the Markdown supplement
makes the current decision available to collaborators.

## Install and obtain images

Use a Python 3.11-3.13 environment. Local CPU verification used Python 3.13.
Run these commands in the repository root (Windows PowerShell):

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Use that environment's Python for subsequent commands (examples below use
`python`). The pinned torch/torchvision pair was checked on CPU. For GPU use,
install a matching build following the [official PyTorch installation selector](https://pytorch.org/get-started/locally/)
and retain the exact installed versions with each run. CUDA training has not
been verified in this update.

Download the Canvas DPD-Small archive and extract all images into
`DPD_Small_dataset/` at the repository root. Raw images and model weights are
excluded from Git.

```text
COS30082_ML_Project/
  DPD_Small_dataset/             # local raw images, not committed
  DPD_Small_train.csv            # 8,000 rows
  DPD_Small_test_seen.csv        # 2,000 rows
  DPD_Small_test_unseen.csv      # 1,448 rows
  dataset.py                    # PyTorch Dataset and DataLoaders
  data_utils.py                 # metadata audit and train-only split export
  requirements.txt
  tests/test_data.py
  artifacts/data/               # metadata audit and split manifests
  eda.py
  eda_summary.txt
  eda_figures/                   # existing EDA figures
  docs/WP1_Status_and_Framework_2026-10-24.md
```

## Audit and reproduce splits

These commands do not require images and do not train a model:

```powershell
python data_utils.py
python eda.py --metadata-only
python -m unittest discover -s tests -v
```

`data_utils.py` validates metadata and writes both split manifests plus
`data_audit.json` to `artifacts/data/`. Manifests include filenames, class IDs,
seed, split version and source CSV SHA-256. Image integrity is explicitly
`not_run` unless requested. Checks cover filename overlap, ID/name consistency,
composition/component consistency and official seen/unseen separation; they do
not detect duplicates with different filenames.

After extracting the images, run:

```powershell
python data_utils.py --check-images
python eda.py
python dataset.py
```

The full audit decodes every metadata-listed image and exits unsuccessfully if
any image is missing or unreadable. Full EDA writes `eda_image_audit.json`, stops
before figures on failure, and regenerates figures only after a successful audit.
`--metadata-only` regenerates statistics while retaining existing figures and
making no image-integrity claims. Real-image verification remains pending in the
current workspace. Automated image tests use temporary synthetic fixtures.

## Load data in PyTorch

```python
import torch
from dataset import get_dataloaders

if __name__ == "__main__":
    train_loader, val_loader, seen_loader, unseen_loader, mappings = get_dataloaders(
        batch_size=32,
        image_size=224,
        seed=42,
        as_dict_targets=True,
        augment_train=True,
        num_workers=0,
        split_dir="artifacts/data",
    )
    images, targets = next(iter(train_loader))
    print(images.shape, images.dtype)       # [B, 3, 224, 224], torch.float32
    print(targets["crop"].dtype)            # torch.int64, IDs 0..54
    print(targets["disease"].dtype)         # torch.int64, IDs 0..174
    print(targets["composition"])          # official global IDs, NOT 0..322

    # In a future independent classifier training loop:
    # logits = crop_model(images.to(device))
    # loss = torch.nn.functional.cross_entropy(logits, targets["crop"].to(device))
```

The five returned objects retain the old order, but loaders are PyTorch
DataLoaders: replace `.take(1)` with `next(iter(loader))` and Keras `.fit()` with
an explicit PyTorch training loop. `as_dict_targets=False` returns targets in
crop, disease, composition order. Labels are int64; images are normalized using
ImageNet mean/std. Set `normalize_imagenet=False` for float images in [0, 1].
Only the training loader shuffles and applies horizontal/vertical flips plus
small affine rotation, translation and scale changes, before normalization.
Validation/test transforms are deterministic. Use the chosen pretrained model's
required preprocessing when building future training code; CLIP requires its
own checkpoint-specific preprocessing. The default 224/ImageNet pipeline is for
supervised baseline development.

Use `num_workers=0` for Windows notebooks. Script-based multiprocessing requires
a `if __name__ == "__main__":` guard. Workers receive seeded generators; construct
the loader once per run. The helper seeds Python, NumPy and Torch, but does not
guarantee deterministic CUDA kernels or identical results across library versions.

## Validation and composition IDs

The default classwise stratified split is made from the original 8,000 training
images only. Each class with at least two images contributes at least one image
to validation and retains at least one in training; all 40 singleton compositions
stay in training. Classwise rounding gives 6,378 train / 1,622 validation images
for seed 42. This replaces the old sklearn split: use the new manifest for every
comparison and do not combine results from different splits.

For simulated unseen validation, pass `split_mode="heldout_composition"`.
For seed 42 and `val_split=0.2`, 65 whole training compositions (1,894 images) are
held out, leaving 6,106 training images. Here 0.2 refers to composition classes.
Every validation crop/disease component is still represented in training. The
function fails if it cannot produce the requested valid holdout.

There are 55 crops, 175 diseases, 323 seen compositions and 10 official unseen
compositions. The 323 seen global composition IDs are non-contiguous within
0-332. `targets["composition"]` always retains the official global ID.

- `seen_comp_id_to_index` / `seen_comp_index_to_id`: global seen ID to/from 0..322.
- `unseen_comp_id_to_index` / `unseen_comp_index_to_id`: global unseen ID to/from 0..9.
- `composition_id_to_pair`: global ID to (crop ID, disease ID).
- `composition_id_to_index` / `composition_index_to_id`: combined candidate mapping.

Do not pass raw composition IDs to a 323-output head. Independent baseline losses
use crop/disease targets; composition evaluation needs an agreed pair-scoring
rule. Unseen metadata is exposed for evaluation mapping, not for training or
model selection. Confirm the final candidate universe and permitted unseen-label
use with the lecturer before freezing the shared evaluation protocol.

Keep official seen and unseen tests untouched during training, prompt selection
and model selection. Creating the loaders does not evaluate them; iterate their
images only for frozen final evaluation.

## Remaining delivery milestones

- Extract and audit real images; visually inspect augmentations on leaves.
- Implement and train independent crop and disease models with configs/logging.
- Screen ResNet, EfficientNet, ConvNeXt and ViT under the same validation protocol;
  record compute and validation results, then select 1-2 feasible full backbones.
- Agree composition ranking with WP2/WP5; deliver final seen/unseen Top-1/5,
  harmonic mean and seen confusion matrix after freezing model selection.
- Complete multi-task and improved CLIP approaches, retrieval and GUI in their
  assigned work packages. Full details are in the linked WP1 checklist.

Submission target from the supplied brief: **27 November 2026, 11:59 PM Malaysia
time**. Report maximum 8 pages; presentation video maximum 8 minutes. Check
Canvas for subsequent announcements.
