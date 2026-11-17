# J2 final model in LeafLens (WP4)

J2 is the team's proposed new approach. It is CLIP ViT-L/14 with its last four visual blocks fine-tuned on all 8,000
DPD-Small training images (Second Run), scored by synthetic crop–disease composition (D1) over all 333 classes, with
fixed calibrated stacking (γ = 8.75 on the 323 seen classes).

- **Official result:** H@1 68.02%, H@5 92.87%, from one frozen evaluation.
- **Baselines:** WP1 and WP2 remain the required baselines. Their results are unchanged.

## Required files (private; obtained from the project owner through authorised access, never committed)

| Field in the GUI | File | SHA-256 |
|---|---|---|
| CLIP ViT-L/14 weights | `ViT-L-14.pt` (official OpenAI release) | `b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836` |
| Frozen J2 checkpoint | `checkpoint.pt` (51 trainable tensors plus heads, 196 MiB) | `e9dbbdf3fc95ba791e8e8129ef38bbd85352a0974c3091d6b7667bd633abb1c9` |
| J2 inference asset | `j2_d1_inference_asset_v1.pt` (D1 downstream heads, 1.7 MB) | `dd6495bb863c6f2e9b9d473f99ad06d7c1efa349cc72b3602b444eef3c6d2783` |

Put the three files in `runs/j2-final-20261117/` (git-ignored). The model picker finds them by hash. You can also
select them with the "Files" button.

## Run

```bash
python gui.py --images <path to DPD_Small_dataset>
```

In the model picker, choose **J2 / final model**, select CPU or CUDA, and click Load. Then open a plant image and
click Analyze.

- The GUI shows the Top-5 crop–disease classes and training-set reference images of the selected class. Reference
  images come from the validated 8,000 training images only.
- **Python packages:** the existing `requirements.txt` plus the pinned official `clip` package (revision `d05afc43…`,
  as for the frozen CLIP and M6 models). Nothing new is required.
- **Speed (RTX 4060 Laptop, 64 training images):**

  | | Load | Prediction (median) |
  |---|---|---|
  | CUDA | 5.3 s | 0.33 s |
  | CPU | 6.8 s | 0.54 s |

## What is verified

| Check | Evidence |
|---|---|
| Identity | Checkpoint, asset and CLIP hashes are pinned. The checkpoint metadata (J2, seed 0, full 8,000, configuration hash) and its exact 51 trainable tensors are checked before use. Any mismatch raises an error; there is no fallback model. |
| Scoring arithmetic | The asset holds the downstream heads captured from one run of the frozen J2 scoring pipeline. They are not refitted or recalibrated. On identical features, the GUI arithmetic reproduces that pipeline to ≤ 6.6e-6 on 3,448 training images, with identical Top-5 lists (`scripts/j2_build_inference_asset.py`). |
| End-to-end GUI path | On 64 training images, the GUI predictor reproduces the frozen J2 scores (`scripts/j2_verify_gui_parity.py`). CUDA: maximum score gap 5.7e-6, identical Top-5 on 64 of 64. CPU (fp32 instead of fp16): identical Top-1 on 64 of 64, identical Top-5 on 62 of 64. |
| Tests | `tests/test_wp4_j2.py` (12 tests, no private files needed). The full suite (675 tests) passes. |

## Input orientation

J2 was trained and officially evaluated on each file's stored pixel orientation. It did not apply the EXIF
orientation tag. The GUI still displays the EXIF-corrected picture, but it hands J2 the stored pixels.

This matters for parity: 8 of the first 64 training images have an orientation tag. Rotating them changed J2's Top-5
for 5 of those 8 images.

## Results are not new evaluations

The GUI is a single-image demo. J2's official numbers come from the one authorised evaluation, recorded in the
Second Run evidence. Using the GUI does not re-evaluate the official test set.
