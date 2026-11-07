# WP5 controlled robustness benchmark — 7 November 2026

Branch `feature/wp5-final-evaluation`, base `7d2852b`. Additive files only:
`evaluation/robustness.py`, `evaluation/robustness_adapters.py`,
`tests/test_robustness.py`, `tests/test_robustness_adapters.py` and this
document. No existing file, protocol, checkpoint, prediction or historical
result was modified. No training and no official-test access.

**Evidence location.** Experiment scripts and outputs are kept outside Git in
the WP5 owner's local evidence folder (named below; not committed), because the
ablation-C scripts import Jonathan's uncommitted WP2 code and the outputs
contain private filenames. Provenance link: the v2 run recorded the SHA-256 of
the robustness module it executed, `dd5b5ec3…7bf3613`, which equals the
committed LF bytes of `evaluation/robustness.py`. Verify against the blob
(`git show HEAD:evaluation/robustness.py | sha256sum`), because a checkout
with `core.autocrlf=true` may write CRLF to disk.

## Why this was needed

The brief (§3.2) asks how the approach "improve[s] robustness when applied to
real-world field images with variations in lighting conditions, backgrounds,
viewpoints, and image quality". No branch (local or remote, including `main`
9010fb7 and `gui`) contains a robustness or perturbation evaluation. Background
sensitivity is partly covered by the existing WP2 Grad-CAM and 9-image occlusion
diagnostics on ablation C (Jonathan's local WP2 evidence, not in this
repository). This suite
covers lighting and image quality, not backgrounds or viewpoint.

## Protocols

Sixteen prespecified conditions: clean plus five families × three severities.

| Family | Severity 1 / 2 / 3 | Definition |
| --- | --- | --- |
| darken | ×0.7 / ×0.5 / ×0.3 | multiply RGB, round, clip |
| brighten | ×1.3 / ×1.6 / ×2.0 | multiply RGB, round, clip (overexposure) |
| gaussian_blur | σ = 0.4% / 0.8% / 1.6% of short side | v1: exact separable Gaussian; v2: Pillow `GaussianBlur(radius=σ)` |
| jpeg | quality 40 / 20 / 10 | Pillow/libjpeg, 4:2:0, no optimize |
| gaussian_noise | σ = 0.02 / 0.05 / 0.10 on [0,1] | seed = SHA-256(suite, seed, filename, condition) |

**v2 `wp5-robustness-suite-v2` (use for every comparison).** Fingerprint
`8f68a0e0…d006b`, pinned in a test. `run_native_suite` decodes each image once
with the declared policy `PIL.Image.open(path).convert("RGB")` (no EXIF
transpose, which is what WP1, WP2, WP3 and ablation C use), corrupts the native
array once, records `pixel_sha256` per image and condition, and passes freshly
built PIL images of those identical pixels to every model's own, unchanged
`predict` (frozen preprocessing + canonical scorer). Pass each model's
historical inference batch size in `batch_sizes`: an undeclared size defaults to
1, and batch size can change floating-point results, which the clean-parity gate
would then expose. Darken/brighten/noise run in row blocks that are tested
bit-identical to the v1 operators; v2 deliberately reuses v1's noise-seed
derivation so the noise stream is identical.

**v1 `wp5-robustness-suite-v1` (superseded exploratory run).** Fingerprint
`7a36aebc…63a3`, unchanged. `run_suite` corrupts the uint8 array after each
model's own resize/crop. Its operators and runner are unchanged so the archived
v1 run stays reproducible; `summarize_conditions` has since gained additive
fields (largest-cluster sensitivity) that do not alter existing values. Do not
report v1 results.

## Corrections made in this review

1. **Identical pixels (scientific defect, fixed by v2).** v1 docs claimed
   "every model sees identical perturbed pixels". False: v1 corrupts after
   model-specific resizing (C 288², WP2 224², CLIP/M6 224 centre crop), so
   architectures receive different arrays. A regression now asserts this v1
   limitation; v2 proves identity by SHA-256 and a two-model test.
2. **Inference precision of the archived C run (reproducibility defect).** v1
   switched C to CUDA autocast from one 32-image near-tie, and the earlier doc
   claimed the historical records were produced under autocast. That was wrong.
   C's own unchanged evaluation code reproduces all 1,894 saved records exactly
   at fp32; with autocast it does not (4 decision and 51 rank changes). v1's
   five changed Top-1 decisions are all fp16 near-ties (logit gaps 0.0000–
   0.0078); none changes composition Top-1/Top-5 correctness. Details (external
   evidence): `robustness_c_dev_20261107T065432Z\AUDIT_2026-11-07.md`.
3. **Duplicate loader.** The v1 script rebuilt C's model/runtime path. The v2
   script reuses C's own `load_multitask_checkpoint`, `configure_runtime`,
   `get_transforms`, `_normalize` and `composition_scores_from_logits`, and
   refuses perturbed inference unless the clean pass reproduces every saved
   field for all 1,894 images.
4. **Statistical framing.** "65 composition clusters" overstated the
   information. See the next section.
5. **Adapter top-k.** The adapter now passes `topk=min(5, K)` explicitly (found
   by the real WP1 predictor test; inconsequential at K = 323).

## Statistics audit

- **Paired resampling:** each image's clean and corrupted outcome is differenced
  before resampling; whole compositions are resampled with replacement and the
  ratio estimator Σdiff/Σn is recomputed. A test proves the vectorized estimator
  equals a brute-force resample loop draw-for-draw.
- **Cluster design (unseen-development fold):** 65 compositions, but Kish
  effective clusters only **8.7** (diseased 6.1, healthy 3.5); cassava mosaic is
  544/1,894 images (28.7%); 30 compositions have < 5 images. Percentile
  intervals with so few effective clusters can under-cover, so treat them as
  approximate. Each delta also reports the estimate without the largest cluster.
- **Interpretation:** an interval describes uncertainty over compositions like
  these for this one fixed checkpoint and image set. It does not cover training
  seeds, other models or field conditions.
- **Multiplicity:** exact McNemar p-values are Holm-adjusted across the 15
  conditions per metric, but assume independent images (anti-conservative under
  clustering): descriptive only. Bootstrap intervals are unadjusted across the
  15 conditions × metrics.
- **Causal language:** corruptions are interventions on the image, so "this
  corruption changed the model's decisions" is supported. Claims about which
  visual features a head uses, or why, are not.

## Development result: ablation C under protocol v2 (reporting version)

Methodology demonstration on Jonathan's own historical WP2 EfficientNet-B3
ablation C (`1b79a10b…`, epoch 5). **Not a selected finalist**; no finalist
weights are available locally. C was trained on exactly the WP5 fitting and
seen-validation images (verified by filename), so only the **unseen-development
fold (1,894 images)** is valid and no H@k exists for C.

Run (external evidence): `robustness_c_dev_v2_20261107T073654Z\` (protocol
written before inference, SHA-256 `ee324851…b06f`; 2,190 s on one RTX 3060).
**Clean-parity gate: 1,894/1,894 on every saved field** (crop, disease and
composition Top-1, every truth rank, Top-5 membership) through v2 decoding and
C's own fp32 stack, before any corrupted image was scored. Clean counts:
composition Top-1 46, Top-5 364, crop head 1,448, disease head 262, false-healthy
193/1,496. Checkpoint hash unchanged; per-image pixel SHA-256 saved.

| Family (mean over 3 severities) | Top-1 decision unchanged vs clean | Crop-head Top-1 (clean 76.5%) |
| --- | --- | --- |
| darken | 88.6% | 76.1% |
| brighten | 73.1% | 74.8% |
| jpeg | 72.3% | 72.2% |
| gaussian_noise | 75.4% | 75.7% |
| gaussian_blur | 39.8% | 52.9% |

Findings (exploratory; intervals are approximate given 8.7 effective clusters):

1. **Decision instability is large and label-free.** Blur s3 leaves 12.6% of
   composition Top-1 decisions unchanged; crop-head hits fall 1,448 → 1,059
   (s2) → 575 (s3). Darkening barely changes the crop head (1,439 at ×0.3).
2. **Composition accuracy barely moves, so accuracy alone hides the
   instability.** Clean Top-5 is 19.2%. Only brighten s3 has a cluster interval
   excluding zero: −2.69 pp [−5.66, −0.43], −3.11 without cassava mosaic. Blur
   s3 is −4.86 pp with Holm-McNemar p = 1.8×10⁻⁶, yet its cluster interval
   [−14.70, +4.45] spans zero: the change is concentrated in few compositions.
   JPEG Top-1 hits even rise (46 → 60 at q40; Holm p = 0.13).
3. **Brightening and blur shift this model's predictions for diseased leaves
   towards "healthy" compositions (pre-registered H2, supported for this
   checkpoint).** False-healthy among 1,496 diseased images: clean 193;
   brighten +4.75 / +7.95 / +8.62 pp; blur +7.62 / +14.30 / +10.29 pp. All six
   intervals exclude zero, and each remains positive without the largest
   cluster (+3.05 to +9.98 pp). Darken s3 (+1.87 [+0.39, +3.38]) and JPEG s1
   (+1.27 [+0.37, +2.81]) are small; noise shows no clear change.
4. **WP2 hypothesis H1 not supported.** The disease head flips more often than
   the crop head in all 15 conditions (e.g. darken s1 8.5% vs 3.1%). After
   matching clean decision margins, the difference spans zero for every blur,
   JPEG and noise severity (blur −1.25 / −4.00 / −4.72 pp; JPEG −3.28 / −2.94 /
   +0.60; noise −4.99 / −5.60 / +0.61). Only brighten s3 (+5.25 [+2.04, +10.13])
   and darken s2 (−2.47 [−3.83, −0.36]) exclude zero, of 15 unadjusted
   intervals. The disease head's extra instability is not distinguishable from
   what its smaller clean decision margins predict (margin = top-1 minus top-2
   logit gap, not a calibrated probability). This does not show why its margins
   are smaller.
5. **Native-resolution application makes noise/JPEG severity depend on capture
   resolution (Q4).** s3 Top-1 consistency by native short side ≤ 300 / 301–999
   / ≥ 1,000 px: noise 47.8 / 70.6 / 89.5%; JPEG 53.6 / 65.1 / 71.4%; blur
   15.5 / 11.6 / 6.7%; darken 86.4 / 88.0 / 88.6%. This pattern is consistent
   with the model's downsampling of large photos attenuating pixel-level noise
   and JPEG artefacts, but that mechanism was not tested separately. Strata also
   differ in composition (26 / 44 / 23 clusters), so they are descriptive, not a
   controlled resolution effect.

Versus the archived v1 run, the qualitative conclusions (1, 3, 4) are unchanged.
Effects that v1 applied after resizing (JPEG/noise) are milder in v2, and JPEG
s2/s3 no longer change false-healthy rates reliably. Report v2.

Limitations: synthetic corruptions, not field images; severities fixed in advance
but not calibrated to real capture conditions; one non-finalist checkpoint, one
seed, unseen-development fold only; backgrounds and viewpoint not covered;
nothing here estimates official-test performance.

## Finalist readiness (`evaluation/robustness_adapters.py`)

No new loader or scorer. `PredictorAdapter` validates any predictor with the
shared `predict(list[PIL], topk)` contract: candidate columns equal the declared
ascending universe, declared score semantics and dtype, finite `[B, K]` scores,
top-k consistent with the scores under stable ties, and (CLIP/M6) the
`frozen_clip` / `m6` variant. `WP2NotebookPredictor` composes the owner's
notebook model, `checkpoint_inference_transform` result, `amp_context` and
`composition_scores` unchanged, with the corrected independent CPU snapshot.

| Finalist | Reused loader | Contract test | Blocked on |
| --- | --- | --- | --- |
| WP1 ResNet50 pair | `IndependentPredictor` (this repo) | **real class**, real preprocessing and float32 scorer, synthetic weights, through `run_native_suite` | both head checkpoints (Fa Zhen / Andy); batch size 2 as exported |
| WP2 ConvNeXt-Base | notebook objects via `WP2NotebookPredictor` | buffer-reusing synthetic model: independent fp32 snapshots, channels_last, owner scorer | checkpoint and notebook objects (Yuen Kai) |
| Frozen CLIP ViT-L/14 | GUI `wp4.clip_predictor.CLIPPredictor` | output-contract replica (variant, float64, semantics, columns, top-k) | approval to install pinned `clip` package; public weights `b8cca3fd…` |
| M6 adapter | same, with adapter checkpoint | same | adapter checkpoint `8835a18b…` (Yuen Kai) |

**Environment requirement.** The identified development packets declare their
inference environments, recorded in `FINALIST_CONTRACTS["…"]["packet_environment"]`:
WP1 torch 2.7.1+cu118; WP2, CLIP and M6 torch 2.5.1+cu121. WP2 additionally
declares bf16, channels_last, batch size 16, `deterministic_algorithms: false`
and cuDNN TF32. WP1's `load_checkpoint` deliberately rejects a different
installed torchvision, so WP1 cannot load in the current WP5 environment (torch
2.9.1 / torchvision 0.24.1). Run each finalist in its recorded environment.
Reduced precision and non-deterministic kernels can flip near-ties, as fp16 did
for ablation C. Agree the clean-parity criterion with each owner before perturbed
inference: exact Top-1/Top-5 decisions where achievable, otherwise a documented,
prespecified tolerance.

**Parity target for WP2.** The gate must compare against the identified WP2
224×224 development export (seen Top-1 609/1,238, H@1 4.4367%), the only WP2
packet with per-image records. The original anonymous export reported 610/1,238;
the cause of the one-hit difference is unresolved and neither record has been
changed. Reproducing 609 in a rerun would not explain the 610.

Running the finalists: build each owner predictor, wrap it in
`PredictorAdapter`, then call `run_native_suite` on both development folds with
the declared batch sizes. Gate on clean parity with each model's identified
development packet before perturbed inference. Then report per condition:
seen/unseen Top-1/5 and H@1/H@5, unseen-diseased exact hits, false-healthy
rate, per-composition errors and decision stability against clean, with
effective clusters per fold. Aggregate H@1 alone can hide opposite movements in
these quantities.

**Finalist identity (context, teammate-reported, not reproduced by WP5).** As of
8 November 2026 the WP2/WP3 owner reports these as negative or non-promoted
ablations: WP3 M9 (healthy-score correction) and M10 (disease visual
prototypes; higher development H@1 but fewer exact unseen-diseased hits), a WP2
320×320 retest, and an in-sample WP2 fitting-capacity check. Finalists are
unchanged: M6 seed45 and the historical selected WP2 224×224 model. The retest's
matched 224 baseline (H@1 3.86%) uses a different protocol, seed and checkpoint
from the selected packet above; do not substitute one for the other.

Decode note for WP4: the GUI applies `ImageOps.exif_transpose` before inference;
the evaluation pipelines do not. 121/3,132 development images (3.9%) carry a
non-identity EXIF orientation, so GUI predictions for such photos can differ from
evaluated behaviour. Not changed here (teammate code).
