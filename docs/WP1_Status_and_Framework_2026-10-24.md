# PyTorch decision and WP1 delivery status

Updated 24 October 2026 (Malaysia time).

This record supplements the historical 16 October team plan. The team has
selected **PyTorch** as the primary implementation framework following the
team discussion and the requested repository update. PyTorch is a team decision;
the supplied assignment brief does not mandate a framework. The dated Word plan
is retained as the historical agreement; its outstanding framework gate is
superseded by this record. Model training and final performance are still pending.

## Ownership

| Work package | Co-owners |
| --- | --- |
| WP1 Data + independent baseline | Andy / Fa Zhen |
| WP2 Multi-task learning | Yuen Kai / Jonathan |
| WP3 Zero-shot + integration | Yuen Kai / Andy |
| WP4 UI + image retrieval | Fa Zhen / Frederick |
| WP5 Evaluation + robustness | Jonathan / Frederick |

For WP1, Andy owns data loading, EDA, training-only splits and augmentation
verification. Fa Zhen owns the two independent crop/disease classifiers and
baseline run configurations. Both compare candidate backbones and review metrics.
Final contributions must follow evidence rather than a preassigned 20% claim.

## Completed in this update

- Replaced the TensorFlow/Keras pipeline with PyTorch Dataset/DataLoader and
  torchvision v2 transforms; retained the five-return-value loading interface.
- Added explicit global composition ID/index mappings. The 323 seen compositions
  have non-contiguous IDs within 0–332; loaders retain these official IDs.
- Added filename overlap, mapping consistency and official composition-isolation
  checks. These are metadata checks, not pixel-level duplicate detection.
- Added deterministic training-only validation, keeping all 40 singleton
  compositions in training and retaining every crop/disease component.
- Added training-only heldout-composition validation for WP3. All components of
  validation combinations remain individually represented in the training fold.
- Exported split manifests with seed, algorithm version, source CSV SHA-256,
  filenames and composition IDs to `artifacts/data/`.
- Fixed EDA to decode every listed image during a full audit, fail on bad inputs,
  and derive verification claims from the actual check. Regenerated the metadata
  summary without claiming image integrity or regenerating existing figures.
- Updated dependencies, setup instructions and the co-owned work-package matrix.

## Compatibility evidence

Local CPU tests used Python 3.13, torch 2.9.1+cpu, torchvision 0.24.1+cpu,
pandas 2.3.3, NumPy 2.2.6 and Pillow 12.0.0. The dependency file pins the tested
core package versions; plotting dependencies have bounded version ranges.

Twelve automated tests passed, covering the official CSVs, mapping round trips,
filename leakage rejection, deterministic splits, singleton handling, component
coverage, split manifests, RGB conversion, normalized float32 tensors, int64
labels, augmentation reproducibility, corruption after the first 50 images,
missing images, truthful EDA summary status, and DataLoader integration with zero
and one worker on Windows.
Image-related tests use temporary synthetic images; they are not DPD benchmarks.

| Split policy (seed 42) | Training images | Validation images | Validation compositions |
| --- | ---: | ---: | ---: |
| Classwise stratified | 6,378 | 1,622 | 283 |
| Heldout composition | 6,106 | 1,894 | 65 |

The new classwise split differs from the previous sklearn split. Per-class
rounding keeps every class with at least two images represented in both folds;
singleton classes stay in training. The realized image fraction is approximate.
All subsequent model comparisons must use the same new manifest. Do not mix
old and new split results. Heldout mode interprets 0.2 as a fraction of composition
classes, not of images. Official test sets remain 2,000 and 1,448 images.

## Remaining WP1 acceptance checklist

- [ ] Extract the real Canvas dataset; run the full decoding audit and inspect
  augmentations on real leaves. Current workspace has no raw images.
- [ ] Implement two separate supervised models: crop (55 outputs) and disease
  (175 outputs), with separate parameters/checkpoints.
- [ ] Implement training entry points, run configurations and checkpoint/logging.
- [ ] Screen ResNet, EfficientNet, ConvNeXt and ViT using the same splits and
  stated budgets; record architecture, pretrained weights, resolution,
  trainable layers, optimizer/scheduler, seed, batch size, steps/epochs,
  validation Top-1/5, wall time, parameters, GPU memory and inference latency.
  MobileNet and Inception-v4 remain optional expansions.
- [ ] Select 1–2 feasible families using validation only; run full independent
  baseline training and retain learning curves and reproducible checkpoints.
- [ ] Agree and implement a composition scoring rule with WP2/WP5. A candidate
  rule is the sum of crop and disease log-probabilities for each valid pair;
  document the final choice before testing.
- [ ] Confirm the final candidate universe and allowed use of unseen category
  names/metadata with the lecturer. The loader's combined 333-label mapping is
  metadata support, not approval of an evaluation protocol.
- [ ] With WP5, report composition Top-1/5 separately for seen/unseen, harmonic
  mean and seen confusion matrix after model-selection policy is frozen.
- [ ] Both co-owners review the code/results and contribute report evidence.

## Integration rules

Images are float32 `[B, 3, H, W]`, normalized with ImageNet mean/std by default.
Targets are int64 crop/disease/global composition IDs. Do not feed raw global
composition IDs into a 323-output seen-only head; use `seen_comp_id_to_index`.
Unseen labels cannot be mapped into that head. Independent baseline loss uses
crop/disease targets, not a composition classification loss.

This preprocessing is for the supervised baselines. WP3 must use the chosen
CLIP checkpoint's own required preprocessing and tokenizer; ImageNet transforms
are not a universal CLIP interface. No CLIP training implementation is included.

Run records must retain code commit, dependency versions, dataset/split hash,
backbone/weight version, seed, hyperparameters, learning curves, hardware,
duration and exported metrics. Official test images must not be used for training,
prompt selection or hyperparameter selection.
