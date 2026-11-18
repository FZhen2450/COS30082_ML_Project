# Second Run J2: the proposed approach in the final submission

**Status (18 Nov 2026):** Group 9 adopted **J2** from the Second Run as its proposed new approach.
- **Baselines:** WP1 (independent ResNet50 crop and disease classifiers) and WP2 (multi-task ConvNeXt-Base) remain the
  required CNN baselines (brief §3.1).
- **Other First Run results:** frozen CLIP and the M6 adapter remain reported as First Run results.

> **This supersedes two statements in the Second Run research documents.** Those documents are kept unchanged as a
> historical record.
> - `docs/R2_Second_Run_Final_Summary.md` §1 says the research "never feeds the First Run submission".
> - Its §10 lists official-test performance under "cannot claim".
>
> Both were written before the single authorised official evaluation (17 Nov 2026) and before this adoption.

## Where everything is

| What | Where |
|---|---|
| J2 in the LeafLens GUI (adapter, tests, parity tools) | This branch, `feature/final-j2-integration-20261118`, PR #39. Guide: [WP4_J2_FINAL_MODEL.md](WP4_J2_FINAL_MODEL.md) |
| J2 training, scoring, guarded official evaluation, frozen configuration and evidence | `main` (merged in #40 from `research/r2-v5-final-model` at `2583f70`): `scripts/r2v5_final.py`, `research_r2_v5/`, `artifacts/r2/v5/`, `docs/R2_v5_*.md` |
| Second Run research history (benchmark, v4 stages, negative results) | Draft PRs #34 and #36 and the `research/r2-*` branches. They are research records and are not merged into `main`. |
| Second Run summary | `docs/R2_Second_Run_Final_Summary.md` |
| Public check of the J2 evidence | `python scripts/r2v5_public_verify.py --tests` |

## Method (frozen configuration `b42fa6e7…`)

1. **Backbone (transfer learning).**
   - OpenAI CLIP ViT-L/14. The first 20 visual transformer blocks are frozen.
   - Blocks 20–23, `ln_post` and `proj` (51 tensors) are fine-tuned on all 8,000 training images.
   - Training: 5 epochs, batch 32, AdamW (backbone 1e-5, heads 1e-3, weight decay 0.05), cosine schedule with one
     warm-up epoch, horizontal flip, crop plus disease cross-entropy, seed 0.
   - Checkpoint SHA-256: `e9dbbdf3…abb1c9`.
2. **D1 compositional scoring.** On L2-normalised, standardised image features, three linear softmax heads are fitted:
   - crop (55);
   - disease (175);
   - composition (333).

   The ten unseen compositions have no training images. They get 64 synthetic features each, by shifting training
   features of the same disease from another crop: x − mean(crop A) + mean(crop B). The score averages the factorised
   score with the composition score:
   ½[log p(crop) + log p(disease) + log p(composition)].
3. **Calibrated stacking.** γ = 8.75 is subtracted from the 323 seen candidates.
   - γ was chosen on development folds only: it is the pooled H_micro optimum over three seeds × six folds, and equals
     the mean of the leave-one-fold-out choices.
   - Ranking covers all 333 candidates, with ties broken by ascending global id.

**Why J2.** On the assignment's metric (H_micro, the development equivalent of H@1), J2 had the best development score
among J1–J3: 66.13% over three seeds, against 61.12% for D1 with frozen features. It also had the best unseen-disease
accuracy (64.91%). J1–J3 are not statistically separable. J4b (J2 plus photometric augmentation) failed its
pre-registered confirmation. Details: `docs/R2_v5_Final_Model.md` §1–2.

## Official result (one evaluation, 17 Nov 2026, frozen model)

| | Top-1 | Top-5 |
|---|---|---|
| Seen (2,000) | 57.10% (1,142) | 87.70% (1,754) |
| Unseen (1,448) | 84.12% (1,218) | 98.69% (1,429) |
| **Harmonic mean** | **H@1 68.02%** | **H@5 92.87%** |

- **Comparison:** the First Run's best official H@1 is WP1's 9.945951%. The comparison is descriptive (different
  development processes, one training seed, one evaluation, First Run scores already known).
- **Audit trail:** `docs/R2_v5_Official_Evaluation_Result.md` and `docs/R2_v5_Official_Evaluation_Procedure.md`

- **Withheld:** the seen-test confusion matrix and per-class official statistics are not published here. The seen-test
  confusion matrix is prepared for the submitted report (brief §3.3).

## The brief's two research questions (§3.2)

- **Generalisation to unseen categories: addressed on DPD-Small.** Calibration plus synthetic composition features
  make the ten unseen compositions reachable. Unseen Top-1 rose from at most 5.59% (First Run) to 84.12%.
  - Part of this comes from crop recognition, since the ten unseen pairs span nine crops.
  - The cost is seen accuracy: 574 of 2,000 seen images were predicted as unseen classes.
- **Robustness to real-world image variation: not established.**
  - A source-exclusion experiment (H_RT, `docs/R2_v4_HRT_Report.md`) lost 45–66 pp on images from a source left out
    of training, relative to matched random removal.
  - Photometric augmentation (J4b) did not reproducibly help.
  - The team reports this as a limitation and does not claim robustness to field images.

## Reproduce

All steps use training data only. None reruns the official evaluation.

1. **Check the published evidence** (no private files). From the repository root:
   ```bash
   python scripts/r2v5_public_verify.py --tests
   ```
   This recomputes H@1 and H@5 from the published counts and checks the frozen configuration and hashes.
2. **Retrain J2** (about 14 minutes on an RTX 4060 Laptop GPU). From the repository root, with `DPD_Small_dataset`
   available:
   ```bash
   python scripts/r2v5_final.py train --out <new folder> --images <DPD_Small_dataset>
   python scripts/r2v5_final.py verify --run <new folder>
   ```
   The seed and configuration are fixed. GPU kernels are not guaranteed bit-identical across machines, so a retrained
   checkpoint can differ from the frozen `e9dbbdf3…`, which is the one evaluated and used by the GUI.
3. **Run J2 in LeafLens:**
   - follow [WP4_J2_FINAL_MODEL.md](WP4_J2_FINAL_MODEL.md);
   - the frozen checkpoint and the inference asset are private and shared with the team, never committed.

## Limitations

- **One training seed and one official evaluation.**
- **Sparse, imbalanced seen classes.** 163 of 323 seen classes had no correct Top-1 image, and 171 classes have at most
  2 test images.
- **Development folds are reused and training-derived.** The development-to-official gap (seen Top-1 about 71%
  against 57.1%) has several possible causes.
- **Robustness to new image sources is not shown** (see above).
