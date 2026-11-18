# Round 2, stage v4: compositional OOD exploration (predeclared plan)

Declared 14 Nov 2026 on branch `research/r2-v4-ood-exploration` (branched from `5bca217`). Nothing in v4 has been scored yet. This document, `research_r2_v4/protocol_v4.json` and the three matrices under `research_r2_v4/matrices/` are committed before any v4 experiment runs.

## 1. Standing and what v4 can and cannot claim
- The v3 freeze stands: D1 is the provisional candidate (nested +2.23 pp, 6/6), F10 is not promoted, and there is no blind claim. v1–v3 files, packets and receipts are not modified.
- v4 evaluates on the same six v3 inner folds, the same candidate policy and the same primary metric, so it compares directly with D1, F10 and B1.
- Those folds already informed v1–v3 choices. Every v4 number is therefore **post-freeze exploratory development cross-validation**. A v4 row that passes its gate is a stronger development candidate, not a confirmed improvement.
- The only route to an independent check is outside v4. That route is new data, or a non-blind evaluation through Jonathan's WP5 importer after the first submission, and either one needs the owner's approval.

## 2. Why these directions (evidence from v1–v3)
- **Unseen disease recognition is the bottleneck.** For D1, unseen crop Top-1 is about 88% but unseen disease Top-1 is only about 58%. Every gain so far came from better disease evidence for held-out pairs: synthesis (D1), text priors (D2) and complementary CNN features (F10).
- **Source and crop are entangled.** 277 of 323 compositions come from a single source. Only 46 compositions span two or more sources. LEACE source erasure destroyed accuracy (C5: H_bal 0.354), so source carries crop and disease signal and cannot simply be removed.
- **Calibration is fragile.** The pooled gamma moves between 6.75 and 15.75 across recipes, and the source stress test showed seen accuracy collapsing when a source becomes novel. In the CPDD example, seen Top-1 dropped from 0.87 to 0.17 at gamma 8.
- **Frozen representations look saturated.** Concatenating a third frozen backbone (G1–G4) never beat F10. The next lever is adapting the representation per fold: partial fine-tuning, controlled augmentation and disentangling objectives.

## 3. Stages and experiment matrix

| Stage | Rows | What changes | Cost |
|---|---|---|---|
| H, source shift (diagnostic) | B1, D1, F10 and every passer | Size-controlled source ablation over 49 (fold, source) pairs, plus a learning curve | frozen features, GPU-light |
| I, frozen-feature methods | I1–I8 | Adaptive calibration (I1, I2), prototype head (I3), HSIC disentanglement (I4), cross-crop SupCon (I5), text-direction synthesis (I6), multi-layer CLIP (I7), multi-crop views (I8) | seconds to minutes per row; I7 and I8 need one local re-extraction each |
| J, per-fold partial fine-tuning | J0–J8 | CLIP-L last-2, last-4 and LoRA; then controlled geometric or photometric augmentation, source-balanced sampling, cross-crop SupCon, crop adversary; ConvNeXt stage 4 | GPU-intensive, see section 4 |
| K, new backbones | not declared | DINOv2, SigLIP, EVA-CLIP and others | blocked, needs downloads (section 5) |

**Matched comparisons.**
- Every I and J row uses D1 as its control and reports B1 and F10 alongside.
- J rows keep the D1 recipe fixed downstream, so only the representation changes.
- J0 (zero epochs) must reproduce D1 within 1e-5 before any J row counts.

**Ablations.**
- Depth: J1 against J2. Parameter-efficient against block tuning: J3.
- Augmentation one family at a time: J4a against J4b.
- Sampling: J5. Each loss alone: J6 and J7.
- Count-aware against crop-aware calibration: I1 against I2.

**Gate and inference.**
- The in-sample gate is v3's gate.
- The nested leave-one-fold-out estimate must also pass, with every mechanical choice (gamma, the best-of-J1–J3 parent) rerun inside the remaining folds.
- Rows with training randomness need a 3-seed mean.
- Every gate row reports a paired composition-level bootstrap CI.
- All rows are reported, including failures.

**Source shift with size control (stage H).**
- Each source ablation is paired with three random removals of the same number of fitting images from other sources, composition-stratified.
- Source effect = ablated − size-matched. Size effect = size-matched − base.
- The 49 usable pairs recur across folds because the seen holdout is shared. The effective sample is 349 unique novel-source seen images in 38 compositions from 16 sources, plus 948 unseen-image evaluations, so inference uses a cluster bootstrap over sources.

## 4. Compute (from the 14 Nov feasibility probe)
The probe used an RTX 4060 Laptop with 8 GiB, 7.44 GiB free at the start, batch 32, AMP fp16 and real training images. Its output is `v4/feasibility_20261114/gpu_probe.json`.

| Configuration | Trainable params | Peak GiB | Images/s |
|---|---|---|---|
| CLIP-L/14 frozen inference (batch 64) | 0 | 1.46 | 101 |
| CLIP-L/14 last 2 blocks | 26.2 M | 2.73 | 81 |
| CLIP-L/14 last 4 blocks | 51.4 M | 3.66 | 69 |
| CLIP-L/14 last 8 blocks | 101.7 M | 5.51 | 54 |
| CLIP-L/14 last 8, grad checkpointing | 101.7 M | 3.27 | 47 |
| CLIP-L/14 LoRA r8 on all MLPs, checkpointing | 2.1 M | 2.53 | 26 |
| CLIP-L/14 full fine-tuning, checkpointing | 304.1 M | 6.18 | 23 (fits, but over the 6.0 GiB limit; not declared) |
| ConvNeXt-B stage 4 | 25.6 M | 0.84 | 322 |
| ConvNeXt-B full | 87.8 M | 5.05 | 118 |

JPEG decode and transform runs at 62 / 161 / 206 images/s with 0 / 4 / 8 workers, so 8 workers keep the GPU fed.

**Estimates**, assuming 5 epochs over about 5,050 fitting images per fold plus feature extraction for about 6,250 fold rows × 2 views:

| Item | Wall time |
|---|---|
| Frozen-feature row (I3–I6, controls) | under 1 min each |
| I1, I2 (post-hoc on D1 scores) | seconds |
| I7 extraction | about 3 min |
| I8 extraction | about 13 min |
| Stage H, per measured row | about 270 fold fits, about 25 min |
| J0 | about 12 min |
| J1 | about 43 min |
| J2 | about 49 min |
| J3 | about 110 min |
| J8 | about 15 min |
| J4a–J7 | about 4 h with a J1/J2 parent; about 9 h if J3 is the parent |
| Stage J screen total | about 6.5–12 h of GPU, run sequentially |
| 3-seed confirmation per passer | +1.5–3.7 h |

**Disk:** per-fold fine-tuned features take about 25 MB each, and no checkpoints are kept, so v4 should stay well under the 60 GiB cap.

## 5. Decisions that need the owner
1. **Stage J go-ahead** (GPU-intensive, estimate in section 7).
2. **Stage K (new backbones)**:
   - Needs `open_clip_torch` or `timm` installed plus weight downloads, for example DINOv2-L/14 (about 1.2 GB), SigLIP-SO400M (about 3.5 GB) or EVA02-CLIP-L (about 1.7 GB).
   - Install and download rules require the owner's explicit authorisation.
   - Those weights are pretrained on external (mostly unlabelled or web) data, as CLIP already is; the owner should confirm that this is compatible with "no external labelled datasets".
3. **Optional robustness folds:** re-partition the 63 inner-eligible compositions with a new seed (for example 2 × 6 extra folds) as a secondary check. This changes the fold set, so it is not declared.

## 6. Risks
- **Adaptive reuse of the inner folds:** handled by labels, nested estimates and seeds, but not removed.
- **Fine-tuning leakage:** each fold's model sees only that fold's fitting rows. Tests must prove that the data loader can never yield seen-val, unseen-val, outer or official-test rows.
- **Small effective samples** for source shift (16 sources), so source-effect claims are descriptive.
- **8 GiB GPU:** full CLIP-L fine-tuning may not fit (see probe). Stage J rows are scoped to what the probe shows fits.
- **Optimistic selection** of the J4–J7 parent: handled in the nested estimate.

## 7. Feasibility results (14 Nov, actually run)
All files are in `workspace/experiments/r2-compositional/v4/feasibility_20261114/`.

| Check | Result | Evidence (sha256) |
|---|---|---|
| Test suite at 5bca217 (CPU) | `Ran 687 tests in 124.795s`, `OK` | `test_suite.log` 05120e54…56b1 |
| Bit-exact reproduction of B1, D1 and F10 on I1–I6 (v3 matrix, v3 features, outer rows poisoned) | 318 numeric values per row; max abs diff 0.0 for all three; PASS at 1e-5 | `repro_v3_stage_f_check.json` 069248a6…6ebb; `repro_v3_stage_f/summaries.jsonl` 89d8552c…4835 |
| Source-shift feasibility | 252 (fold, source) pairs. 229 are ablatable, and all 229 have a size-matched control. 49 pairs have ≥ 10 evaluation images: 349 unique novel-source seen images, 38 compositions, 16 sources, 948 unseen-image evaluations | `source_feasibility.json` 40fb9e65…9428 |
| GPU probe | section 4 | `gpu_probe.json` f0c59f99…b011 |

**Notes:**
- The full test suite also runs First Run tests (`test_notebook.py`, `test_wp3b_adapter.py`). Those tests create and delete temporary folders under `workspace/experiments/_verification/wp3b-m2/`; the folder is empty afterwards. From now on v4 runs only `tests/test_r2_*.py` and its own tests.
- No v4 method has been implemented or scored. Everything above is a reproduction or a resource check.
