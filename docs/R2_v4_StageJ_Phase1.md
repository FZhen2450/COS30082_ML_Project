# R2 v4 Stage J Phase 1: J1–J3, seed 0, six folds (14 Nov 2026)

**Status:** post-freeze exploratory development cross-validation on the reused v3 inner folds I1–I6. These are seed-0 screen results only; no three-seed confirmation has been run, and nothing is promoted. D1 remains the provisional R2 candidate.

**Scope held:**
- not run: J4–J8, H_RT and extra seeds;
- untouched: First Run, the official test sets, PR #34 and historical evidence;
- no merge and no tag.

## 1. Pre-training verification
- **Commits:**
  - `ca63114` (Amendment 2): its parent is `1b2fcdb`; it touches only `amendment_2.json`, its declaration and the declare script.
  - `ff19706` (workflow): its parent is `ca63114`; it touches only v4 paths.
  - `e96d471` (Phase 1 receipt and evaluation extensions plus the parent-rule test) was committed before training.
  - All three declarations verify OK.
  - Amendment 2 (10:08 MYT) predates every Stage J run. The first audit started after 10:20, and no J1–J3 model existed before 11:45.
- **Tests:** `Ran 91 tests … OK` before the extensions; `Ran 92 tests in 274.184s … OK` at `e96d471`.
- **Superseded attempts kept:** the v1 audit (batch-composition drift) and the 20-step smoke that failed closed.
- **Zero-filled layout:** `LayoutChunks` calls `ImageAccess.get` only for fold rows. Other slots are zero tensors, and their row IDs are used only to slice outputs that are then discarded.
  - Extraction reads no labels.
  - Training labels (crop and disease) come from fitting rows only.
  - The D1 downstream uses the v3 candidate ontology (names and IDs of F ∪ H) and the fold's own truth columns, exactly as v3 does.
- **Baselines:** the J0 and J8 zero-epoch receipts re-hash correctly (6/6 each), and the audit receipt reports `pass: true` (packets equal the registered D1 and F10 packets, max diff 0.0).
- **Nested parent rule:** the held fold is excluded when its inner parent is chosen, which is tested by making J1 perfect on the held fold only; that fold's inner parent does not change.

## 2. Execution
- **Settings:** J1 → J2 → J3 strictly sequential (one GPU job), seed 0, I1–I6, the declared 5 epochs and hyperparameters. No validation-guided choice or tuning, and no extra seeds.
- **Results:** every fold ok on its first attempt (18/18), all from commit `e96d471` with a clean worktree.

| Row | Wall time | s / fold | img/s | Peak GPU (train / total) | AMP skipped / steps | Grad checks finite | Tensors updated | Frozen unchanged | Reload diff |
|---|---|---|---|---|---|---|---|---|---|
| J1 last-2 | 53 min | 517–527 | 74 | 2.84 / 3.85 GiB | 2 / 785–790 | 18/18 | 27/27 | yes | 0.0 |
| J2 last-4 | 59 min | 576–591 | 63 | 3.96 / 4.33 GiB | 2 / 785–790 | 18/18 | 51/51 | yes | 0.0 |
| J3 LoRA r8 | 2 h 05 | 1,242–1,256 | 24 | 2.53 / 3.42 GiB | 2–3 / 785–790 | 18/18 | 96/96 | yes | 0.0 |

**Training loss** (mean over folds, epochs 1–5):

| Row | Epoch 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| J1 | 6.05 | 1.01 | 0.37 | 0.24 | 0.20 |
| J2 | 5.79 | 0.77 | 0.26 | 0.16 | 0.13 |
| J3 | 6.56 | 1.47 | 0.53 | 0.35 | 0.30 |

- **No validation curve:** by declaration all fitting rows train and no evaluation row is read during training.
- **Scheduler warning:** PyTorch's "lr_scheduler.step() before optimizer.step()" appears because AMP skips the first optimizer step. The schedule's first value is dropped identically in every fold.

## 3. Results vs D1 (LOFO γ; pp differences; bootstrap = paired composition-cluster, 2,000 resamples)

| | D1 | F10 (ref) | **J1** | **J2** | **J3** |
|---|---|---|---|---|---|
| H_bal | 66.20 | 68.32 | 69.00 (+2.80) | **69.86 (+3.66)** | 69.51 (+3.31) |
| H_micro | 61.56 | 64.0 | 64.36 | 65.13 | 64.34 |
| AUSUC | 58.02 | 60.44 | 59.50 (+1.48) | 60.64 (+2.62) | 60.60 (+2.58) |
| Seen Top-1 | 71.62 | 73.0 | 75.08 | 75.36 | 73.88 |
| Unseen macro | 62.89 | 65.3 | 64.96 | 66.42 | 66.65 |
| Unseen disease | 58.37 | 61.1 | 59.89 | 60.95 | 60.03 |
| Unseen crop | 87.66 | 88.4 | 87.41 | 88.40 | 88.86 |
| Within-crop disease transfer | 75.2 | 78.0 | 77.11 | 77.93 | 79.23 |
| Healthy shortcut Δ | – | – | +0.61 | +1.03 | +0.77 |
| False-healthy, seen / unseen | 2.36 / 1.41 | 2.2 / 1.5 | 2.28 / 2.02 | 2.23 / 2.44 | 2.31 / 2.18 |
| Folds better / worse | – | 5/1 | 5/1 (I2 −0.20) | **6/0** | 5/1 (I5 −1.74) |
| Bootstrap Δ, 95% CI | – | [+0.02, +4.60] | [+0.78, +5.04] | [+0.98, +6.53] | [+1.34, +5.68] |
| Nested vs D1 (single row) | – | +0.89 (4/6) | +2.80 (5/6) | +3.66 (6/6) | +3.31 (5/6) |
| v3 gate (seed 0) | – | pass | pass | pass | pass |

**Per-fold Δ H_bal (pp):**

| Row | I1 | I2 | I3 | I4 | I5 | I6 |
|---|---|---|---|---|---|---|
| J1 | +2.16 | −0.20 | +2.71 | +5.46 | +3.53 | +3.13 |
| J2 | +4.21 | +1.66 | +3.29 | +7.43 | +2.96 | +2.40 |
| J3 | +5.60 | +2.87 | +4.30 | +6.23 | −1.74 | +2.61 |

LOFO γ is 7.75–8.25 in every fold. H_bal at γ = 0 stays very low (J1 7.5, J2 7.1, J3 7.7), so calibration remains essential. Curves are in `curves/`.

**Nested selection and the mechanical parent:**
- **Stage nested**, D1 → best gate-passing of {J1, J2, J3} chosen on five folds: **+2.68 pp, 5/6 held folds better**, which meets the gate thresholds. Held folds select J2 four times and J3 twice; I5 (J3) is −1.74.
- **Parent by the declared rule** (best of J1–J3 by seed-0 mean H_bal): **J2_clipL_last4**.
- **Parent-selection audit:** the inner parent is J2 for held folds I1, I2, I3 and I6, and J3 for I4 and I5. Under Amendment 2, J4a–J7 (children of J2) would therefore be **excluded** from the nested estimate in held folds I4 and I5. A child is never credited in a fold whose own inner selection would not have picked its parent.

## 4. Source diagnostic H_RF (representation fixed; 49 pairs per row, 0 failed; cluster bootstrap over 9 / 14 sources)

| Row (γ) | Novel-source seen: source effect | Size effect | Unseen on source s | H_bal source / size effect |
|---|---|---|---|---|
| D1, Stage H (8) | −49.3 [−57.3, −40.5] | −0.2 | −5.4 | −0.13 / −0.56 |
| J1 (8) | −37.9 [−46.6, −28.9] | +0.1 | +1.1 | −0.02 / −0.28 |
| J2 (7.75) | −31.2 [−40.3, −23.0] | −0.4 | +2.2 | −0.01 / −0.47 |
| J3 (8) | −40.2 [−49.7, −29.4] | −0.0 | −1.0 | −0.08 / −0.44 |

**Do not read these as robustness gains.** For J rows the fine-tuned backbone was trained **with** source s, and only the downstream is refitted without it. For D1 the frozen CLIP backbone never saw any DPD source, so its H_RF is a full measure. The smaller J effects are therefore expected by construction. Only H_RT (retraining without s), declared for three-seed passers, can test source robustness.

## 5. Negative findings and limits
- **Seed 0 only.** The v3 noise level (D2 seeds spread about 0.75 pp) is below these gains, but nothing is confirmed until seeds 1–2.
- **The gain is generic, not variant-specific.** All three variants gain 2.8–3.7 pp. Fine-tuning is what helps; depth and LoRA differences are within noise, and the J2-over-J3 parent choice rests on a 0.35 pp difference.
- **Gains are spread** across seen accuracy (J1/J2 +3.5–3.7), unseen macro (+2.1 to +3.8) and AUSUC. Unseen disease Top-1 rises only 1.5–2.6 pp, and unseen crop is flat. The bottleneck from v3 (disease transfer) moves least.
- **False-healthy predictions on unseen images rise** from 1.41% to 2.0–2.4% (shortcut +0.6 to +1.0 pp; gate limit +5).
- **Regressions:** J3 regresses on I5 (−1.74) and J1 on I2 (−0.20).
- **Reused folds:** these folds informed v1–v3 and Stage I. With D1 as the control and a gain of about 3 pp, the nested estimate (+2.68) is the defensible summary, but it remains development evidence.
- **Comparison with F10:** J2 beats F10 (68.32) by 1.5 pp in-sample. F10 was not promoted in v3, so this comparison is descriptive.
- The novel-source seen stratum in the base folds has only 28 image-fold instances and is not interpretable. H_RF is the source measure, with the caveat above.

## 6. Evidence
`artifacts/r2/v4/stage_j_phase1_20261114/` contains:
- `evaluation.json`;
- `curves/`;
- 18 receipts;
- `hrf/` (pairs and summary);
- logs (paths redacted), `code_start.txt` and `SHA256SUMS`;
- `private_files_SHA256SUMS.txt`: sha256 of every checkpoint, feature cache and packet, which stay private in `workspace/experiments/r2-compositional/v4/stage_j_phase1_20261114/`.

## 7. Next decisions (not started)
1. **Three-seed confirmation.** All three rows passed the seed-0 screen, so the declared next step is seeds 1 and 2 for J1, J2 and J3, then the three-seed gate and nested estimates. That is about 8 h on Katana.
2. **H_RT** for any three-seed passer, on the three declared pairs.
3. **J4a–J7** on parent J2 (declared rule), and J8.
