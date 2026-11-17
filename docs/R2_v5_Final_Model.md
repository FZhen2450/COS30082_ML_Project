# Second Run final model (final sprint, 17 Nov 2026)

> **Development results only.** Every development number below comes from cross-validation on reused folds built from
> the DPD-Small *training* set (v3 inner folds I1–I6). None of it is an official-test result. No official-test image,
> label, CSV or score was read. First Run's official scores are historical comparison targets only.

## 1. Final model and why it was chosen

**J2:** CLIP ViT-L/14 with its last 4 visual blocks fine-tuned, then the unchanged D1 synthetic-composition downstream,
scored over all 333 candidates with a fixed γ = 8.75.

- **Highest assignment-aligned development H_micro among J1–J3:** 66.13 % (three seeds, γ chosen by H_micro on the
  other five folds).
- **Highest unseen-disease accuracy among J1–J3:** 64.91 %.
- **Simplest validated recipe:** default flip augmentation, it was the v4 declared parent, and it is confirmed against
  D1 at three seeds.
- **J3** has the highest H_bal, but on the assignment metric it ranks below J1 and J2 (H_micro 64.75).
- **J4b** (66.36) failed its registered confirmation against J2, so it was not considered.
- **Uncertainty:** the J1–J3 differences are small (about 1.4 pp) and not statistically established. J2 is chosen as
  the strongest practical candidate, not a proven best.

## 2. Assignment-aligned development comparison

The official H@1 (brief §3.3; First Run `evaluation/metrics.py`) is the harmonic mean of image-weighted seen Top-1 and
image-weighted unseen Top-1. On development folds that is exactly H_micro, while H_bal uses the unseen
composition-macro accuracy instead.

Three-seed means (D1 is deterministic). LOFO = γ chosen on the other five folds; percentages throughout.

| Model | Policy | **H_micro** | H_bal | Seen T1 | Seen T5 | Unseen T1 | Unseen T5 | Unseen macro | Unseen crop | Unseen disease | Within-crop disease | Healthy shortcut | γ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| D1 | γ = 0 | 4.66 | 7.32 | 86.08 | 98.82 | 2.45 | 28.25 | 3.85 | 83.82 | 5.30 | 15.02 | 11.31 | 0 |
| D1 | LOFO H_micro | **61.12** | 64.91 | 66.44 | 96.05 | 59.75 | 97.35 | 65.85 | 87.29 | 61.93 | 78.56 | 1.09 | 8.67 |
| J1 | γ = 0 | 4.51 | 7.53 | 85.81 | 98.58 | 2.35 | 26.00 | 4.01 | 83.24 | 5.77 | 16.45 | 10.45 | 0 |
| J1 | LOFO H_micro | **65.46** | 68.91 | 71.65 | 96.86 | 61.41 | 97.16 | 67.50 | 87.75 | 63.58 | 79.41 | 1.40 | 8.58 |
| **J2** | γ = 0 | 4.62 | 6.41 | 85.50 | 98.56 | 2.42 | 25.98 | 3.39 | 83.80 | 5.89 | 15.27 | 9.66 | 0 |
| **J2** | LOFO H_bal (v4) | 65.05 | 69.56 | 74.64 | 96.97 | 58.82 | 97.22 | 66.33 | 88.96 | 60.91 | 77.39 | 2.45 | 8.15 |
| **J2** | LOFO H_micro | **66.13** | 69.43 | 70.99 | 96.65 | 62.82 | 97.92 | 68.99 | 89.03 | 64.91 | 80.09 | 1.75 | 8.75 |
| J3 | γ = 0 | 5.95 | 8.14 | 85.29 | 98.51 | 3.17 | 27.19 | 4.34 | 85.20 | 6.37 | 16.14 | 12.12 | 0 |
| J3 | LOFO H_micro | **64.75** | 69.44 | 69.93 | 96.18 | 61.90 | 97.78 | 70.35 | 89.28 | 63.67 | 81.71 | 1.41 | 8.56 |
| J4b* | LOFO H_micro | 66.36 | 70.23 | 71.74 | 97.06 | 62.66 | 97.44 | 69.81 | 89.40 | 64.68 | 80.33 | 1.81 | 8.79 |

\* J4b failed its registered three-seed confirmation against J2 and is shown for reference only.

**J2 H_micro by fold** (seed mean): I1 67.69, I2 68.06, I3 61.78, I4 75.00, I5 69.84, I6 54.39.
**By seed:** 65.92 / 65.64 / 66.82.

**Calibration sensitivity:** without calibration (γ = 0), every model scores only **4.5–6.0 %** H_micro, the same
range as First Run's official H@1 of 1.8–9.9 %. With calibration it reaches 61–66 %. Calibration is the dominant
factor in the development metric.

**Remaining differences from the official setup:**

| | Development folds | Official |
|---|---|---|
| Candidates | 301 (290–291 seen + 10–11 held-out) | 333 (323 seen + 10 unseen) |
| Unseen images | 309–333 over 10–11 compositions | 1,448 over 10 compositions |
| Seen evaluation | 874–879 held-out training images | 2,000 test images |
| Sources | the same as training | possibly different (unknown, and H_RT shows source sensitivity) |

## 3. Final checkpoint

| | |
|---|---|
| Run | `<E>/v5/final_J2_full8000/run/` |
| `checkpoint.pt` | sha256 `e9dbbdf3fc95ba791e8e8129ef38bbd85352a0974c3091d6b7667bd633abb1c9` (196 MiB; trainable tensors + heads) |
| `features.pt` (8,000 training images, 2 views) | sha256 `eee9759f48ec5988e6c58e8ffc2c3d64ba6bd1c99ef6b2bafb07e96a38c8ba25` |
| `receipt.json` | sha256 `983f71953b76af912fa7978bd8bf0786b6a23f4adff9ea88d6428647783658cd` |
| `verification.json` | sha256 `534a13811f007580d9c15d9767282820577218ada8693e51f2699161a69d26f4` |

## 4. Full-training configuration and results

- **Configuration:** `research_r2_v5/final_config.json`.
  - J2: `clip_l14`, last 4 blocks, flip augmentation, 5 epochs, batch 32.
  - AdamW with backbone lr 1e-5 and heads lr 1e-3, weight decay 0.05, cosine schedule with 1 warm-up epoch, AMP.
  - Composition-balanced sampler; crop + disease cross-entropy.
  - Configuration sha256 `b42fa6e7…0053`.
  - Seed 0 (torch seed 0). Initialised from the original pretrained CLIP weights.
- **Data:** all 8,000 training-CSV images, 8,000 fitting rows and no evaluation rows. The guarded image loader refuses
  any non-training file.
- **Run:**
  - 814 s, 1,250 optimizer steps.
  - Epoch-mean loss 5.05 → 0.58 → 0.22 → 0.13 → 0.11.
  - 27 of 27 gradient checks finite; 2 AMP-skipped steps (epoch 1); 51 of 51 trainable tensors updated.
  - Frozen parameters unchanged; reload difference 0.0.
  - Peak 3.95 GiB allocated by training, 6.8 GiB device memory; mean GPU utilisation 89.5 %; maximum 88 °C.
- **Code:** commit `aec9d39` with a clean worktree; `jrun.py` unchanged (`433d4058…`).

## 5. Frozen calibration and candidate universe

- **Candidates:** 333 global composition ids 0–332, from First Run's pinned final candidate payload
  (`artifacts/r2/v5/final_candidates_cos30082_v1.json`, sha256 `ba17e941…`; class-level rows only).
  - The 323 seen rows equal the training CSV.
  - The 10 unseen ids (39, 84, 107, 124, 133, 136, 144, 166, 276, 303) are exactly the ids absent from training, and
    each has a crop and a disease observed in training.
- **Calibration:** score − 8.75 for the 323 seen candidates; the 10 unseen candidates are not penalised.
  - γ = 8.75 is the pooled H_micro optimum over J2's six development folds × three seeds, and also the mean of the
    per-fold LOFO choices (range 8.25–9.0).
  - Development data only.
- **Ranking:** descending calibrated score; ties broken by ascending global id (First Run tie rule).

## 6. Inference entry point

- `scripts/r2v5_final.py verify --run <E>/v5/final_J2_full8000` (done).
- `scripts/r2v5_final.py official-eval …` is guarded and one-shot; see `docs/R2_v5_Official_Evaluation_Procedure.md`.
- **Library entry:** `research_r2_v5.final`:
  - `load_model` (checkpoint → model);
  - `extract_query` (images → features);
  - `raw_scores` (features → [n, 333] D1 scores, deterministic);
  - `calibrate`, `topk`, `metrics` and `harmonic` (official contract).

## 7. Official evaluation readiness

**Ready, not executed.** The frozen candidate and the guarded one-shot command are prepared. The command refuses
unless `R2V5_OFFICIAL_EVAL_AUTHORIZED=1` and `--owner-authorization` are both given. It also refuses if any verified
hash has changed or the output directory exists.

## 8. Verification evidence

| Check | Result |
|---|---|
| Training isolation | only training-CSV rows; full-data fold has no evaluation rows; guarded loader refuses non-training files (test) |
| Checkpoint reload | run-time reload 0.0; independent fresh-model re-extraction of training rows 0–63 = cache (max diff 0.0) |
| Query path | the official-eval image path reproduces the cached features exactly (0.0) |
| Finite and deterministic | 333-column scores finite; two runs bit-identical |
| Candidate order and labels | payload hash, ids 0–332, seen rows = training CSV, unseen ids = absent ids (tests) |
| Seen/unseen formulas | Top-1/Top-5 and the harmonic mean equal First Run `evaluation/metrics.py` on random data with ties (test) |
| Top-1/Top-5 consistency | Top-1 = first of the stable Top-5 = argmax (verify) |
| Calibration without held-fold leakage | LOFO γ chosen on the other folds only; final γ from development folds only, never official labels |
| Full-data smoke test | 320 training images (already fitted): Top-1 89.7 %, Top-5 100 % at γ; 10.3 % predicted as an unseen id. Pipeline sanity only |
| Tests | `test_r2v5_final` 11 OK; `test_r2v4_j` 19 OK (30 in 312.6 s) |
| Security | no network or package changes; official CSV paths are explicit arguments; `torch.load` only on our own checkpoint |

## 9. Git

- Branch `research/r2-v5-assignment-aligned`, created from `1a2f4b0` (the published v4 commit).
- `aec9d39`: implementation; results commit to follow.
- Not pushed. v4 (`06ccf43`, `1a2f4b0`) and First Run are unchanged.

## 10. Risks and expected generalisation limits

- **Development ≠ official.**
  - The development folds come from the training sources.
  - H_RT showed a 45–66 pp loss on images from a source excluded from training.
  - If official test images differ in source or acquisition, official accuracy can be far below the 66 % development
    H_micro.
- **γ transfer.** γ was tuned on fold models fitted on about 5,000 images with 291 seen candidates. The final model is
  fitted on 8,000 images with 323 seen candidates, so score margins on new images may shift and the unseen/seen balance
  may differ.
- **A single training seed** for the final model. Development seed spread for J2 H_micro was 65.6–66.8.
- **Disease transfer remains the bottleneck:** unseen disease 64.9 % vs unseen crop 89.0 % on development.
- **Not blind.** The candidate payload (class names and ids) comes from First Run's pinned contract, which the brief
  permits as semantic information. First Run's official scores are known to the project, so the broader project has
  no fully blind test.

## 11. Recommendation

**Ready for one-shot official evaluation**, pending owner authorisation. On development folds, the model is the
strongest validated Second Run candidate on the assignment metric. Development evidence suggests calibration is what
separates about 5 % from about 60 %+ H@1. Whether the official score exceeds First Run's 9.95 % is not established
until the single authorised evaluation.
