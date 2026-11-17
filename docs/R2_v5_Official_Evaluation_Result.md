# Second Run v5: one-shot official DPD-Small evaluation result

**Answer: yes.** Second Run v5's official H@1 is **68.02 %**, against First Run's best official H@1 of 9.945951 %.
Classification: **Improved.**

The comparison is descriptive. First Run and Second Run were developed under different research processes, and First
Run's official scores were already known to the broader project, so this is not a fully blind test. Second Run
development never used official data, and nothing is tuned after this result.

## Run

| | |
|---|---|
| Authorisation | owner, 17 Nov 2026: exactly one official evaluation of the frozen v5 J2 model |
| Executed | 2026-11-17 20:15:54 → 20:17:31 MYT; exit 0; 97 s total (seen part 54.6 s, unseen part 39.2 s) |
| GPU | peak 1.8 GiB allocated by inference (4.7 GiB device memory); mean utilisation 72.8 %; maximum 87 °C |
| Code | `a52dbc2`, clean worktree; verification record `verification_07e05ef.json` (sha256 `dd42ee75…`) |
| Attempts | **one**. No repeat, no retry, no change afterwards |

**Gate A**, all passed before inference:
- the authorisation guard refused a probe that lacked the environment variable;
- frozen hashes matched the verification record;
- the output directory did not exist;
- seen CSV: 2,000 rows over 323 seen compositions; unseen CSV: 1,448 rows over the 10 unseen compositions;
- every crop and disease label agrees with the candidate payload;
- file names are unique, none is shared with training or between the parts, and every image is present.

## Frozen model and scoring

| | |
|---|---|
| Model | CLIP ViT-L/14, last 4 visual blocks fine-tuned (J2), trained on all 8,000 training images for 5 epochs, seed 0; checkpoint `e9dbbdf3fc95ba791e8e8129ef38bbd85352a0974c3091d6b7667bd633abb1c9` |
| Composition scoring | the unchanged D1 synthetic-composition scorer, fitted on the 8,000 training images |
| Candidates | all 333 global composition ids 0–332 (payload `ba17e941…`): 323 seen; 10 unseen (39, 84, 107, 124, 133, 136, 144, 166, 276, 303) |
| Calibration | γ = 8.75 subtracted from the 323 seen candidates. It was fixed from development folds before any official access and not changed |
| Ranking | descending calibrated score; ties broken by ascending global id |

## Official results

| | Top-1 (correct / n) | Top-5 (correct / n) | Crop Top-1 | Disease Top-1 |
|---|---|---|---|---|
| Seen | **57.10 %** (1,142 / 2,000) | 87.70 % (1,754 / 2,000) | 84.25 % | 62.10 % |
| Unseen | **84.12 %** (1,218 / 1,448) | 98.69 % (1,429 / 1,448) | 87.09 % | 84.19 % |

- **H@1** = 2 × 0.5710 × 0.84116 / (0.5710 + 0.84116) = **0.680238 (68.02 %)**.
- **H@5** = 2 × 0.8770 × 0.98688 / (0.8770 + 0.98688) = **0.928700 (92.87 %)**.

**Independent recomputation** (`independent_recomputation.json`), from the saved predictions and the CSV labels, with
no re-inference:
- direct counting from the saved top-5 ids;
- re-ranking the saved calibrated scores, which reproduces the saved top-5 exactly;
- First Run's `evaluation/metrics.py` (`topk_accuracy`, `harmonic_mean`);

All three agree with the reported values to 1e-12.

**Seen confusion matrix** (333 × 333, `seen_confusion_333x333.npy`):
- it counts 2,000 images; the diagonal is 1,142, equal to the Top-1 count;
- row sums equal the class counts;
- **574 seen images (28.7 %) are predicted as one of the 10 unseen classes.** This is the cost of the seen-class
  penalty γ;

**False-healthy predictions:**
- unseen images (all diseased) predicted healthy: 1 of 1,448 (0.07 %);
- diseased seen images predicted healthy: 1.88 % of 1,647.

## Comparison with First Run (historical official H@1)

| Model | Official H@1 |
|---|---|
| **Second Run v5 (J2 full-data + D1 + γ 8.75)** | **68.02 %** |
| First Run WP1 independent ResNet50 (best) | 9.95 % |
| First Run frozen CLIP | 6.62 % |
| First Run M6 CLIP adapter | 5.96 % |
| First Run WP2 ConvNeXt-Base | 1.76 % |

## Interpretation and limitations

- **What drives the gain.** The gain comes from seen/unseen calibration together with composition synthesis:
  - the model predicts an unseen id for 97.5 % of unseen images, and is correct for 84.1 %;
  - on development folds, the uncalibrated (γ = 0) models scored 4.5–6 % H_micro, the same range as First Run;
  - **the price is seen accuracy:** 57.1 % official vs 71 % on development, with 28.7 % of seen images pulled into
    unseen classes.
- **Why unseen accuracy is high.** Official unseen accuracy (84 %) exceeds the development estimate (63 %). With only
  10 unseen classes spread over 9 crops, once the model leans toward the unseen classes, recognising the crop nearly
  identifies the class: unseen crop accuracy is 87.1 % and correct-crop-but-wrong-disease is only 3.0 %. The development
  folds had more crop overlap among their held-out classes.
- **Single run.** One training seed and one evaluation, with no confidence interval for the official score.
- **Not fully blind.** First Run's official scores were known to the project, and the 333-candidate list and its 10
  unseen class names are the brief-permitted semantic information First Run already used.
- **Calibration as a deployment choice.** γ trades seen accuracy for unseen recall. In deployment, the expected share
  of unseen classes would set the right balance.

## Evidence (SHA-256)

| File | sha256 |
|---|---|
| `official_evaluation.json` | `0110cb988437dbf16a32eb7b45ebbba85eae71382340b1ba5861b4a09226263c` |
| `seen_confusion_333x333.npy` | `faf047144d20c60fc8aed221b18c526535a26df3eb226ceb5c7ea5eb60b4d124` |
| `seen_calibrated_scores.npy` (private, per image) | `abbe5d390c8d04c32242cb7c190a8809c117d257736216f682ce6c4405996809` |
| `unseen_calibrated_scores.npy` (private, per image) | `556e5206262635624b2f92ade2b39d01e6ebb14a95bc051da5cc214cc43d6742` |
| `seen_top5_ids.npy` (private, per image) | `1bb40afcac45488e35bbb91233bc295ad637ba2bd5957237266e404075053bbc` |
| `unseen_top5_ids.npy` (private, per image) | `a428f9cc0e80d88441449524afd2dda7dccf5bce692b95000057e48319da1d97` |
| `independent_recomputation.json` | `1e4e0d878a8877a3cc84173526bc5302784a93abc06d9bfc19a986bcd90e3009` |
| `gate_a.txt` | `900d563eec68cad74fa3458c911371980eaf4861fa197842ae77ac0daf2593a9` |

- **Raw outputs:** kept unchanged and read-only in `<E>/v5/official_eval_ONE_SHOT/`.
- **Per-image packets:** these hold per-image official predictions and stay private. Only aggregate files are
  committed or packaged.

## Final model status

**Frozen. Second Run training and evaluation are complete:** no further training, calibration or official evaluation.
