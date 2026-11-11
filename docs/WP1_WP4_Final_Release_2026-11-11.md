# WP1 and WP4 final-333 release - 11 November 2026

**12 November follow-up:** the [acceptance supplement](WP1_WP4_Acceptance_Supplement_2026-11-12.md)
adds state identities, replaces final-mode prompt re-encoding with the original
shared bank, and verifies synthetic added-candidate selection/fallback/export.
The dated results below retain their original scope.

This release implements the [student-reported lecturer clarification](Lecturer_Clarification_2026-11-11.md)
without altering the historical development split, checkpoints or official results.
Full-training execution and its new official metrics are recorded separately
from the engineering checks below.

Review source: `feature/wp1-wp4-final333-20261111`. Publication is separate from
merging into main, independent peer operation, or the team's submission sign-off.

## WP1 recipe and current execution

The frozen recipe is `configs/wp1/resnet50_full8000.json`: two independent
ResNet50 classifiers initialized separately from the existing ImageNet V2
weights, seed 42, batch size 2, AdamW learning rate/weight decay 0.0001, native
ImageNet V2 preprocessing and the unchanged training augmentation policy.
All 8,000 original training records are consumed in each epoch. Crop runs for
two epochs and disease for three: the development-selected task budgets are
fixed before this run. The last epoch is saved as `final.pt`; no old validation
fold or official-test metric chooses the checkpoint.

The source identity, exact ordered training membership, pretrained file hash,
recipe and training source hashes were frozen before optimization. All 8,000
images passed decoding and file-hash auditing. The run output is private:
`runs/wp1-full8000-20261111/`. Both tasks completed; `release.json` verifies
unchanged training-image and frozen-source bytes. The recorded training and
post-verification interval was 1,840.83 seconds. New checkpoint identities are
registered in `artifacts/models/wp1_full8000_release.json`; no weights are in Git.

The new protocol, `artifacts/data/final_full8000_333_v1.json`, contains 323
compositions in the 8,000-image fitting pool and all 333 original global IDs
in the candidate space. The ten official unseen compositions do not acquire
training images. Candidate and prompt hashes retain the verified official
semantic definitions.

```powershell
& 'D:/COS30082_WP1_Runtime/venv/Scripts/python.exe' -B scripts/run_wp1_full8000.py --pretrained D:/COS30082_WP1_Runtime/torch-cache/hub/checkpoints/resnet50-11ad3fa6.pth --output NEW_RUN --execute-training
```

Without `--execute-training`, this command prepares/audits only. Always choose
a new output directory. No weights or packages are downloaded by the command.
The current run must finish before starting another GPU job.

## WP1 official evaluation and verification

`scripts/evaluate_wp1_full8000.py` verifies the frozen full-training release
before reading official metadata. It ranks both complete test folds against
the same 333 classes using native float32 crop/disease log-probability sums.
It exports full score matrices, exact named Top-5 predictions, confusion
matrices, Overall/Seen/Unseen counts and Top-1/Top-5, plus H@1/H@5. Overall is
the combined correct count divided by 3,448. Exact ties use ascending global ID.

The official image path uses untransposed PIL RGB plus native preprocessing,
matching the declared historical evaluation input policy. Ordinary GUI previews
use EXIF-corrected RGB and batch one; these are separately labelled paths.
Checkpoint and image hashes are checked again after inference. Historical
official results remain intact. No new checkpoint is selected from these scores.

```powershell
& 'D:/COS30082_WP1_Runtime/venv/Scripts/python.exe' -B scripts/evaluate_wp1_full8000.py --release runs/wp1-full8000-20261111/release.json --output NEW_OFFICIAL_RUN --execute-evaluation
```

Without `--execute-evaluation`, only a plan is written and official metadata or
images are not opened. A new official run is not an untouched blind test:
previous official outcomes have already been reported.

### Completed official result

Run: `runs/wp1-full8000-official333-20261111/`. All three populations rank
against the same 333 candidates; the checkpoint pair was frozen before this run.

| Population | Images | Top-1 hits | Top-1 | Top-5 hits | Top-5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Overall | 3,448 | 977 | 28.3353% | 1,789 | 51.8852% |
| Seen | 2,000 | 896 | 44.8000% | 1,436 | 71.8000% |
| Unseen | 1,448 | 81 | 5.5939% | 353 | 24.3785% |

H@1 = **9.94595%**; H@5 = **36.39844%**. Exact fractions and checkpoint hashes
are in [the aggregate result](../artifacts/results/wp1_full8000_official333_20261111.json).
The final crop SHA-256 is `16c8f8ee1933e9333d49f8e8e33b88dd27c914d0b839920b35d9b3996f4cd6f5`;
the disease SHA-256 is `f4c05a331794e4dab06f52877780a76a1ff35229eb5d157f9597c8c1719429e4`.

The producer-side saved-score replay passed **29 checks**, reproducing all
3,448 Top-5 rows, both named prediction CSVs, both confusion matrices and all
reported metrics exactly. This is not a second person's model-inference replay.
Checkpoint, image and official-CSV bytes were unchanged through evaluation.

Compared with the original 4,868-fitting WP1 release, Seen and Overall Top-1
increased, while Unseen Top-1 and H@1 decreased (historical H@1: 13.1945%).
The different training populations and budgets are explicit. No causal or
significance claim, test-based checkpoint replacement or extra tuning follows
from this result. The historical pair and result packets remain unchanged.

```powershell
python -B scripts/verify_wp1_full8000.py --run runs/wp1-full8000-official333-20261111 --receipt NEW_REPLAY_RECEIPT.json
```

## WP4 final-333 behavior

The current three-column layout and Fit whole image preview are retained.
The header identifies `FINAL / 333 CLASSES`. Model selection stays inline.
The launcher selects the final protocol by default; pass the original
development protocol explicitly to reproduce the development-323 mode.

- WP1 final inference accepts correctly assigned full-8,000 checkpoints whose
  source, training membership, candidates and prompts match the final contract.
  Historical development WP1 weights are rejected in final mode.
- Frozen CLIP encodes the 333 approved prompts with the pinned native model.
- Historical M6 is loaded only after validating its unchanged hash and original
  323-candidate training contract. Final inference then uses a separately named
  333-row text bank. Its actual training scope remains
  `historical_development_4868` in provenance and the UI labels it legacy-4868.
  This does not claim that M6 has been retrained on 8,000 images. A new full-8,000
  adapter remains the WP3 owner's handoff.
- Final references come only from the 8,000-image training source. Unsupported
  pairs use explicitly labelled same-disease/different-crop references, or state
  that none are available. Query paths and pixel-identical copies are excluded.
- Export separates reference-training scope from model-training provenance.
  A single-image GUI result is not an official dataset accuracy estimate.
- Final WP1 automatic discovery uses `artifacts/models/wp1_full8000_release.json`
  after actual training completion. It will not silently substitute legacy
  weights when the final release is absent. Explicit manual files must still
  pass the final loader's contract checks.

Run on this machine with the existing environment and package directory:

```powershell
./scripts/launch_wp4.ps1 -Python 'D:/COS30082_WP1_Runtime/venv/Scripts/python.exe' -ClipPackages runs/wp4-completion-20261109/clip_packages
```

The optional final rehearsal harness checks actual model loading, 333 ordered
score columns, canonical Top-5, fitting-only reference membership, labelled
unseen fallback, JSON export and normal/minimum screenshots. It accepts only
original training filenames and defaults to CPU:

```powershell
python -B scripts/rehearse_wp4_final333.py --model m6 --output NEW_REHEARSAL --screenshots
```

Actual M6 rehearsal: `runs/wp4-final333-m6-20261111/`; the declared peach/curl
training image still has incorrect Top-1 and truth absent from Top-5. The
successful engineering checks do not imply a disease-recognition success.
Frozen CLIP rehearsal and the new WP1 rehearsal have separate receipts.

All three actual final-333 GUI rehearsals passed on CPU, including real native
image inference, canonical score/rank agreement, export and unseen fallback:
`runs/wp4-final333-frozen-20261111/`, `runs/wp4-final333-m6-20261111/` and
`runs/wp4-final333-wp1-full8000-20261111/`. Normal and 1280 x 840 screenshots
were inspected. The new WP1 peach/curl training example remains wrong at Top-1,
with its truth at rank 2; this is an engineering check, not a test accuracy.

The combined applicable WP1/WP4 regression suite passed **55/55 tests without
skips**. Receipt: `runs/wp1-wp4-final333-regression-20261111.log`. This is a
scoped Windows run; no new full Katana-suite pass is claimed.

## Remaining team work

Frederick's independent operation, peer review, the WP3 owner's full-training
adapter, WP2's full-training results, report/deck integration and actual team
recording/signatures are separate tasks. Do not copy automated checks into a
human review field or claim the adapter's greater-than-6.62% goal is achieved.
