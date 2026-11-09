# WP1 completion and official owner handoff — 9 November 2026

The frozen WP1 pair completed its first official shared-333 inference in the
original compatible producer environment. This fills the previously missing WP1
row; it does not change any model selection or the three earlier official runs.

## Source and fixes

Fetched GitHub and fast-forwarded the working checkout from `4397646` to main
`bbdabfdc665c711a02b3ca94cc285eb9ed86f97a`. Existing local work was preserved in
the named stash `codex-preserve-local-before-main-pull-20261109`, then restored.
The GUI manifest-path conflict uses main's existing reconciled manifest.
The WP1 follow-up is prepared on branch `fix/wp1-final-handoff-20261109`, which
also contains the separate tested WP4 commit. Publication is distinct from
merging into main or posting an issue reply.

- The development-audit CLI now defaults to the existing reconciled manifest,
  resolves its input defaults from the repository, and requires a new output
  path. Existing audit files cannot be overwritten.
- Two audit tests previously skipped a nonexistent historical manifest. They
  now exercise the actual checked-in reconciled protocol. Two CLI regressions
  check execution from another directory and preservation of existing evidence.
- On Windows, the redirected-path security test uses a real NTFS junction;
  other platforms retain the file symlink. It still requires rejection of an
  actual filesystem redirect, without requiring Windows symlink privileges.
- `scripts/verify_wp1_official_export.py` independently replays the owner's
  saved native-FP32 score format, checks identities/ranks/metrics, reproduces
  the complete WP5 report and compares all ten report CSVs. It does not convert
  these label-bearing packets into PR #24's label-free format.

## Frozen model and protocol

Two independent ResNet50 classifiers, with separate parameters: 55 crops and
175 diseases. The declared baseline used three epochs on 4,868 fitting images;
crop epoch 2 and disease epoch 3 were selected by seen-validation component
Top-1, with earliest-epoch tie handling. No four-backbone training comparison
is claimed. Development folds remain 4,868 / 1,238 / 1,894, with 323 original
candidate IDs. The accepted 3,132-row development bundle was reverified,
including nine malformed-packet checks; H@1 remains 1.3553%, H@5 15.5783%.

Official inference ranks the same 333 original global IDs in both folds,
using the exact cross-model manifest supplied by the team. This is the
owner-authorized team interpretation; lecturer confirmation of shared versus
restricted candidate lists remains outstanding.

The original torchvision version guard remains active. Actual producer runtime:
Python 3.13.13, Torch 2.7.1+cu118, Torchvision 0.22.1+cu118, NVIDIA RTX 3050
Laptop GPU. Preprocessing remains ResNet50 ImageNet V2, shorter-edge resize
232, center crop 224, bilinear interpolation and ImageNet normalization.
Composition scores remain native float32 log P(crop|image) + log P(disease|image),
with ascending-global-ID ties. They are ranking scores, not calibrated confidence.

| Frozen file | SHA-256 |
|---|---|
| Crop epoch 2 | `f0c82ca78153f492078a3a39a11eac1a847cc85eb496cf3b3d4d74db752c7d14` |
| Disease epoch 3 | `f497e19da62b68dfbcfbae58b9a5318a26b1989ba59b7a486223cad0d56e31a5` |

Both hashes match before and after inference. The owner notebook's pair
fingerprint and WP5's richer identity fingerprint use different payloads;
neither replaces these raw checkpoint-file hashes.

## First official result

| Fold | Images | Top-1 hits | Top-1 | Top-5 hits | Top-5 |
|---|---:|---:|---:|---:|---:|
| Official seen | 2,000 | 647 | 32.35% | 1,085 | 54.25% |
| Official unseen | 1,448 | 120 | 8.2873% | 441 | 30.4558% |

H@1 = **13.1945%**; H@5 = **39.0110%**. Full precision is retained in the
saved metrics. This result must not trigger test-based reselection of the
team's preselected primary model.

One owner inference attempt completed in 202.2394 seconds including reporting;
the two inference loops took 105.6374 and 62.8320 seconds. Peak CUDA allocation
was 224,432,640 bytes. These are this run's measurements, not general latency
claims. No real-data training, tuning or weight/prompt/candidate changes occurred.
All 3,448 image-file SHA-256 values match before and after the run; this is not
an atomic per-decode proof.

## Verification and handoff

- **48/48 WP1-scoped tests passed**, with no skips: independent models,
  shared protocol, export paths, audit, official export validation and data.
- Independent saved-score replay reproduced all 3,448 named Top-5 rows,
  exact metrics, complete WP5 JSON and all ten report CSVs, including the
  323-by-333 seen confusion matrix and unseen confusion matrix.
- The configured default-off notebook ran without creating official run
  outputs. All function/class ASTs match the supplied owner notebook;
  only machine configuration and the deliberate execution flag changed.
- A broader Windows regression attempt at the pulled checkpoint ran 530
  tests, with 2 failures, 44 errors and 2 skips. It encountered WSL path
  assumptions, missing native CLIP, Git ownership and symlink privilege.
  The WP1 audit skips/privilege issue was subsequently fixed and the 48-test
  WP1 suite passed. The remaining broader suite has not been rerun and is
  **not** claimed passing. The previous Katana 621/642 receipts remain historical.

Private evidence root: `runs/wp1-final-20261109/`.
Official run: `official_runs/20261109T004313Z_0df3eb32dcaf/`.
The team ZIP contains the complete run, executed notebook, compatible replay
source, receipts and instructions; raw images, model weights and environments
are excluded. A saved-score replay is not a second-human model-inference replay.

Replay without image/model inference:

```powershell
python -B scripts/verify_wp1_official_export.py --run runs/wp1-final-20261109/official_runs/20261109T004313Z_0df3eb32dcaf --receipt NEW_RECEIPT.json
```

Jonathan has independently accepted the saved WP1 results: **86/86 checks**
passed without numerical discrepancies, as recorded in
[Issue #8](https://github.com/FZhen2450/COS30082_ML_Project/issues/8#issuecomment-6732629416).
Together with the earlier 212-check audit, all 13,792 four-model official
Top-5 rows reproduced. WP1 is the observed official H@1 leader; 114/120 correct
unseen hits occur in three compositions, and its exploratory composition-level
comparison intervals versus frozen CLIP and M6 include zero. The original
model selections remain unchanged.

Remaining team actions: the team records lecturer protocol
clarification, human production-augmentation/background-bias review, and
submission/video/signature approvals. These cannot be represented as completed
by source tests or by this automated run.

## Brief consultation paragraph

For WP1, we established the shared data pipeline and verified the 8,000-image
development split. We trained two separate ResNet50 classifiers for 55 crops
and 175 diseases, then combined their log-probability scores to rank crop–disease
compositions. We froze crop epoch 2 and disease epoch 3 before official testing.
Their first official evaluation, using the team's shared 333-candidate protocol,
achieved 32.35% seen Top-1, 8.29% unseen Top-1 and 13.19% H@1. We also exported
complete predictions, confusion matrices and provenance for WP5 verification.
The candidate-policy interpretation still needs Dr. Lee's confirmation.
