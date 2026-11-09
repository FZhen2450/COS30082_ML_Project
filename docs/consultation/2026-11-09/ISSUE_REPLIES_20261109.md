# Prepared GitHub issue replies

These are ready-to-paste drafts. They have not been posted by this task.
The source pushes are verified; neither branch is merged into main.

## Issue 2

WP1 source follow-up is now published on `fix/wp1-final-handoff-20261109`:
[7314644](https://github.com/FZhen2450/COS30082_ML_Project/commit/73146444ace8637426042e3bfb76aa9bb3b8585c).

This publishes the audit CLI/path and output-preservation fixes, named
development reference aliases, the official saved-score replay tool and its
corruption tests. The combined WP1/WP4 publication suite passed **95/95 tests,
no skips**. Earlier 48-test WP1 and 79-test WP4 receipts retain their own scope;
this is not a full-repository/Katana CI-pass claim.

Jonathan's independent **86/86** WP1 verification is now reflected in the
completion record and updated consultation copies. The first frozen crop-e2 /
disease-e3 official run, checkpoints, FP32 scoring and metrics remain unchanged.
No new inference, retraining or reselection was performed for publication.

The source is on a review branch, not merged into main. Protocol clarification,
human visual/bias acceptance and team recording/declarations remain separate.

## Issue 8

Thanks Jonathan. Your **86/86** WP1 verification and earlier **212/212**
three-model audit are now incorporated in my updated report copy, slide-3 notes
and consultation preparation. All **13,792** saved model-image predictions and
the original metrics remain unchanged.

I will describe WP1 as the **observed official H@1 leader (13.19%)**. Its 114/120
correct unseen hits are concentrated in three compositions, and the exploratory
composition-level intervals versus frozen CLIP and M6 include zero. I will leave
the detailed uncertainty explanation to Jonathan and candidate/fitting-scope
questions to Yuen Kai. Original model selections remain unchanged.

The native-FP32 owner-format replay tool is published at
[7314644](https://github.com/FZhen2450/COS30082_ML_Project/commit/73146444ace8637426042e3bfb76aa9bb3b8585c). Updated copies are
[here](https://github.com/FZhen2450/COS30082_ML_Project/tree/fix/wp1-final-handoff-20261109/docs/consultation/2026-11-09). These outputs are not relabelled as PR #24's label-free exports;
the original producer provenance and separate pair-fingerprint definitions remain.
No new official inference or post-test tuning occurred.

## Issue 13

WP4 follow-up source is now pushed to `fix/wp4-completion-20261109`:
[5c9d1f9](https://github.com/FZhen2450/COS30082_ML_Project/commit/5c9d1f9b05f7de9f9c86082ecd6bf1f676048d21). The combined WP1/WP4 branch also contains it:
[7314644](https://github.com/FZhen2450/COS30082_ML_Project/commit/73146444ace8637426042e3bfb76aa9bb3b8585c). Neither branch is merged into main.

The implementation includes three real model-picker paths, native image
inference, fitting-only references, labelled same-disease/different-crop fallback,
query-copy exclusion and provenance-aware JSON export. Additional fixes reject
Boolean/floating global IDs and Boolean scores, bind WP1 file/input identities,
and prevent delayed Tk destruction on a worker thread.

**Verification:** 79/79 applicable isolated regressions passed; the combined
WP1/WP4 publication suite subsequently passed **95/95**, both without skips.
Actual WP1, frozen CLIP and M6 picker/inference/reference/export rehearsals also
passed on the predetermined seen-development peach/curl image. M6 still returns
15/10/293/292/4, missing truth 219; this is an engineering success with an
incorrect prediction, not an accuracy claim.

The three top-prediction reference counts were 4, 1 and 4 for WP1, frozen CLIP
and M6. Sparse fitting classes are not padded with invented examples. Normal and
minimum-size LeafLens-only captures, real fallback browsing and exports were
checked. Private images, weights and score evidence remain outside Git.

Checklist implementation requirements 2-8 are verified. For item 1, inspection
and GUI reuse are verified; I do not claim a missing historical pre-implementation
plan post was made. The dated completion audit and demo runbook record the exact
source scope and commands.

Frederick, please independently rehearse the published source before recording.
Keep this issue open for real peer review and the team video. The GUI remains
development-323; historical 6,264-row parity, ordinary EXIF/CPU execution and
official-333 results remain distinct. No model/scorer/selection or threshold
changed, and no official-test inference ran for this follow-up.

## Issue 3 optional short cross-reference

The WP4 inference-interface item is implemented: frozen CLIP/M6 return original
ordered global IDs, canonical FP64 scores and provenance through the normal
LeafLens picker, preserving WP1 support. Published follow-up:
[5c9d1f9](https://github.com/FZhen2450/COS30082_ML_Project/commit/5c9d1f9b05f7de9f9c86082ecd6bf1f676048d21); combined publication:
[7314644](https://github.com/FZhen2450/COS30082_ML_Project/commit/73146444ace8637426042e3bfb76aa9bb3b8585c). Applicable combined tests passed 95/95 with
no skips. See Issue #13 for the separate peer-review and recording gates.
