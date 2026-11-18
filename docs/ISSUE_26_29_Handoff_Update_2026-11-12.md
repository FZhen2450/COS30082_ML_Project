# Ready-to-post drafts - 12 November 2026

Not posted. Source for all updates:
https://github.com/FZhen2450/COS30082_ML_Project/tree/feature/wp1-wp4-final333-20261111
This is a review branch, not merged into main.

## Issue #26 - WP1

WP1 owner-side full-8,000 refits and provenance supplements are complete.
Two independent ResNet50 models started from verified ImageNet V2 weights:
crop 2 epochs, disease 3 epochs, 8,000 images per epoch, batch 2, seed 42,
AdamW lr/weight decay 0.0001. Terminal checkpoints followed the fixed recipe.

Both checkpoint-file and serialization-independent state SHA-256 identities
are available. The new `artifacts/models/wp1_full8000_state_identity_v1.json`
includes the versioned definition and per-tensor manifest. Original weights
and file hashes are unchanged. Training identities, initialization/runtime/source
records, native scorer and private weights are available in the owner handoff.

The earlier official-333 run is producer-verified, pending WP5 common-contract
acceptance:

| Population | Images | Top-1 | Top-5 |
| --- | ---: | ---: | ---: |
| Overall | 3,448 | 28.3353% | 51.8852% |
| Seen | 2,000 | 44.8000% | 71.8000% |
| Unseen | 1,448 | 5.5939% | 24.3785% |

H@1 9.94595%; H@5 36.39844%. Producer saved-score replay passed 29 checks.
Historical results are preserved; no subsequent official-score tuning occurred.
Please assess the existing packet against the common contract before requesting
additional official inference. This is not a common four-model comparison or
independent human replay.

Next: Andy's provenance review and Jonathan's packet/replay acceptance under #30.
Applicable WP1/WP4 regressions pass 61/61, no skips. Evidence:
`docs/WP1_WP4_Acceptance_Supplement_2026-11-12.md`.

## Issue #29 - WP4

LeafLens owner-side final-333 changes and added acceptance checks are complete.

- Accepted layout retained: inline model choice, whole-image preview, 60/40
  image/Top-5 center and permanently visible scrolling references.
- All 333 global candidate IDs; semantic hash matches #25:
  `ba17e94119e8b2b7a4b18603e25405d2f52bf4a93396973f80491800697eb62f`.
- Training-only references use all 8,000 original training images, exclude query
  and pixel-identical copies, and label same-disease/different-crop fallbacks.
- Final CLIP models now load the original official NPZ bank with file SHA-256
  `1e1183371b7ad3b1687f7a140af1fea89e1fa6744ac98f1a30ad6fdae19be22b`.
  Stored float64 rows are used directly without re-encoding or renormalization;
  file and tensor identities are recorded separately in exports.
- The synthetic added-candidate check covers Analyze -> Top-1 candidate 39 ->
  selected-result/class alignment -> labelled fallback -> JSON export. All
  synthetic evidence is labelled and carries no accuracy claim.
- Real Frozen CLIP and historical M6 training-image rehearsals pass using the
  original bank. The earlier full-training WP1 rehearsal remains available.
- Applicable WP1/WP4 regressions pass 61/61, no skips.

Setup: `python scripts/fetch_official_clip_text_bank.py`, or set
`LEAFLENS_CLIP_TEXT_BANK`. The GUI never downloads assets. Evidence/setup details:
`docs/WP1_WP4_Acceptance_Supplement_2026-11-12.md`.

Remaining: WP3 supplies the selected full-8,000 adapter/schema/hashes for
integration; current M6 is explicitly legacy-4,868. Frederick records his own
independent rehearsal. Jonathan's common encoder/runtime/score acceptance is
separate from this bank/GUI check. This update does not close #29/#30 or sign
for another owner's acceptance.

## Issue #25 - short coordination note

WP1/#26 and WP4/#29 supplements are ready on the review branch linked above:
unchanged full-8,000 WP1 weights now have versioned file/state identities;
LeafLens uses the original hash-pinned official 333-row CLIP bank and passes
the complete synthetic added-candidate selection/fallback/export check.
Applicable regressions: 61/61, no skips.

Next owners: Andy for WP1 provenance, Jonathan for common-contract packet
acceptance/independent replay, WP3 for the selected full-training adapter, and
Frederick for his independent GUI receipt. No new official inference or
historical-result replacement occurred in this supplement. Detailed evidence:
`docs/WP1_WP4_Acceptance_Supplement_2026-11-12.md`.
