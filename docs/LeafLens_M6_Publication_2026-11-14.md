# Full-8,000 M6 source handoff — 14 November 2026

Tracking: [#29](https://github.com/FZhen2450/COS30082_ML_Project/issues/29).
This publishes the existing 12 November owner-side integration for independent
review and Frederick's GUI rehearsal. Base: PR #32 head
`90e70c40b1e565a48b37bc6801da9466352ad592`; its history and Andy's provenance
sign-off are retained. This change is intended to be reviewed on top of #32.

## Implementation

- Final-mode discovery selects M6 checkpoint `a6bd8980…07f0` by content hash and
  verifies the ordered 8,000-image training identity. Development mode retains
  the historical adapter. Switching protocol clears the old adapter selection.
- The loader validates the full-training contract, fixed terminal epoch,
  recipe, frozen encoder and finite architecture-compatible state.
- Final inference retains the original official 333-row FP64 text bank without
  re-encoding or renormalizing it. The GUI/export identifies the loaded adapter
  as full-8,000 or historical according to verified provenance.
- The rehearsal harness records finite score checks, stable Top-5, all ordered
  candidate IDs, raw source hashes and canonical score/export evidence.

The production source and rehearsal script are unchanged from the successful
12 November rehearsal. The publication adds the verification records below.
Private model files, full score packets, exports and images are excluded.

## Verification

On 14 November the existing producer-compatible desktop runtime ran:

```powershell
& 'D:/COS30082_WP1_Runtime/venv/Scripts/python.exe' -B -m unittest discover -s tests -p 'test_wp4*.py' -v
```

Result: **42 tests passed, no failures, errors or skips**, 16.759 seconds.
Python 3.13.13, Torch 2.7.1+cu118, Torchvision 0.22.1+cu118, NumPy 2.2.6,
CPU. The initial sandbox run had 9 Tk skips; the desktop run exercised all 9.
The complete desktop log is
[WP4_M6_Regression_2026-11-14.txt](verification/WP4_M6_Regression_2026-11-14.txt).
Only the WP4 suite was rerun; this is not a full-repository or Katana pass.

An additional read-only check matched **19 raw SHA-256 values**: all six
rehearsal companion files, the twelve source files named in the receipt and
the local full-8,000 M6 checkpoint. Full expected/actual values and the original
desktop-log hash are in
[the publication verification record](../artifacts/results/leaflens_m6_publication_20261114.json).
The published text log uses LF newlines; its raw hash can differ from the
original PowerShell log, whose bytes are retained locally.

The patch was then applied with `git apply --check` and `git apply` in an
isolated checkout at exactly `90e70c4`. The same **42 tests passed without
skips**, in 21.377 seconds, in that publication checkout. Its complete log is
[the isolated verification log](verification/WP4_M6_Isolated_Regression_2026-11-14.txt).

The 12 November native inference receipt remains
`61cbdb695313ab45f50f57587c00ce13c78ca454b65c0d97d831cd2288ef3703`.
All companion hashes and runtime/provenance are in
[the original rehearsal index](../artifacts/results/leaflens_full8000_m6_rehearsal_20261112.json).
The training query was misclassified. Successful integration does not establish
accuracy or cross-runtime score parity. No training, new inference or official
test-image access occurred during publication.

## Independent rehearsal setup

1. Check out this branch on top of #32. Preserve the existing Python/CLIP
   environment; use the pinned CLIP revision in the original rehearsal record.
2. Obtain the private M6 handoff already sent by Yuen Kai. Checkpoint SHA-256:
   `a6bd8980bd091df53e567f9e67601d94d1738c998cfd137fd1270cbec9dc07f0`.
   Package SHA-256:
   `1f79d37bfadb4c96aeef5590f66791532a31f01f49dedd45d42a7b9eae7da798`.
3. Retain the pinned CLIP weights and original official text bank. Run
   `python scripts/fetch_official_clip_text_bank.py` if needed, or set
   `LEAFLENS_CLIP_TEXT_BANK` to the original NPZ. Bank SHA-256:
   `1e1183371b7ad3b1687f7a140af1fea89e1fa6744ac98f1a30ad6fdae19be22b`.
4. Start `python gui.py`, choose the final-333 protocol, select M6 and load
   the verified assets. Use original training images for the rehearsal.
   Confirm the full-8000 label, complete Top-5, resize behavior, training-only
   references, added-candidate fallback and matching JSON export.
5. Record the exact source revision, checkpoint/bank hashes, runtime, query,
   window dimensions, observed checks and limitations in Frederick's own
   receipt. The automated harness command is in
   [the original rehearsal guide](LeafLens_M6_Full8000_Rehearsal_2026-11-12.md).

## Remaining handoffs

- Frederick's independent human rehearsal is due **15 November MYT**.
- Jonathan reported Top-5 clipping on shorter laptop windows and has a separate
  local layout patch. That unpublished patch is not included here; review it
  against this branch and repeat the affected resize checks before acceptance.
- WP1 independent saved-score verification is already recorded on #30 (89
  checks); a further 90 checks cover the manifest semantics and frozen CLIP.
  The original serialized Katana manifest was not retained, which remains a
  provenance limitation despite matching reconstructed semantics.
- Yuen Kai owns delivery of the immutable 13 November WP2/M6 score packets,
  independent raw hashes and runtime/release evidence. Jonathan owns their
  intake and final four-model comparison, due **17 November MYT**. Frozen CLIP
  reuse has conditional owner approval; final scorer/runtime compatibility is
  still subject to WP5 verification.
- Fa Zhen's WP1/GUI assets for #31 are available in the existing release and
  rehearsal records. Andy/Frederick coordinate report/demo/forms by 20 November.
  The report's eight-page limit needs clarification for references/appendices;
  eight main pages alone does not establish compliance.

These peer checks remain open. Publication does not merge #32, #24 or #35,
accept official results, or activate Jonathan's separate WP5 audit-gate package.
