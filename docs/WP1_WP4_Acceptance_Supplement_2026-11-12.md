# WP1/WP4 acceptance supplement - 12 November 2026

Review branch: `feature/wp1-wp4-final333-20261111`; trackers #26/#29.
This supplements the [11 November release](WP1_WP4_Final_Release_2026-11-11.md).
Training recipes, weight files and historical results are unchanged.

## WP1 state identities

The [public sidecar](../artifacts/models/wp1_full8000_state_identity_v1.json)
records file and state-dictionary hashes separately, with a per-tensor manifest.
Namespace: `cos30082-state-dict-sha256-v1`. The hash includes sorted tensor names,
dtype/shape metadata and contiguous little-endian raw bytes with explicit length
framing, including parameters and persistent buffers. Checkpoint serialization
and metadata are excluded. The full definition is in the sidecar and
`checkpoint_identity.py`. Both states contain 320 tensors.

| Task | State-dictionary SHA-256 |
| --- | --- |
| Crop | `19917af4b944c2c03c05547e749867035a06665e6a07008b7b6f2fbe08ae5d04` |
| Disease | `f9a82d429cc51c1c4f9c7bfe2c84745b7e945906ca20640bba806b5bde59c8a9` |

File hashes still match the full-8,000 release registry. Use
`python scripts/export_wp1_state_hashes.py --help` to regenerate to a new path.
Existing outputs are preserved. No retraining or official-score selection occurred.

## Original official CLIP bank

The shared #25 Drive bank was downloaded and verified. Its exact NPZ file hash is
`1e1183371b7ad3b1687f7a140af1fea89e1fa6744ac98f1a30ad6fdae19be22b`.
Its float64 `[333,768]` tensor has raw C-order byte hash
`5bbb5f715620c65fa76b2f32c84b72dddf0d9ceae8d43c7e9720a2240ff999b6`.
These are different hash definitions, recorded separately in the
[bank identity](../artifacts/models/official333_clip_text_bank_identity.json).

Final CLIP loading validates exact file bytes, fields, precision, global IDs
0..332, finite values and unit norms. Stored rows are used without prompt
re-encoding or renormalization; native image normalization remains unchanged.
Historical development behavior is preserved. Old M6 remains explicitly
development-4,868; a new text bank does not make its adapter full-training.

Run `python scripts/fetch_official_clip_text_bank.py` explicitly, or set
`LEAFLENS_CLIP_TEXT_BANK`. The binary is excluded from Git. GUI startup never
downloads it and rejects a missing/mismatched bank. Bank agreement does not
establish historical saved-score or shared image-encoder/runtime parity.

## GUI checks

`python -B -m unittest discover -s tests -p 'test_wp[14]*.py' -v`
passed **61 tests, no skips**. Added coverage includes state identities, strict
bank validation, no final-mode text re-encoding and real GUI event handling.

`scripts/rehearse_wp4_added_candidate.py` injects a labelled synthetic scorer
only inside the harness. Candidate 39 (pepper bell / bacterial spot), absent
from training references, is Top-1. Real Analyze, ranking selection, class
alignment, labelled same-disease/different-crop fallback, training/query exclusion
and JSON export pass. Synthetic screenshots, exports and receipts are labelled;
this establishes no neural inference or accuracy. No production fake option exists.

Private evidence:

- `runs/wp4-added-candidate-synthetic-20261112/`: complete synthetic path,
  export, receipt and visually inspected 1280 x 840 screenshot.
- `runs/wp4-official-bank-frozen-20261112/`: real Frozen CLIP training-image
  inference/export with original-bank provenance.
- `runs/wp4-official-bank-m6-20261112/`: corresponding real historical M6 check.

Native rehearsals compare the GUI with its canonical scorer, verify all 333
columns and JSON exports, and exercise added-class browser fallbacks. They read
no official test images. The synthetic harness separately covers an added class
arriving through prediction selection, rather than only class browsing.

## Pending peer/team acceptance

- Andy: review WP1 initialization, recipe and ordered training identities (#26).
- Jonathan/WP5: accept the existing producer packet under the common manifest,
  column/tie/runtime contract and perform independent saved-score replay (#30).
- WP3 owner: supply the selected full-8,000 adapter, schema and hashes for
  integration and rehearsal (#28/#29).
- Frederick: independently operate the GUI and record his own receipt (#29).

The owner-side WP1 official result remains Overall Top-1 28.3353%, Seen 44.8%,
Unseen 5.5939%, H@1 9.94595%, pending WP5 acceptance. No additional official
inference, common four-model completion or peer sign-off is claimed here.
