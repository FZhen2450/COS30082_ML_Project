# Fa Zhen: WP1 / WP4 submission handoff — 14 November 2026

For report/slide editors and the demo coordinator, under issue #31.
Group: **9**, following Yuen Kai's 14 November allocation-sheet confirmation.
This is a section/asset handoff, not the final team report or submission ZIP.
Round 2 is excluded. The final team table awaits Jonathan's WP5 acceptance.

## Report text: independent baseline and final refit

WP1 uses two independent ResNet50 classifiers, one for crop and one for disease,
each initialized from verified ImageNet V2 weights. The final refit consumed all
8,000 original training images per epoch, using seed 42, batch size 2 and AdamW
with learning rate and weight decay both 0.0001. The crop model ran for two
epochs and the disease model for three, following the development-selected
budgets. The fixed terminal checkpoints were retained; official-test results
were not used to choose an epoch or replace a checkpoint.

Both official folds rank the same 333 ordered crop–disease candidates. The
native FP32 composition score is the sum of the crop and disease log
probabilities; exact ties use ascending global candidate ID. The official
image path uses untransposed PIL RGB and the declared native preprocessing.
The GUI uses a separately documented EXIF-corrected RGB, batch-one path, so its
engineering checks do not establish equality with every producer runtime.

The full-training WP1 packet records the counts below. Producer replay
reproduced all 3,448 Top-5 rows; Jonathan separately reported 89 independent
saved-score checks without discrepancies. Subsequent common-manifest semantic
verification also passed. This does not establish independent checkpoint-byte
verification or remove the limitation that the original serialized Katana
common manifest was not retained. Use the final WP5-approved table in the
submitted report.

| Population | Rows | Top-1 hits | Top-1 | Top-5 hits | Top-5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Overall | 3,448 | 977 | 28.3353% | 1,789 | 51.8852% |
| Seen | 2,000 | 896 | 44.8000% | 1,436 | 71.8000% |
| Unseen | 1,448 | 81 | 5.5939% | 353 | 24.3785% |

H@1 is 9.94595% and H@5 is 36.39844%. Overall accuracy uses pooled integer
counts; each harmonic mean combines its Seen and Unseen rates. Historical
4,868-fitting results remain separately labelled. These full-training results
show substantially weaker unseen-composition performance than seen accuracy;
they do not establish uniform crop-independent disease recognition.

## Report text: LeafLens interface and retrieval

LeafLens supports a shared 333-candidate final mode in a three-column desktop
interface: inline model/asset controls, a whole-image preview with Top-5
rankings, and scrollable training references. Assets are checked by SHA-256
and contract before loading. The published M6 integration distinguishes the
full-8,000 adapter from the historical adapter in its provenance and displayed
model label. Final CLIP inference uses the original hash-pinned 333-row FP64
text bank without re-encoding or renormalizing the stored rows.

Reference retrieval uses only the original training pool. It excludes the
query and pixel-identical copies, and labels same-disease/different-crop
fallbacks when an exact pair has no training example. Ranking values are
model scores rather than calibrated confidence percentages. JSON export
preserves the predictions and provenance.

Owner-side native training-image rehearsals verified model loading, scoring,
ranking, retrieval and export. PR #37 passed 42 WP4 tests without skips both
locally and in its isolated publication checkout. Yuen Kai separately reported
42 WP4, 22 GUI parity, 21 WP1 and 11 data-audit tests passing in another
Windows runtime. Frederick's independent human rehearsal remains pending.
Small-window Top-5 clipping is a known issue on the published build; the
separate proposed fix requires actual source review and validation.

## Slide content: WP1 (one slide)

- Independent crop/disease ResNet50 heads; verified pretrained initialization.
- Final training: 8,000 images, fixed crop-2 / disease-3 epoch budgets.
- Common official task: 333 candidates, 2,000 Seen + 1,448 Unseen images.
- WP1 Top-1: Overall 28.34%, Seen 44.80%, Unseen 5.59%; H@1 9.95%.
- Saved-score replay verified; final team comparison remains WP5-controlled.
- Limitation: high seen accuracy does not imply unseen disease transfer.

## Slide content: LeafLens (one slide)

- Whole-image preview, inline model loading and Top-5 with JSON export.
- Hash-verified full-8,000 M6 and original official 333-row CLIP bank.
- Training-only references; query exclusion and clearly labelled fallbacks.
- Scores are ranking values, not confidence percentages.
- Native example can be incorrect; show the limitation honestly.
- Small-window fix and independent laptop acceptance remain pending.

## Speaking script (about 70–90 seconds; rehearse and time it)

My contribution covers the independent baseline and LeafLens integration.
For WP1, we train separate crop and disease ResNet50 classifiers. The final
models use all 8,000 training images with fixed budgets of two and three
epochs. We combine their log probabilities and rank the same 333 candidate
pairs for both official folds. WP1 achieves 44.8 percent Seen and 5.59 percent
Unseen Top-1 accuracy, giving a harmonic mean of 9.95 percent. The gap shows
that recognizing familiar examples remains much easier than transferring to
unseen crop–disease combinations.

In LeafLens, users load verified model assets, select a leaf image and inspect
the Top-5 results alongside training references. References exclude the query,
and same-disease examples from another crop are explicitly labelled. The
displayed values are ranking scores, not confidence percentages. Our native
rehearsal example is misclassified, so we present it as an integration check
rather than evidence of accuracy. Automated loading, ranking and export checks
passed, while independent laptop acceptance and the small-window layout review
are tracked separately.

## Demo procedure for Frederick

1. Record source revision, Python/Torch/Tk versions, device, window size and
   display scaling. Record checkpoint/text-bank hashes, with no private paths
   or credentials in the public receipt.
2. Until the small-window fix is reviewed, use a window at least 1280×840 and
   Windows scaling of 100–125%, per Yuen Kai's compatibility note.
3. Launch `python gui.py` from the reviewed build with existing compatible
   dependencies and private assets. Confirm `FINAL / 333 CLASSES`.
4. Load M6. Confirm `M6 full-8000 / final-333` and the adapter identity
   `a6bd8980…07f0`. Do not replace it with the historical `8835a18b…35cc`.
5. Use an original training image, check all five rows and the ranking-score
   note, choose another ranking, inspect references and export JSON.
6. Browse candidate 39 to inspect its labelled same-disease/different-crop
   fallback. If synthetic scores are used to force an added candidate into
   Top-1, label that segment synthetic and keep it separate from native M6.
7. Check resize behavior, export/prediction agreement and any failures. The
   team video must be at most eight minutes and include all five speakers.
8. Frederick records his own observed outcome on #29. This guide and the
   owner's automated rehearsal cannot stand in for his human acceptance.

## Screenshot captions for the private handoff ZIP

- `01_wp1_native_training_query.png`: native WP1 final-333 inference on training
  image `PWV2_peach_leaf_curl_6128.jpg`; Top-1 is incorrectly coffee/miner,
  while peach/curl is rank 2. Engineering demonstration, not an accuracy test.
- `02_m6_native_training_query.png`: the same training query using full-8,000
  M6; Top-1 is incorrectly apple/rust and truth is absent from Top-5. The native
  scores and JSON export match. Do not caption this as successful diagnosis.
- `03_labelled_reference_fallback.png`: class browsing shows same-disease,
  different-crop training references. It is a retrieval demonstration rather
  than evidence that M6 predicted the added candidate.

All three are the preserved 12 November-or-earlier owner screenshots of the
pre-layout-fix build. They must not be labelled screenshots of the pending fix.

## Evidence and publication

- WP1 recipe: `configs/wp1/resnet50_full8000.json`.
- WP1 aggregate counts: `artifacts/results/wp1_full8000_official333_20261111.json`.
- WP1 state identities and release: `artifacts/models/wp1_full8000_state_identity_v1.json`
  and `artifacts/models/wp1_full8000_release.json`.
- Andy's independent provenance sign-off is preserved in PR #32.
- M6 source/rehearsal handoff: PR #37 and
  `docs/LeafLens_M6_Publication_2026-11-14.md`.
- Source handoff Drive: https://drive.google.com/drive/folders/1aPB3thwyNAzmeViMzUUKbwKMnaRuQ-mg
- WP5 acceptance and limitations: issue #30 comments dated 12–14 November.

WP2/M6 original score packets supplied on 14 November contain labels and stay
private. Local archive/ledger integrity verification does not replace
Jonathan's independent numerical and provenance acceptance. Round 2 results
are exploratory and are not substituted into this First Run handoff.

Before final submission, the editor must confirm the page-limit interpretation,
use the final accepted four-model table, verify Group 9 cover/member details,
collect signatures and verify video/access/submission receipts with the team.
