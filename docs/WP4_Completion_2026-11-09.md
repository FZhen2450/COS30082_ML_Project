# WP4 engineering completion audit — 9 November 2026

Issue: [#13](https://github.com/FZhen2450/COS30082_ML_Project/issues/13).
Base: main `bbdabfdc665c711a02b3ca94cc285eb9ed86f97a`.

Publication update: the WP4 source is pushed to `fix/wp4-completion-20261109`
at [5c9d1f9](https://github.com/FZhen2450/COS30082_ML_Project/commit/5c9d1f9b05f7de9f9c86082ecd6bf1f676048d21).
The combined WP1/WP4 branch `fix/wp1-final-handoff-20261109` contains it at
[7314644](https://github.com/FZhen2450/COS30082_ML_Project/commit/73146444ace8637426042e3bfb76aa9bb3b8585c).
Both GitHub refs were read back and matched. A new combined publication regression
passed **95/95**, no skips; the 79-test receipt below retains its original scope.
Neither branch is merged into main, and no peer approval or issue closure is implied.

The original issue checkboxes lag its comments: PR #20 is merged, its GUI is
integrated on main, and the 8 November comment records full historical-path
development parity. The implementation requirements below are present. This
follow-up repairs additional presentation, export and Tk lifecycle issues and
provides repeatable real-model rehearsals. It does not certify another student's
review, independent operation or video recording.

## Issue #13 deliverable audit

| Requirement | Verified state and evidence |
|---|---|
| Inspect interfaces/schema and reuse the existing GUI | `Catalog`, `present_prediction`, the picker, native loader and M6 envelope were inspected. Existing LeafLens is reused. PR #20 records the original implementation; this dated document records the follow-up file scope. A separate pre-implementation plan post in the original issue cannot be independently established from its visible timeline. |
| Verified explicit local CLIP/M6 loading | Present in `wp4/clip_predictor.py`: exact package revision and artifact hashes, strict native preprocessing/mappings, safe CPU adapter loading, frozen native encoder, no model download or random fallback. Real CPU picker rehearsal completed. |
| Canonical outputs, names, ties and score meaning | 323 original ordered IDs; native FP32 WP1 and FP64 CLIP/M6 scores. Exact/near ties and nonfinite values tested. Additional regressions reject Boolean/floating IDs and Boolean scores. Query filename, crop/disease and selected model are visible; raw scores are not confidence percentages. |
| Separate hash namespaces | Existing CLIP/M6 bridge retains historical and catalog identities after checking exact ordered IDs/pairs/names. New WP1 GUI provenance includes actual checkpoint-file hashes and source/split/catalog hashes. |
| Explicit development-only scope | The GUI and exports remain development-323. All rehearsal queries are allowlisted training-derived records; no official image/label result is used for UI selection or thresholds. |
| Fitting-only, honest reference retrieval | Query and pixel-identical copies are excluded. All three real rehearsals verify fitting membership. Real unsupported apple/alternaria class browsing shows clearly labelled cabbage/cauliflower same-disease examples, not invented exact apple matches. Sparse classes may show fewer than four references. |
| Synthetic failures and UI thread boundaries | 79/79 applicable isolated regressions passed without skips. Includes wrong hashes/labels/columns/dimensions, non-contiguous IDs, exact/near ties, NaN, safe missing assets, no-download import, model-switch state, closing during work and reference isolation. |
| Asset handoff and manual smoke procedure | Updated CLI handles WP1, frozen CLIP and M6 through the real picker, with no injected predictor. Normal and minimum-size application-window screenshots, full scores, JSON export and source/runtime identities are saved privately. See the demo/rehearsal runbook. |

The first original checkbox includes a historical process requirement, not just
an implementation feature. Do not claim an absent pre-implementation issue post
was made. Functional verification and the current file inventory are explicit.

## Follow-up source scope

- `wp4/service.py`: validate ID/score types before presentation; bind WP1 loader
  provenance to before/after checkpoint-file hashes without changing its scorer.
- `wp4/gui.py`: export actual EXIF-corrected RGB input identity and WP1 provenance;
  keep completion callbacks on the UI thread and pass worker-only data to threads.
- `wp4/model_picker.py`: release StringVars and callbacks on the UI thread;
  make dialog destruction idempotent.
- `scripts/rehearse_wp4_m6.py`: backward-compatible M6 default, optional WP1 and
  frozen-CLIP modes, real fallback browsing, minimum-size checks and Windows
  application-HWND capture. No source/evaluation threshold or model changes.
- `scripts/launch_wp4.ps1`: explicit Python/runtime launcher and optional existing
  CLIP package directory; does not install packages or alter the caller's environment.
- Focused service/picker tests and these two dated runbooks.

## Verification and recovered failure

The initial combined Windows regression exposed delayed Tk variable destruction
on a worker thread and terminated with `Tcl_AsyncDelete`. The evidence log is
retained. Worker closures now snapshot model/catalog/query data rather than the
application; queues contain only data/errors, while UI callbacks stay on the UI
thread. Picker destruction clears variables and callbacks on that thread, and
closing cancels UI callbacks and safely discards late worker results.

**79 tests passed, no skips**, in a separate main-based Git checkout, using the
same module order that exposed the crash. Scope: WP4 service/native predictor/
picker, independent baseline, shared protocol, data, WP1 export paths and native
GUI parity guards. This is an applicable combined suite, not a new full 621/642
Katana run. No broader Windows-suite pass is claimed.

Runtime: Python 3.13.13 / Torch 2.7.1+cu118 / Torchvision 0.22.1+cu118.
The GUI rehearsals use CPU and ordinary EXIF-corrected RGB, single-image input.
Existing pinned CLIP/ftfy/regex package files were copied byte-for-byte from an
existing local review environment into an ignored support directory; Torch and
Torchvision were not copied or overridden. No package was installed or upgraded.

## Real GUI evidence

Predetermined seen-development query: `PWV2_peach_leaf_curl_6128.jpg`, truth
peach/curl, original ID **219**. Each model is loaded through the normal picker,
performs actual image inference and is checked against its canonical predictor.
No cached training embedding or injected predictor stands in for image encoding.

| Model | Top-5 global IDs | References for first prediction | Honest interpretation |
|---|---|---:|---|
| WP1 | 102, 97, 104, 103, 219 | 4 | Incorrect Top-1; truth at rank 5. |
| Frozen CLIP | 293, 292, 53, 291, 41 | 1 | Truth absent from Top-5; sparse fitting references are shown honestly. |
| M6 | 15, 10, 293, 292, 4 | 4 | Incorrect Top-1; truth absent from Top-5. |

These are engineering checks of one declared development image, not accuracy
estimates. Keep M6's uneven healthy/diseased transfer and false-healthy limitation
in the presentation. No claim is made that normal CPU/batch-1/EXIF decoding
equals the historical CUDA/batch-8/no-EXIF path. The earlier 6,264-row parity and
bounded robustness receipts retain their original runtime/scope and are not rerun.

Pinned private files remain:

- CLIP: `b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836`.
- M6: `8835a18b9eb933ae224bd17866f18bc56406e24d8612743f4e7c88848a8135cc`.
- WP1 crop: `f0c82ca78153f492078a3a39a11eac1a847cc85eb496cf3b3d4d74db752c7d14`.
- WP1 disease: `f497e19da62b68dfbcfbae58b9a5318a26b1989ba59b7a486223cad0d56e31a5`.

Private evidence root: `runs/wp4-completion-20261109/`. Final real-model
receipts are in `wp1-final-rehearsal/`, `m6-final-rehearsal/` and
`frozen_clip-final-rehearsal/`. Earlier experiments/logs are retained separately;
only verified LeafLens application-window captures belong in the team handoff.

## Remaining human/review gates

- [ ] Frederick independently launches the final source and records his real
  operation/review receipt; this automated rehearsal cannot sign for him.
- [ ] Chia/Jonathan review the source/provenance handoff; do not invent approval.
- [ ] The team records actual speakers/narration, verifies names/IDs and signs
  required forms. Screenshots and a script are not a completed team video.

The shared final candidate interpretation remains a separate lecturer discussion;
the GUI continues to use the frozen development scope. Keep Issue #13 open for
the genuine review/recording gates rather than closing it from synthetic tests.
