# WP3A integration preview — 1 November 2026

Base: 69c50bef944e06b18dcc13c598164ec04d92d1ea. Tracking: [Issue #3](https://github.com/FZhen2450/COS30082_ML_Project/issues/3).
This is a local validation composition of merged main, the reviewed WP2 dependency set, and incremental WP3A source. No Git merge or commit was performed.

## Explicit ownership and future PR boundary

The top-level development_data.py and other WP2 files in this preview are byte-identical WP2-owned dependency overlays. No WP3-specific development-data copy or alternate splitter was created. Full and incremental inventories are separate; the future WP3A PR must contain only the WP3 increment after WP2 is available on its base.

The reviewed WP3 notebook/test/status document are retained byte-for-byte. New tests/test_wp3_shared_integration.py verifies WP1 candidate-to-prompt alignment, final v2/native preprocessing, identical-score WP5 metrics, units, H@1/H@5, exact/near ties and historical source identity. No model/data/scoring implementation is changed.

The README contains all merged-main content, then the WP2 update, then a WP3A append. Relative to the WP2 preview it is append-only. This avoids discarding the merged team README and makes the intended WP2-first sequence explicit.

## Preserved CLIP protocol and results

Policy: label_only_single_template_v2.
Prompt/table SHA-256:753672c7964bcb20a1346145f21a72b40a19f55590a006b5783823f23c13f537.
All 323 prompts, global IDs, native tokenizer and native CLIP preprocessing remain unchanged. The review copy is unexecuted with all 9 expensive controls False.

Official OpenAI CLIP revision:d05afc436d78f1c48dc0dbf8e5980a9d471f35f6.
The existing owner-run development results remain ViT-B/32 H@1=3.2178% and ViT-L/14 H@1=8.0801%, versus original WP2 H@1=4.4371%. Those runs were not repeated or modified here.

The 31 October WP3 status document remains historical evidence. This page supersedes its former external-PYTHONPATH and CSV/WP5 gate notes: the preview now contains an explicit common WP2 dependency, and merged checkout rules restore the historical CSV source bytes.

## Verification and remaining work

The full WP3A integration preview passed 284/284 tests in 31.155s without an external PYTHONPATH. This includes merged-main WP1/WP5 tests, all accepted WP2/WP3 tests, 11 new WP2 integration tests and 9 new WP3 integration tests. The 22 new-main tests beyond the old 12-test foundation remain intact; no teammate tests were replaced.

WP3B trainable improvements/ablations, new adapters, GUI integration, final frozen evaluation and actual source publication remain separate tasks. The publication sequence is WP2 first, then the WP3-only increment; no duplicated WP2 commits should appear in that second PR.

## Confirmed shared contract

The project owner confirmed Fa Zhen's formal ordered-filename reconciliation passed. This integration run independently reproduced it with the merged shared_protocol.py, using a reference exported from accepted WP2 code and verified against the historical ordered-membership digest, not generated from WP1 output.

Result: ordered_filenames_match_supplied_reference.
Counts: 4,868 fitting / 1,238 seen validation / 1,894 internally unseen validation.
All three lists match exactly, including order. Candidate columns are the same 323 ascending non-contiguous global IDs; fitting retains 55 crops and 175 diseases.

Working training-source SHA-256:9f1b729ab08f120c9e8d3a5681aa1ccbdf1d9210f9aaa6143959363dcb106a7d.
Historical candidate SHA-256:4cb0a24a96cf647f32311817fa7b082a46d3cf00c20ae61f4287e83cc898dfaf.
Historical split SHA-256:f0c7ba212449fd68f1d19c0f84cc91bf4cc1af29ae405128e5198bce89f25f85.
WP1 candidate namespace: wp1-canonical-json-candidate-records-v1.
WP1 candidate SHA-256:8217be059e3a4a144f89081ee43b7cd042bb5a62c370197bf0e3fca1c6d28bf3.
WP1 split namespace: wp1-canonical-json-fold-filenames-v1.
WP1 split SHA-256:edc29a67dc9de9c454e5f548389b4bffd601ebc06423e63b42385be053b7ceff.
The digests have different serializations; matching filenames and component tables establish compatibility. Neither namespace replaces the historical hashes.

Merged WP1 .gitattributes is inherited unchanged. Both fresh preview worktrees reproduce all three historical CRLF CSV hashes. No replacement .gitattributes, metadata rewrite, split regeneration in the repository, or legacy hash alias was added.

The checked-in WP1 protocol/runbook retain their dated pre-reference status. They were not edited. A new reconciled artifact was generated outside Git for verification. WP1 training/evaluation callers should explicitly select that accepted reconciled artifact until its owner publishes an updated default/status; the exact-membership gate itself is resolved.

## Metrics and numerical boundaries

Merged evaluation.metrics returns fractions in [0,1]. Use percentages only for display. Stable candidate-column ties match ascending-global-ID ties because the candidate columns are verified ascending.

The integration checks pass canonical notebook score matrices directly to WP5 evaluate_compositions. They do not substitute the optional WP5 probability-product helper for WP2's accepted float64 sum-log-softmax scorer, and do not replace direct CLIP cosine similarities with two-head scores.

A synthetic extreme-logit probe found the optional product helper on float32 inputs underflows to all-zero candidate scores and can change Top-1. The historical log scorer preserves distinctions. A deliberately artificial H input of 1e-300 gives 1e-300 in the stable WP2 implementation and 0 in the merged WP5 implementation; actual fold accuracies have minimum positive value at least 1/1894. These edge cases are recorded, not silently fixed.

Normal Top-1/5 and H@1/H@5 agree for the operational contract. H comparisons in tests allow 1e-15 diagnostic floating-point difference only; no tolerance was introduced into ranking, exact ties, checkpoint selection or historical results. WP5 summarize_seen_unseen returns H@1; its existing harmonic_mean function is used to check H@5.

## Published team code preserved

[WP1 PR #4](https://github.com/FZhen2450/COS30082_ML_Project/pull/4) and [WP5 PR #1](https://github.com/FZhen2450/COS30082_ML_Project/pull/1) are merged in this base. The WP1 implementation, scripts, configs, manifests and WP5 module/tests are not edited.

The GUI is published on the separate origin/gui branch at 439764618093e22e68e936ab93f9ff3dd5bf1a51, not merged into main. Its service currently loads the independent-classifier adapter. No GUI code was copied, imported or launched.

Future WP2/WP3 GUI adapters must supply ordered global candidate IDs, scores and an explicitly namespaced protocol fingerprint. The GUI Catalog checks WP1's candidate digest; adapters must not relabel the historical 4cb0... model digest as that different namespace. Keep both identities and verify the mapping. GUI adapter implementation is a separate WP4 coordination task.

## Boundaries

Use existing Katana WSL as04_yolo_katana. No dependency changes, trained-weight acquisition, GPU training, full image inference or official-test evaluation occurred in integration preparation. Tests use metadata and bounded synthetic CPU fixtures; existing legacy tests may read official CSV metadata as contract fixtures, not for model selection.

Original experimental/preparation worktrees and historical records are preserved. There are no staged changes, new commits, pushes or PRs. These local previews are for independent review, not authorization to publish.
