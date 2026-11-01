# WP3A frozen CLIP reference and development results

Prepared 31 October 2026. Co-owners: Yuen Kai CHIA and Andy Wei Jian CHU.
Tracking: [Issue #3](https://github.com/FZhen2450/COS30082_ML_Project/issues/3).
[Verified result update](https://github.com/FZhen2450/COS30082_ML_Project/issues/3#issuecomment-6541381541).
Supervised comparison: [Issue #2](https://github.com/FZhen2450/COS30082_ML_Project/issues/2).

## Source and dependency scope

notebooks/wp3_clip_zero_shot.ipynb visibly implements metadata loading, the development split, prompts, image decoding, native preprocessing/tokenization, frozen encoders, normalization, cosine scoring, ranking, Top-1/5, H@1/5, per-pair/error diagnostics and result export. It does not import project ML processing scripts. Tests are in tests/test_wp3_clip.py.

The unit tests use development_data.py as a WP2 reference. This preparation branch starts from main, as requested, so that module is supplied only through an explicit validation PYTHONPATH pointing to the WP2 preparation worktree. It is not copied into the WP3 change set. The future WP3 PR should follow the WP2 merge, or use an explicitly approved stacked base.

The publication notebook has outputs/counts cleared and all 9 expensive controls False. Its current inert CONFIG retains the owner's last selected L/14 model; choosing B/32 remains explicit. All function/class ASTs and non-control Python AST are unchanged. Owner-executed originals and results are preserved.

## Model and environment provenance

Official OpenAI CLIP 1.0, pinned revision d05afc436d78f1c48dc0dbf8e5980a9d471f35f6.
Existing approved additions: ftfy 6.3.1 and regex 2025.9.18. Existing Katana environment: Python 3.10.20, Torch 2.5.1+cu121, Torchvision 0.20.1+cu121. No packages were installed or changed in preparation; do not install the different root requirements pins wholesale into the shared environment.

Frozen ViT-B/32 has 151,277,313 parameters and 512-dimensional embeddings. ViT-L/14 has 427,616,513 parameters and 768-dimensional embeddings. Both have zero trainable parameters in this workflow.

Official weight SHA-256:

- ViT-B/32: 40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af
- ViT-L/14: b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836

Assets remain outside Git and full hashes are verified before local loading. Missing/mismatched assets fail; no random fallback or implicit download is used. The optional acquisition control remains False. Native transforms use shorter-edge bicubic resize 224, center crop 224, RGB and CLIP mean/std; native BPE uses 77 tokens with truncation disabled. This is not the WP2 ImageNet pipeline.

## Fixed development and prompt contracts

Source 8,000 records; fitting 4,868 / seen validation 1,238 / internally unseen validation 1,894; 55 crop and 175 disease components; 323 identical ordered global candidate IDs. Seed 42, with no official-test development use.

Candidate SHA-256: 4cb0a24a96cf647f32311817fa7b082a46d3cf00c20ae61f4287e83cc898dfaf.
Split SHA-256: f0c7ba212449fd68f1d19c0f84cc91bf4cc1af29ae405128e5198bce89f25f85.
Final prompt policy: label_only_single_template_v2.
Policy/table SHA-256: 753672c7964bcb20a1346145f21a72b40a19f55590a006b5783823f23c13f537.

The label-only article rule corrected 23 diseased prompts, leaving all 37 healthy prompts unchanged. Native token shape is [323,77], maximum non-padding length 18. Both backbones use the exact same table and tokens, with separate model-specific text embeddings.

Scores use float64 CPU scaled L2 normalization and direct image/text cosine similarities. Exact ties are resolved by ascending global ID. Do not substitute WP2's crop/disease-logit formula for direct CLIP candidate scores. H=2SU/(S+U), with zero for S=U=0.

## Actual owner-executed results

| Approach | Seen Top-1 | Seen Top-5 | Internally unseen Top-1 | Internally unseen Top-5 | H@1 | H@5 |
|---|---:|---:|---:|---:|---:|---:|
| WP2 original ConvNeXt-Base | 49.2730% | 79.9677% | 2.3231% | 9.4509% | 4.4371% | 16.9040% |
| WP3A CLIP ViT-B/32 | 4.1195% | 14.3780% | 2.6399% | 8.5533% | 3.2178% | 10.7259% |
| WP3A CLIP ViT-L/14 | 7.0275% | 22.4556% | 9.5037% | 31.1510% | 8.0801% | 26.0981% |

Run IDs:

- B/32: 20261031T064747Z_ViT-B-32_e819582b
- L/14: 20261031T071549Z_ViT-L-14_b879af52

Both processed 1,238 seen and 1,894 internally unseen images, batch 8. B/32 hit counts: seen Top-1 51, Top-5 178; unseen Top-1 50, Top-5 162. L/14: seen 87/278; unseen 180/590. Saved per-pair CSV counts independently reproduce the weighted metrics and harmonic means.

Recorded image-fold inference totals: B/32 43.21 s, peak allocated 377.25 MiB; L/14 76.82 s, 979.87 MiB. These include decoding/transforms/transfers/encoding/scoring for the two folds, excluding model/text setup. They are owner measurements, not new preparation benchmarks.

L/14 leads these frozen development references on H@1, while WP2 retains much stronger seen recognition. This does not complete the trainable improved approach, prove absence of public pretraining overlap or establish final official-test performance. Disease transfer remains weak; crop/disease agreement of the top-ranked composition is not the same as independent component-head accuracy.

## CSV provenance blocker

Clean-main checkout bytes are LF. Historical run/manifest source hashes use the row-identical CRLF file. Current worktree training SHA-256: 31d19a852f74b6f1095ea13121243a95074d50576d3d5fc5b48baa0ff6eb0994.
Recorded historical source SHA-256: 9f1b729ab08f120c9e8d3a5681aa1ccbdf1d9210f9aaa6143959363dcb106a7d.
Candidate/split/prompt hashes are unchanged. Historical WP2 comparison guards correctly reject the raw-byte mismatch. The proposed scoped attribute rule is tested only in a disposable checkout and awaits WP1 approval. No rule, alias or metadata rewrite was adopted here.

## Manual workflow and tests

Use a fresh as04_yolo_katana VS Code WSL kernel for each model. Execute notebook sections in order; inspect counts/hashes/prompt table; choose CONFIG.model; explicitly enable verified local model loading and text encoding; re-encode all 323 prompts per model; enable full development inference only when the owner chooses. Expected text shapes are [323,512] and [323,768]. Do not reuse v1 or another model's embeddings. All expensive controls are delivered False.

Preparation worktrees omit ignored raw images and weights; unit tests need neither. Full development inference and training were not executed in preparation.

The WP3 worktree passed 61/61 applicable tests (12 inherited data tests plus 49 WP3 tests) with the prepared WP2 reference supplied via PYTHONPATH. Standalone publication remains dependent on WP2's development_data.py being available. A separate combined WP2+WP3 validation directory passed 242/242 tests in 26.308 s without an external PYTHONPATH. The external handoff records all exact commands.

Andy coordination, WP3B's one proposed factorized adapter and controlled ablations, WP5 contract integration, WP4 inference/retrieval and final frozen evaluation remain open. No trainable improvement is implemented in WP3A. Future PR text should use Refs #3 rather than auto-closing the broader tracker.
