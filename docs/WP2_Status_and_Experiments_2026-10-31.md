# WP2 supervised development workflow and experiment evidence

Prepared 31 October 2026. Co-owners: Yuen Kai CHIA and Jonathan Zheng Li CHAI.
Tracking: [Issue #2](https://github.com/FZhen2450/COS30082_ML_Project/issues/2).
[Published comparison update](https://github.com/FZhen2450/COS30082_ML_Project/issues/2#issuecomment-6541380729).
WP3 remains separately tracked in [Issue #3](https://github.com/FZhen2450/COS30082_ML_Project/issues/3).

## Source scope and interfaces

The primary executable WP2 workflow is notebooks/wp2_backbone_comparison.ipynb. Dataset decoding, transforms, split construction, B3/Base model construction, losses, training/validation, metrics, checkpoint selection and diagnostics remain visible there.

The Python modules retain accepted engineering references and team interfaces:

- development_data.py: training-only three-way split and loader references.
- multitask_model.py: shared ResNet18 engineering reference.
- multitask_backbones.py: offline EfficientNet-B0 and ConvNeXt-Tiny engineering wrappers.
- train_multitask.py: weighted loss, bounded epoch functions and state-dictionary checkpoints.

ResNet18/B0/Tiny references do not imply inclusion in the final full-training comparison. EfficientNet-B3 and ConvNeXt-Base experiments are implemented in the notebook. Their output contract is raw crop_logits[B,55], disease_logits[B,175] and shared features[B,D]. Reusable loss/epoch logic is independent of D. Checkpoint metadata, mappings and persistent model buffers are validated.

Publication preparation changes only notebook execution state: outputs/counts are cleared and expensive controls are False. ML function/class ASTs and all other Python AST, apart from those Boolean control assignments, are preserved. Original executed notebooks, checkpoints and experiment records remain outside this preparation branch.

## Environment and execution

Use Katana Ubuntu WSL, Conda as04_yolo_katana: Python 3.10.20, Torch 2.5.1+cu121, Torchvision 0.20.1+cu121, RTX 4060 Laptop 8 GB. These are existing verified environment versions; no packages were installed during preparation.

Root requirements.txt remains the older WP1 CPU-reference specification and pins different Torch/Torchvision versions. Do not install it wholesale into the shared YOLO/ML Conda environment. Dependency policy changes require separate team approval.

The notebook publication copy has every expensive control disabled, including RUN_M5E_PILOT. No model weights are downloaded automatically and no real training or full inference was executed during preparation. Time-consuming operations remain manual VS Code notebook actions. Preparation worktrees do not contain the ignored raw-image dataset; attach/configure the approved data location only for a separately authorized manual run.

## Development contract and model selection

Only the original 8,000 training records create the seed 42 split:

- fitting 4,868 images, with all 55 crop and 175 disease components;
- seen validation 1,238 images;
- internally unseen validation 1,894 images;
- 258 fitting compositions, 65 entire held-out compositions;
- 323 identical ordered original global composition candidates for both folds.

Candidates SHA-256: 4cb0a24a96cf647f32311817fa7b082a46d3cf00c20ae61f4287e83cc898dfaf.
Split membership SHA-256: f0c7ba212449fd68f1d19c0f84cc91bf4cc1af29ae405128e5198bce89f25f85.

Loss is 0.5 crop cross-entropy plus 0.5 disease cross-entropy. Composition scoring remains the accepted PyTorch float64 sum of crop/disease log-softmax scores. Exact score ties use ascending global composition ID.

Checkpoint selection remains highest development H@1, then lowest sample-weighted combined seen/unseen validation total loss, then earliest epoch. Comparisons use full precision and no tolerance. M5E early stopping monitors improvement over global-best H@1 including warm-up, with a minimum fine-tuning grace period. Historical selections are not retroactively changed.

## Recorded results

These are owner-executed training-derived development results, not official-test performance.

| Approach | Seen Top-1 | Seen Top-5 | Internally unseen Top-1 | Internally unseen Top-5 | H@1 | H@5 |
|---|---:|---:|---:|---:|---:|---:|
| Original EfficientNet-B3 | 46.7690% | 76.8982% | 1.4256% | 8.9757% | 2.7668% | 16.0751% |
| Original ConvNeXt-Base | 49.2730% | 79.9677% | 2.3231% | 9.4509% | 4.4371% | 16.9040% |
| M5E capped disease sampler pilot | 79.8061% | 94.4265% | 1.5839% | 14.6779% | 3.1062% | 25.4066% |

Original EfficientNet-B3 completed 23 epochs; original ConvNeXt-Base completed 97. Both selected epoch 1 warm-up checkpoints. Actual budgets and resolutions differ and remain recorded; this is not an assertion of identical realized compute.

The M5E paired pilot used fresh verified initialization and the agreed finite 17-epoch budget. The capped disease sampler selected epoch 15 but did not improve H@1; its 3.1062% result is closed as a negative primary-metric result. Lower loss or higher Top-5 alone is not a success criterion. The original ConvNeXt-Base H@1=4.4371% remains the supervised development reference.

M5D identified weak disease transfer across held-out compositions rather than a candidate-mapping defect. Per-pair/support/error diagnostics and stage-specific resolution metadata remain in the notebook. Historical checkpoint files and version 1 provenance remain unchanged.

## Unresolved byte-provenance gate

This worktree intentionally uses clean-main LF CSV blobs. The training source SHA-256 is 31d19a852f74b6f1095ea13121243a95074d50576d3d5fc5b48baa0ff6eb0994.
Historical manifests/checkpoints/runs record the equivalent CRLF working-copy SHA-256 9f1b729ab08f120c9e8d3a5681aa1ccbdf1d9210f9aaa6143959363dcb106a7d.
Parsed rows, labels, ordering, candidate and split hashes are identical. Historical raw-source comparison remains strict and will reject a mismatched raw hash.

A narrowly scoped text eol=crlf attribute rule was tested only in a disposable checkout. It is NOT present in this publication worktree and requires WP1 approval. Do not rewrite old hashes, regenerate manifests or stage newline-only CSVs to hide the discrepancy. Fresh-checkout provenance must be resolved before publication approval.

## Verification and integration

This preparation worktree passed 193/193 synthetic/structural/data-contract tests using the existing Conda Python, CPU-only execution and two threads. A separate combined WP2+WP3 validation directory passed 242/242 tests in 26.308 s. Full logs and original/source-preservation proofs are in the external preparation handoff. No real-image inference or training was used for verification.

Jonathan's [WP5 PR #1](https://github.com/FZhen2450/COS30082_ML_Project/pull/1) remains separate/unmerged. Notebook-local metrics represent the agreed specification; do not merge or duplicate WP5 code as part of this source preparation. WP4 GUI/retrieval integration, final frozen official evaluation and a reviewed publication PR remain separate work.

Use Refs #2 for a future PR; the tracker contains pending integration and should not be automatically closed.
