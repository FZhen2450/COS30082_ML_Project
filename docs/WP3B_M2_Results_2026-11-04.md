# WP3B M1/M2 evidence and M2 source publication — 4 November 2026

Tracking: [WP3 #3](https://github.com/FZhen2450/COS30082_ML_Project/issues/3). This source publication is stacked on [WP3A draft #6](https://github.com/FZhen2450/COS30082_ML_Project/pull/6), head `300f1d78bb962f67fa2dbdb7b76ed23301abd7df`; WP2 parent is [draft #5](https://github.com/FZhen2450/COS30082_ML_Project/pull/5). Do not merge or retarget without parent-integration checks.

Yuen Kai CHIA manually executed the recorded experiments in VS Code WSL. WP3 co-owners are Yuen Kai CHIA and Andy Wei Jian CHU; publication does not imply Andy authored or reviewed this implementation. AI assistance prepared engineering source/tests and independently checked saved records; experimental ownership and human review remain explicit.

## Published source versus local experimental evidence

This increment publishes only `notebooks/wp3b_trainable_adapter.ipynb`, `tests/test_wp3b_adapter.py` and this document. Notebook outputs/execution counts are cleared and expensive controls are False. Function/class ASTs match the original executed notebook. Existing M1/M2 notebooks, run directories, predictions and checkpoints remain byte-preserved outside Git. No training or image inference was repeated for publication.

M2 uses official frozen OpenAI CLIP ViT-L/14 and native 224 preprocessing, a 768->64->768 ReLU visual residual (99,136 parameters), zero up-projection and scale 0.1. Text/CLIP parameters stay frozen. The objective is 0.5 crop CE + 0.5 disease CE + unit-feature squared-L2 identity regularization. Component prototypes average only the 258 fitting-composition text embeddings. Primary inference ranks the original 323 direct candidate texts, not component-product scores. Exact ties use ascending global IDs. Training consumes only 4,868 fitting images. No optimizer resumption is implemented.

## Verified historical results

All values are percentages; unseen is training-derived validation, not the official test.

| Reference | Seen Top-1 | Unseen Top-1 | H@1 | H@5 |
| --- | ---: | ---: | ---: | ---: |
| Frozen ViT-L/14 | 7.0275 | 9.5037 | 8.0801 | 26.0981 |
| M1 factorized frozen diagnostic | 4.3619 | 2.7455 | 3.3699 | 20.5077 |
| M2 epoch 1 | 16.1551 | 11.2460 | 13.2608 | 43.1441 |
| M2 selected epoch 5 | 23.1018 | 9.3981 | 13.3608 | 41.7819 |

M1 remains an unchanged negative diagnostic. M2 selected epoch 5 according to highest H@1, then lower sample-weighted validation loss, then earliest epoch. Epoch 1's higher unseen accuracy/H@5 does not authorize retrospective selection. M2 per-image hits reproduce seen 286/1,238 and unseen 178/1,894. Its unseen diseased hits collapse from frozen 79/1,496 to 16/1,496, while healthy hits improve 101/398 -> 162/398; false-healthy errors rise 58 -> 490. This is a higher-H@1 result with a serious disease-transfer limitation, not uniformly improved recognition. M2 unseen tomato 2/333 and cassava 0/663 contrast with frozen 0/333 and 3/663. Predicted fitting-composition counts rise 1,263 -> 1,600/1,894. Paired frozen/M2 unseen: 60 both-correct, 120 frozen-only, 118 M2-only, 1,596 both-wrong.

M2 epoch-1 subgroup/pair predictions were not exported; only its aggregate history and checkpoint exist. Do not invent subgroup results from the aggregate. Owner-only later inference is required. No statistical significance, final test result or cause of failure is established.

## Provenance and reproducibility

Training source: 8,000 rows; ordered folds 4,868 / 1,238 / 1,894, seed 42; 55 crops, 175 diseases, same 323 non-contiguous candidate IDs. No official-test tuning. Historical source hash `9f1b729ab08f120c9e8d3a5681aa1ccbdf1d9210f9aaa6143959363dcb106a7d`; split `f0c7ba212449fd68f1d19c0f84cc91bf4cc1af29ae405128e5198bce89f25f85`; candidate `4cb0a24a96cf647f32311817fa7b082a46d3cf00c20ae61f4287e83cc898dfaf`; v2 direct-prompt hash `753672c7964bcb20a1346145f21a72b40a19f55590a006b5783823f23c13f537`. WP1 hashes have separate serialization namespaces.

M1 completed run `20261102T075353Z_b017e93531c2`; M2 `20261104T042104Z_a4708eb25dca`. M2 full execution contract `9cdb7abc6ea17c34b63181b9f459d2a3c82b746922ddee77158bcdf6ae83e3bc`. Checkpoints are adapter-only state dictionaries plus the fixed text bank, class/candidate mappings, immutable contract and selection record; loading is CPU `weights_only=True`, with no network/backbone reconstruction. Publication source references the canonical Katana workspace; paths are explicit local requirements, not portable hidden downloads.

Use the established `as04_yolo_katana` environment (Python 3.10.20 / Torch 2.5.1+cu121 / Torchvision 0.20.1+cu121 / OpenAI CLIP revision d05afc436d78f1c48dc0dbf8e5980a9d471f35f6). Do not install main README reference pins wholesale. From this checkout:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 /home/user1/anaconda3/envs/as04_yolo_katana/bin/python -B -m unittest discover -s tests -v
```

Full publication validation: 320 tests passed (284 inherited WP2/WP3A + 36 M2). Tests use synthetic CPU features/images and default-off notebook execution; they do not load CLIP weights or evaluate real images. All substantive model/data/loss/training/scoring/checkpoint code is visible in the notebook. The owner's canonical workspace contains local verified images/weights and historical run evidence; these are deliberately excluded from Git.

Future owner runs require a fresh kernel, a unique declaration and explicit flags. No experiment is authorized by opening/Run All with defaults. The final 333-candidate versus separate-candidate official evaluation remains unconfirmed. Proposed M3 work must preserve this historical record and declare a new intervention before new validation results.
