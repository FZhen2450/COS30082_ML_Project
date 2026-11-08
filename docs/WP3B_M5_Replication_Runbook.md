# WP3B-M5 — Owner-run cached adapter replication

## Purpose and fixed scope

This notebook measures training-seed variability of the accepted M4 unbounded versus cap 0.20 comparison. It does not introduce a new architecture, objective, prompt or selection rule. No new M5 result is claimed before owner execution.

Source notebook: `notebooks/wp3b_seed_replication.ipynb`. All substantive metadata/split, prompt, cache, adapter, loss, optimizer, epoch, scoring, selection, checkpoint and reporting processing is visible in its cells. It imports installed libraries, not project ML processing modules. The original M4 notebook remains a historical record.

Parent: PR15 / `df2c9dd22657456780162b065d721b18e17a3810`; PR15 is an unmerged correction to M4 PR14. Keep this implementation stacked until parent integration is coordinated; do not merge or retarget blindly.

## Predeclared experiments

| Training seed | Arm | Relative cap | Fitting epochs |
| ---: | --- | ---: | ---: |
| 43 | control | None | 5 |
| 43 | bounded | 0.20 | 5 |
| 44 | control | None | 5 |
| 44 | bounded | 0.20 | 5 |

Each arm starts from a fresh identity adapter. Both arms of each seed have identical initial parameters and fitting-loader order. Different training seeds change initialization/order, never the dataset partition. Split seed 42 remains frozen, with 4,868 fitting / 1,238 seen validation / 1,894 internally unseen validation examples, 55 crops, 175 diseases, 258 fitting compositions and 323 ordered original global candidate IDs.

Fixed: ViT-L/14 raw768-dimensional cache;768→64→768 residual adapter with99136 parameters; scale0.1; component CE weights0.5/0.5; identity weight1; same-crop fitting contrast weight1/temperature0.07; AdamW LR1e-4/weight decay1e-4; batch8; no augmentation/AMP/scheduler; five epochs and epoch0 eligibility. Full-precision H@1, then minimum sample-weighted validation loss, then earliest epoch select checkpoints. Disease gates are reported after selection, not used to substitute a different epoch.

Native preprocessing and label-only promptv2 remain unchanged. Composition scoring is the accepted float64 direct CLIP cosine ranking with ascending global-ID exact ties. This is323-candidate development; any future shared333 final evaluation is separately frozen and is not executed here.

## Existing inputs

The notebook prefills the verified canonical cache path and its SHA-256:
`9dbb07ae03ba38d003f5e4c610498a9a8c5ecb57475ec2f04f8e8481deeaa38d`.

It verifies the full artifact, raw tensors, labels, exact ordered filenames, source/split/candidate/prompt hashes, pinned CLIP provenance and native preprocessing, all 8,000 current approved image byte hashes, and exact historical frozen Top-5 parity. It does not decode/encode images, load a CLIP encoder or download weights. The installed OpenAI CLIP package supplies only native transform provenance verification.

Seed42 results are read-only historical references. Their summary/declaration/prediction/history/prompt/selected-checkpoint hashes are pinned in `HISTORICAL_REFERENCE`. Full recorded predictions and checkpoint contracts are checked again for reporting. The historical selection remains control epoch1 and bounded epoch3. The bounded seed 42 disease gate remains failed even if later seeds are favorable.

Do not edit the cache, historical files or expected digests to bypass an error. A changed image or provenance mismatch is a stop-and-diagnose condition.

## Exact manual execution in VS Code WSL

1. Open `/home/user1/COS30082 Applied Machine Learning/00 Assigments/AS02 ML Project/workspace/worktrees/wp3b-m5-replication/notebooks/wp3b_seed_replication.ipynb`.
2. Select the existing `as04_yolo_katana` Python3.10.20 interpreter and start a fresh kernel. Save the notebook before execution. Do not reuse the historical M4 kernel state or overwrite the M4 notebook.
3. Inspect configuration. The cache path/checksum are already filled. Keep the declared seeds, split, budget and model settings unchanged. Default CUDA is explicit; unavailable CUDA raises an error instead of silently changing devices.
4. Set only `LOAD_FEATURE_CACHE=True`, `RUN_REPLICATIONS=True`, `BUILD_REPLICATION_REPORT=True` in the configuration cell.
5. Run All. Cache checks occur first. All four contracts are declared before the first optimizer step. The four five-epoch arms then run sequentially, each with epoch0 and full development validation after every fitting epoch.
6. Inspect the final selected-run table, paired differences, descriptive statistics, gate outcomes and plots. Save the executed notebook. Preserve the reported group directory under `workspace/experiments/wp3b-m5-replication/`.

With the default three flagsFalse, Run All reads training metadata and displays identities only. It starts no epochs, validation sweep, cache/checkpoint load, image access, download, report write or output directory. Codex verification uses that default state and tiny synthetic CPU fixtures; the real four-arm run remains an owner action.

The prior fitting-only phases were around 28–29 seconds per five-epoch arm, excluding validation/setup. Treat this as a planning observation, not a timing guarantee. Cached-adapter memory readings do not represent end-to-end CLIP-encoder memory use. CPU memory readings are recorded as unavailable rather than zero; report reopening preserves this distinction.

## Outputs and interruption policy

A unique timestamp/UUID group directory contains a complete four-arm `replication_declaration.json` plus separate `seed_43_control`, `seed_43_bounded`, `seed_44_control`, `seed_44_bounded` directories. Each has its declaration, direct prompts, initial-state digest, safe adapter checkpoints, epoch history, all 18,792 epoch-level Top-5 prediction records, selected predictions and summary.

Successful completion produces `completed.json` with artifact fingerprints. Failure/interrupt produces `failed.json` and re-raises; it never claims a complete comparison. The same kernel rejects a repeated runner call. There is no automatic resume/retry or checkpoint overwrite. Preserve an interrupted directory, diagnose the cause and begin a fresh deliberate attempt if justified.

Reporting verifies recorded artifacts, recomputes hit counts, checks complete budgets and the selected epoch, and checks identical epoch0 initialization within each seed. Partial/missing/duplicate arms or changed files are rejected.

Report files contain all six selected runs including historical seed 42, paired bounded-minus-control differences, per-arm mean/sample standard deviation (n=3), epoch histories, subgroup trajectories, per-pair diagnostics and an explicit gate interpretation. All accuracies are fractions internally. Plots display percentages with denominators labelled. Existing report files are re-read and compared, never silently rewritten.

To re-open a completed report in a fresh kernel, leave `RUN_REPLICATIONS=False`, set `REPORT_GROUP_DIRECTORY` to its absolute group path and `BUILD_REPLICATION_REPORT=True`. The prefilled configuration must remain identical. Cache loading is unnecessary for a recorded-result report; the report loader reads checksummed records and small adapter checkpoints on CPU.

## Interpretation boundaries

Higher H@1 does not by itself demonstrate disease transfer. Report unseen-diseased composition hits / 1,496, healthy accuracy / 398, false-healthy errors, fitting-composition preference, per-pair support and H@5. Include seed 42's failed gate rather than pooling it away. Three seeds describe training randomness on one split; they are not independent dataset replications or a formal significance test.

Do not relabel bounded seed 42 epoch1 as selected after noticing its better disease count. Any future safety-gated eligibility policy or disease-preservation intervention is a separate predeclared experiment. No official-test images or per-image labels enter this notebook.

## Lightweight verification

From this worktree using the existing Conda environment:
```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /home/user1/anaconda3/envs/as04_yolo_katana/bin/python -B -m unittest discover -s tests -p 'test_wp3b_replication.py' -v
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /home/user1/anaconda3/envs/as04_yolo_katana/bin/python -B -m unittest discover -s tests -v
```

Results and preservation evidence are recorded in the canonical `workspace/handoffs/WP3B_M5_20261105/` handoff. No package installation is part of setup or verification.

## Oral-defense Q&A

1. **Why keep split seed 42 while changing training seeds?** It isolates initialization and fitting order as the source of variability. Changing the partition would change the comparison task.
2. **Why train both arms with each seed?** Pairing gives both caps the same starting parameters and sample order, reducing confounding from unrelated random differences.
3. **Why can cached embeddings replace repeated CLIP encoding?** CLIP is frozen, native preprocessing is unchanged and there is no augmentation. The verified raw embeddings are constants; adapter gradients flow only through the visible residual network.
4. **Does the best H@1 checkpoint guarantee better disease recognition?** No. Healthy-example gains can raise aggregate accuracy. That is why per-pair and disease-safety outcomes are reported without changing checkpoint selection retrospectively.
5. **What can three seeds establish?** They show observed variation and paired effects for this fixed split and budget. They do not guarantee statistical significance or final-test performance.
