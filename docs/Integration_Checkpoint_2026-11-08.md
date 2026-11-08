# Integration checkpoint — 8 November 2026

This checkpoint combines the WP1 audit and independent-classifier workflow, WP2 multi-task models, frozen CLIP and the WP3B adapter foundation, WP5 evaluation, and the current WP4 GUI. Original branch histories and experimental records are retained. No new training or official-test evaluation was performed during integration.

Track remaining work in Issues [2](https://github.com/FZhen2450/COS30082_ML_Project/issues/2), [3](https://github.com/FZhen2450/COS30082_ML_Project/issues/3), [7](https://github.com/FZhen2450/COS30082_ML_Project/issues/7), [8](https://github.com/FZhen2450/COS30082_ML_Project/issues/8), and [13](https://github.com/FZhen2450/COS30082_ML_Project/issues/13).

## Data and evaluation contract

The separate `artifacts/data/development_seed42_reconciled_v1.json` is an exact copy of the independently reconciled owner reference. Its ordered filenames match the original checked-in manifest: **4,868 fitting, 1,238 seen validation, and 1,894 internally unseen validation**. The earlier manifest, whose status still says pending, is retained unchanged. We preserve all 323 ascending global composition IDs, 55 crop classes, and 175 disease classes.

The WP1 catalog hash (`8217be…`) and historical WP3 tuple hash (`4cb0a2…`) describe different JSON namespaces. They do not imply different classes. The training CSV byte SHA-256 remains `9f1b729ab08f120c9e8d3a5681aa1ccbdf1d9210f9aaa6143959363dcb106a7d`. The merged `.gitattributes`, raw CSVs, data utilities, independent-classifier implementation, dependency file, and canonical metric formulas are unchanged.

The development comparison uses complete reconciled folds, one already-selected run per model, and no reselection. Historical training and search budgets differ.

| Selected model | Seen Top-1 | Internally unseen Top-1 | H@1 | H@5 |
|---|---:|---:|---:|---:|
| WP1 independent pair | 53.3926% | 0.6864% | 1.3553% | 15.5783% |
| WP2 ConvNeXt-Base, epoch 1, 224 pixels, identified export | 49.1922% | 2.3231% | 4.4367% | 16.9040% |
| Frozen CLIP ViT-L/14 | 7.0275% | 9.5037% | 8.0801% | 26.0981% |
| M6, seed 45, cap 0.20, epoch 1 | 18.3360% | 19.4826% | 18.8919% | 48.2234% |

**M6 remains the primary and GUI model.** Healthy classes supply most of its gain: unseen diseased-composition hits improve from 79 to 100 out of 1,496, while healthy hits improve from 101 to 269 out of 398. False-healthy predictions on diseased images rise from 58 to 463. This is not uniform disease-transfer improvement. Later exploratory variants with larger H@1 failed their adoption gates. The independent diseased-transfer confidence interval includes zero: improvement is not established; this is not proof of equal performance or no effect.

WP2's 609 identified seen hits versus 610 historical hits remains unresolved. Neither historical record nor its provenance gap is rewritten. The matching resolution pilot's 6→5 counts refer to composition hits in fresh 224/320 arms, not disease-head accuracy or the historical selected checkpoint.

## Reliability corrections

- WP2 CPU diagnostics take explicit detached copies of each batch, preventing reused output buffers from overwriting earlier logits. Future exports snapshot the saved notebook and its output-independent AST. Historical records remain unchanged.
- The M3 design now links its completed experiment and failed disease-safety result. Reviewed M4 residual, current-image-hash and full-fold guards are retained. M5 replication and M6 one-epoch replication source remain default-off.
- The actual GUI predictor uses a bounded streaming SHA helper compatible with Katana's Python 3.10. Model hashes, preprocessing, adapter arithmetic and scoring are unchanged. Its default catalog uses the separate reconciled manifest.
- WP5's robustness adapter rejects fractional, boolean and textual global IDs before integer conversion. The new `notebooks/gui_native_clean_parity.ipynb` provides a complete, task-bound clean-first gate. The low-level `run_native_suite` primitive alone is not that gate.
- Andy's original audit artifact is preserved. A separate Katana audit scanned all 8,000 training-source files: all were readable, all had unique file-byte hashes, and no exact duplicates were found. The deterministic first-200-filename aHash screen found no candidates; it cannot establish absence of near duplicates elsewhere. Audit paths reject traversal and outside symlink redirects. Failed heuristic decodes are recorded. Approval of a 192-pixel preview policy does not establish equivalence to production transforms. Human visual and bias review remain separate tasks.

## Actual GUI clean-parity verification

Both real `CLIPPredictor` variants completed both development folds with untransposed RGB decoding, CUDA native FP16 image encoding in batches of eight, text batches of 32, the saved FP32 M6 adapter, and CPU FP64 scoring. Both direct text banks match historical SHA-256 `7f402c8dbebe821b1e1f2d799c9a0ce0e0d56a10a54aad563c6cd79d510df732`.

All **6,264 model–image ordered Top-5 predictions match exactly**, with zero mismatches. Frozen CLIP reproduces seen Top-1/Top-5 hits of 87/278 and unseen hits of 180/590. M6 reproduces 227/533 and 369/1,038. The pass took 184.198 seconds and peaked at 1,970,383,360 allocated GPU bytes. Historical packets store ranks rather than full score matrices, so this establishes all available rank and identity fields, not historical full-score equality.

The ordinary GUI applies EXIF orientation correction and normally runs CPU/batch-one inference. That remains a separately labelled input/runtime path; the clean pass does not claim it is numerically identical to historical evaluation.

## Bounded native robustness

Only after the complete clean gate passed, the actual predictors processed a deterministic subset: the first four filenames per composition in each fold, retaining **533 seen and 218 internally unseen images**. The fixed WP5 severity-one conditions are darkening, brightening, blur, JPEG compression and Gaussian noise. Each model receives identical native corrupted pixels before its unchanged preprocessing.

The run completed **9,012 model–image predictions**, including 7,510 corrupted predictions, in 321.194 seconds. Peak allocated GPU memory was 1,970,383,360 bytes. Seventy-two independent checks reproduce the saved summary exactly through WP5 metrics; all score arrays are finite FP64.

| Condition | Frozen CLIP H@1 | M6 H@1 | Frozen unseen Top-1 | M6 unseen Top-1 |
|---|---:|---:|---:|---:|
| Clean subset | 13.0357% | 16.1379% | 13.7615% | 19.2661% |
| Darken | 11.1317% | 15.2211% | 11.0092% | 17.4312% |
| Brighten | 12.8616% | 16.3316% | 14.6789% | 21.1009% |
| Blur | 10.3819% | 14.2179% | 9.6330% | 16.0550% |
| JPEG | 13.3257% | 16.7348% | 15.5963% | 20.6422% |
| Gaussian noise | 11.3615% | 16.1379% | 11.4679% | 19.2661% |

Blur causes the largest observed M6 H@1 decrease in this fixed probe. M6's H@1 exceeds frozen CLIP in each tested condition. This is a class-capped development sensitivity diagnostic, not full-fold accuracy, field-image robustness, causal evidence, or checkpoint selection. Subset batching differs from full-fold encoding; comparisons use the subset's own clean condition. No severities, prompts, scorers or models were tuned after seeing these results.

## Verification and environment

The complete integrated CPU suite passed **621 tests** in 120.633 seconds, with no skips. All prior assertions remain. RED→GREEN logs, initial baseline/fixture failures, commands and source hashes are retained in the private handoff. Existing dataset regressions read official CSV metadata as schema fixtures; the new ML workflow never opens official images or uses official labels for fitting or selection.

All nine publication notebooks have valid Python syntax, empty outputs/execution counts and disabled expensive controls. Weights, checkpoints, caches, raw images, credentials and executed experiment notebooks are excluded from Git. The dirty experimental workspace and frozen assets remain intact.

Katana retains `as04_yolo_katana`: Python 3.10.20, torch 2.5.1+cu121, torchvision 0.20.1+cu121 and RTX 4060 Laptop 8 GB. **Do not install the different current requirements pins over this environment.** Other platforms need separately verified environments. Immutable model assets are linked through Issue #8; verify hashes before reconstruction.

## Remaining gates

Development selection remains full-precision highest H@1, then lowest sample-weighted combined validation loss, then earliest epoch, without a tolerance. A shared 333-candidate official evaluation for both official folds is the team's separately frozen working protocol. Dr. Lee's recorded verbal answer, “Seen, unseen, Harmony”, did not formally specify shared versus separate candidates. Semantic information is allowed; official images/truths are excluded from training and selection.

Official Top-1/Top-5, H@1/H@5 and the seen confusion matrix remain pending separate frozen execution. Preserve that boundary when returning to improvement experiments. The main report is at most eight pages including figures/tables; references and appendices are excluded under the recorded lecturer clarification. Prioritize honest methods, subgroup errors, GUI retrieval labels/limitations, reproducibility and the demonstration. Consultation with Dr. Lee can clarify expectations; no numerical accuracy threshold has been guaranteed.
