# Completed M3 paired pilot and next decision — 4 November 2026

Owner-executed notebook checked directly against declarations, checkpoint state and every saved prediction. This supersedes the earlier no-results status in the dated M3 design document; it does not rewrite historical records. Research #9, parent source PR #11; next research #12.

| Selected arm | Epoch | Seen Top-1 | Unseen Top-1 | H@1 | H@5 | Diseased unseen correct | False-healthy diseased errors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Unchanged M3 control / historical M2 | 5 | 23.1018% | 9.3981% | 13.3608% | 41.7819% | 16/1,496 | 490/1,496 |
| Same-crop conditional term | 1 | 18.0937% | 12.3020% | 14.6461% | 45.1578% | 71/1,496 | 873/1,496 |

Healthy composition hits remain 162/398. The additional 55 diseased hits account for the unseen Top-1 increase, but many other diseased cases move to healthy predictions. Frozen CLIP still has 79 diseased hits and only 58 false-healthy errors. Therefore the intervention improves the primary development metric yet **fails the predeclared disease-safety gate**. Do not call it uniformly superior or a complete solution. Preserve epoch1 selection: the existing exact H@1/base-loss/earliest rule selects it correctly. Do not select a different epoch retrospectively based on subgroup statistics.

Both arms process 4,868 fitting examples in each of five epochs, plus the same 1,238/1,894 validation folds and 323 ordered candidates. Each epoch file contains 18,792 rows across six epochs. All epoch H@1/H@5 values independently reproduce from global-ID ranks. The control reproduces historical M2 selected weights bit-for-bit; executed M3 function/class ASTs equal published source. One declaration-only attempt was preserved and excluded. The original executed notebook, checkpoints, ZIP exports and result CSVs remain unchanged.

M3 epoch5 H@1 is 11.6151%, below selected epoch1. Mean unseen unit-feature squared displacement rises .1540 -> .2832 over epochs1->5. The fixed residual multiplier .1 does not constrain movement when learned projection norms grow. This is evidence for testing a hard movement bound, not causal proof that distortion alone explains all disease failures. No longer training run is justified by this pilot.

The next isolated M4 experiment preserves the M3 objective and compares an unbounded residual with a hard relative norm cap .20; both share one verified frozen-feature cache. No new M4 result exists. Its protocol and manual steps are in the M4 notebook and design document. Do not tune radius or stopping policy repeatedly on these same folds. The current primary winner is an experimental development model, with explicit disease limitations; official-test/generalization conclusions remain pending the separately frozen final policy.

Yuen Kai CHIA manually ran the experiments; WP3 co-owner Andy Wei Jian CHU and WP5 Jonathan coordinate audits via #7/#8. AI assistance independently verified saved records and prepared source changes. No unperformed authorship/review is attributed. The owner no longer requires external ChatGPT review; this does not remove reproducibility, testing or scientific reporting duties.
