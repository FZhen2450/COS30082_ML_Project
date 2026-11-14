# R2 v4 Stage J: workflow readiness (14 Nov 2026)

**Status:** post-freeze exploratory. No J1–J8 model has been trained on full folds, and no J1–J8 evaluation metric has been computed. The full six-fold sweep waits for owner approval. Amendment 2 (`research_r2_v4/amendment_2.json`, commit `ca63114`) was declared before any Stage J scoring and changes no selection rule.

## 1. Workflow (one row, seed and fold)
Implemented in `research_r2_v4/jrun.py`; the CLI is `scripts/r2v4_stage_j.py`.

1. **Data boundary.** Fitting rows must be disjoint from seen-val, unseen-val, outer-exclusion and outer-seen rows. Images come through `ImageAccess`, which:
   - refuses any row outside the current phase's allowed set, and any filename not in the training CSV;
   - checks before opening anything.
2. **Seeds.** `1000*seed + fold_index` seeds torch, numpy and the sampler; workers are offset by 100 per worker; cudnn benchmark is off.
3. **Training** on fitting rows only:
   - composition-balanced sampler;
   - declared optimiser, schedule, augmentation and extra losses;
   - AMP;
   - gradient checks at steps 1 and 5, every 50 steps, and the last step.
4. **Fail-closed checks** (`JCheckError`):
   - every trainable tensor has a gradient at every checked step;
   - every trainable tensor gets a non-zero gradient at least once;
   - checked gradient norms are finite, except AMP-skipped steps, which may be at most 5% of steps and never 3 checks in a row;
   - the training loss stays finite;
   - frozen parameters are byte-identical (sha256 digest) after training;
   - every trainable tensor changed;
   - training with no trainable tensor is refused.
5. **Checkpoint.** Saves the trainable parameters, heads and metadata, then reloads into a freshly built model. It requires exactly the expected keys, matching metadata, and an identical output on a fixed probe batch (max diff must be 0.0).
6. **Extraction from the reloaded model.** Covers fold rows only (fitting, seen-val and unseen-val), in the **v3 batch layout** (`LayoutChunks`): consecutive 32-row chunks of all 8,000 rows, with non-fold slots zero-filled and never opened.
7. **Feature cache.** Stored with metadata: row, seed, fold, config sha256, checkpoint sha256, rows sha256, backbone and schema. A cache is loaded only if every field matches.
8. **D1 downstream.** Unchanged recipe on the fold's features (J8: [L2 v3 CLIP-L, L2 fold ConvNeXt]). The packet is validated against `fold_contract` before and after saving.
9. **Receipt.** Written atomically, last, only if everything passed. A failure writes `failure.json` in its own attempt directory; reruns get a new attempt.
10. **Evaluation** (`evaluate`) and the source co-endpoint (`hrf`) refuse any row or seed without six verified receipts. Verification re-hashes the checkpoint, features and packet.

## 2. Tests (`Ran 91 tests in 175.204s`, OK: 66 R2 + 7 v4 H/I + 18 v4 J)
`tests/test_r2v4_j.py` covers:
- **Forbidden access:** seen-val, unseen-val and outer rows are refused during training and nothing is loaded; access with no phase is refused; overlapping fitting rows are refused; layout chunks load only fold rows.
- **Stale or mismatched feature cache:** wrong checkpoint sha, seed, fold, config, rows, or an unreadable file are all refused.
- **Corrupted checkpoint:** a truncated file, a missing key, an altered weight (output check) and a metadata mismatch are all refused; a good checkpoint reloads with diff 0.0.
- **Gradient monitor:** a missing gradient, 3 consecutive non-finite checks, a tensor that never gets a non-zero gradient, and more than 5% skipped steps all fail.
- **Fail-closed fold:** a backend that mutates a frozen parameter fails, and so does one with no trainable parameters. No receipt is written in either case.
- **Incorrect candidate order:** a reversed `cand_ids` is refused by `validate_packet`.
- **Synthetic six-fold dry run:** a tiny synthetic backend, real folds and the real D1 downstream:
  - 2 rows × 6 folds ok;
  - evaluation against the registered D1;
  - H_RF on 2 pairs, which re-derives the Stage H index sets;
  - every file the process opened was recorded: no official-test file and no image file was opened;
  - deleting one fold's packet, or failing one receipt, makes evaluation refuse the row.

## 3. Real-data audits (no full training)

| Audit | Result |
|---|---|
| J0 through the full workflow, all 6 folds | features bit-identical to v3 `clip_l14`; packets equal the registered v3 D1 packets (max abs diff 0.0; candidate IDs and truth identical) |
| J8 at zero epochs, all 6 folds | ConvNeXt features bit-identical to v3 `tv_convnext_base`; packets equal the registered v3 F10 packets (max abs diff 0.0) |
| J1 smoke (fold I6, metric-free) | attempt 1 (20 steps) **failed closed** (AMP skipped 2/20 > 5%); attempt 2 (100 steps) ok: 27/27 tensors updated, 4/4 finite checks, reload diff 0.0, packet validated; no metric computed |

**Superseded audit** (`stage_j_audit_20261114`, kept): fold-only extraction batches gave features one fp16 ULP away from v3 (7.8e-3 CLIP, 1.95e-3 ConvNeXt) and D1 packet differences of up to 0.44. Candidate IDs and truth were identical. The fix was the v3 batch-layout extraction. The audit also shows that the D1 downstream amplifies fp16-level feature noise, so J rows must use the same extraction path as their control.

## 4. Proposed execution order (for approval)
1. J1 and J2, seed 0 (about 45–55 min each alone).
2. J5 and J4b on the mechanically chosen parent (best of J1–J3; J3 must also exist for the rule, see step 4). They target the dominant source-shift failure.
3. J8 (about 20 min).
4. J3 (about 2 h), needed to complete the declared parent rule. **Alternative:** run J3 before step 2 so the parent is final before any child trains.
5. J4a, then J6 and J7 (lowest prior).
6. H_RF for every completed row (about 25 min each with frozen fold features).
7. Seeds 1 and 2 only for rows whose seed-0 screen passes the gate; H_RT only for three-seed passers.

**Recommendation:** run J1, J2 and J3 first, so the parent is fixed by the declared rule before any J4a–J7 run. Total seed-0 time on Katana is about 8–9 h.
