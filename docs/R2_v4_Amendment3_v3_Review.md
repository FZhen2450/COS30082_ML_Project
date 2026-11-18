# R2 v4 amendment 3 v3: final gate hardening (review package)

**Scope:** CPU-only methodological repair, ready to declare once approved.
- No amendment declared.
- No training.
- No real-data child evaluation.
- No official-test access.
- No First Run change.
- No historical output changed.

## 1. Git

- **Start:** `a795469`, clean. Review ZIP `e086e6da…adfb` verified.
- **End:** the repair commit (see the final report). It is separate from all earlier commits.
- **v1 and v2 proposals,** H_RT and Phase 1/2 artifacts and all amendments are unchanged.

## 2. Files changed

| File | Change |
|---|---|
| `research_r2_v4/child_inputs.py` (new) | Verified-input boundary (finding A) |
| `research_r2_v4/child_gate.py` | Identity enforcement in `check_pair`/`check_seeds`. Primary endpoint locked to H_bal (finding C). `diagnostics()` and `promotion_decision()` added. Bootstrap field renamed. Holm removed (finding B). Interpretation text (finding D). |
| `tests/test_r2v4_child_gate.py` | Synthetic tables now carry identity records. New tests: same-label image swap, unverified or misidentified inputs, endpoint lock, secondary cannot promote, diagnostics. The Holm test was removed and the bootstrap field renamed. |
| `tests/test_r2v4_child_inputs.py` (new) | Boundary tests on real synthetic training runs |
| `docs/R2_v4_Amendment3_PROPOSAL_v3.md` (new) | Updated proposal |
| `docs/R2_v4_Amendment3_v3_Review.md` (new) | This document |
| `artifacts/r2/v4/stage_hrt_20261116_post_review_v3/` (new) | Test log, `strata_units.json`, SHA256SUMS |

## 3. Findings and how each was addressed

- **A. Identity and provenance:** done; see proposal §1.
  - `child_inputs.verified_pair` enforces all eight requested items.
  - The gate refuses tables without its identity record.
  - **Same-label swap in the fold order:** refused through the identity hash. The contract hash and label arrays are shown to be identical.
  - **Same-label swap inside a packet:** refused through re-derivation. The test shows that receipt, contract, label and sidecar checks **alone accept it**, so re-derivation is mandatory.
  - Real-data scoring must go through `verified_pair` (re-derivation on by default).
- **B. Uncertainty:** `p_le_0` became `bootstrap_sign_proportion_le_0`, labelled "descriptive; not a p-value; not multiplicity-adjusted". Holm was removed, and a test asserts it is absent.
  - Bootstrap CIs are kept.
  - An optional formal paired cluster-randomisation test is proposed with its explicit null, and is not implemented.
  - Published v4 uncertainty values are unchanged.
- **C. Endpoint lock:**
  - `primary_gate` refuses any endpoint other than H_bal.
  - `diagnostics()` has no pass/fail.
  - `promotion_decision()` accepts only the primary H_bal result. Diagnostics, alternative-endpoint results and the secondary pipeline estimate are all refused (tested).
- **D. Interpretation:** an exploratory paired child-versus-fixed-J2 comparison. J2 was chosen after seeing all six folds, so this is not nested or independent. The "conservative bias" claim is withdrawn. The pipeline estimate stays secondary, with its cost and full parent-variant requirement.

## 4. Primary gate and diagnostics

See proposal §2:
- **Primary:** H_bal; +1 pp; ≥ 5/6; AUSUC Δ ≥ 0; shortcut Δ ≤ +5 pp; screen at seed 0; confirmation on exactly {0, 1, 2}.
- **Diagnostics:**
  - unseen disease, crop and within-crop Top-1; seen Top-1; H_micro;
  - the calibration-matched H_bal difference;
  - source-stratified seen Top-1;
  - H_RT-population accuracy;
  - the bootstrap CI.

## 5. Verification

- **New and changed tests:** 17 gate tests and 9 boundary tests, run on CPU (`CUDA_VISIBLE_DEVICES=""`). Full R2 suite: `Ran 160 tests in 1194.773s … OK` (log sha256 `941f90df…6bc1`).

| Required negative test | Test |
|---|---|
| Same-label image-order swap | `test_same_label_image_swap_is_refused` (gate); `test_same_label_swap_in_the_fold_order_is_refused` and `test_same_label_swap_inside_a_packet_is_refused` (boundary) |
| Invalid or missing provenance | `test_missing_or_invalid_provenance_is_refused`; `test_unverified_or_misidentified_inputs_are_refused`; `test_stale_or_foreign_feature_cache_is_refused`; `test_training_code_must_match`; `test_exactly_one_registered_modification` |
| Incorrect seeds and checkpoint identities | `test_seed_and_checkpoint_identity`; `test_incomplete_or_incompatible_seeds_fail` |
| Alternate promotion endpoint | `test_alternative_endpoint_cannot_gate_or_promote`; `test_secondary_estimate_cannot_promote` |
| Null child | `test_null_child_fails_and_old_rule_passed_it`; `test_bootstrap_of_null_child_is_degenerate_at_zero` |
| Fewer than five improving folds | `test_fold_count_threshold` (4/6 fails) |
| AUSUC regression / healthy-shortcut violation | `test_ausuc_and_shortcut_constraints` |
| Missing parent variant | `test_missing_parent_variant_fails_closed` |

- **Positive fixtures:**
  - a genuinely improving synthetic child passes screen and confirmation;
  - a verified real synthetic child/parent pair feeds the gate end to end;
  - each of J4a–J7 is exactly one registered modification of J2.
- **Byte identity:** the Phase 1, Phase 2, H_RT and H_RT post-review `SHA256SUMS` all verify; declarations verify; a fresh Phase 2 evaluation from the registered runs is identical to the published `evaluation.json` except for its timestamp. No registered code changed (only `child_gate.py`, an unregistered proposal module).

## 6. J4b and J5 decision matrix (nothing executed)

**Order:**
1. Declare amendment 3 v3.
2. Implement the child runner (sidecar plus `check_registered_fold`) with tests.
3. **J4b** screen.
4. **J5** only after the J4b review, and only if submission time allows.
5. J7 and J8 are not prioritised.

| Step | J4b photometric (priority 1) | J5 source-balanced (priority 2, conditional) |
|---|---|---|
| Hypothesis | Photometric invariance reduces dependence on source appearance | Within-crop and within-disease source balancing narrows the dominant vs other-source gap |
| Screen (seed 0, 6 folds on the J2 configuration) | ≈ 1 h | ≈ 1 h |
| Primary decision | Repaired child-vs-J2 H_bal gate. A screen fail stops the row (diagnostics still reported). A screen pass leads to seeds 1–2 (≈ 2 h) and the confirmation gate. | same |
| Diagnostic 1 | Other-source seen Top-1, child − J2 (628 image-fold instances = **111 unique images, 30 compositions, 15 sources, 40 clusters**; not independent) | same: its primary mechanism diagnostic |
| Diagnostic 2 | Unseen disease Top-1, child − J2 | same |
| Diagnostic 3 | H_RT populations: J4b retrained on the 3 registered source-excluded fitting sets (≈ 25 min). Report **accuracy under source exclusion**: J4b_abl − J2_abl on the same novel-source images. | H_RF only (≈ 20 min), no claim about exclusion |
| Claim of a reduced source-exclusion **penalty** | Only with J4b matched-random arms as well (≈ 25 min): DiD = (J4b_abl − J4b_rnd) − (J2_abl − J2_rnd), paired cluster bootstrap over the novel-source images' compositions. Without them, no penalty claim. | Not claimed |
| Total | ≈ 1.5–4 h | ≈ 1.3–3.5 h |

**Interpretation grid** (per row; exploratory throughout; nothing above D1):

| Primary gate | Mechanism diagnostics | Reading |
|---|---|---|
| pass (confirmed) | improved, CI excludes 0 | "Improves on J2 in development H_bal, with the intended mechanism observed" |
| pass | not improved | "Development gain without evidence for the stated mechanism" |
| fail | improved | "Mechanism signal without a development H_bal gain". A robustness observation only; it cannot promote. |
| fail | not improved | Negative result, reported |

## 7. Unresolved issues blocking declaration or real-data execution

1. **Declaration:** owner and independent approval of v3 is required, then `r2v4_declare.py --amendment 3` with the code hashes in proposal §5. **This blocks everything below.**
2. **No child runner yet.** It must call `jrun.run_fold` on the **registered** fold (`check_registered_fold`), then write `child_identity.json`, in new attempt directories.
   - For J4b H_RT arms, it must reuse `hrt.run_arm` with `declared=` the J4b configuration and J4b's own seed-0 fold models as base.
   - CPU work, about half a day, with tests.
3. **Re-derivation on real data is unexercised.** It must be checked on the J2 parent packets (CPU; about 1–2 min per fold) before any child is scored. A failure there blocks real-data scoring.
4. **No formal multiplicity test:** only the optional randomisation test is proposed. Until it is approved and implemented, child results carry bootstrap CIs and descriptive sign proportions only.
5. **Statistical scope:** reused folds, adaptive child choice and the J2 comparator selected on all folds. Every result stays exploratory.
6. **Time:** First Run's 27 Nov 2026 submission takes precedence.
