# R2 v4: amendment 3 final repair, declaration and J4b seed-0 screen (17 Nov 2026)

**Status:** post-freeze exploratory development cross-validation on the reused v3 inner folds; **adaptive** (J4b was chosen after H_RT).
- Only J4b seed 0 was run.
- No J5, J7, J8, extra seeds or H_RT arms.
- No official test data; no First Run access.
- No push, merge or tag.
- A seed-0 pass is **not** a confirmation.

## 1. Git

| | |
|---|---|
| Start | `2aa5175` (clean; review ZIP `76849108…a740` verified) |
| `d3bd4b8` | impl: final repair (A1–A3), child runner, preflight CLI, declare-tool update, tests |
| `65dd3fd` | plan: **amendment 3 declared** (2026-11-17 01:35 +08:00), before any J4b scoring |
| Results commit | this report and `artifacts/r2/v4/stage_j4b_20261117/` |
| Worktree | clean before training, after training, and at evaluation (launcher checks) |

## 2. Review defects and fixes

| Defect | Fix (`d3bd4b8`) | Test |
|---|---|---|
| A1: `verified_pair(..., rederive=False)` accepted a same-label packet swap | Re-derivation is mandatory. No `rederive` parameter exists on `load_verified` or the production `verified_pair` (passing one is a `TypeError`). The test-only `_pair(rederive=False)` produces tables marked `rederived: False`, which the gate refuses. | `test_same_label_swap_inside_a_packet_is_refused`, `test_unrederived_tables_cannot_reach_the_gate`, `test_production_path_has_no_bypass_or_override` |
| A2: a forged result dictionary passed `promotion_decision` | `promotion_decision(child_tables, parent_tables, mode)` recomputes the H_bal gate from the tables. Tables must carry an identity **sealed** by `child_inputs` (an HMAC over the identity record and the table's arrays). The reviewer's exact forged dictionary is refused, as are a copied identity and an altered table. | `test_fabricated_promotion_result_is_refused` |
| A3: parent and configuration overrides were reachable | The production pair fixes J2 (registered Phase 1/2 runs). The child must be in `AUTHORISED_CHILDREN = (J4b,)` with its declared configuration and exactly `augment=photometric`. Overrides exist only in the test path. | `test_production_path_has_no_bypass_or_override`, `test_wrong_parent_configuration_is_refused`, `test_exactly_one_registered_modification` |

## 3. Tests

- **Command:** `CUDA_VISIBLE_DEVICES='' python -m unittest discover -s tests -p 'test_r2*.py' -v`
- **Result:** `Ran 170 tests in 1270.328s … OK`. Log sha256 `d569f8cc…b0a4`.
- **Child tests:** 36 in total (gate 19, inputs 11, runner 6). They demonstrate:
  - tampered same-label packets are refused even when re-derivation is "skipped";
  - fabricated promotion results are refused;
  - a wrong parent or an alternate child configuration is refused;
  - a null child fails;
  - wrong fold, image order, candidate order, seed and checkpoint identity fail closed;
  - the AUSUC and shortcut limits are unchanged;
  - H_bal is the only promotion endpoint;
  - runner refusals: dirty worktree, protected directories, missing declaration or authorisation, failures and rejections kept, retries in new attempt directories.
- **Historical evidence:**
  - the SHA256SUMS of Phase 1, Phase 2, H_RT and both post-review folders all verify;
  - `r2v4_declare.py --verify` (base and amendments 1–3) passes;
  - a fresh Phase 2 evaluation is identical except for its timestamp;
  - `jrun.py` is unchanged (`433d4058…`).

## 4. J2 parent preflight (Gate C)

`scripts/r2v4_child.py preflight --seed 0`, from clean `d3bd4b8`, device cuda (the original scoring device):

| Fold | Result | max |re-derived − stored| |
|---|---|---|
| I1–I6 | pass | 1.9073e-06 each (tolerance 1e-5) |

The receipt is `preflight_J2_seed0.json` (sha256 `78416fa2…ce48`). It records the receipt, checkpoint, features and packet hashes, contract, identity and ordered file-name hashes per fold. The same check passed again inside the J4b evaluation.

## 5. Amendment 3 declaration (Gate E)

- **Files:**
  - `research_r2_v4/amendment_3.json`: sha256 `ba3f2442…3456`.
  - `research_r2_v4/DECLARATION_AMENDMENT_3.json`: sha256 `95358ee8…9f54`.
- **Built from** `d3bd4b8` with a clean worktree, and committed in `65dd3fd` at 2026-11-17 01:35:05 +08:00. Training started at 01:35:26.
- **Contents:**
  - the exact J2 and J4b configurations; the modification `{augment: photometric}`;
  - the primary gate and diagnostics; the verification requirements;
  - disclosures: reused folds, adaptive selection, calibration dependence, fixed-parent selection, non-independent counts;
  - `authorised_now`: J4b, seed 0, screen;
  - **24 sha256 code, test and document pins.**
- **Tool change:** `r2v4_declare.py` now verifies the pins (`--verify --amendment 3`) and never rewrites a declaration. Earlier declarations verify unchanged.

## 6. J4b seed-0 run (Gate F)

- **Run:** `r2v4_child.py run --child J4b_best_photometric_aug --seed 0` started at 01:35:26 and ended at 02:33:29 (58 min). Six folds, one GPU job.

| Fold | Attempt | Seconds | Peak GPU train / total | Steps | AMP skipped | Grad checks | Tensors changed | Reload | Torch seed | Re-derivation |
|---|---|---|---|---|---|---|---|---|---|---|
| I1 | 1 | 561 | 3.95 / 4.31 GiB | 785 | 2 | 18/18 | 51/51 | 0.0 | 0 | 1.9e-6 |
| I2 | 1 | 571 | 3.96 / 4.33 | 790 | 2 | 18/18 | 51/51 | 0.0 | 1 | 1.9e-6 |
| I3 | 1 | 579 | 3.96 / 4.33 | 790 | 2 | 18/18 | 51/51 | 0.0 | 2 | 1.9e-6 |
| I4 | 1 | 578 | 3.96 / 4.33 | 790 | 2 | 18/18 | 51/51 | 0.0 | 3 | 1.9e-6 |
| I5 | 1 | 582 | 3.96 / 4.33 | 790 | 2 | 18/18 | 51/51 | 0.0 | 4 | 1.9e-6 |
| I6 | 1 | 581 | 3.96 / 4.33 | 790 | 2 | 18/18 | 51/51 | 0.0 | 5 | 1.9e-6 |

- **All folds:** each passed on its first attempt; there were 0 failures, 0 rejections and 0 retries. Every fold has its `child_identity.json`.
- **Integrity:** the J2 evidence fingerprint is identical before and after the run. All runs are from `65dd3fd` with `jrun.py` `433d4058`.
- **Training loss by epoch:** 6.2 → 1.3 → 0.56 → 0.39 → 0.32. J2 was 5.8 → 0.13; J4b ends higher, as expected with augmentation.
- **GPU:** 119 samples; peak 6.2 GiB device memory; 89% mean utilisation; maximum 88 °C.

## 7. Screen result: J4b vs J2 (seed 0; each row's own LOFO γ chosen without the held fold)

| Fold | J4b H_bal | J2 H_bal | Δ (pp) | AUSUC J4b / J2 | Shortcut J4b / J2 | γ J4b / J2 |
|---|---|---|---|---|---|---|
| I1 | 61.58 | 58.83 | **+2.75** | 48.78 / 46.71 | 1.80 / 2.40 | 7.75 / 7.75 |
| I2 | 75.23 | 74.37 | +0.87 | 68.51 / 67.80 | 0.65 / 1.29 | 8.25 / 8.25 |
| I3 | 69.01 | 67.65 | +1.36 | 57.17 / 55.23 | 1.27 / 1.91 | 8.25 / 7.75 |
| I4 | 78.50 | 76.84 | +1.66 | 73.44 / 70.28 | 0.96 / 1.27 | 8.25 / 8.25 |
| I5 | 73.35 | 71.37 | +1.98 | 64.18 / 61.87 | 1.59 / 1.59 | 8.25 / 7.75 |
| I6 | 69.95 | 70.10 | **−0.15** | 62.99 / 61.98 | **11.97 / 6.15** | 7.75 / 8.25 |
| **Mean** | **71.27** | **69.86** | **+1.41** | +1.87 pp | +0.60 pp | |

**Primary gate:**

| Criterion | Result | Threshold |
|---|---|---|
| Mean Δ H_bal | +1.41 pp | ≥ +1.0 ✓ |
| Folds improved | 5/6 | ≥ 5 ✓ |
| Δ AUSUC | +1.87 pp | ≥ 0 ✓ |
| Δ shortcut | +0.60 pp | ≤ +5 ✓ |

- **Decision:** `promotion_decision` gives **screen pass**. That means seeds 1–2 may be considered **only after independent review**. This is not a confirmation.
- **Exploratory bootstrap:** 95% CI of the H_bal difference is [+0.14, +2.85] (paired composition-cluster, 2,000 resamples). The descriptive sign proportion is 0.0135, which is not a p-value.
- **Calibration-matched (diagnostic, J4b scored at J2's γ):** +1.72 pp, positive in all six folds (I6 +0.89). The gain is not a calibration artefact.

## 8. Mechanism diagnostics (no pass/fail; cannot substitute for the gate)

| Diagnostic | J4b | J2 | Δ (pp) | Units |
|---|---|---|---|---|
| Unseen disease Top-1 | 61.13 | 60.95 | **+0.18** (folds −3.6 to +7.0) | 6 folds |
| Unseen crop Top-1 | 89.46 | 88.40 | +1.06 | 6 folds |
| Unseen composition macro | 68.41 | 66.42 | +2.00 | 6 folds |
| Within-crop disease (unseen) | 78.81 | 77.93 | +0.87 | 6 folds |
| Seen Top-1 | 75.76 | 75.36 | +0.40 | 6 folds |
| Seen Top-1, dominant source | 78.0 | 77.4 | +0.63 | 4,608 image-folds = 809 unique images, 195 compositions, 39 sources |
| **Seen Top-1, other fitting source** | **60.7** | **62.1** | **−1.43** | 628 image-folds = **111 unique images, 30 compositions, 15 sources, 40 clusters** |
| Seen Top-1, novel source | 39.3 | 35.7 | +3.57 | 28 image-folds = 5 unique images (uninterpretable) |
| False-healthy, seen diseased / unseen | 2.03 / **3.04** | 2.23 / 2.44 | −0.20 / **+0.60** | 6 folds |

**Reading (observations, not causes):**
- **The registered mechanism is not supported.** Photometric augmentation was meant to reduce dependence on source appearance. On seen images from a non-dominant source the child is 1.4 pp **worse**, though this rests on 111 unique images in 40 clusters, so it is weak evidence either way.
- **Disease transfer is unchanged:** +0.18 pp, with fold signs mixed.
- **Where the gain is:** unseen composition macro, unseen crop, AUSUC, and calibration-matched H_bal.
- **I6 shortcut:** the healthy-shortcut rate nearly doubles in I6 (6.2 → 12.0%). It is within the mean limit and should be watched at seeds 1–2.
- **Interpretation grid:** this is "gate pass at screen; development gain without evidence for the stated mechanism".

## 9. Failures, retries and limitations

- **Failures in this stage:** none, with no retries and no superseded evidence.
- **Earlier stage:** H_RT had 15 rejected and 1 interrupted attempts, all kept.
- **Single seed:** one J4b seed against one J2 seed. The margin over the threshold (+0.41 pp) is about the size of the J-row seed SD (0.25–0.47 pp), so a three-seed confirmation could fail.
- **Fold reuse:** the folds are reused, so the result is not blind or independent.
- **Adaptive choice:** J4b was chosen after H_RT, and this is the first child row tried.
- **Calibration:** dependence remains (γ ≈ 8).
- **Fixed parent:** J2 was selected on all six folds, so this is not a nested estimate.
- **Not run:** no H_RT arms for J4b were run or authorised, so nothing is claimed about source exclusion.

## 10. Recommendation on seeds 1–2

- **The declared rule makes seeds 1–2 eligible after independent review, and they are cheap:**
  - about 2 h of training;
  - a J2 preflight for seeds 1 and 2 first (about 1 min);
  - no new code.
- **The scientific case is narrow.** The H_bal gain (+1.41, calibration-matched +1.72, CI excluding 0) is consistent and worth confirming, because a single-seed margin of 0.41 pp over the threshold is exactly what the three-seed rule exists to test.
- **Do not frame J4b as a source-robustness method.** The mechanism diagnostics argue against that. Running J4b's H_RT arms is **not** recommended unless seeds 1–2 confirm the H_bal gain **and** a source-exclusion hypothesis is declared separately.
- **If the 27 Nov First Run deadline leaves no slack,** stopping J4b here is also defensible. Report it as "screen pass, unconfirmed, mechanism not observed".
