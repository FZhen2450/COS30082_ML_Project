# R2 v4 H_RT: retrained source exclusion for J1–J3 (16 Nov 2026)

**Status:** post-freeze exploratory, diagnostic co-endpoint (amendment 2). Never a gate, and never used to select or to change the J4–J7 parent.
- The data are DPD-Small training images only, on the reused v3 inner folds. No official-test file or inference was used.
- D1, Phase 1 and Phase 2 outputs are unchanged. The Phase 2 evaluation was re-run and is identical.
- No J4–J8 run, no merge, no tag, no push.

## 1. Git

| | |
|---|---|
| Start | `176d3e3` (Phase 2 results), clean. Phase 2 ZIP sha256 `ece9e138…da7e` verified. |
| `d9c9e8d` | `impl(r2-v4)`: H_RT workflow, fail-closed checks, 17 tests, plan. Committed before any H_RT run. |
| `3334b2b` | `analysis(r2-v4)`: descriptive parent-rule audit script (see §8 on how this file broke the first launch) |
| Results commit | this report, the amendment-3 proposal, `scripts/r2v4_hrt_supplement.py` and `artifacts/r2/v4/stage_hrt_20261116/` |
| GitHub | PR #36 still at `1b2fcdb`; nothing pushed |

## 2. Files and rationale

| File | Change | Why |
|---|---|---|
| `research_r2_v4/hrt.py` (new) | Registered pairs (rule re-derived), arm folds, independent arm checks, `ExclusionAccess`, `run_arm`, `verify_arm`, `evaluate_row`, bootstrap | Reuses `jrun.run_fold`, which is unchanged (sha256 `433d4058…`), so the base model and both arms are trained by identical code |
| `scripts/r2v4_hrt.py` (new) | `scope` / `run` / `evaluate` | – |
| `research_r2_v4/confirm.py` | `receipt_problems` requires a positive optimizer-step count before the AMP threshold | Requested; tightening only (Phase 2 evaluation reproduced identically) |
| `tests/test_r2v4_hrt.py` (new, 17) | Synthetic end-to-end runs of both arms and negative tests | Gate C |
| `docs/R2_v4_HRT_Plan.md` | Gate A facts and design, written before the code | Gate A |
| `scripts/r2v4_parent_rule_audit.py` (new) | Descriptive Gate F audit | Gate F |
| `scripts/r2v4_hrt_supplement.py` (new) | Attempt audit, D1 counterpart arms, composition-coverage breakdown | Post hoc and descriptive |

## 3. Method and invariants

- **Scope (registered, re-verified):**
  - rows J1, J2, J3;
  - pairs (I6, CDDM), (I1, MTID), (I2, CPDD), re-derived from `source_feasibility.json`;
  - seed 0;
  - arms: **ablated** (every fitting image of the source removed) and **random** (the saved Stage H matched-removal index set `<fold>_<source>_seed0.json`, sha256-checked against `indexsets_SHA256SUMS` and re-derived with `matched_removal(seed=0)`, exact for all three);
  - **base** = the Phase 1 seed-0 fold model.
- **Training:**
  - Each arm is trained from the original pretrained weights, with the declared row config (only the row id changes) and the declared five epochs.
  - `torch_seed` is the base's (I6 → 5, I1 → 0, I2 → 1).
  - The receipt's frozen-parameter digest before training equals the base's, which confirms the same pretrained initialisation.
  - No checkpoint is reused: every arm has its own checkpoint, different from the base's and the other arm's.
- **Image guard:** `ExclusionAccess` refuses every removed fitting row in every phase before opening it, on top of the phase allow-lists. No arm opened a removed row.
- **Extraction:** v3 batch layout (`LayoutChunks`) from the reloaded arm checkpoint. Feature-cache and checkpoint metadata name the arm, so a cache or checkpoint from another pair or arm is refused.
- **Downstream:** the unchanged D1 recipe. Each packet is validated against the arm fold's contract.
- **Comparable population:**
  - Every arm (base, ablated, random) is restricted to the ablated fold's candidate universe and evaluation rows (`common.restrict`, as H_RF and Stage H).
  - Universes nest: ablated ⊆ random ⊆ base.
  - Novel-source seen images are therefore always from compositions that still have fitting images.
- **γ:** 0 and each row's seed-0 pooled LOFO γ (J1 8, J2 7.75, J3 8), as H_RF.
- **Uncertainty:**
  - Paired cluster bootstrap within each pair (2,000 resamples, seed 0).
  - Novel-source seen Top-1: clusters are the compositions of the novel-source images.
  - H_bal: seen and unseen compositions are resampled.
  - Aggregates are an equal-weight mean of the three pairs, with a stratified bootstrap.

## 4. Commands and tests

    python scripts/r2v4_hrt.py scope --out <E>/stage_hrt_20261116/pre/scope.json
    python scripts/r2v4_stage_j.py evaluate --out <E>/stage_hrt_20261116/pre/confirm_regression --runs <phase1> <phase2>   # IDENTICAL to Phase 2
    python -m unittest discover -s tests -p 'test_r2*.py' -v      # Ran 134 tests in 823.850s, OK (117 earlier + 17 H_RT)
    bash <E>/stage_hrt_20261116/run_hrt.sh          # launch 1 (stopped; see section 8)
    bash <E>/stage_hrt_20261116/run_hrt_retry1.sh   # retry 1 -> 'r2v4_hrt.py run' per row, then 'r2v4_hrt.py evaluate'
    python scripts/r2v4_hrt_supplement.py --runs <E>/stage_hrt_20261116/runs --eval <E>/.../evaluation/evaluation.json --out <E>/.../supplement.json

**Gate C negative tests** (`tests/test_r2v4_hrt.py`):
- registered-pair tampering;
- index set: manifest sha mismatch, non-reproducing content, and wrong fold or source;
- incomplete source removal;
- random arm: emptied composition, source rows present, wrong size;
- excluded row opened (and refused even with a wrong allow-list);
- training on the un-ablated fold refused before any image is opened;
- duplicate attempts; permuted candidate universe;
- feature cache and checkpoint swapped in from the other arm;
- receipt changed after the success marker; marker naming another arm;
- torch seed; configuration; non-finite gradients; zero optimizer steps;
- frozen weights changed; reload difference; different initialisation; fitting-row count;
- marker claiming excluded rows were opened; wrong removed-rows hash; wrong base model;
- failure recorded with no marker, and the retry getting a new attempt directory.

## 5. Execution and provenance

- **All 18 registered arms are ok:** 9 per arm type, a single successful attempt each, every check passed.

| Row | Seconds per arm (I6 / I1 / I2) | Steps (I6 / I1 / I2) | Peak GPU train / total | AMP skipped | Tensors changed | Reload diff |
|---|---|---|---|---|---|---|
| J1 | 395–412 / 513–514 / 517–518 | 485 / 725 / 730 | 2.85 / 3.86 GiB | 1–2 | 27/27 | 0.0 |
| J2 | 447–449 / 555–565 / 545–548 | 485 / 725 / 730 | 3.96 / 4.33 GiB | 1–2 | 51/51 | 0.0 |
| J3 | 836–837 / 1,144 / 1,149–1,150 | 485 / 725 / 730 | 2.53 / 3.42 GiB | 1–2 | 96/96 | 0.0 |

- **GPU:**
  - Launch 1: 343 samples, peak 6.0 GiB, 93% mean utilisation, 89 °C maximum.
  - Retry 1: 399 samples, peak 6.2 GiB, 92%, 89 °C.
- **Epoch losses:** in receipts and `evaluation.json`. Ablated arms end slightly higher than base. For I6 the end loss is J1 0.33 vs 0.20, J2 0.22 vs 0.13 and J3 0.53 vs 0.30, from 38% fewer images.
- **Code:** code hashes are identical across both launches.

## 6. Composition support (registered design, unchanged)

| Pair | Fitting base → arm | Fitting compositions, ablated / random | Emptied, ablated / random | Evaluation universe / seen / unseen | Novel-source seen | Source-s unseen | Random match levels (comp / crop / disease / any) |
|---|---|---|---|---|---|---|---|
| I6 / CDDM | 5,062 → 3,112 | 266 / 291 | **25** / 0 | 276 / 676 / 309 | 135 | 253 | 231 / 435 / 501 / 783 |
| I1 / MTID | 5,042 → 4,661 | 290 / 291 | 1 / 0 | 300 / 856 / 333 | 53 | 0 | 53 / 196 / 51 / 81 |
| I2 / CPDD | 5,057 → 4,698 | 286 / 291 | 5 / 0 | 296 / 855 / 309 | 39 | 0 | 201 / 121 / 2 / 35 |

**Confounding (as the reviewer raised):**
- Excluding a source also removes supervision inside the compositions it supplies, and for I6 it removes 25 whole compositions. The matched-random arm removes the same number of images but mostly from other compositions: only 231 of 1,950 are same-composition picks in I6.
- The emptied compositions and their 202 seen-val images are excluded from **every** arm's evaluation, so no arm is scored on a composition it has no images of.
- The remaining confound is reduced per-composition and per-class supervision in the ablated arm. Section 7.3 examines this descriptively.

**Lineage:**
- In I1/MTID, one kept fitting image (`SCLD_sugarcane_viral_4947.jpg`, another source and another crop) shares dhash lineage group 33 with removed MTID maize images. This is recorded in `scope.json`, not removed, and the registered index set is unchanged.
- Group 33 contains **fitting images only**.
- No evaluation image shares a lineage group with any fitting image in any fold (`r2v4_verify_inputs.py`; separately, 0 novel-source evaluation images share lineage with kept or removed fitting images). Training/evaluation isolation is intact.

## 7. Results

### 7.1 Per pair: source effect = ablated − matched random (pp, selected γ; 95% CI)

| Row | Pair | Novel-source seen Top-1: source effect | Size effect (random − base) | Ablated − base | H_bal: source effect | H_bal size effect | AUSUC source effect | H_RF (s0) novel source | D1 Stage H novel source |
|---|---|---|---|---|---|---|---|---|---|
| J1 | I6 / CDDM | **−64.4** [−84.1, −47.8] | −2.2 | −66.7 | −1.5 [−12.6, +8.4] | −7.4 | +0.0 | −50.4 | −63.0 |
| J1 | I1 / MTID | **−45.3** [−91.7, −6.9] | 0.0 | −45.3 | −1.5 [−7.7, +2.7] | +0.6 | −1.4 | −41.5 | −49.1 |
| J1 | I2 / CPDD | **−53.8** [−70.3, −33.3] | 0.0 | −53.8 | +0.7 [−3.0, +5.3] | +0.4 | −2.0 | −46.2 | −61.5 |
| J2 | I6 / CDDM | **−60.7** [−80.4, −43.6] | −3.7 | −64.4 | +0.6 [−9.6, +10.2] | −8.0 | +1.9 | −34.8 | −63.0 |
| J2 | I1 / MTID | **−47.2** [−87.5, −13.8] | +1.9 | −45.3 | −2.0 [−5.5, +1.2] | +1.2 | −1.0 | −41.5 | −49.1 |
| J2 | I2 / CPDD | **−51.3** [−64.9, −32.1] | −5.1 | −56.4 | −0.2 [−3.8, +4.7] | −1.6 | −1.3 | −28.2 | −61.5 |
| J3 | I6 / CDDM | **−65.9** [−83.7, −50.6] | +1.5 | −64.4 | −3.1 [−14.6, +7.2] | −6.6 | −1.9 | −57.8 | −63.0 |
| J3 | I1 / MTID | **−54.7** [−95.8, −20.7] | 0.0 | −54.7 | −1.0 [−4.6, +3.6] | 0.0 | −0.5 | −54.7 | −49.1 |
| J3 | I2 / CPDD | **−56.4** [−78.3, −35.1] | 0.0 | −56.4 | +0.7 [−2.2, +3.9] | −2.2 | −0.7 | −46.2 | −61.5 |

**Aggregate (equal weight over the 3 pairs; shown only after the per-pair table):**

| Row | Novel-source seen source effect | H_bal source effect |
|---|---|---|
| J1 | −54.5 [−75.4, −35.6] | −0.8 [−5.1, +3.1] |
| J2 | −53.1 [−71.9, −36.0] | −0.5 [−4.5, +3.2] |
| J3 | −59.0 [−78.8, −42.1] | −1.1 [−5.1, +2.7] |

Full per-arm outcomes are in `evaluation.json`, for every row, pair and arm at γ 0 and the selected γ:
- seen and unseen Top-1 and Top-5, H_bal, H_micro, H@5, AUSUC;
- unseen crop and disease, within-crop;
- false-healthy;
- source-s unseen.

**Notable per-arm facts:**
- **I6/CDDM ablated:**
  - The 253 CDDM unseen images lose crop recognition: unseen crop Top-1 falls from 99% to 70–77%.
  - False-healthy on unseen images rises from 1.6% to 18–20%.
  - Seen Top-1 falls by about 13 pp in **both** removal arms (base 74–77 → 56–64), which is the cost of losing 38% of the images.
- **I1 and I2:** whole-fold H_bal, seen and unseen accuracy move by ≤ 3 pp. The novel-source images are the only place the exclusion shows.
- **γ = 0:** novel-source accuracy in the base arm is about 90–98% even without calibration, and falls to 31–62% in the ablated arm.

### 7.2 Does the fine-tuned gain over D1 survive retraining without the source? (`supplement.json`)

For each pair, D1 was recomputed on the same three arms. It reproduces the registered Stage H values exactly: max |diff| 0.0.

Below, J and D1 are compared on identical images and candidates, at J's γ (pp, J − D1):

| Pair | Row | Novel-source seen, base arm | **Novel-source seen, ablated arm** [95% CI] | H_bal, base | H_bal, ablated |
|---|---|---|---|---|---|
| I6 / CDDM | J1 | +5.2 | −2.2 [−6.1, 0.0] | +2.6 | +0.7 |
| | J2 | +3.0 | −0.7 [−8.1, +3.4] | +1.9 | +1.9 |
| | J3 | +0.7 | −4.4 [−10.1, −0.9] | +2.1 | −0.7 |
| I1 / MTID | J1 | 0.0 | +3.8 [0.0, +6.9] | +1.5 | +1.3 |
| | J2 | +3.8 | +7.5 [+4.2, +10.3] | +4.2 | +4.3 |
| | J3 | 0.0 | −5.7 [−6.9, −4.2] | +5.0 | +4.8 |
| I2 / CPDD | J1 | +2.6 | +7.7 [−4.7, +20.0] | −0.8 | +1.3 |
| | J2 | +2.6 | +5.1 [−5.4, +12.1] | +2.1 | +1.1 |
| | J3 | +2.6 | +5.1 [−7.1, +14.3] | +3.4 | +3.0 |

**Absolute novel-source seen Top-1 with the source excluded:**

| Pair | J1 | J2 | J3 | D1 |
|---|---|---|---|---|
| I6 / CDDM | 23.7 | 25.9 | 21.5 | 25.9 |
| I1 / MTID | 47.2 | 50.9 | 37.7 | 43.4 |
| I2 / CPDD | 25.6 | 23.1 | 23.1 | 17.9 |

With the source present (base arm), all four models score 77–96%.

### 7.3 Composition-coverage breakdown (post hoc, descriptive; novel-source seen images, selected γ)

The bins use the share of each image's composition's fitting images that the ablated arm removed. The random arm removes a different share, shown as its mean. Cells are Top-1 % for J1 / J2 / J3, with D1 in the last column.

| Pair | Ablated removed | Images (compositions) | Random removed (mean) | Base | Ablated | Random | D1 ablated |
|---|---|---|---|---|---|---|---|
| I6 | 0–25% | 6 (3) | 48% | 83 / 83 / 67 | 50 / 33 / 50 | 83 / 83 / 100 | 67 |
| I6 | 25–50% | 22 (5) | **55%** | 95 / 86 / 91 | **27 / 32 / 27** | **91 / 82 / 82** | 32 |
| I6 | 50–75% | 20 (1) | 28% | 55 / 65 / 35 | 0 / 0 / 0 | 45 / 50 / 50 | 0 |
| I6 | 75–100% | 87 (8) | 9% | 98 / 98 / 98 | 26 / 30 / 23 | 98 / 97 / 97 | 28 |
| I1 | 75–100% | 53 (2) | 17% | 92 / 96 / 92 | 47 / 51 / 38 | 92 / 98 / 92 | 43 |
| I2 | 0–25% | 5 (1) | 16% | 80 / 80 / 80 | 60 / 60 / 60 | 80 / 80 / 80 | 60 |
| I2 | 25–50% | 19 (4) | **41%** | 79 / 79 / 79 | **32 / 26 / 21** | **79 / 74 / 79** | 16 |
| I2 | 50–75% | 15 (2) | 40% | 80 / 80 / 80 | 7 / 7 / 13 | 80 / 73 / 80 | 7 |

- **Observed:** in the I6 25–50% bin (22 images, 5 compositions) and the I2 25–50% bin (19 images, 4 compositions), the random arm removed a **similar or larger** share of those compositions' images than the ablated arm. Yet it keeps 74–91% accuracy while the ablated arm falls to 21–32%. D1 shows the same pattern.
- **Interpretation (not a causal claim):** within these bins the collapse is not explained by fewer images of the composition, and is consistent with dependence on source-specific appearance. The bins are tiny (1–8 compositions), I1/MTID cannot be separated at all (all 53 images are in one bin, 2 compositions), and source is confounded with composition-specific image content (disease stage, background, capture).

## 8. Failures, retries and superseded evidence

There are 34 attempt directories, all kept (`supplement.json` → `attempts`):

| Status | Count | What |
|---|---|---|
| ok | 18 | All registered arms: J1 I6/ablated is attempt 1 (launch 1); every other arm is attempt 2, except J3 I2/random, whose first attempt was in retry 1 |
| rejected after training | 15 | Launch 1, arms 2–16. Each trained fully but failed closed with `run started from a dirty worktree`, because the receipt recorded `worktree_clean: false`. |
| interrupted | 1 | J3 I2/ablated attempt 1, killed when the operator stopped launch 1; `interrupted.json` |

- **Cause of the rejections:**
  - I wrote the untracked `scripts/r2v4_parent_rule_audit.py` into the worktree about 6 min into launch 1.
  - The check worked as designed: no rejected attempt was used, and the code path itself was identical (training code hashes unchanged).
  - I stopped launch 1, marked the interrupted attempt, committed the file (`3334b2b`) and relaunched (`run_hrt_retry1.sh`).
  - Every retried arm went to a new attempt directory; nothing was overwritten.
  - Cost: about 2.8 GPU-hours.

## 9. Decision on source robustness

**Not supported.** The registered H_RT shows that the J1–J3 development gains **do not survive genuine source exclusion** on the images from the excluded source:
- Retraining the representation without a source drops accuracy on that source's seen images by 45–66 pp relative to matched random removal, in every row and pair. Every CI excludes 0.
- That is larger than H_RF (−28 to −58) and about as large as frozen D1's Stage H drop (−49 to −63).
- With the source excluded, fine-tuned and frozen models reach similar novel-source accuracy (J − D1 between −5.7 and +7.7 pp). Most CIs span 0; J2 is above D1 on I1 (+7.5 [+4.2, +10.3]), and J3 is **below** D1 on I1 and I6. There is no consistent fine-tuning advantage.
- The J-over-D1 whole-fold H_bal advantage largely persists in the ablated arms (−0.7 to +4.8 pp). It comes from images of the sources still present.

**What this supports:**
- Partial fine-tuning improves recognition **within the sources it was trained on** (in-distribution development folds, Phase 2).
- It does not supply source-independent features. Whole-fold H_bal barely registers the exclusion, because each novel-source set is a small part of the evaluation rows (39–135 of 676–856).
- Three pairs at one seed, with composition and source confounding (§6, §7.3), allow only a qualitative conclusion. It should not be stated as a general robustness property, either way.

## 10. Gate F: readiness for J4a–J8 (nothing executed)

See `docs/R2_v4_Amendment3_PROPOSAL.md` and `parent_rule_audit.json`. On the seed-0 tables:
- **Inner parents:** I1, I2, I3, I6 → J2; I4, I5 → J3. A J2 child is eligible in **4 of 6** held folds, while the gate needs 5.
- **Null child:** a child identical to J2 passes its single-row nested gate vs D1 (+3.66 pp, 6/6). Its gate measures the parent, not the modification.
- **Perfect child:** selected only in 4 folds. The stage nested estimate (+24.18, 5/6) includes J3 fallback folds (+0.75 pp of it).

**Readiness:**
- **J4a–J7 are not ready.** The rule must first be amended prospectively. The proposal covers:
  - the child gated against its parent;
  - nested-complete training on J3 for I4/I5;
  - attribution reporting.
  - It needs about 1 h 40 per child at screen.
- Given H_RT, the children targeting the observed failure modes should go first: J5 (source-balanced) and J7 (crop adversary for disease transfer).
- **J8** (ConvNeXt + frozen CLIP, against D1 and F10) is independent of the parent rule and is ready, but it is not started.

## 11. Evidence

`artifacts/r2/v4/stage_hrt_20261116/` contains:
- `evaluation.json`, `supplement.json`, `scope.json`, `parent_rule_audit.json`;
- `receipts/`: every attempt's `receipt.json` with its `hrt_receipt.json`, `failure.json` or `interrupted.json`;
- `confirm_regression/` (Phase 2 re-evaluation, identical);
- `gpu_monitor*.csv`, `code_start*.txt`, the launch scripts;
- `logs/` (redacted), including the 134-test log;
- `SHA256SUMS`;
- `private_files_SHA256SUMS.txt`: checkpoints, feature caches and packets of all attempts, kept private in `workspace/experiments/r2-compositional/v4/stage_hrt_20261116/runs/`.
