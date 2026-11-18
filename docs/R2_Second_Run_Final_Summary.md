# AS02 Second Run (Research Round 2): final research summary

> **All results are development cross-validation on reused folds built from the DPD-Small *training* set** (v3
> benchmark, inner folds I1–I6). They are **not official-test results**. No official-test image, label, score or inference
> was used at any point. The research is post-freeze and exploratory: the folds informed earlier rounds (v1–v3), and some
> rows were chosen adaptively.

## 1. Objective and assignment relevance

* **Question:** can a model recognise crop–disease **compositions it never saw in training**, while staying accurate on seen
  compositions?
* **Primary endpoint:** H_bal, the harmonic mean of seen Top-1 and unseen composition-macro Top-1.
* **Relevance to AS02:** this is the assignment's unseen-composition problem. R2 is an independent research track whose
  findings may be cited as supplementary development evidence. It never feeds the First Run submission.

## 2. Development results (H_bal, LOFO-calibrated γ)

| Method | What it is | Result | Status |
|---|---|---|---|
| D1 | frozen CLIP ViT-L/14 features + synthetic-composition downstream | 66.20; v3 nested +2.23 pp vs B1 (6/6) | provisional reference |
| J1 | D1 + fine-tune last 2 CLIP blocks | 69.15 (+2.95 vs D1), 6/6, 3 seeds | **confirmed** |
| J2 | D1 + fine-tune last 4 blocks | 69.56 (+3.36), 6/6, 3 seeds | **confirmed**, parent by rule |
| J3 | D1 + LoRA-r8 MLP | 69.90 (+3.70), 6/6, 3 seeds | **confirmed** |
| Stage (D1 → best of J1–J3) | nested, every choice made on 5 folds | +3.36 pp, 6/6 | confirmed |
| J4b | J2 + photometric augmentation | vs J2: seed 0 +1.41 (5/6); three seeds **+0.86 (4/6)** | **failed confirmation** |

* J1–J3 are **statistically indistinguishable** (e.g. J2 − J3 −0.34 pp, CI [−2.51, +1.65]).
* The finding is "partial CLIP fine-tuning helps by about 3–3.7 pp". It is not a claim about which variant is best.

## 3. Registered evaluation procedures

* **Before scoring:** the protocol and four amendments were declared and hashed before the scoring they govern.
  Amendments 3–4 also pin code by sha256.
* **Gate:**
  * mean H_bal gain ≥ +1 pp;
  * ≥ 5/6 folds improved;
  * AUSUC change ≥ 0;
  * unseen healthy-shortcut change ≤ +5 pp;
  * confirmation = exactly seeds {0, 1, 2}, with the gate applied to per-fold seed means.
* **Calibration:** each row's (config, γ) is chosen by leave-one-fold-out on the other five folds.
* **Child rows** (J4a–J7) are compared with the fixed parent J2 on all folds. Diagnostics have no pass/fail. Bootstrap
  CIs are exploratory.
* **Provenance:**
  * single-attempt receipts with file hashes;
  * packet re-derivation from feature caches;
  * ordered image identity and lineage isolation checks;
  * sealed tables, with the decision recomputed from verified inputs.

## 4. J4b confirmation failure

| | Result | Bar |
|---|---|---|
| H_bal change vs J2 | +0.86 pp | ≥ +1.0 ✗ |
| Folds improved | 4/6 | ≥ 5 ✗ |
| AUSUC change | +0.83 | ≥ 0 ✓ |
| Healthy-shortcut change | −0.11 pp | ≤ +5 ✓ |

* **By seed:** +1.41 / +1.55 / **−0.37** pp. The seed-to-seed spread is larger than the effect.
* **Bootstrap 95% CI (exploratory):** [+0.07, +1.69]. Calibration-matched: +0.66 pp.
* The result is kept as a registered negative; thresholds unchanged.

## 5. Calibration dependence

* Selected γ (the seen-class score penalty) is about 7.75–8.75 for every J row.
* At γ = 0, H_bal is only 5–9%, because uncalibrated models almost never predict unseen compositions.
* Every reported number is conditional on this LOFO calibration. Child comparisons also report a calibration-matched
  sensitivity.

## 6. Source sensitivity (H_RT) and confounding

* **Registered source-exclusion retraining** (J1–J3; pairs I6/CDDM, I1/MTID, I2/CPDD; seed 0; ablated vs size-matched
  random removal):
  * seen accuracy on the excluded source's images drops **45–66 pp** beyond random removal, in every row and pair;
  * with the source excluded, J ≈ D1.
* **Conclusion:** the fine-tuning gains do **not** transfer to an unseen image source, so source robustness is not
  supported.
* **Confounding:** source is entangled with composition content (disease stage, background, capture), and removing a
  source also removes compositions. The collapse is consistent with source-specific appearance, but it is not a causal
  attribution.

## 7. Model and source-stratified limitations

* **J4b, seen images from a non-dominant fitting source:** −2.44 pp in all three seeds (628 image-folds = 111 unique
  images, 40 composition-source clusters).
* **Novel-source seen images:** only 5 unique images, so they can't be read.
* **Unseen disease:** fold-level effects range from −2.0 to +5.7 pp.
* **Fold I6:** the healthy shortcut rises with J4b (+2.59 pp, seed mean).
* **Folds:** few compositions per fold, reused across rounds, not independent. Image-fold counts are not independent
  observations.

## 8. Reproducibility evidence

* **Runs:**
  * J1–J3: 54 seed-folds;
  * H_RT: 18 arms;
  * J4b: 18 seed-folds.
* **Receipts:** every run has one, recording config hash, torch seed `1000·seed + fold`, gradient/AMP/frozen/reload
  checks, code identity and a clean worktree.
* **Re-derivation:** child and parent packets re-derive from their feature caches within 1.9e-6 (tolerance 1e-5).
* **Tests:** 186 R2/v4 tests pass, including fail-closed negative tests.
* **Failures:** all retained (e.g. 15 H_RT arms failed closed on a dirty worktree and were retried as attempt 2).
* **Public evidence:** sanitized derivatives of the private originals (see `docs/R2_v4_Publication_Notice.md`).

## 9. Why experiments stopped

* The registered question for J4b was answered (negative).
* H_RT already showed the source-shift limitation that J4–J7 were meant to address.
* More rows or seeds would expand an adaptive search on reused folds without an independent test.
* The First Run submission (27 Nov 2026) has priority.
* The research was frozen by owner decision at private commit `06ccf43`.

## 10. What can and cannot be claimed in the assignment report

**Can claim** (as development cross-validation on training-derived folds):
* partial CLIP fine-tuning improved H_bal over frozen features by about 3–3.7 pp, reproducibly over three seeds;
* results depend on seen/unseen calibration;
* photometric augmentation did not reproducibly improve on J2;
* performance collapses on an image source excluded from training;
* the evaluation was predeclared, multi-seed and provenance-checked.

**Cannot claim:**
* any official-test performance or superiority;
* that J1, J2 or J3 is the single best variant;
* source or domain robustness;
* a mechanism for J4b;
* that these are blind, independent or confirmatory results in the statistical sense.
