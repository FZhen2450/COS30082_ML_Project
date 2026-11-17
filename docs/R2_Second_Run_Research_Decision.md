# AS02 Second Run (Research Round 2): final research decision

**Scope:** the independent R2 research track on DPD-Small training images only (compositional plant-disease recognition
with unseen crop–disease compositions). Every number here comes from **development cross-validation** on the registered inner
folds (v3 benchmark, I1–I6). No official-test image, label, score or inference was used. Nothing here feeds the First Run
submission (27 Nov 2026). These results are post-freeze, exploratory and not blind.

## 1. Decision

* **Best supported method: partial fine-tuning of CLIP ViT-L/14** (J1 last-2 blocks, J2 last-4 blocks, J3 LoRA-r8 MLP), with
  the frozen-feature D1 synthetic-composition downstream and LOFO-calibrated γ.
  * All three variants are confirmed over D1 at three seeds: +2.95, +3.36 and +3.70 pp H_bal; 6/6 folds; AUSUC +2.1 to +3.5.
  * The stage-level nested estimate is +3.36 pp, 6/6.
  * They are **indistinguishable from one another** (J2 − J3 −0.34 pp, CI [−2.51, +1.65]).
  * J2 stays the declared parent by rule; this is not a claim that J2 is the best variant.
* **J4b (J2 + photometric augmentation): not confirmed.** The seed-0 screen passed (+1.41 pp, 5/6). The declared three-seed
  confirmation failed: +0.86 pp, 4/6 folds, seed 2 −0.37 pp. Recorded as a negative result; thresholds unchanged.
* **D1** stays the provisional frozen-feature reference: its v3 nested estimate is +2.23 pp at 6/6 over the B1 control.
* **Recommended: freeze the research now** (section 9). Run no further rows (J4a, J5–J8), seeds, H_RT arms or architectures.

## 2. Development results (H_bal, harmonic mean of seen Top-1 and unseen composition-macro Top-1; LOFO γ)

| Stage | Comparison | Result | Status |
|---|---|---|---|
| v3 | D1 vs B1 (nested) | +2.23 pp, 6/6 | provisional candidate |
| v4 Stage I | 11+ rows vs D1 | no row passes; stage nested −0.29 pp | negative |
| v4 J Phase 1–2 | J1 / J2 / J3 vs D1 (66.20), 3 seeds | 69.15 / 69.56 / 69.90 (+2.95 / +3.36 / +3.70); 6/6 each | **confirmed** |
| v4 J Phase 2 | stage nested (D1 → best of J1–J3) | +3.36 pp, 6/6 | confirmed |
| v4 amendment 3 | J4b vs J2, seed 0 (screen) | +1.41 pp, 5/6, AUSUC +1.87 | screen pass |
| v4 amendment 4 | J4b vs J2, seeds 0–2 (confirmation) | **+0.86 pp, 4/6**, AUSUC +0.83, shortcut −0.11 | **fail** |

J4b vs J2 diagnostics (three-seed means; no pass/fail):

| Measure | Δ |
|---|---|
| H_micro | +0.70 |
| Seen Top-1 / Top-5 | −0.42 / +0.35 |
| Unseen Top-1 (micro) / Top-1 (macro) / Top-5 | +1.31 / +1.76 / −0.41 |
| Unseen crop / disease / within-crop disease | +0.50 / +1.24 / +1.14 |
| Calibration-matched H_bal | +0.66 |
| Bootstrap CI (exploratory) | [+0.07, +1.69] |

## 3. Source-shift findings

* **Stage H (D1):** removing one image source costs 44–57 pp of novel-source seen Top-1, beyond a size-matched random removal
  (9 sources).
* **H_RF (J representation fixed, downstream refitted without the source):** −31 to −40 pp, vs −49 for D1. This
  understates the effect by construction, because each backbone was trained with the source present.
* **H_RT (registered source-exclusion retraining, J1–J3, 3 pairs):**
  * accuracy on the excluded source's seen images drops by **45–66 pp** against matched random removal, in every row and
    pair (every CI excludes 0);
  * with the source excluded, J ≈ D1.
  * **The fine-tuning gains do not transfer to unseen sources; source robustness is not supported.** Whole-fold H_bal is
    barely affected, because novel-source images are few.
* **J4b:** the photometric-robustness mechanism is **not supported**.
  * Non-dominant fitting-source seen accuracy is −2.44 pp, negative in all three seeds (111 unique images, 40
    composition-source clusters).
  * The novel-source stratum is 5 images, so it cannot be read.
  * No J4b H_RT was run, so no source claim is made.
* Source is confounded with composition-specific content (disease stage, background, capture). The H_RT collapse is
  consistent with dependence on source-specific appearance, but it is not a causal attribution.

## 4. Calibration dependency

* Every reported H_bal uses a calibrated γ (the seen-class score penalty), chosen per held fold by LOFO on the other five
  folds. Ties go to the smaller γ.
* Selected γ is about 7.75–8.75 for every J row. At γ = 0, H_bal is only 5–9%, because uncalibrated scores almost never
  predict unseen compositions.
* All conclusions are therefore conditional on this calibration. The calibration-matched sensitivity (child scored at the
  parent's γ) is reported for every child comparison.
* For J4b it is +0.66 pp, so its small gain is not a γ artefact, but it is also below the threshold.

## 5. Reproducibility evidence

* **Fold runs:** 18 seed-0 J folds and 36 seed-1/2 J folds; 18 H_RT arms; 6 J4b seed-0 folds and 12 J4b seed-1/2 folds.
  Every fold has a receipt with:
  * file hashes (checkpoint, features, packet);
  * the declared configuration hash, torch seed `1000·seed + fold`, gradient/AMP/frozen/reload checks;
  * code identity, recorded from a clean worktree.
* **Verification on load:**
  * every child packet is re-derived from its own feature cache in the registered order (max diff 1.9e-6, tolerance 1e-5);
  * J2's seeds 0–2 were re-derived before child scoring;
  * the ordered evaluation image identity, fold contracts, lineage isolation and candidate mapping are checked;
  * tables are sealed, and the decision is recomputed from them, so a fabricated result is refused.
* **Declarations:** protocol v4, then amendments 1, 2, 3 and 4. Each was declared before the scoring it governs and pins
  its predecessors. Amendments 3–4 pin their code (24 and 28 files), and amendment 4 was added without editing anything
  amendment 3 pins.
* **Tests:** 186 R2/v4 tests pass (CPU), including synthetic negative tests for every fail-closed path.
* **Failures:** every failure is kept, for example H_RT launch 1 (15 arms failed closed on a dirty worktree, retried as
  attempt 2). No failure or retry occurred in the J4b stages.

## 6. Comparison methodology

* **Primary endpoint:** H_bal.
* **Gate:** mean gain ≥ +1 pp; ≥ 5/6 folds improved (strict); AUSUC change ≥ 0; healthy-shortcut change ≤ +5 pp.
* **Confirmation:** exactly seeds {0, 1, 2}, with the gate applied to per-fold seed means.
* **Calibration:** each row uses its own LOFO (config, γ) choice.
* **Child rows** (amendment 3) are compared with the one fixed parent J2 on every fold. Diagnostics carry no pass/fail.
* **Uncertainty:** paired composition-cluster bootstrap CIs are exploratory; the bootstrap sign share is descriptive, not a
  p-value; no multiplicity adjustment.
* **Disclosures:**
  * the folds are reused, so the work is adaptive and not blind;
  * the J2 parent was chosen on all six folds;
  * image-fold counts are not independent (628 image-folds = 111 unique images).

## 7. Academic assignment relevance

These points can go into the AS02 discussion as **supplementary development evidence**. They do not replace the First Run
results and add no official-test claim.

* **Unseen-composition generalisation:** a CLIP ViT-L/14 backbone with partial fine-tuning and a composition-synthesis
  downstream improves the harmonic seen/unseen accuracy over frozen features by about 3–3.7 pp. The gain is reproducible
  across seeds and confirmed by a nested estimate.
* **Calibration:** results depend on seen/unseen calibration (γ), a standard generalized zero-shot learning issue.
* **Negative results:** photometric augmentation did not reproducibly help, and fine-tuning does not confer robustness to
  an unseen image source. Both are well-supported negative results with explicit protocols.
* **Methodology:** predeclared gates, three-seed confirmation, source-exclusion controls and fail-closed provenance can be
  cited as the evaluation design.

## 8. Limitations

* Development folds only: no independent test of any R2 method.
* Small numbers of compositions per fold.
* Source is confounded with composition content.
* Calibration dependence.
* J1–J3 are statistically indistinguishable.
* The J4b result rests on three seeds, and the seed-to-seed spread (−0.37 to +1.55 pp) exceeds the effect.

## 9. Recommended final research freeze

1. Freeze at the results commit of this report. Do not start J4a, J5–J8, further seeds, H_RT arms, other architectures or any
   official-test inference.
2. Use J1–J3 (reported as "partial CLIP fine-tuning", parent J2 by rule) as the best supported R2 development method, D1
   as the frozen reference, J4b as a confirmed negative, and H_RT as the source-shift finding.
3. Publication of the local commits (PR #36 is at `1b2fcdb`) needs separate owner approval. Nothing is pushed.
4. The First Run's 27 Nov submission keeps priority. R2 needs no further compute.

## 10. Evidence index

| Item | Location | sha256 |
|---|---|---|
| Amendment 4 | `research_r2_v4/amendment_4.json` / `DECLARATION_AMENDMENT_4.json` | `53c4b33a…faea` / `1e7ad336…cfd9` |
| Amendment 3 (unchanged) | `research_r2_v4/amendment_3.json` / `DECLARATION_AMENDMENT_3.json` | `ba3f2442…3456` / `95358ee8…9f54` |
| J4b three-seed evaluation | `artifacts/r2/v4/stage_j4b_confirm_20261117/evaluation.json` | `21d1de61…28a1` |
| J4b seed-0 screen | `artifacts/r2/v4/stage_j4b_20261117/` | review ZIP `8f82a773…a71b` |
| Test log (186 OK) | `artifacts/.../stage_j4b_confirm_20261117/logs/tests_gateB.log.txt` | `5b57250b…acb98` |
| J1–J3 three seeds | `artifacts/r2/v4/stage_j_phase2_20261115/`, `docs/R2_v4_StageJ_Phase2.md` | review ZIP `ece9e138…da7e` |
| H_RT | `artifacts/r2/v4/stage_hrt_20261116*/`, `docs/R2_v4_HRT_Report.md` | review ZIP `a30400d8…468e` |

* **Git (local, branch `research/r2-v4-ood-exploration`):** `d9ed896` → `42d6474` (implementation) → `9122366` (amendment
  4 declared) → results commit. Nothing is pushed.
* **Commands:** see `docs/R2_v4_J4b_ThreeSeed_Confirmation.md` section 10 and each stage report.
