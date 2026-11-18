# R2 v4 after H_RT: reviewer summary (17 Nov 2026)

**Disposition:** H_RT evidence conditionally accepted by the independent review. No result, attempt, receipt or checksum has been changed since `9659270`.

**Labels:** everything below is post-freeze exploratory development cross-validation on the reused v3 inner folds of the DPD-Small training set. None of it is an official-test result.

## 1. Three-seed development improvement (Phase 2, confirmed)

- **Size of the gain:** J1, J2 and J3 (partial fine-tuning of CLIP ViT-L/14) beat frozen-feature D1 (66.20) by +2.95, +3.36 and +3.70 pp H_bal.
  - Every row: 6/6 folds on the per-fold seed mean; nested +2.8 to +3.7 pp.
  - Three-seed bootstrap CIs: J1 [+1.19, +5.11], J2 [+0.83, +6.33], J3 [+1.63, +6.26].
- **The variants are indistinguishable:** J2 − J3 is −0.34 [−2.51, +1.65].
- **Where the gain comes from:** mostly unseen-composition macro (+3.3 to +5.6) and seen Top-1. Unseen disease rises only +2.5 to +2.9 pp, to about 61%.
- **Calibration dependence:** at γ = 0, H_bal is 5–9%.

## 2. Full source retraining (H_RT, diagnostic)

**Design:** 3 rows × 3 registered pairs × {source-excluded, size-matched random removal}, seed 0, retrained from the pretrained weights; 18/18 arms ok.
- **Effect on novel-source seen Top-1** (source-excluded minus random):

| Row | Mean of 3 pairs | Range across pairs |
|---|---|---|
| J1 | −54.5 pp | −45 to −64 |
| J2 | −53.1 pp | −47 to −61 |
| J3 | −59.0 pp | −55 to −66 |

- **These are differences between two removal arms. They are not isolated causal effects of source style.**
- **Per-pair intervals are wide:** for example, I1/MTID J1 is [−91.7, −6.9]. That pair rests on 53 images from 2 compositions.
- **One seed per arm:** no training-seed variability is included.

## 3. Whole-fold versus excluded-source images

- **Whole fold:** H_bal source effects are −3.1 to +0.7 pp, and every CI spans 0. Each novel-source set is a small part of the evaluation rows (39–135 of 676–856 seen-val images).
- **Excluded-source images, no advantage for fine-tuning:** with the source excluded, J and D1 reach similar accuracy on them (J − D1 from −5.7 to +7.7 pp, mixed sign).
  - J2 is above D1 on I1 (+7.5 [+4.2, +10.3]).
  - J3 is below D1 on I1 and I6.
- **Whole fold, gain persists:** the J-over-D1 H_bal advantage largely persists in the source-excluded arms (−0.7 to +4.8 pp). It comes from images of the sources that remain.

## 4. Composition-coverage confounding

- **I6/CDDM:**
  - Excluding CDDM removes 1,950 of 5,062 fitting images and **empties 25 compositions**. Those compositions and their 202 seen-val images are removed from every arm's comparable evaluation population.
  - In the surviving compositions, the two arms have **different supervision coverage**. The source-excluded arm loses CDDM images within compositions. The random arm removes mostly other compositions' images: only 231 of 1,950 picks are same-composition matches.
- **I1/MTID (1 emptied, 53 of 381 same-composition picks) and I2/CPDD (5 emptied, 201 of 359):** less confounded. All of I1's novel-source images still fall in compositions that lost ≥ 75% of their images in the excluded arm.
- **Lineage:** one dhash lineage link (MTID maize images and `SCLD_sugarcane_viral_4947.jpg`, different crop) involves fitting images only. It is recorded, not removed, and no training/evaluation isolation is affected.

## 5. Observations versus causal explanations

- **Observed:**
  - Seen-image accuracy collapses when an image's source is absent from training.
  - The collapse is as large for fine-tuned as for frozen representations.
  - In small strata where the random arm removed a similar or larger share of a composition's images, the random arm keeps 74–91% while the excluded arm falls to 21–32%. These strata are 22 images in 5 compositions (I6) and 19 images in 4 compositions (I2).
- **Not established:** that "source style" causes the drop. Source is confounded with composition-specific content (disease stage, background, capture protocol) and with per-composition supervision.
- **H_RT is not general:** three sources, one seed and small strata cannot establish robustness or non-robustness across all sources.

## 6. Remaining generalisation limitations

1. All development numbers reuse the folds that informed v1–v3, Stage I and Phase 1. Nothing is blind or independent.
2. No Second Run model has an official DPD-Small test result, and none is compared with First Run's official WP1 baseline here (First Run is off-limits).
3. Unseen-disease transfer remains the weakest component (about 61% Top-1).
4. Performance depends on LOFO calibration (γ ≈ 8).
5. Source novelty is a dominant failure mode that fine-tuning did not reduce.
6. The J4a–J7 rule as registered cannot attribute child effects (Gate F audit). A repaired proposal is in `docs/R2_v4_Amendment3_PROPOSAL_v2.md`; it is not declared.
