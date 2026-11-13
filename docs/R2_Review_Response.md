# Round 2 review response (13 Nov 2026)

This responds to the owner's review of #33 and #34. All v1 results, files and labels are preserved unchanged. New evidence is added beside them.

## 1. Locked-fold independence (v1)

The reviewer's counts reproduce exactly (`scripts/r2_audit_v1_independence.py` → `artifacts/r2/audit_v1_independence.json`). The audit also shows the overlap is total:

| v1 fold | Unseen-val | In fitting of all 5 screening folds | Scored as screening seen-val | Untouched by screening | Seen-val also scored in screening |
|---|---:|---:|---:|---:|---:|
| L1 | 739 | 628 | 111 | **0** | 1,083 / 1,083 |
| L2 | 344 | 292 | 52 | **0** | 1,142 / 1,142 |
| L3 | 346 | 293 | 53 | **0** | 1,141 / 1,141 |

All 8,000 images were fitted on or scored during v1 screening. v1 L1–L3 are therefore **cross-validation confirmation folds**: they are composition-disjoint *within each run* and were scored once with a frozen recipe and γ. They are **not** an isolated outer evaluation. The v1 recipe and γ were selected using models fitted on the L-fold unseen compositions, and the v1 seen holdout is shared by every fold.

Consequence: no image in the training data is untouched with respect to D2. Benchmark v2 (section 5) isolates an outer population **by data** from a mechanical replay of the selection. It cannot remove the researcher's v1 exposure: every outer composition was analysed in v1. Only new data would give a fully fresh test. The official sets are excluded, being previously examined and non-blind.

## 2. Separate metrics for D2 (v1 cross-validation folds L1–L3; `artifacts/r2/metrics_v1_cv_locked*.{json,csv}`)

| Mean over L1–L3 | D2 γ=0 | D2 γ=8 | Control B1 γ=0 | B1 γ=8 | B1 γ=10.75 (its own) |
|---|---:|---:|---:|---:|---:|
| H_bal@1 (seen micro vs unseen composition-macro) | 12.8 | **73.1** | 8.6 | 63.9 | 68.6 |
| Image-weighted H@1 (seen micro vs unseen micro) | 5.8 | **68.4** | 4.5 | 52.3 | 64.3 |
| H@5 (micro) | 48.1 | 97.9 | 25.0 | 92.3 | 96.6 |
| Seen Top-1 | 85.0 | 71.8 | 84.5 | 79.1 | 68.9 |
| Unseen Top-1, micro | 3.1 | 66.0 | 2.3 | 39.4 | 60.9 |
| Unseen Top-1, composition-macro | 7.1 | 74.5 | 4.5 | 54.2 | 68.4 |
| Within-crop unseen accuracy (true crop given) | 14.9 | 82.4 | 11.9 | 63.9 | 80.5 |
| Healthy-shortcut rate | 3.7 | 1.1 | 8.5 | 6.0 | 3.2 |
| AUSUC (γ-free) | 69.0 | | 61.4 | | |

**Calibration accounts for most of the measured value.** For D2, calibration alone moves H_bal from 12.8 to 73.1 (+60.3 pp). The representation difference between D2 and the control is +4.2 pp at γ=0, +9.2 pp at equal γ=8, +4.5 pp with each at its own γ, and +7.6 AUSUC points. Per-composition Top-1 at each γ is in `metrics_v1_cv_locked_per_composition.csv`. The zero-shot reference has a different score scale (its own γ is 1.75), so its γ=8 numbers are not meaningful.

## 3. Calibration robustness (training-derived only; no γ was re-selected)

Files: `artifacts/r2/calibration_v1_D2.json`, `calibration_v2inner_D1.json`, `calibration_shift_v2inner_*.json`, `v2_outer_gamma_diagnostic.json`.

| Stress | Evidence | Reading |
|---|---|---|
| Fold to fold | v1 D2: per-fold optimal γ 7.0–9.0, and γ=8 costs ≤ 1.3 pp on 8 folds. v2-inner D1: optimum 6.5–9.0, γ=8 costs 0.1–3.1 pp | Stable within the training distribution |
| Composition bootstrap | Pooled-γ 95% interval [7.5, 8.5] (v1 D2) and [7.1, 8.5] (v2-inner D1) | γ is not driven by a few compositions |
| Number of unseen candidates | Optimal γ rises as candidates fall: k=3 → 9.0–9.25, k=5 → 8.4–8.5, k=8 → 8.1–8.25, all (10–11) → 7.5–8.0. γ=8 costs up to 4.4 pp at k=3 | Official has 10. Fine there; fragile for a much smaller candidate set |
| Crops and diseases (fixed γ) | Unseen macro accuracy p10–p90 spans 53–100% by crop (worst: strawberry, coffee, rose) and 38–95% by disease (worst: spider mites, alternaria, target spot) | One global γ hides large per-concept differences |
| Source datasets, seen cost | γ=8 lowers seen Top-1 by a median of 10.7 pp per source and up to 63 pp (CDRD) | Some sources' seen images are routinely pulled into held-out pairs |
| **Source shift** (v2 inner; refit with one source removed from fitting; 29 ablations, 19 skipped for concept coverage) | Seen-val images of that source (pairs still present via other sources): **90.4 → 35.5% at γ=0** and 72.0 → 17.7% at γ=8, with 58.7% pulled into held-out pairs. Unseen images of that source: 55.4 → 54.8% at γ=8 | **Strongest evidence so far of a source-style shortcut.** γ=8 is a good operating point only when seen images come from familiar sources |
| Fitting-set size (v2 inner, composition-stratified) | D1 optimal γ 7.75 / 7.75 / 8.0 at 50 / 75 / 100%; B1 10.25 / 10.25 / 10.5 | Slight upward drift with more data. An 8,000-image refit would plausibly sit near 8–8.5 (extrapolation, not measured) |
| Outer transfer (post-hoc, no selection) | O1 optimum 9.0 (registered 8: −3.2 pp); O2 optimum 8.0 (0 pp) | Same scale as between inner folds |

v1 folds contain **no** source-novel unseen images: every held-out composition shares its source with fitting compositions. The source-shift test above was built because of that gap. Whether the official images share training sources is unknown and was not checked, since that would mean reading official filenames.

## 4. Independent implementation review (`scripts/r2_verify_implementation.py` → `artifacts/r2/verify_implementation_S1.json`)

| Check (real data, fold S1) | Result |
|---|---|
| Transductive invariance: replacing half the validation rows with noise leaves every other row's scores unchanged (B1, C4 erasure, D2), so standardisation, LEACE, crop means, synthesis and heads use fitting rows only | PASS, max change 0.0 |
| Factorised score = log(WP5 `composition_scores_from_logits`) | PASS, max diff 7e-15, identical Top-1 |
| Synthesis: x − μ(source crop) + μ(target crop), with composition-balanced crop means re-derived independently in pandas; sources never come from the target crop; sources are fitting rows only | PASS, max error 4e-16 |
| Candidate masks: unseen candidates are exactly the held-out set; no penalised candidate lacks fitting images in any v1 fold | PASS |
| Score scaling: the fused score is a mean of log-probability terms | PASS. Disclosure: D2's factorised term is a sum of four log-probabilities (two heads plus two text priors), so its scale differs from B1's. This is why their γ values differ |
| Reproducibility: a re-run equals the saved Stage E scores | PASS, max diff 2e-6 (float32 storage), identical Top-1 |
| Training vs inference: returned heads are in eval mode; repeated scoring is identical | PASS |
| Lineage controls | `validate_benchmark` / `validate_benchmark_v2` (tests) |

**Guard defect (fixed).** The Stage E guard compared `--frozen-recipe-sha256` with the hash of whatever matrix was supplied, so any replacement matrix plus its own hash passed. Protected folds now unlock only through `--frozen-recipe NAME`, which `research_r2/guards.py` checks against the committed `research_r2/frozen_recipes.json` registry (path, file hash, benchmark semantic hash and row ids). Requesting a locked or outer fold directly raises an error. `stage_e_frozen_v1` is registered retrospectively, and the registry records that its v1 run predates this guard. Tests: `tests/test_r2_review.py`.

## 5. Benchmark v2 with an outer population isolated by data (`research_r2/benchmark_v2.py`)

- **Outer population:** O1 and O2, 11 diseased compositions each, chosen so that concept coverage survives removing all 22. An outer seen holdout of 669 images. Lineage closure gives an exclusion set X of 1,751 images.
- **Inner population:** 6,249 images. Inner eligibility, 6 inner folds of 10–11 held-out pairs, and the inner seen holdout are all built from it alone.
- **Validator:** X and the inner population partition the 8,000 images, no lineage group spans both, inner folds contain zero outer rows, and outer crops and diseases are all covered by inner data. Semantic SHA-256 `2689b007…8a68`. A test rebuilds v2 deterministically.
- **Enforcement:** during inner runs, the features of all X rows are overwritten with NaN, so any use would fail scoring.
- **Order of commits** (each pushed before the next stage ran):
  1. benchmark, protocol and guards (`f8f1875`);
  2. inner replay plus mechanical selection, and registration of `outer_v2_frozen` (`c06dc4e`);
  3. a single outer scoring.

### Inner replay (unchanged v1 matrices on I1–I6)
| Step | Δ H_bal | Folds better | Δ AUSUC | Gate (≥ 5/6) |
|---|---:|---:|---:|:---:|
| D1 (hallucinated composition head) vs B1 | +2.46 | 6/6 | +3.3 | **pass** |
| D2 (D1 + text priors; the v1 choice) vs D1 | +0.03 | 3/6 | +2.2 | **fail** |
| D3 (D1 + 50% erasure) vs D1 | +1.03 | 3/6 | +0.4 | fail |

The rule selects **D1**. Other v1 Stage C passers do not replicate either: C7 text priors −1.08, C4 erasure +0.64 at 4/6. C9/D1 is the only intervention that passes in both v1 and v2.

### Outer confirmation (scored once; each row at its own pooled inner γ)
| Mean of O1, O2 | D1 (selected) | D2 (v1 choice) | B1 control | Zero-shot |
|---|---:|---:|---:|---:|
| γ | 8.0 | 8.0 | 10.5 | 1.75 |
| H_bal@1 | **67.1** | 69.1 | 62.7 | 12.6 |
| Image-weighted H@1 | **67.3** | 65.5 | 64.9 | 8.8 |
| H@5 (micro) | 98.3 | 97.8 | 97.2 | 29.7 |
| Seen Top-1 | 75.6 | 73.1 | 72.1 | 8.4 |
| Unseen Top-1 micro / macro | 60.6 / 60.6 | 59.6 / 65.5 | 58.9 / 55.6 | 14.7 / 25.8 |
| Within-crop unseen | 74.8 | 76.2 | 72.6 | 45.3 |
| Healthy shortcut | 2.3 | 1.1 | 2.4 | 0.3 |
| AUSUC | 58.2 | 60.5 | 54.0 | 3.6 |
| H_bal at γ=0 | 5.9 | 7.1 | 4.5 | 6.9 |

Per fold, D1 vs B1 H_bal: O1 65.7 vs 62.1, O2 68.6 vs 63.2.

Paired composition bootstrap, 5,000 resamples (`artifacts/r2/v2_outer_bootstrap.json`):
- **D1 − B1:** H_bal **+4.5 pp, 95% CI [+1.1, +8.6]**; image-weighted H +3.0 [−1.2, +9.4].
- **D2 − B1:** +6.4 [+1.3, +13.9]; image-weighted +1.4 [−3.4, +16.8].
- **D2 − D1:** +1.8 [−2.5, +7.7]; image-weighted −1.5.
- **Per composition, D1 vs B1:** 8 better, 6 worse, 8 tied of 22.

### Conclusion of the review
1. The **hallucinated composition head (D1) survives** a data-isolated outer evaluation on the balanced metric: +4.5 pp H_bal, with an interval excluding zero. Its image-weighted gain is smaller and not established. It improves 8 of 22 outer compositions and worsens 6.
2. The **text priors that made D2 the v1 choice do not replicate** (inner gate fails; outer D2 − D1 interval spans zero). The v1 claim "D2 > D1" is withdrawn as unsupported. The v1 numbers themselves stand as recorded.
3. **Calibration is the dominant effect** (5.9 → 67.1 H_bal for D1 on the outer folds). It is stable within the training distribution but strongly sensitive to source shift on the seen side.
4. **This is not a fully independent test.** It isolates by data, but every outer composition was analysed in v1, and the candidate set was designed after v1 Stage C. No population inside the 8,000 images remains untouched by the researcher.

## 6. Submission separation
No official image, filename or label was read. No official inference was run, and the accepted four-model comparison is unchanged. PR #34 stays a draft and unmerged, and no team branch, issue deliverable or Round 1 file was modified.

## 7. Proposed next research step (prepared only; nothing run)
The v2 outer population is now spent for this candidate family. Further development should use the v2 inner folds, plus the source-ablation stress test as an extra gate.

1. **Source-robust learning, aimed at the evidenced shortcut.** Candidates: source-balanced or group-DRO weighting across sources within a composition; style and colour randomisation of frozen-feature views; per-source feature whitening estimated on fitting rows only. Primary check: novel-source seen accuracy in the source-ablation test, without losing inner H_bal.
2. **Source-aware or confidence-adaptive calibration.** For example, γ scaled by the image's fitting-set novelty (distance to the nearest fitting prototype), selected on inner folds and judged on the source-ablation stress test. A single global γ is fragile there.
3. **Keep D1 as the frozen-feature reference.** Do not pursue text priors further without a new hypothesis.
4. **Larger steps that need coordination or authorisation:** a partial CLIP fine-tuning pilot, and DINOv2 / SigLIP / BioCLIP frozen features.

Any official run of D1 would need explicit owner approval and must be labelled non-blind. It should report γ=0 and the registered γ side by side, through the WP5 importer as a separately labelled model.

---
**v3 addendum (13 Nov 2026).** See `docs/R2_v3_Candidate_Policy_and_Phase2.md`. v2 inner folds scored the 22 outer compositions as penalised fitting candidates without fitting images. The corrected benchmark v3 leaves every v2 conclusion unchanged: D1 is selected again, with a nested estimate of +2.23 pp at 6/6.
