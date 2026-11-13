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
