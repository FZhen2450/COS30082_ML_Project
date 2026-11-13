# Round 2 v3: candidate-universe correction and Phase 2–3 research (13 Nov 2026)

Starting point: PR #34 at `74766e5`. v1 and v2 files and results are preserved unchanged. The v2 outer folds are treated as already examined: they were not re-scored, and nothing below is a claim about them. No official image, filename or label was read, and no official inference was run.

## 1. The defect (confirmed)

`make_fold()` set `fit_mask = not held-out`. In v2 the inner population excludes the 22 outer compositions entirely, so in **every v2 inner fold those 22 compositions were scored as fitting ("seen") candidates and penalised by γ, although they had no fitting image** (`scripts/r2_audit_v2_candidates.py` → `artifacts/r2/audit_v2_candidates.json`).

- v1 folds and the v2 outer folds had no such candidates.
- No validation truth was ever one of them.

Three different things had been conflated:

| Class | Meaning | v2 inner handling | Correct handling |
|---|---|---|---|
| F, fitting | has fitting images | scored, penalised | scored, penalised |
| H, evaluation held-out | evaluated as unseen; name known | scored, not penalised; synthetic features | same |
| E, externally excluded | no fitting images, not evaluated (outer compositions during inner development) | **scored and penalised as F** | **not scored**; name not used; no synthesis |

Measured impact on the saved v2 inner scores (column deletion, heads unchanged):
- **Trained rows (B1, D1, D2):** no Top-1 prediction ever landed on an E candidate at the registered γ. H_bal, image-weighted H, Top-5 and the selected γ are identical, and AUSUC changes by ≤ 0.01 pp. At γ=0, E candidates took a few Top-5 slots (B1 unseen Top-5 20.28 vs 20.71).
- **Frozen zero-shot:** 3.2% of seen and 2.9% of unseen Top-1 predictions went to E candidates. H_bal 14.41 → 14.52, H@5 42.7 → 44.6.

Column deletion cannot capture training-side effects: the D1 composition head was trained with 22 empty classes, and zero-shot normalisation included them. A full re-run was therefore required (section 3).

## 2. Candidate-universe policy (`research_r2/candidates.py`)

The universe for each fold is F ∪ H, and E is removed. This is the only choice that both matches the official structure (F = 323 trained, H = 10 named unseen, E = ∅) and keeps the outer population isolated. Treating E as H would give about 33 unseen candidates among 323 (official: 10 of 333) and would use outer names during inner development.

Invariants checked on every fold:
- F and H are disjoint;
- every F candidate has fitting images and no H candidate does;
- seen-val truth lies in F and unseen-val truth in H;
- the stored universe equals the re-derived universe, and tampering raises an error.

`derive_fold()` keeps the policy for fitting subsamples and source ablations: a composition emptied by an ablation moves from F to E.

**Benchmark v3** (`artifacts/r2/benchmark_v3.json`, semantic SHA-256 `0e85819c…7e42`) has v2's image partitions unchanged, with a per-fold universe added: inner folds have 301 candidates (290–291 F, 10–11 H, 22 E) and outer folds 323 (E empty).

Saved v3 scores carry their `cand_ids`. All score consumers (`scorer_from_npz`) use them; the legacy v1/v2 code path is kept so history reproduces. On that path the current code reproduces saved v2 B1/D1/D2 scores on every inner fold to ≤ 2e-6 (`scripts/r2_verify_legacy_repro.py`).

## 3. Phase 1: correction replay (unchanged v1 matrices on v3 inner folds; protocol `e25bbc3`)

Across all 31 rows, the largest change from v2 is 0.52 pp H_bal (D2; D1 +0.16) and 0.24 pp AUSUC (C3). γ is unchanged except D2 (8.0 → 8.25).

| Gate step | Δ H_bal | Folds better | Δ AUSUC | Pass |
|---|---:|---:|---:|:---:|
| D1 vs B1 | +2.62 | 6/6 | +3.38 | ✓ |
| D2 vs D1 | +0.39 | 3/6 | +2.17 | ✗ |
| D3 vs D1 | +0.97 | 4/6 | +0.33 | ✗ |

**The selection is unchanged: D1.**

The nested leave-one-fold-out estimate re-runs the whole selection without each held fold, then scores the held fold:
- the selected recipe beats the control on **6/6** held folds;
- mean H_bal is 65.8 vs 63.6 (**+2.23 pp**);
- D1 is selected in 5 of 6 runs, D2 in 1.

**Conclusion:** the defect was real and is now fixed, but it did not change any v2 conclusion.

## 4. Phase 2: source robustness, synthesis, disentanglement and representation (11 rows vs D1; matrix declared in `e25bbc3`)

| Row | Idea | H_bal | Δ vs D1 | Folds | Δ AUSUC | Gate |
|---|---|---:|---:|---:|---:|:---:|
| D1 | control | 66.20 | | | | |
| F1 | equal weight per source within each crop and disease (heads) | 66.14 | −0.07 | 3/6 | +0.37 | ✗ |
| F2 | equal weight per source within each composition (composition head) | 66.60 | +0.40 | 5/6 | −0.03 | ✗ |
| F3 | composition-controlled source-style offset augmentation | 66.84 | +0.64 | 4/6 | +0.99 | ✗ |
| F4 | synthesis shifted only along the LEACE crop subspace | 65.51 | −0.69 | 1/6 | −0.38 | ✗ |
| F5 | synthesise seen pairs too (removes the real/synthetic cue) | 66.64 | +0.43 | 4/6 | −0.30 | ✗ |
| F6 | within-composition noise on synthetic features | 66.22 | +0.02 | 3/6 | −0.17 | ✗ |
| F7 | crop-centred disease head (disentanglement) | 65.86 | −0.34 | 2/6 | −1.35 | ✗ |
| F8 | F1 + F2 | 65.97 | −0.23 | 1/6 | +0.22 | ✗ |
| F9 | composition head weighted ×2 in fusion | 66.56 | +0.36 | 4/6 | +0.11 | ✗ |
| **F10** | **D1 on concatenated CLIP-L/14 + ConvNeXt-B features** | **68.32** | **+2.12** | **5/6** | **+2.42** | **✓** |
| F11 | flip test-time augmentation | 66.08 | −0.12 | 3/6 | −0.34 | ✗ |

- **F10:** image-weighted H is 63.97 vs 61.56. Per fold it gains +7.4 (I1), +2.0, +1.6, +1.7, +0.3 and −0.3 pp. Its **nested estimate is only +0.89 pp (4/6 held folds better)**, below the gate's own thresholds, so selection optimism is about 1.2 pp and the gain is concentrated in one fold.
- **F7 and the text priors share a pattern:** both raise γ=0 within-crop disease transfer (D1 15.0 → F7 18.5) without improving calibrated H_bal.

## 5. Phase 3: representation diversity (declared after Phase 2, `44d98ef`)

| Row | H_bal | Δ vs F10 | Folds | Δ AUSUC |
|---|---:|---:|---:|---:|
| + EfficientNet-B3 | 68.11 | −0.21 | 3/6 | −1.11 |
| + CLIP-B/32 | 67.59 | −0.73 | 1/6 | −0.16 |
| + ViT-B/16 (ImageNet) | 68.06 | −0.26 | 2/6 | −0.23 |
| CLIP-L + EfficientNet-B3 instead of ConvNeXt | 65.92 | −2.40 | 1/6 | −3.75 |

No row passes. The complementary signal is specific to ConvNeXt-B, not to "any CNN" or to more backbones.

## 6. Source-shift stress (refit with one source removed from fitting; v3 inner; 29 ablations, 19 skipped for concept coverage)

Seen-val images of the removed source (479; their compositions keep fitting images from other sources) and unseen-val images of that source (1,000):

| Row | Seen Top-1, source in fitting → removed, γ=0 | Same at γ=8 | Seen pulled into held-out pairs at γ=8 | Unseen Top-1, in fitting → removed, γ=8 |
|---|---|---|---:|---|
| D1 | 90.4 → 35.7 | 72.0 → 17.3 | 59.1% | 55.5 → **55.0** |
| F10 | 91.0 → 30.9 | 74.7 → 15.0 | 61.2% | 55.9 → **46.9** |

D1 reproduces the v2 finding (unseen 55.4 → 54.8 there): source novelty collapses the *seen* side, and γ=8 compounds it, while unseen recognition is unaffected. **F10 is less source-robust than D1**: 4.8 pp worse on novel-source seen images at γ=0, and 9.0 pp of unseen accuracy lost when the source is removed. A plausible reading is that part of the ConvNeXt complement is source-style information. This is a further reason not to promote F10. The optimal γ is again stable under fitting-set size (D1 7.75 / 7.75 / 8.0; F10 7.5 / 7.5 / 8.0 at 50 / 75 / 100%).

## 7. Verification problem found and fixed during this round

While adapting `scripts/r2_calibration_shift.py` to the v3 universe, the unseen-side comparison used base-fold truth indices for ablated-fold predictions, which have a smaller universe. The first v3 stress outputs therefore reported a spurious collapse of unseen accuracy (55.5 → 8.5%).

The contradiction with v2 (55.4 → 54.8%) exposed it. The fix uses the ablated fold's own truth indices. The flawed files are kept under `artifacts/r2/v3/superseded/` with a README. The fixed script reproduces the committed v2 stress numbers exactly on the legacy benchmark (48 ablations, 520 numeric fields, maximum difference 0; `artifacts/r2/v3/stress_v2_repro_check.txt`).

## 8. Standing of candidates and limitations
- **D1 remains the provisional research candidate.** It passes the gate in v1, in v2 (inner and outer) and in v3, and its v3 nested estimate is +2.23 pp at 6/6. It is still a development result.
- **F10 is promising but not established.** It passes the in-sample gate, but its nested estimate (+0.89 pp, 4/6) does not meet the gate thresholds, its gain is concentrated in fold I1, it is **less** robust to source shift than D1, and it has had no independent confirmation. Promotion would need evidence outside this development data.
- **No untouched confirmation population exists.** All v3 evidence is development cross-validation on images that were all used in v1 and v2. The v2 outer folds are spent. Nested estimates are the strongest remaining within-data safeguard.
- **Calibration stays the dominant effect** and remains sensitive to source shift on the seen side (section 6). None of the source-robust training variants tested here changed that.
- **Scope:** all methods use frozen features with linear heads. Fine-tuning and other pretrained backbones (DINOv2, SigLIP, BioCLIP; downloads not authorised) remain untested.

## 9. Next steps (not run)
1. **A genuinely independent check** of D1 versus F10 would need new data or the owner's decision on a non-blind official run after the first submission. Report γ=0 and the registered γ side by side through the WP5 importer.
2. **Source robustness needs an objective that changes the representation,** not just reweighting. Candidates: partial fine-tuning with source-adversarial or style-randomised views, with the source-ablation test as a gate.
3. **Calibration that adapts to source novelty**, for example γ scaled by distance to the nearest fitting prototype, judged on the stress test.
4. **Disease-transfer signals** (text priors, F7) improve uncalibrated within-crop transfer. A joint objective that keeps this under calibration is untested.
