# Research Round 2 — compositional generalisation: design record

Status: Stage A–C (research, controls, frozen-feature screening). Isolated from the first
submission: no Round 1 checkpoint, packet, notebook, GUI asset or PR branch is modified.
No official-test image, filename or label is read by any Round 2 code (`research_r2/data.py`
refuses both official CSVs by name; tests enforce this).

## 1. Evidence that motivated Round 2 (checked, not from summaries)

| Observation | Source |
| --- | --- |
| Official first-run H@1: WP1 ResNet50 pair 9.95%, frozen CLIP 6.62%, M6 adapter 5.96%, ConvNeXt 1.76% (non-blind; already examined) | `workspace/experiments/official-final-full8000-20261112/.../comparison.csv` |
| On the Round 1 development split the order was nearly reversed (WP1 1.36%, adapter 18.89%) | `four-model-development-report/20261107T045821Z/comparison.md` |
| The Round 1 unseen-validation fold holds out 65 of 323 candidates (20%), 12 of them **healthy** (21% of its images); one composition (cassava mosaic, 544 images) is 29% of the fold. The official unseen set is 10 of 333 candidates, **all diseased** | `artifacts/data/heldout_composition_seed42.json`; brief §4.2; DPD paper §3.4 |
| M6 dev gains came mostly from healthy held-out pairs (319 of 354 net hits) | Issue #3, 5 Nov |
| Frozen-CLIP cross-crop disease retrieval collapses from 25.4% to 2.4% Top-1 when the query's own composition is excluded | Issue #3, M11 |
| 277 of 323 training compositions come from a single source dataset (filename prefix); the majority crop per source predicts crop for 64% of images | this audit |
| All 10 official unseen classes pair a crop known from 3–22 other diseases with a disease known on 2–12 other crops | brief + training CSV |

Conclusion: the Round 1 development benchmark measured something different from the
official task (large held-out candidate share, healthy pairs, one dominant composition),
so selections made on it did not transfer. Round 2 first builds a benchmark that mirrors
the official structure, then compares methods on it.

## 2. Literature and hypotheses (Stage A)

| Family | Key references | Hypothesis tested here | Cost |
| --- | --- | --- | --- |
| Calibrated stacking for generalised ZSL | Chao et al., ECCV 2016; standard in GZSL/CZSL (Purushwalkam et al., ICCV 2019 report AUC over a bias sweep) | Factorised scores are biased toward seen pairs; one bias selected on training-only folds fixes the decision rule | inference only |
| Concept erasure ("erase details" interpretation) | LEACE, Belrose et al., NeurIPS 2023; INLP, Ravfogel et al., ACL 2020 | Removing linearly decodable crop identity from disease-head input forces crop-invariant disease evidence | closed form |
| Domain/source invariance | Group DRO (Sagawa et al., ICLR 2020); DomainBed (Gulrajani & Lopez-Paz, 2021) | Source-dataset style is a shortcut for compositions | closed form |
| Balanced / re-weighted learning | Cui et al., CVPR 2019; Menon et al., logit adjustment, ICLR 2021 | Dominant compositions teach the disease head crop context | free |
| Feature-space compositional synthesis | f-CLSWGAN (Xian et al., CVPR 2018); attribute–object hallucination (Wei et al., ICCV 2019) | Shifting disease features to other crops' means simulates unseen pairs | cheap |
| Vision–language priors | CLIP (Radford et al., 2021); CSP (Nayak et al., ICLR 2023); DFSP (Lu et al., CVPR 2023); DPD-paper FF-CLIP (Liaw et al., ICPR 2024) | Crop-agnostic disease text carries knowledge independent of co-occurrence | cheap |
| Backbone families | CLIP ViT-L/14 & B/32, ImageNet ViT-B/16, ConvNeXt-B, EfficientNet-B3 (all local) | Representation choice dominates head choice | ~3 min/backbone |
| Lecturer group's own unseen-plant work | Chai et al., ICIP 2023 (pairwise feature learning); Chai et al., Neurocomputing 2024 (self-supervised CL-ViT); DPD paper Table 4 (full DPD: H 5.2–13.0%) | Self-supervised / masked objectives may be what "erase details" referred to; not assumed | GPU-heavy |
| Not available locally (would need an authorised download) | DINOv2 (Oquab et al., 2023), SigLIP (Zhai et al., 2023), BioCLIP (Stevens et al., 2024) | Expected strong frozen features; listed as a follow-up only | — |

"Erase details" is investigated under several separate interpretations (label/composition
masking via the held-out folds, linear concept erasure of crop or source, background/style
invariance) and none is attributed to Dr. Lee.

## 3. Benchmark and leakage control (`research_r2/benchmark.py`)

- **Population:** the 8,000 training images only (CSV SHA-256 `9f1b72…6a7d`).
- **Eligible held-out compositions (87; 3,197 images; 34 diseases; 33 crops):** diseased, at least 5 images, crop present in another composition, disease present on at least one other crop. This mirrors the published structure of the 10 official unseen classes. Healthy compositions are never held out.
- **Folds:** 8 composition-disjoint folds of 10–11 held-out compositions (≈3.4% of 323 candidates, matching 10/333). They are built by seeded, size-balanced greedy assignment, which keeps every held-out crop and disease present in fitting and every held-out disease observed on another crop. A seeded role permutation then makes **S1–S5 screening** folds and **L1–L3 locked** folds.
- **Seen validation:** one global holdout G of about 15% per composition, sampled by lineage group. Compositions with fewer than 4 images contribute none. Per fold, seen-val is G minus the held-out compositions.
- **Lineage:** connected components of (identical SHA-256) or (dHash-64 Hamming ≤ 4) over training images. 42 linked pairs were found, 10 of them across compositions. Groups are never split between fitting and any validation set. Held-out images linked to fitting images are dropped (1 image in S1, 1 in D0).
- **Candidates:** always all 323 training-source compositions in ascending global ID order. The fold's held-out pairs are the "unseen" candidates.
- **Legacy fold D0:** the Round 1 65-composition holdout, materialised under the same rules. It is labelled `legacy_diagnostic_previously_examined` and is never used for selection.
- **Validation:** `validate_benchmark` asserts disjointness, completeness, composition-disjointness, concept coverage, lineage cleanliness, zero healthy held-out pairs, and an exact partition of the eligible set. The semantic SHA-256 `b6db7509…4c54` is pinned in `research_r2/protocol.json` and replayed by a test.

## 4. Predeclared selection policy (`research_r2/protocol.json`, committed before scoring)

Primary: mean over S1–S5 of **H_bal@1**, the harmonic mean of seen-val micro Top-1 and
unseen-val *composition-macro* Top-1. It is computed after calibrated stacking with a
**leave-one-fold-out γ**: the γ for each fold is chosen on the other four folds. Gate
versus a matched control: at least +1.0 pp mean, improvement in at least 4/5 folds, no
AUSUC loss, and at most +5 pp healthy-shortcut rate. Locked folds are scored once, for the
Stage E recipe only, with γ fixed from S1–S5.

Diagnostics added during Stage B, before any Stage C row was run:
- **within-crop unseen accuracy:** only the true crop's candidates are ranked, so the held-out disease must beat that crop's fitting diseases on disease evidence alone;
- **exact AUSUC:** the γ-free quality of the representation.

## 5. Results (development only; training-source images; all numbers reproducible from `artifacts/r2/results_*.jsonl`)

All rows use frozen backbones with linear heads fitted on each fold's fitting images. H_bal is a percentage, given as the mean over S1–S5 with LOFO γ. AUSUC is γ-free. wc0 is within-crop unseen accuracy at γ=0.

### Stage B controls
| Row | H_bal | H_bal γ=0 | AUSUC | wc0 | Reading |
|---|---:|---:|---:|---:|---|
| B0 frozen CLIP-L zero-shot | 12.6 | 8.5 | 4.6 | 28.5 | Language alone gives some crop-agnostic disease signal but very low seen accuracy (8%) |
| B1 factorised linear heads | 63.3 | 5.6 | 51.5 | 11.3 | Uncalibrated, 94–100% of unseen images go to a fitting pair; one training-fold γ recovers balance |
| B2 flat seen-composition head | 13.8 | 0.0 | 6.8 | 0.0 | Memorisation control: cannot transfer |

### Stage C screen (control B1; gate = ≥+1 pp, ≥4/5 folds, no AUSUC loss, ≤+5 pp shortcut)
| Row | H_bal | Δ vs B1 | folds | AUSUC | Gate |
|---|---:|---:|---:|---:|:---:|
| A1 CLIP B/32 | 58.2 | −5.1 | 1 | 45.6 | ✗ |
| A2 ImageNet ViT-B/16 | 51.5 | −11.8 | 0 | 36.3 | ✗ |
| A3 ConvNeXt-B | 57.2 | −6.1 | 0 | 42.4 | ✗ |
| A4 EfficientNet-B3 | 54.8 | −8.5 | 0 | 40.1 | ✗ |
| C1 composition-balanced disease head | 61.4 | −1.9 | 1 | 51.8 | ✗ |
| C2 balanced, both heads | 61.3 | −2.1 | 1 | 51.5 | ✗ |
| C3 LEACE crop erasure (full) | 57.0 | −6.4 | 0 | 44.8 | ✗ |
| C4 LEACE crop erasure 50% | 64.5 | +1.2 | 4 | 53.4 | ✓ |
| C5 LEACE source erasure | 31.9 | −31.4 | 0 | 17.0 | ✗ |
| C6 crop-shift synthesis, disease head | 60.6 | −2.8 | 0 | 48.3 | ✗ |
| C7 crop-agnostic text priors | 64.7 | +1.3 | 4 | 55.1 | ✓ |
| C8 composition text prior | 63.8 | +0.4 | 3 | 53.7 | ✗ |
| C9 hallucinated composition head | 67.6 | +4.3 | 5 | 56.2 | ✓ (γ* at grid edge) |
| C10 MLP heads | 64.2 | +0.9 | 3 | 53.3 | ✗ |
| C11 weight decay 5e-3 | 63.5 | +0.2 | 4 | 51.7 | ✗ |

### Stage D interactions (matrix committed before execution)
| Row | H_bal | Control | Δ | folds | AUSUC | Gate |
|---|---:|---|---:|---:|---:|:---:|
| D1 C9 with scale-normalised fusion | 67.3 | B1 | +4.0 | 5 | 56.2 | ✓ |
| **D2 D1 + text priors** | **70.8** | D1 | +3.5 | 5 | **61.8** | ✓ |
| D3 D1 + 50% erasure | 68.7 | D1 | +1.4 | 4 | 57.4 | ✓ |
| D4 D2 + 50% erasure | 71.3 | D2 | +0.5 | 3 | 62.5 | ✗ |
| D5 factorised + text + erasure | 66.7 | C7 | +2.1 | — | 56.9 | (no synthesis) |
| D6 / D7 D2 with synthesis seeds 1, 2 | 71.4 / 71.0 | D1 | +4.1 / +3.7 | 5 / 5 | 62.3 / 62.2 | ✓ stable |
| D8 D2 with 4× synthesis | 71.3 | D2 | +0.5 | 4 | 62.7 | ✗ |
| D9 D2 with h-flip TTA | 70.6 | D2 | −0.2 | 2 | 62.2 | ✗ |
| D10 D1 recipe on ConvNeXt-B | 60.8 | A3 | +3.6 | — | 48.2 | mechanism generalises across backbones |

### Stage E selection and locked confirmation
Selection rule: take the most complex configuration in which every added component passed
the gate against its own ablation. That is **D2** (D1 > B1, D2 > D1; D4, D8 and D9 do not
pass against D2). The recipe file `research_r2/matrices/stage_e_frozen_v1.json`
(SHA-256 `51c4dd07…c883`, commit `f341991`) was frozen before L1–L3 were scored. Locked
folds were scored exactly once, with γ = 8.0 fixed from S1–S5.

| Locked fold | D2 H_bal | B1 H_bal | Zero-shot H_bal | D2 seen | D2 unseen macro | D2 H5 (micro) | D2 healthy shortcut |
|---|---:|---:|---:|---:|---:|---:|---:|
| L1 | 78.8 | 76.5 | 13.0 | 75.7 | 82.2 | 98.4 | 1.4 |
| L2 | 70.9 | 63.1 | 13.9 | 70.1 | 71.8 | 98.1 | 1.2 |
| L3 | 69.6 | 66.2 | 12.6 | 69.6 | 69.6 | 97.1 | 0.9 |
| **Mean** | **73.1** | 68.6 | 13.2 | | | | |

D2 beats the matched control on all three locked folds (+2.3, +7.8, +3.4 pp); AUSUC is
69.0 vs 61.4. The legacy diagnostic D0 (65 held-out pairs, 12 healthy, previously examined)
gives H_bal 47.1 vs 33.3. It is not comparable with Round 1 numbers, because D0 is
re-materialised under Round 2 lineage and holdout rules.

### Error analysis (D2, all 8 new folds, 87 held-out compositions)
- Unseen outcomes: 62.7% correct. Errors are 19.0% right crop but a fitting pair (the crop-dependent-disease signature), 10.8% another held-out pair of the same crop, and 7.5% wrong crop. B1 gives 59.0 / 18.2 / 11.8 / 11.0.
- D2 beats B1 on 43 compositions, loses on 22, ties on 22.
- Hardest diseases: spider mites (0%), alternaria leaf spot (5%), target spot (17%), frog-eye leaf spot (22%; B1 63%, a real regression). Per-disease accuracy hardly depends on how many crops a disease is seen on (Spearman 0.13).

## 6. Interpretation and honesty notes
1. **Calibration is the largest single effect, and it is a decision-rule change, not a representation change.** Every factorised model is strongly seen-biased at γ=0. Calibrated stacking is standard GZSL practice; here γ is chosen only on training-source folds and is stable across folds. It must be reported as a separate, explicit inference step, with the whole seen/unseen curve (AUSUC) shown and the γ=0 numbers alongside.
2. **Representation gains are real but smaller.** At equal calibration, D2 raises the γ-free AUSUC from 51.5 to 61.8 and the γ=0 within-crop disease transfer from 11.3% to 16.0%. Crop-dependent disease bias remains the dominant error.
3. **The unseen task is partly crop recognition.** In both our folds and the official set (10 unseen classes over 9 crops), once seen pairs are down-weighted the crop nearly identifies the pair. Within-crop accuracy is the stricter disease-transfer diagnostic.
4. **Hypothesis outcomes.** Crop-entangled disease features are *supported*: full erasure hurts, partial erasure helps, and most residual errors are same-crop seen pairs. Source-style shortcut is *not* remedied by erasure, because source identity carries real crop information; the shortcut itself is not ruled out. Composition imbalance is *not supported* as a remedy target.
5. **Transfer to the official test is not established.** Round 1 seen-val (53%) overstated the official seen accuracy (44.8%) for WP1. Distribution shift between DPD source datasets could move the best γ. Any official run of D2 must be labelled **non-blind**, because the official sets were examined in Round 1.

## 7. Compute
Feature extraction for 8,000 images on the RTX 4060 Laptop (8 GB): CLIP-L/14 171 s at 1.18 GiB peak; B/32 39 s; ViT-B/16 45 s; ConvNeXt-B 47 s; EfficientNet-B3 34 s. Stage B/C used 537 s of row time, Stage D 511 s and Stage E 122 s. A full 8,000-image refit of D2 needs under 1 minute after extraction. Official inference (3,448 images) would be about 75 s of CLIP-L encoding.

## 8. Next experiments (not run)
1. **Owner-approved full refit of the frozen D2 recipe** on all 8,000 images over Jonathan's shared 333 manifest, with γ = 8.0. Emit a label-free packet for his importer and report it as a separately labelled, non-blind fifth model. Report γ=0 and γ=8 side by side.
2. **Partial fine-tuning pilot** (last 2–4 CLIP-L blocks or LoRA) with the D2 heads and synthesis loss, on S1–S5 only. Estimated at ≈3–4 min per epoch per fold. Coordinate first: M12 showed LoRA can trade unseen accuracy for seen accuracy.
3. **Authorised downloads only:** DINOv2 / SigLIP / BioCLIP frozen features. The backbone axis was the largest representation effect among the families tested.
4. **Targeted remedies** for spider-mite, alternaria and frog-eye failures, e.g. disease-description prompts with recorded provenance, following the DPD paper's taxonomy text.
