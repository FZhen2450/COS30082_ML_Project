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
