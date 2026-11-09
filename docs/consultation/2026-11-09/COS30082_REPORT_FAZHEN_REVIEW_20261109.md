<!-- PAGE 1 -->
# 1. Methodology

Recognising Unseen Crop-Disease Compositions

COS30082 Applied Machine Learning - Assignment 02
Andy Wei Jian CHU; Fa Zhen NG; Yuen Kai CHIA; Frederick; Jonathan Zheng Li CHAI
Audited report draft with frozen official results, 9 November 2026

Official shared-333 evaluation is complete for all four frozen systems; WP1 ran once in its recorded producer-compatible environment. Team recording and signed submission details remain pending. This is an updated submission draft.

## 1.1 Task and motivation

Plant disease recognition must identify both the crop and its disease. A model can recognise individual components yet fail when their combination was absent from supervised fitting. We compare two independent crop/disease classifiers, a shared-backbone multi-task classifier, frozen CLIP and a small trainable CLIP adapter. DPD explicitly motivates evaluation beyond familiar crop-disease combinations [1].

## 1.2 Data and development isolation

The supplied DPD-Small task contains 55 crop labels, 175 disease labels, 323 training-source compositions and 10 additional official unseen compositions. The official partitions contain 8,000 training, 2,000 seen-test and 1,448 unseen-test images. Model development uses only the 8,000 training-source records.

| Training-derived fold | Images | Composition role |
| --- | --- | --- |
| Fitting | 4,868 | 258 fitting compositions |
| Seen validation | 1,238 | 225 compositions represented in fitting |
| Internally unseen validation | 1,894 | 65 compositions absent from fitting |

Seed 42 first holds out 65 complete compositions while retaining every crop and disease component in fitting. A classwise split of the remaining compositions produces fitting and seen validation. The three ordered filename lists are disjoint and cover all 8,000 records. Both validation folds rank the same 323 ascending original global composition IDs, which are non-contiguous; they are never renumbered as label identities.

The 65 internally unseen development compositions are distinct from the 10 officially unseen classes. Neither official-test images nor their performance are used for fitting, prompt design, tuning or checkpoint selection. A separate final protocol is required before accessing those images.

Source code: https://github.com/FZhen2450/COS30082_ML_Project
Reproducible integration checkpoint: bbdabfdc665c711a02b3ca94cc285eb9ed86f97a
Recorded demonstration link: pending team recording and access verification.


---

<!-- PAGE 2 -->
# 1. Methodology - supervised baselines

## 1.3 Independent crop and disease classifiers (WP1)

Two ResNet50 networks [2] are trained independently: one outputs 55 crop logits and the other 175 disease logits. Each uses ImageNet-1K V2 initialization, a replaced linear classifier and its own optimizer updates. Each completed three fitting epochs using AdamW, learning rate 10^(-4), weight decay 10^(-4), batch size 2 and seed 42. Crop epoch 2 and disease epoch 3 were selected separately by seen-validation component Top-1; exact ties favour the earlier epoch. Each network minimises cross-entropy. Fitting augmentation uses horizontal/vertical flips and a random affine transform (rotation up to 18 degrees, translation 5%, scale 0.95-1.05) before the native recipe.

The crop model has 23,620,727 parameters and the disease model 23,866,607. Evaluation uses RGB images, bilinear shorter-edge resizing to 232, antialiasing, a 224 centre crop and ImageNet normalization. The published native composition scorer sums crop and disease log probabilities in float32. Only valid candidate pairs compete. Component accuracy and composition accuracy are separate quantities. The frozen pair's first official run and Jonathan's independent 86-check verification are recorded in Appendix A.4.

## 1.4 Shared-backbone multi-task model (WP2)

WP2 uses a single ImageNet-1K V1 ConvNeXt-Base [3] feature extractor. Its pooled, normalized 1,024-dimensional representation feeds separate 55-output crop and 175-output disease linear heads. Both outputs remain raw logits. Sharing reduces duplicate feature extraction and lets both tasks update the same representation, but it does not guarantee transferable disease features.

L_WP2 = 0.5 CE(crop_logits, crop_truth) + 0.5 CE(disease_logits, disease_truth)

Cross-entropy internally combines log-softmax and the negative log likelihood of the correct class. Applying softmax inside the model is unnecessary. The total loss sends gradients through both heads into the common backbone. Each head retains independent parameters. Epoch losses are weighted by the actual number of processed samples, including partial batches.

EfficientNet-B3 [4] and ConvNeXt-Base completed 23 and 97 epochs respectively, using head warm-up followed by progressive fine-tuning. Both selected warm-up epoch 1. Their completed compute budgets differ; the comparison does not isolate architecture alone. ConvNeXt-Base is the stronger frozen supervised reference. ResNet18, EfficientNet-B0 and ConvNeXt-Tiny remain engineering references, rather than additional completed full-training comparisons.

| Selected WP2 preprocessing | Recorded value |
| --- | --- |
| Authoritative epoch / resolution | Epoch 1, 224 x 224 |
| Validation transform | Square bilinear RGB resize, antialiasing |
| Normalization | ImageNet mean / standard deviation |
| Fitting augmentation | Horizontal/vertical flips; small affine transforms |
| Historical planned endpoint | 320 x 320; not the selected checkpoint resolution |

The original runs use AdamW, seed 42 and equally weighted crop/disease losses. Exact run-specific stage settings remain in the immutable run records. Appendix A lists the configuration/provenance handoff. Legacy metadata is preserved; reconstruction uses the authoritative epoch-specific resolution. A later matched 224/320 retest did not establish reliable unseen-disease improvement.


---

<!-- PAGE 3 -->
# 1. Methodology - proposed approach

## 1.5 Frozen CLIP reference

CLIP aligns image and text representations using large-scale pretrained image-text supervision [5]. Our frozen reference uses official ViT-L/14 weights and native CLIP preprocessing: bicubic shorter-edge resize to 224, a 224 centre crop, RGB conversion and CLIP normalization. Tokenization uses the pinned official implementation. CLIP pretraining differs substantially from the supervised baselines, which limits causal interpretation of the four-system comparison.

One deterministic prompt describes each candidate pair (323 in development, 333 in the official task). Healthy pairs use "a photo of a healthy {crop} plant". Diseased pairs use "a photo of {a/an} {crop} plant affected by {disease}". Labels undergo NFKC normalization, underscore replacement, whitespace collapse and lowercasing; no symptoms or performance-driven aliases are added. Policy label_only_single_template_v2 fixes the article from the normalized crop label before full baseline evaluation.

## 1.6 Identity-initialized bounded visual adapter (M6)

Motivated by residual adaptation of CLIP [6], our proposed approach learns only a 768-to-64-to-768 ReLU visual bottleneck with 99,136 trainable parameters. The CLIP image encoder, text encoder and text bank stay frozen. Zero initialization of the upper projection makes epoch 0 reproduce frozen CLIP. The full pretrained encoder is still required at inference; a small trainable adapter does not make the whole deployed model small.

u = z / ||z||; r = W_up ReLU(W_down u + b_down) + b_up
delta = 0.1 r; y = z + ||z|| delta min(1, 0.20 / ||delta||)

The zero-residual case uses a factor of one. The finite residual update satisfies ||y - z|| / ||z|| at most 0.20, subject to floating-point arithmetic. Explicit guards reject non-finite residuals, norms and outputs. The cap limits representation movement; it is not an accuracy guarantee. Unit-normalized adapted features retain direct-composition text scoring at inference.

## 1.7 Fixed component-aware fitting objective

L_fit = 0.5 CE_crop + 0.5 CE_disease + L_identity + L_same-crop
L_identity = mean ||normalize(y) - normalize(z)|| squared

Crop/disease text prototypes are normalized means of direct text embeddings from the 258 fitting compositions, equally weighted by composition. At temperature 0.07, component cross-entropy encourages the correct crop and disease. A same-crop conditional term distinguishes the true fitting composition from other fitting compositions with the same crop; a crop with only one eligible composition contributes zero conditional loss. All fitting samples remain in its mean denominator.

The matched cap/no-cap ablation uses one fitting epoch per seed 45, 46 and 47, batch 8, AdamW learning rate 10^(-4) and weight decay 10^(-4). Each pair shares the verified frozen-feature cache, initialization and fitting order. Validation uses the unchanged component-plus-identity objective, without the training-only conditional term. The seed-45 representative rule was recorded after the development ablation and before official evaluation, rather than choosing the best-scoring seed.


---

<!-- PAGE 4 -->
# 1. Methodology - evaluation and application

## 1.8 Common task, native scorers

For valid pair (c,d), WP2 computes log_softmax(crop_logits)[c] + log_softmax(disease_logits)[d] in PyTorch float64. WP1 preserves its original float32 scorer. CLIP and M6 rank all direct-composition image/text cosine similarities using their canonical CPU float64 scorer; any shared positive CLIP logit scale leaves the ranking unchanged. Scorers retain their historical precision rather than being numerically substituted to force agreement.

Score columns explicitly identify the same ordered global candidates. Stable descending ranking breaks exact score ties by ascending global ID. The shared WP5 reporting interface accepts complete canonical scores or validated rank records. Rank-only historical inputs preserve producer decisions but cannot independently prove full-score or exact-tie equality.

Top-k = correct composition-in-first-k predictions / processed fold count
H@k = 2 S_k U_k / (S_k + U_k); H@k = 0 when S_k = U_k = 0

Top-1/Top-5 are composition accuracies. Independent head diagnostics and the components of a ranked composition are labelled separately. The exact denominators are 1,238 seen and 1,894 internally unseen images. WP2/M6 checkpoint selection maximizes full-precision development H@1, then minimizes sample-weighted combined validation loss, then chooses the earliest epoch. No rounding tolerance is introduced. M6 includes identity epoch 0; every matched arm selected epoch 1. WP1 uses its distinct component selection rule.

## 1.9 Provenance, reproducibility and uncertainty

Ordered filenames, truths, mappings, prompts, source bytes, model states and run settings accompany the shared records. Historical CRLF source bytes and the approved checkout attributes are preserved; canonical candidate and manifest hashes use different declared namespaces. The four-model table compares frozen selected systems on complete identical folds, not matched training/search budgets. The repeated development folds informed exploratory research, so they are not an untouched estimate of final generalization.

WP5 cluster-bootstrap intervals resample composition groups, with 10,000 replicates, rather than pretending images from one composition are independent. Uneven support gives an unseen effective composition count of about 8.7 despite 65 nominal groups. Intervals on repeatedly inspected development folds remain exploratory.

## 1.10 GUI and fitting-only reference retrieval

The LeafLens desktop GUI accepts a local image, runs a real selected predictor, displays five candidate crop-disease pairs and exports predictions with protocol provenance. It supports the WP1 pair, frozen CLIP and M6; WP2 is evaluated offline and is not currently selectable. Each predictor uses its native preprocessing. Cosine/ranking scores are not calibrated confidence.

References come only from the fitting fold. Exact crop-disease matches are shown where available; otherwise same-disease examples from other crops are clearly labelled. RGB-histogram ordering offers illustrative visual references, not learned semantic retrieval. The query and pixel-identical images are excluded. A plausible gallery can accompany a wrong diagnosis, so reference similarity is not evidence that the model is correct. A separately versioned local follow-up verifies all three model-picker paths; its source and publication status are recorded in Appendix A.5.


---

<!-- PAGE 5 -->
# 2. Results and Discussion

## 2.1 Four-model development comparison

| Frozen system | Seen T1 | Seen T5 | Unseen T1 | Unseen T5 | H@1 | H@5 |
| --- | --- | --- | --- | --- | --- | --- |
| WP1 independent ResNet50 | 53.39 | 77.54 | 0.69 | 8.66 | 1.36 | 15.58 |
| WP2 shared ConvNeXt-Base | 49.19 | 79.97 | 2.32 | 9.45 | 4.44 | 16.90 |
| Frozen CLIP ViT-L/14 | 7.03 | 22.46 | 9.50 | 31.15 | 8.08 | 26.10 |
| M6 capped adapter, seed 45 | 18.34 | 43.05 | 19.48 | 54.80 | 18.89 | 48.22 |

Table 1. All values are percentages on the same 323-candidate development task: 1,238 seen and 1,894 internally unseen records per system. WP2 uses its identified export, not the older anonymous aggregate.

![Figure](../figures/development_comparison.png)
Figure 1. Seen accuracy alone hides failure on held-out crop-disease pairs. H@1 summarizes their balance.

The two supervised systems recognise seen pairs substantially better than CLIP-based systems, yet their internally unseen Top-1 falls to 0.69% and 2.32%. This pattern is consistent with weak compositional disease transfer. In contrast, M6 raises unseen Top-1 from frozen CLIP's 180/1,894 to 369/1,894 and seen Top-1 from 87/1,238 to 227/1,238. Its H@1 is 18.89%, the highest among these four frozen finalists, compared with 8.08% for frozen CLIP.

The M6-minus-frozen H@1 difference is 10.81 percentage points, with an exploratory composition-cluster 95% interval of [4.36, 18.39]. This supports an improvement on the measured development task, while the selection history and heterogeneous pretraining prevent a general causal claim about architecture or unseen disease recognition.

## 2.2 Historical WP2 discrepancy retained

The original selected ConvNeXt checkpoint reported 610 seen Top-1 hits (49.27%); its new filename-identified export reports 609 (49.19%). The checkpoint and selected epoch are unchanged. Unseen Top-1 and both Top-5 counts agree. Different batch/BF16/backend execution is a possible explanation, not a proven cause; old aggregates cannot identify which decisions changed. We retain both records without retuning to restore a preferred number. The identified result is used consistently in this report.

The WP2 disease-balanced sampler reached H@1 3.11%, below the original supervised reference, and is retained as a negative primary-metric result. Lower fitting loss or better Top-5 alone does not establish success on the predeclared H@1 goal.


---

<!-- PAGE 6 -->
# 2. Results and Discussion - ablations and errors

## 2.3 Matched test of the residual cap

| Seed | Unbounded H@1 | Cap-0.20 H@1 | Disease hits: control / cap |
| --- | --- | --- | --- |
| 45 | 15.27% | 18.89% | 91 / 100 |
| 46 | 15.20% | 18.86% | 91 / 105 |
| 47 | 15.45% | 19.19% | 83 / 95 |
| Mean | 15.31% | 18.98% | Per seed: 1,496 diseased images |

Table 2. One fitting epoch per arm. Within each seed the objective, order, cache, initialization, optimizer and evaluation are fixed; only the cap differs. Repeated seeds evaluate the same images.

The cap improves H@1 in all three matched pairs by 3.62-3.74 percentage points. Its effect is not uniform: unseen Top-5 decreases in all three pairs. Across repeated predictions, 319 of the 354 additional exact unseen Top-1 hits are healthy and 35 are diseased. These are repeated decisions on the same validation images, not 354 newly sampled images.

## 2.4 Healthy versus diseased transfer

| Unseen subgroup | Images | Frozen exact hits | M6 exact hits |
| --- | --- | --- | --- |
| Healthy compositions | 398 | 101 (25.38%) | 269 (67.59%) |
| Diseased compositions | 1,496 | 79 (5.28%) | 100 (6.68%) |
| All internally unseen | 1,894 | 180 (9.50%) | 369 (19.48%) |

Healthy cases account for 168 of M6's 189 net additional unseen hits over frozen CLIP. On diseased images the improvement is 1.40 percentage points with an exploratory cluster interval of [-2.5, 4.7]. This does not establish improved disease transfer. M6 predicts a healthy pair for 463 diseased images, versus 58 for frozen CLIP. Its global H@1 improvement must be reported together with this failure mode.

The unseen distribution is highly uneven: cassava mosaic alone contributes 544 images. M6 obtains 13 exact hits there, compared with two for frozen CLIP, while many low-support compositions remain difficult. Across the four finalists, 24 of 65 held-out compositions (139 images) have no correct Top-1 prediction. Per-pair support and background/source biases warrant human review; filename-prefix correlations are hypotheses, not confirmed acquisition-source labels.

## 2.5 Follow-ups that did not meet adoption rules

Fitting-only healthy-score correction reduced false-healthy errors but lowered H@1 (18.89% to 18.82%). Disease-prototype scoring raised H@1 to about 20.46% while reducing unseen Top-1 to 16.63% and disease hits to 77, so it was not adopted. A late-block visual LoRA probe similarly raised H@1 to about 20.31% but reduced unseen Top-1 to 17.48%; it failed the declared adoption rule. These exploratory results do not replace the frozen representative.

The bounded adapter is a constrained supervised adaptation of pretrained CLIP, not unsupervised learning. Only fitting labels drive gradient updates; unseen composition semantics supply candidate text. Better fitting loss or a larger backbone alone did not solve cross-crop disease transfer.


---

<!-- PAGE 7 -->
# 2. Results and Discussion - robustness and integration

## 2.6 Bounded native-image robustness

A fixed support-bounded subset selects up to four lexically ordered filenames per composition, then preserves fold order: 533 seen and 218 internally unseen images. The unchanged WP5 severity-1 suite applies darkening (0.7), brightening (1.3), Gaussian blur (sigma 0.004 times the shorter edge), JPEG quality 40 and Gaussian noise (sigma 0.02 times 255; seed 20261107). Both models receive identical corrupted pixels before their unchanged native preprocessing.

![Figure](../figures/robustness.png)
Figure 2. Composition H@1 on the same fixed 751-image subset. Values are subset diagnostics, not the full-fold development results.

M6 has higher H@1 than frozen CLIP under every measured condition. Its clean subset H@1 is 16.14%, falling to 14.22% under blur, the largest observed drop (1.92 percentage points). These measurements cover 9,012 model-image predictions, including 7,510 corrupted predictions. They are a bounded sensitivity check; they do not establish field robustness, statistical significance or an optimum severity. Different subset class weights and batch boundaries require comparison with the subset's own clean baseline.

## 2.7 Production-predictor clean parity

The actual GUI CLIP/M6 predictor reproduced every ordered Top-5 row in the historical development records: 2 models times (1,238 + 1,894) = 6,264 model-image rows, with zero mismatches. The check used the historical untransposed RGB path, CUDA native FP16 image encoding with batches of 8, text batches of 32, FP32 adapter arithmetic and CPU FP64 scoring. Text-bank hashes also matched.

Clean parity took 184.20 seconds; peak CUDA allocated memory was 1,970,383,360 bytes (about 1.84 GiB). This is a specific validation workload rather than a general latency promise. Historical rank-only packets cannot prove complete-score equality. The ordinary GUI corrects EXIF orientation and may use CPU/single-image execution, so that distinct path is not claimed to be numerically identical.

## 2.8 Integrity and software verification

The merged integration checkpoint passed 621 automated CPU tests in a clean checkout and again from an extracted source archive. All nine published notebooks have cleared outputs/counts and expensive controls disabled by default. The full fitting-source audit decoded 8,000 images and found 8,000 distinct raw-byte hashes, with no exact duplicate groups. A deterministic 200-image perceptual-hash screen found no candidates; this small heuristic screen cannot rule out near duplicates elsewhere.

These are engineering checks, separate from predictive accuracy. Human augmentation/symptom-preservation and bias acceptance remain necessary. No final-test claim follows from passing tests, a convincing GUI example or a clean byte-level audit.


---

<!-- PAGE 8 -->
# 2. Results and Discussion - official evaluation

## 2.9 Frozen official test results

| Frozen system | Seen T1 | Seen T5 | Unseen T1 | Unseen T5 | H@1 | H@5 |
| --- | --- | --- | --- | --- | --- | --- |
| WP1 independent ResNet50 | 32.35 | 54.25 | 8.29 | 30.46 | 13.19 | 39.01 |
| WP2 ConvNeXt-Base | 29.70 | 54.40 | 0.83 | 15.33 | 1.61 | 23.92 |
| Frozen CLIP ViT-L/14 | 8.25 | 28.25 | 5.52 | 24.10 | 6.62 | 26.01 |
| M6 capped adapter, seed45 | 18.80 | 48.15 | 3.45 | 19.48 | 5.83 | 27.73 |

Table 3. Composition percentages; 2,000 seen / 1,448 unseen images; all four rank the same 333 candidates. WP1 completed a separate frozen run (20261109T004313Z_0df3eb32dcaf) in Torchvision 0.22.1+cu118. Values are display-rounded only.

All project-supervised fitting used 4,868 records. Of the 2,000 course-seen test images, 486 (24.3%) belong to the 65 compositions absent from that fitting fold. Top-1 on fitting-supported / absent pairs is 41.9% / 2.7% for WP1, 38.7% / 1.6% for WP2, 8.0% / 9.1% for frozen CLIP and 20.1% / 14.8% for M6. Frozen CLIP received no project fitting updates; its pretraining exposure is unknown.

Shared 333 is the owner-authorized team interpretation. The lecturer's recorded answer did not explicitly settle shared versus separate lists. We retain this qualification and the first attempt.

## 2.10 Development gains did not transfer to official unseen disease pairs

Official observed H@1 ranks WP1 (13.19%) > frozen CLIP (6.62%) > M6 (5.83%) > WP2 (1.61%), unlike development (M6 > CLIP > WP2 > WP1). M6 increases seen exact hits over CLIP from 165 to 376 but reduces unseen hits from 80 to 50. It gains nine and loses 39 unseen decisions. Only ten of these losses are healthy predictions; 30 occur in cotton bacterial blight and grape black rot, and the truth remains in M6's Top-5 for 24 of the 39.

Official unseen is entirely diseased, whereas internally unseen development included 398 healthy images. M6 predicts healthy pairs for 152 official unseen images (10.50%), frozen CLIP for eight (0.55%) and WP2 for 518 (35.77%). M6's predicted crop is correct more often than frozen CLIP's (857 versus 641), but its disease less often (82 versus 112). These are components of the ranked composition, not independent heads. The findings support a disease-transfer limitation without establishing a sole cause. WP1 scores 120 official unseen hits, of which 114 occur in three compositions; WP1 and CLIP have no overlapping correct unseen images. WP1 disease Top-1 is 12.1% on development unseen versus 11.7% officially, while its correct-crop share conditional on a correct disease differs. These are descriptive, not causal, findings.

## 2.11 Verification, uncertainty and conclusion

Saved-score producer replays and independent NumPy-only WP5 audits (212 three-model checks plus 86 WP1 checks) reproduce all 13,792 model-image Top-5 predictions and reports. Appendices B-C show the selected M6 seen-test matrix and all ten unseen-pair outcomes; complete labelled matrices accompany the handoff. A post-run image-byte check matches all 3,448 official files to the released manifest; it is not an atomic per-decode proof.

Only ten unseen compositions are available (Kish effective count 5.4). M6 is worse in five pairs and better in two, with three ties. The exploratory composition-cluster 95% interval for the M6-minus-frozen H@1 difference is [-1.90, +1.76] percentage points (estimate -0.78; 10,000 resamples, seed 20261107). Small-cluster percentile intervals can under-cover. We conclude that M6 did not improve observed official H@1; reliable inferiority is not established. Exploratory composition-cluster intervals for WP1 minus CLIP [+6.58 pp, 95% interval -4.12 to +15.01] and WP1 minus M6 [+7.36 pp, -4.42 to +16.10] cross zero; the WP1 minus WP2 interval [+11.58 pp, +0.64 to +19.75] does not. These small-cluster intervals are tentative, not proof of population-level superiority.

M6 improves development H@1, official seen accuracy and H@5, but not observed official H@1. Original weights, prompts, scoring and selection decisions remain unchanged. The team video and signed forms remain pending.


---

<!-- PAGE 9 -->
# References

[1] A. Y. H. Chai et al. (2025). Deep-Plant-Disease Dataset Is All You Need for Plant Disease Identification. ACM Multimedia, 7 pages. https://doi.org/10.1145/3746027.3758192

[2] K. He, X. Zhang, S. Ren and J. Sun (2016). Deep Residual Learning for Image Recognition. CVPR. https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html

[3] Z. Liu et al. (2022). A ConvNet for the 2020s. CVPR. https://openaccess.thecvf.com/content/CVPR2022/html/Liu_A_ConvNet_for_the_2020s_CVPR_2022_paper.html

[4] M. Tan and Q. V. Le (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. ICML, PMLR 97. https://proceedings.mlr.press/v97/tan19a.html

[5] A. Radford et al. (2021). Learning Transferable Visual Models From Natural Language Supervision. ICML, PMLR 139, 8748-8763. https://proceedings.mlr.press/v139/radford21a.html

[6] P. Gao et al. (2021; revised 2025). CLIP-Adapter: Better Vision-Language Models with Feature Adapters. Preprint and International Journal of Computer Vision. https://arxiv.org/abs/2110.04544

## Project evidence

Source checkpoint: https://github.com/FZhen2450/COS30082_ML_Project/tree/bbdabfdc665c711a02b3ca94cc285eb9ed86f97a

Integration record: docs/Integration_Checkpoint_2026-11-08.md at the source checkpoint. Four-model filename-level import: workspace/handoffs/FOUR_MODEL_DEVELOPMENT_20261107. Matched ablation source: notebooks/wp3b_one_epoch_replication.ipynb. GUI parity/robustness: workspace/experiments/gui-native-parity/20261108T052037Z_bf07ee0888d0.

This consolidated draft incorporates Jonathan's WP5 report-section v3 and the WP1 producer provenance, then updates their status with the 8 November integration evidence. Historical drafts and raw experiment records remain unchanged. References and this appendix are outside the eight main pages under the student-recorded lecturer guidance of 3 November; verify subsequent Canvas instructions before submission.


---

<!-- PAGE 10 -->
# Appendix A. Reproduction and provenance

## A.1 Fixed development identities

Raw training CSV SHA-256 (historical CRLF checkout):
9f1b729ab08f120c9e8d3a5681aa1ccbdf1d9210f9aaa6143959363dcb106a7d
Historical candidate-ID/pair payload SHA-256:
4cb0a24a96cf647f32311817fa7b082a46d3cf00c20ae61f4287e83cc898dfaf
Historical ordered split SHA-256:
f0c7ba212449fd68f1d19c0f84cc91bf4cc1af29ae405128e5198bce89f25f85
Prompt-policy v2 SHA-256:
753672c7964bcb20a1346145f21a72b40a19f55590a006b5783823f23c13f537

## A.2 Model and execution identities

Official OpenAI CLIP revision: d05afc436d78f1c48dc0dbf8e5980a9d471f35f6. ViT-L/14 weight SHA-256:
b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836
M6 seed 45 selected epoch 1 adapter SHA-256:
8835a18b9eb933ae224bd17866f18bc56406e24d8612743f4e7c88848a8135cc
Selected WP2 epoch 1 SHA-256:
7a0460b7d03bc08d19b68ca06cdd266746953322149a3edad177cf29992d7c04
WP1 crop epoch 2 SHA-256:
f0c82ca78153f492078a3a39a11eac1a847cc85eb496cf3b3d4d74db752c7d14
WP1 disease epoch 3 SHA-256:
f497e19da62b68dfbcfbae58b9a5318a26b1989ba59b7a486223cad0d56e31a5

Katana: Python 3.10.20, PyTorch 2.5.1+cu121, Torchvision 0.20.1+cu121, RTX 4060 Laptop 8 GB, Ubuntu WSL. WP1 producer: Python 3.13.13, PyTorch 2.7.1+cu118, Torchvision 0.22.1+cu118, RTX 3050 Laptop. Different environments are disclosed; do not overwrite the Katana Conda environment with repository reference pins.

## A.3 Source, artifacts and configuration

The linked source and private artifact receipts retain executable notebooks, reference modules, tests, manifests and verified asset locations. Raw images and model weights remain outside Git.

WP2 run_metadata.json records AdamW (weight decay 0.01), two planned head warm-up epochs (LR 0.001, batch 64), then backbone/head LR 0.00001/0.0001, effective batch 64, five progressive epochs 224 to 320, BF16/channels-last and clipping 1.0. It completed 97 epochs, selecting epoch 1. Named export: evaluation 224, batch 16. Unpreserved historical executable bytes are not reconstructed.

## A.4 Official final-evaluation provenance

First official attempt: 20261108T113128Z_294ff0a10d7e; 154.47 seconds including reporting, peak allocated GPU 1,026,800,640 bytes. Final candidate digest ba17e94119e8b2b7a4b18603e25405d2f52bf4a93396973f80491800697eb62f; final prompt digest c9dd871d61d4141b33583ddd71961c309fd679a8edab088cb28ba20eccef88f2. Semantic identity manifest: 36cdf1ecdbec0522f3ce83250dc21eacc499f8f61db25955eed7a3d578e77606. Original records are preserved. WP1 ran separately as 20261109T004313Z_0df3eb32dcaf, preserving crop epoch 2, disease epoch 3 and the native float32 scorer. Jonathan independently verified the saved outputs with 86/86 checks and no numerical discrepancies, completing the four-model audit alongside the earlier 212/212 checks. This saved-score verification does not claim an independent second-member checkpoint-byte verification or a new model-inference run. Protocol clarification, team recording and student declarations remain pending.
Frozen CLIP/M6 share one newly encoded 333-prompt bank, float64 tensor SHA-256:
5bbb5f715620c65fa76b2f32c84b72dddf0d9ceae8d43c7e9720a2240ff999b6
Development-bank byte equality is not claimed. Candidate/prompt digests identify the source reconstruction files; run copies are row-identical re-serializations. WP1 owner pair fingerprint cbdddec2…af2518 hashes its two head-file digests, while WP5 fingerprint dfde18f0…ba27c8 additionally encodes head ids and selected epochs (crop e2, disease e3); both identify the same frozen files but are not interchangeable.


## A.5 Published WP1 and WP4 follow-up on 9 November

The follow-up source is published on `fix/wp1-final-handoff-20261109` at
[7314644](https://github.com/FZhen2450/COS30082_ML_Project/commit/73146444ace8637426042e3bfb76aa9bb3b8585c), including the separate WP4 source
commit [5c9d1f9](https://github.com/FZhen2450/COS30082_ML_Project/commit/5c9d1f9b05f7de9f9c86082ecd6bf1f676048d21) on `fix/wp4-completion-20261109`.
Both remote refs were read back from GitHub and matched the local commits.
These branches are not merged into the sealed main checkpoint.

The original WP4 source snapshot passed 79/79 applicable isolated regressions.
The combined WP1/WP4 publication suite subsequently passed 95/95 tests without
skips. These are scoped engineering checks, distinct from main's historical
621-test receipt. The follow-up hardens global-ID/score validation, WP1 GUI
export provenance and Tk worker/dialog teardown. Actual normal-picker rehearsals
for WP1, frozen CLIP and M6 verified native image inference, fitting-only
references, labelled different-crop fallback, normal/minimum-size layouts and
JSON export on the predeclared seen-development peach/curl image.

The GUI remains development-323 with EXIF-corrected RGB and single-image input.
The known-failure rehearsal does not establish accuracy or new historical-path
parity. No new model inference, training, scorer change or official retuning was
performed for publication. Frederick's independent operation, peer review,
actual team recording and declarations remain genuine human completion gates.


---

<!-- PAGE 11 -->
# Appendix B. Official seen-test confusion matrix

Selected M6 seed45/epoch 1. This overview includes all 323 seen truth rows and 333 allowed prediction columns. Every one of the 2,000 official seen images contributes exactly one count. Colour uses log(1+raw count) to show sparse errors; this is separate from the supplied row-normalized table.

![Figure](../figures/official_seen_matrix.png)
Figure B1. Complete selected-M6 seen-test matrix overview. Axis labels identify original global IDs; the full seen_confusion.csv contains every truth/prediction ID, crop-disease label, raw count, truth support and normalized value. Equivalent matrices for WP1, WP2 and frozen CLIP are included in the results handoff.

The overview is not intended for reading every individual cell at report size. Use the labelled CSV for exact lookup, including zero-support rows. No unseen truth image is mixed into this seen-test matrix. The same 333 prediction universe is retained for the separate unseen report.


---

<!-- PAGE 12 -->
# Appendix C. Official unseen composition outcomes

All 1,448 official unseen images belong to these 10 diseased compositions. Entries below are exact Top-1 composition hits, not disease-head accuracy. Every row competed against all 333 candidates; zero hits are retained. Global IDs and label spellings remain unchanged.

| Global ID | Composition | Images | WP1 hits | WP2 hits | Frozen hits | M6 hits |
| --- | --- | --- | --- | --- | --- | --- |
| 39 | pepper_bell_bacterial_spot | 220 | 49 | 1 | 14 | 8 |
| 84 | cherry_aphis_spp | 72 | 0 | 0 | 0 | 0 |
| 107 | corn_brown_spot | 35 | 0 | 0 | 1 | 0 |
| 124 | cotton_bacterial_blight | 125 | 0 | 0 | 40 | 25 |
| 133 | cucumber_downy_mildew | 32 | 0 | 0 | 0 | 0 |
| 136 | cucumber_powdery_mildew | 37 | 1 | 2 | 4 | 3 |
| 144 | grape_black_rot | 271 | 46 | 0 | 18 | 8 |
| 166 | mango_anthracnose | 95 | 0 | 0 | 0 | 0 |
| 276 | sugarcane_rust | 89 | 19 | 6 | 0 | 2 |
| 303 | tomato_early_blight | 472 | 5 | 3 | 3 | 4 |

Total exact unseen hits: WP1=120, WP2=12, frozen CLIP=80, M6=50. The full report also includes Top-5, crop/disease support and healthy/diseased prediction diagnostics. These post-evaluation breakdowns describe errors; they do not select prompts, weights, architectures or a new checkpoint. Half of frozen CLIP's and M6's unseen hits come from cotton bacterial blight.

M6 has a narrower improvement pattern than its development headline suggested. On the official task the aggregate unseen decrease persists despite its higher seen accuracy. The small number of unseen classes and uneven support limit population-wide inference; no statistical significance or universal disease-transfer claim is made.


---
