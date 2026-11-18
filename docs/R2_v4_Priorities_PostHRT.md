# R2 v4 after H_RT: which J child to run next (proposal; post-hoc adaptive; nothing executed)

**Status:** this prioritisation uses the H_RT result, so it is **adaptive research, not a blind confirmation**. Any child chosen here is a new exploratory row, and the number of children tried will be disclosed.

**Assignment goals it serves:** better unseen crop–disease recognition, and robustness to real-world image variation.

**Facts used** (`priority_stats.json`, Stage I / v3 results, H_RT):
- **Source structure:** the fitting data have 42 sources.
  - Only about 38 of 291 fitting compositions per fold have more than one source; they hold 41% of fitting images. The dominant source supplies 89% of a composition's images, weighted by images.
  - 29 of 55 crops (90% of images) and 41 of 175 diseases (70%) are multi-source.
- **Seen-val Top-1 by image source, J rows at seed 0:**
  - the composition's dominant source: 75–77% (4,608 image-folds);
  - another fitting source: 61–65% (628 image-folds);
  - a source absent from the composition's fitting images: 25–39% (28 image-folds, too few to read).
- **H_RT:** when a source is wholly absent from training, novel-source accuracy collapses by 45–66 pp for fine-tuned and frozen models alike.
- **Prior attempts on frozen features (all failed their gates):**
  - source balancing within crop and disease (v3 F1, −0.07) and within composition (F2, +0.40);
  - feature-level style-offset augmentation (F3, +0.64, the best F row);
  - disentanglement: F7 crop-centred disease head −0.34; Stage I I4 and I4b factor projections −3.2 and −2.9 (both lowered unseen disease); I5 cross-crop SupCon −15.2.
- **J1–J3 disease transfer:** unseen disease Top-1 rose only 2.5–2.9 pp, to about 61%.

## Hypotheses

| | Hypothesis | What it can and cannot address | Prior evidence | Feasibility (registered row, J2 parent) |
|---|---|---|---|---|
| **J4b** photometric augmentation (ColorJitter 0.3/0.3/0.3/0.05, grayscale 0.1, blur p 0.2) | Invariance to lighting, colour and camera appearance reduces reliance on source-specific appearance | Can act even when a source is wholly absent, if part of the source signature is photometric. Cannot remove background or composition cues. | The only image-level, representation-changing variant of the best frozen-feature idea (F3). Untested end to end. | High: same cost as J2, no new code path beyond the transform (already implemented) |
| **J5** source-balanced loss weights within crop and within disease | Equalising source contributions within a concept reduces dominant-source shortcuts | Acts only where a concept has ≥ 2 sources (90% of images by crop, 70% by disease). **Cannot help when the source is absent**, which is the H_RT failure. Note the registered J5 balances within **crop and disease**, not within composition: within composition, 87% of compositions are single-source. | The frozen-feature equivalent (F1) was null | High: same cost as J2 |
| **J7** crop adversary (GradReverse λ 0.1 on a disease projection Pd(z)) | Removing crop information from the disease pathway improves disease transfer to unseen crops | The registered J7 feeds **z** (the encoder embedding), not Pd(z), to the D1 downstream. The adversary shapes z only indirectly, so the design tests its own hypothesis weakly. | Four frozen-feature disentanglement attempts failed, two of them by lowering unseen disease | Medium: same cost; a stronger test would need a redesigned row (downstream on Pd(z)), which is not registered |

## Recommendation (at most two)

1. **J4b first.**
   - Hypothesis: photometric invariance learned during fine-tuning lowers the source-exclusion drop.
   - Primary development gate: the amendment-3 v2 child-vs-J2 H_bal gate, six folds.
   - Mechanism endpoint, beyond the parent: novel-source seen Top-1 in the H_RT source-excluded arms, child minus J2, on the three registered pairs. Also other-fitting-source seen Top-1, six folds.
   - This is the only candidate whose mechanism could address the failure H_RT exposed.
2. **J5 second.**
   - Hypothesis: within-concept source balancing narrows the 10–17 pp dominant vs other-source gap on seen images.
   - Primary development gate: the same child-vs-J2 gate.
   - Mechanism endpoint: other-fitting-source seen Top-1, six folds (628 image-folds), child minus J2.
   - It cannot address absent sources, and its frozen-feature analogue was null, so expectations are low. It is cheap and directly measurable.
3. **J7 is not recommended now.** The registered design uses z downstream, so the adversary's effect on transfer is indirect, and the four prior disentanglement attempts failed. A properly targeted disease-transfer test would be a new, separately declared row.

**J8** (ConvNeXt stage 4 + frozen CLIP) addresses neither goal directly and stays deferred.

## Resources (Katana, one GPU job at a time; J2 ≈ 9.7 min per fold)

| Step | J4b | J5 |
|---|---|---|
| Screen (seed 0, 6 folds) | ≈ 1 h | ≈ 1 h |
| Confirmation if the screen passes (seeds 1–2) | ≈ 2 h | ≈ 2 h |
| Mechanism endpoint | H_RT source-excluded arms: 3 trainings ≈ 25 min (+ 3 random arms ≈ 25 min if the source effect is to be decomposed) | none extra (six-fold strata) + H_RF ≈ 20 min |
| Optional pipeline estimate (child on J3, 6 folds) | ≈ 2 h | ≈ 2 h |
| **Total** | **1.5–5.5 h** | **1.3–5.3 h** |

**Before any of this:** owner approval to declare amendment 3 (v2), and implementation of child runs and the gate on real data with fail-closed tests (about half a day of CPU work).

**Timing:** First Run's 27 Nov 2026 submission remains the primary obligation, so this should be time-boxed.
