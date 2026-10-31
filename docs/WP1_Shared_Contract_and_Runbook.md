# WP1 shared contract and execution guide

Prepared 31 October 2026 (Malaysia time), from upstream main
`38afc2c071876e768f961b50577702a0311578ed`.

Applied to the real repository on 31 October 2026 after updating main to
`cead983`. See [integration status](WP1_Integration_2026-10-31.md) for current
checks and remaining inputs. The original handoff and its historical evidence
are retained in [the archive](wp1-handoff-2026-10-31/README_FIRST.original.md).

## Delivery boundaries

This implementation provides three scoped CRLF rules, read-only source-byte
verification, a training-only three-fold protocol exporter/validator, separate
supervised classifiers, four screening configurations, checkpoint reconstruction,
a PIL-image inference adapter and a development evaluator using WP5 metrics.
There is no DPD training result, GPU benchmark or official-test result in this
delivery. The subsequent real-image audit decoded all 11,448 images successfully;
see the integration status for the record. Synthetic tests do not substitute for
real experiments.
The dated 24 October checklist is historical; use this guide for current code
availability and the verification report for actual test outcomes.

Issues #2/#3 describe accepted experiments on a separate unpublished WP2/WP3
worktree. Their source artifacts were unavailable during preparation. Preserve
those artifacts and their original hashes. Do not replace them with this export.

## CRLF convention and fresh checkout

Only `/DPD_Small_train.csv`, `/DPD_Small_test_seen.csv` and
`/DPD_Small_test_unseen.csv` use `text eol=crlf`. Git blobs remain the original LF
objects; working files must be the exact historical CRLF bytes. No universal
CSV/text rule or Git configuration change is introduced.

After the proposed patch has been committed in a review branch, use a fresh Git
clone, rather than a GitHub ZIP archive, and run:

```powershell
python scripts/verify_csv_checkout.py
python scripts/check_fresh_checkout.py
```

The second command initializes three fresh temporary repositories, copies Git
objects and performs `read-tree`/`checkout-index`, with `core.autocrlf=false`,
`true` and `input`. It checks immutable blob IDs, working hashes, rows, scoped
attributes, historical manifest source hashes and clean working trees. Use
`--method clone` to exercise local clone transport as well. The current Windows
sandbox blocks Git's shell transport; object-copy fresh checkout was verified.
This verifies fresh Git checkout bytes, not remote transport or publication.

The training LF SHA-256 is
`31d19a852f74b6f1095ea13121243a95074d50576d3d5fc5b48baa0ff6eb0994`.
Its historical CRLF SHA-256 is
`9f1b729ab08f120c9e8d3a5681aa1ccbdf1d9210f9aaa6143959363dcb106a7d`.
The checked-in `stratified_seed42.json` and `heldout_composition_seed42.json`
remain unchanged. No past experiment provenance is rewritten.

## Training-only development protocol

```powershell
python shared_protocol.py
python shared_protocol.py --verify
```

The exporter reads only `DPD_Small_train.csv`. It first holds out whole
compositions using the existing `pytorch-v1` policy, then stratifies the remaining
records, both with seed 42. The resulting counts are 4,868 fitting, 1,238 seen
validation and 1,894 internally unseen validation; there are 258 fitting, 225
represented seen-validation and 65 unseen-validation compositions. Fitting
retains 55 crop and 175 disease components. Every fold ranks the same 323 global
training-source candidates in ascending global ID order.

**Reconciliation is still required.** On the computer containing accepted WP2
and WP3 artifacts, export these exact lists, in their original order, as JSON:

```json
{
  "fitting_images": ["original_fitting_filename_1", "..."],
  "seen_validation_images": ["original_seen_filename_1", "..."],
  "unseen_validation_images": ["original_unseen_filename_1", "..."]
}
```

Then run:

```powershell
python shared_protocol.py --reference accepted_wp2_wp3_filenames.json --output artifacts/data/development_seed42_reconciled.json
```

An ordering or membership difference raises an error. Investigate it using the
accepted artifacts; do not reorder historical records to make a test pass.
Different existing exports are never overwritten. Independent DPD training and
the evaluator refuse an unreconciled protocol.

This export intentionally uses new, explicitly named hash serializations:
canonical JSON with sorted keys, compact separators, ASCII and no trailing LF.
Candidate records include labels and component IDs. Its candidate hash is NOT
Issue #2/#3's `4cb0a24a...` hash; its split hash is NOT Issue #3's `f0c7ba21...`
hash. Those original byte formats cannot be inferred from a digest alone.
Matching filename lists establish the tested partition, not equality of these
different digest formats. Keep both hash namespaces in integration records.

## Model and inference interface

| Item | Contract |
| --- | --- |
| Independent models | Separate crop 55-output and disease 175-output networks, optimizers and checkpoints |
| Backbone choices | ResNet50, EfficientNet-B3, ConvNeXt-Base, ViT-B/16 |
| Supervised preprocessing | Corresponding torchvision DEFAULT weight resize/crop/normalization recipe; RGB; training-only augmentation |
| CLIP preprocessing | WP3's own accepted native CLIP transform/tokenizer; do not use this supervised recipe |
| Targets | int64 crop/disease/global composition IDs |
| Candidate columns | 323 ascending global IDs for development, paired with explicit crop/disease IDs |
| Independent score | `log_softmax(crop)[crop_id] + log_softmax(disease)[disease_id]` |
| CLIP score | Direct candidate-text similarity, retaining WP3's accepted implementation |
| Exact ties | Ascending global ID, obtained by stable sorting ascending-ID columns |
| Metric scale | Fractions in [0,1]; convert to percentages only for presentation |
| Prediction output | `candidate_ids[K]`, `scores[B,K]`, `topk_composition_ids[B,k]`, score semantics and candidate hash |

`IndependentPredictor.predict()` accepts a list of PIL images and preprocesses
each checkpoint separately. It does not expose probabilities as calibrated
confidence. Training-image retrieval should resolve returned global IDs through
the candidate table and search only permitted fitting/training images. Retrieval
and GUI are still WP4 implementations; this adapter does not complete them.

The current adapter covers development candidates only. Do not silently use it
for final official unseen evaluation: the final candidate universe (including
permitted unseen-label metadata) needs a separate frozen protocol and acceptance.

## Real images and augmentation review

On your dataset machine, extract Canvas images into `DPD_Small_dataset/` in the
project root. Keep images and audit previews out of Git. Use a new output folder:

```powershell
python data_utils.py --check-images --output-dir runs/wp1-image-audit
python eda.py
python scripts/preview_augmentations.py --protocol artifacts/data/development_seed42_reconciled.json --output runs/wp1-augmentation-review
```

The full audit decodes every listed image; failures must be resolved before
acceptance. Full EDA regenerates figures only after successful decoding. The
preview samples fitting images only, produces original/augmented leaf views and
leaves `review.json` pending for Andy/Fa Zhen to review symptoms and plausibility.
These checks do not detect differently named pixel duplicates.

## Independent screening and training

Install `requirements.txt` in Python 3.11-3.13; use a matching CUDA build on the
GPU machine. The four `configs/wp1/*.json` files use the same seed, three epochs,
batch size 8, AdamW learning rate 0.0001 and weight decay 0.0001. Native model
resolutions differ. Equal epochs are not equal compute; record duration/memory,
and predeclare any common reduced batch size if GPU capacity requires it.
These are screening budgets, not tuned final configurations.

Run each task separately, using new output directories:

```powershell
python independent_baseline.py --config configs/wp1/resnet50.json --task crop --protocol artifacts/data/development_seed42_reconciled.json --output runs/wp1/resnet50/crop --device cuda
python independent_baseline.py --config configs/wp1/resnet50.json --task disease --protocol artifacts/data/development_seed42_reconciled.json --output runs/wp1/resnet50/disease --device cuda
```

Repeat for the other three config files. `pretrained=true` may download official
torchvision weights on first use. Keep the cache and record installed versions.
Each run records configuration, code commit/dirty state, source hashes, dependency
versions, hardware, native preprocessing, parameter count, per-epoch loss/Top-1/5
and duration, selected epoch, CPU-safe state dict and peak GPU allocated memory.
The component checkpoint is selected by seen-validation component Top-1, with
earliest epoch winning ties. This is a declared baseline policy; architecture
selection still requires pair-level development H@1. There is no automatic final
test evaluation, resume support, scheduler or adaptive early stopping.

## WP5 integration and architecture selection

`evaluate_independent.py` imports `evaluation.metrics` from Jonathan's WP5 PR #1,
now merged into main at `cead983`. The synthetic bridge check passed against the
merged module. This delivery does not modify the WP5 implementation.

Log-sum and probability-product scores are mathematically rank-equivalent, not
guaranteed bitwise equivalent in finite precision. The synthetic bridge check
must not be treated as proof for every trained model. Agree the final numeric
scoring implementation with WP2/WP5 before recording cross-model comparisons.

```powershell
python evaluate_independent.py --crop runs/wp1/resnet50/crop/best.pt --disease runs/wp1/resnet50/disease/best.pt --protocol artifacts/data/development_seed42_reconciled.json --output runs/wp1/resnet50/development.json --device cuda
```

The evaluator reports development seen/unseen Top-1/5, H@1/H@5, per-pair counts
and decode/preprocess/inference time. Setup is excluded from this timing; report
hardware/batch size and distinguish cold from warm behavior. It never evaluates
official test images. Choose 1-2 feasible backbone families using development
H@1 and stated compute limits, then predeclare full-run budgets before training.
Keep all screening runs, including negative results. A final seen confusion
matrix and official seen/unseen evaluation remain for the frozen final protocol.

## What is needed outside this workspace

1. Accepted WP2/WP3 ordered filename lists and original digest serialization on
   the owner-run VS Code/WSL machine: reconcile the export before any comparison.
2. Real Canvas images: run decoding, full EDA and human augmentation acceptance.
3. CUDA GPU and pretrained cache: run the two independent tasks for each family;
   retain logs/weights locally and share textual metrics/provenance for review.
4. Jonathan/WP5: PR #1 is merged; confirm candidate columns, scoring
   equivalence, tie order, metric field names and hash namespaces.
5. Frederick/WP4: connect the PIL prediction adapter to the actual GUI/retrieval
   implementation and test that returned global IDs display the correct labels.
6. Team/lecturer: approve the final official-test candidate universe and unseen
   metadata policy. Nothing in this guide establishes that approval.

The current local review branch is `feature/wp1-shared-data-contract`. Remote
publication, review and merge are separate from the local validation evidence.
