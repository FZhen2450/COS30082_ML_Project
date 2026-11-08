# WP5 reporting implementation and release checks — 7 November 2026

Prepared in `COS30082_WP5_Release`, branch `feature/wp5-final-evaluation`,
base `69c50bef944e06b18dcc13c598164ec04d92d1ea`.
All additions are uncommitted. Existing tracked source, tests, protocols and
experimental results are unchanged. No training, official-test inference,
package installation, staging, commit, push, merge, or GitHub messages occurred.

## Inspection and scope

Read [Issue #8 and both comments](https://github.com/FZhen2450/COS30082_ML_Project/issues/8),
including [the final hash freeze](https://github.com/FZhen2450/COS30082_ML_Project/issues/8#issuecomment-6653018470).
Inspected `evaluation/metrics.py`, `shared_protocol.py`,
`evaluate_independent.py`, `independent_baseline.py`,
`scripts/verify_wp5_bridge.py`, existing tests, and the WP1 runbook.
Read the published WP2/WP3 shared integration tests as remote references.
No files were copied from the WP2 experiment checkout or edited on PR branches.

Existing functionality: stable Top-k over explicit global-ID columns, fractional
accuracy, H@1, WP1's training-only protocol validation, and the independent
evaluator's H@5/per-pair hits. The additive reporting module provides the missing
records, validation, diagnostics, pairing, matrices and exports. No existing API,
scorer, checkpoint selection, historical precision or tiny-H behavior changed.

The initial plan was reporting module + synthetic tests + this handoff. Two small
scripts were added for a reproducible no-training regression run and the user's
subsequently authorized class-only reconstruction.

## Added files

| File | Purpose |
| --- | --- |
| `evaluation/reporting.py` | Validated immutable records, reports, comparison, matrices, exports, hash verification |
| `tests/test_evaluation_reporting.py` | 39 synthetic/ingress regression tests, including nine P2 regressions |
| `scripts/run_wp5_regressions.py` | Discover all local tests; explicitly exclude only the two optimizer-update tests |
| `scripts/reconstruct_final_tables.py` | Reproduce the frozen class-only candidate/prompt payloads without model or test-image access |
| `docs/WP5_Reporting_2026-11-07.md` | This API, evidence, and blocker handoff |
| `artifacts/wp5/final_reconstruction_2026-11-07/verified_reconstruction_candidates.json` | Actual canonical candidate bytes matching Issue #8 |
| `artifacts/wp5/final_reconstruction_2026-11-07/verified_reconstruction_prompts.json` | Actual canonical prompt bytes matching Issue #8 |
| `artifacts/wp5/final_reconstruction_2026-11-07/reconstruction_provenance.json` | Inputs, checksums, source revision, and explicitly no approval |

## APIs and provenance

`Candidate` retains original composition/crop/disease IDs and all three labels.
`CandidateSpec` includes protocol ID/version, development/final mode, explicit
ordered candidates, source/split hashes and namespaces, semantic source, explicit
healthy disease membership, tie rule, synthetic flag and final policy reference.
It rejects malformed/non-integer IDs, duplicate IDs/pairs, conflicting labels,
nonascending columns and incorrect fingerprints. No local renumbering occurs.

`Sample` carries the exact original filename/sample ID, fold and global truth ID.
`EvaluationManifest` freezes fitting/seen/unseen samples, independent truth axes,
expected denominators and the identity source. Truth axes partition candidates.
Real development requires 1,238/1,894 rows and 4,868 fitting records. Real final
requires 2,000/1,448 rows, 323/10 truth classes, the verified ten unseen IDs, and
the original fitting records for support diagnostics. Counts do not establish
identity approval: callers must supply an owner-approved truth manifest.

`development_manifest(protocol, source_csv, healthy_disease_names=...)` delegates
to WP1's training-source validator and additionally requires seed 42 and accepted
ordered-filename reconciliation. It rejects official CSV filenames before opening
them. This is the only reporting adapter that reads input metadata. There is no
default official-test reader or inference entry point.

`RunProvenance` declares model/run/checkpoint ID and SHA-256, preprocessing,
code/environment versions, canonical score semantics/precision, comparison budget
and optional prompt hash. The prompt hash must always be a string: empty where
optional, otherwise a lowercase SHA-256 digest. Scores belong to run provenance so different canonical
model scorers can share one model-independent candidate task. For an independent
crop/disease pair, the checkpoint ID/hash can identify a separately frozen asset
manifest containing both checkpoint identities; preserve the individual hashes
and transforms there, and describe that manifest in preprocessing provenance.

`PredictionRecord` contains immutable `Sample`, `CandidateSpec`, `RunProvenance`,
explicit score-column global IDs, declared `available_k`, and exactly one of:

- The complete canonical finite score vector.
- Unique ranked global IDs covering at least `min(5,K)` candidates.

Optional `independent_crop_id` / `independent_disease_id` are the original head
decisions, separately labeled and counted only where provided. The reporter does
not derive independent-head accuracy from the chosen composition.

`records_from_scores(spec, provenance, samples, scores, candidate_ids)` checks
NumPy/Torch matrix shape and declared float32/float64 precision, then makes
immutable copies. It only ranks supplied values: WP2 float64 PyTorch
log-softmax sums and WP3 direct float64 cosine scores remain untouched. Exact ties
follow stable supplied columns; the boundary requires ascending global IDs.
Direct constructors and deserialization validate the original exact integer
ratio against the declared float32/float64 representation before storing a
Python float. Inexact values, integer precision loss, nonfinite values,
overflow and underflow to zero are rejected; accepted values are not rounded.
The matrix adapter's existing dtype checks remain unchanged.
Float32 values are represented exactly as Python floats, without recomputing
scores. Rank-only inputs inherit upstream tie decisions; missing score values
cannot independently prove that the upstream ranker followed its declared rule.

Candidate specs, identity manifests and predictions expose `to_dict()` /
`from_dict()` with versioned schemas. WP1 candidate hashing remains ASCII compact
sorted-key JSON over candidate records, namespace
`wp1-canonical-json-candidate-records-v1`. Final candidate hashing uses UTF-8
compact sorted-key JSON over `{"schema":"cos30082-final-candidates-v1","rows":[...]}`.
Prompt hashing uses the documented `{"policy":...,"rows":[...]}` payload.
Reporting spec/manifest fingerprints are additional identities and never replace
the original WP1 or historical WP2/WP3 hashes.

`build_report(manifest, records)` rejects empty, missing, duplicate, unexpected,
wrong-truth/wrong-fold or mixed-provenance rows. It reports seen/unseen Top-1/5,
correct/processed/expected counts, H@1/H@5, combined micro accuracy, per-crop,
per-disease and per-composition composition hits, healthy/diseased results,
false-healthy counts, selected-composition component consistency, fitting-only
support and prediction preferences. Support below five is explicitly flagged;
this is a descriptive sparsity warning, not a confidence interval or significance
claim. Zero-support accuracies and normalized cells are JSON null / CSV blank.

`compare_models(manifest, records_a, records_b)` strictly aligns both complete
sets to original identities/truth/fold/spec. It rejects incompatible budgets and
never drops rows for an intersection. It reports both-correct/A-only/B-only/neither
for Top-1/5 by fold and composition, gains/losses, H deltas and aligned health /
false-healthy comparisons. Different canonical model scorers are allowed; the
fractional composition-hit metric policy must match. Results describe paired
outcomes without causal or significance claims.

`confusion_matrix(records, truth_ids, prediction_ids, candidates=...)` has
independent ordered truth/prediction axes and explicit labels. It retains
zero-support rows and prediction-only classes; count totals and diagonal hits
are available for ingress/report checks.

`export_report(report, new_directory)` creates a new directory and writes
`report.json` plus summary, harmonic, group, health, component and labeled
confusion CSVs. Paired reports export pairing, health comparison and harmonic
delta CSVs. JSON includes the candidate task, identities, counts, missing-data
policy and run provenance. An existing output directory always raises; no
historical result is overwritten. Raw images, weights and per-epoch tensors
are not exported.

Synthetic usage (run from repository root):

```python
import sys
sys.path.insert(0, "tests")
from test_evaluation_reporting import fixture
from evaluation.reporting import build_report, compare_models, export_report

manifest, provenance, canonical_scores, records = fixture()
report = build_report(manifest, records)
assert report["harmonic_mean_top1"] == 0.5
export_report(report, "artifacts/wp5/my_new_synthetic_report")  # must not exist
paired = compare_models(manifest, records, records[::-1])  # identity-order check
assert paired["harmonic_mean_top1_b_minus_a"] == 0.0
```

Real input example: load an owner-approved serialized manifest through
`EvaluationManifest.from_dict(...)`, construct the frozen `RunProvenance`, pass
the original sample rows and canonical matrix to `records_from_scores(...)`,
then call `build_report` / `compare_models`. Do not manufacture a manifest from
whichever prediction rows happen to be present: that defeats missing-row checks.

## Independent final reconstruction

The user authorized reconstruction after confirming the original frozen files
were unavailable. Downloaded only the metadata archive from the
[authors' Zenodo record](https://zenodo.org/records/16879271), checked its published
MD5, and opened only `csv/default/train.csv`. Discarded filenames immediately;
retained class mappings only. No test archive member, images or weights were
opened. All 323 course-training class mappings matched on original global ID,
both component IDs and all names. The ten additional IDs are
39, 84, 107, 124, 133, 136, 144, 166, 276 and 303.

Inspected the published [WP3A notebook at pinned commit 300f1d7](https://github.com/FZhen2450/COS30082_ML_Project/blob/300f1d78bb962f67fa2dbdb7b76ed23301abd7df/notebooks/wp3_clip_zero_shot.ipynb).
Reproduced its policy object, normalization, article rule, prompt fields and
column order without executing the notebook. The 323-candidate development
prompt hash also reproduced; every original prompt string remained identical.

| Payload / input | Actual SHA-256 |
| --- | --- |
| Reconstructed canonical candidate file | `ba17e94119e8b2b7a4b18603e25405d2f52bf4a93396973f80491800697eb62f` |
| Reconstructed canonical prompt file | `c9dd871d61d4141b33583ddd71961c309fd679a8edab088cb28ba20eccef88f2` |
| Preserved development prompt payload | `753672c7964bcb20a1346145f21a72b40a19f55590a006b5783823f23c13f537` |
| Course CRLF training CSV | `9f1b729ab08f120c9e8d3a5681aa1ccbdf1d9210f9aaa6143959363dcb106a7d` |
| Downloaded public metadata ZIP | `b05e34861ba044a43039347345de5eff299f6cfb5a6540574d5182c5d891a3a7` |

Both final digests exactly match Issue #8. The saved payload files are themselves
canonical UTF-8 bytes without a trailing LF, so their raw file SHA-256 digests
also match. This verifies reconstructed payload identity, not final-policy
approval. `verify_final_payloads(...)` explicitly returns `approval_granted=False`.

Actual reconstruction command (a rerun must choose another new directory):

```powershell
python -B scripts/reconstruct_final_tables.py --output artifacts/wp5/final_reconstruction_2026-11-07
```

Exit 0; verification `payload_hashes_match_issue_8`; 333 candidates, 323 matched
course classes, 323 preserved development prompt strings, no official-test
access, no model loading. Independent raw `Get-FileHash` and a separate call to
`verify_final_payloads` confirmed both complete saved-file digests.

## Validation evidence

Windows Python 3.13.7; NumPy 2.2.6; torch 2.9.1+cu128;
torchvision 0.24.1+cu128; pandas 2.3.3. Existing interpreter:
`C:/Users/60103/Documents/GitHub/COS30082_ML_Project/.venv/Scripts/python.exe`.
CPU-only runs: `CUDA_VISIBLE_DEVICES=''`, OMP/MKL/OpenBLAS threads 2, Python `-B`.
No environments or packages were changed.

```powershell
$env:CUDA_VISIBLE_DEVICES=''
$env:OMP_NUM_THREADS='2'
$env:MKL_NUM_THREADS='2'
$env:OPENBLAS_NUM_THREADS='2'
python -B -m unittest discover -s tests -p test_metrics.py -v
python -B -m unittest discover -s tests -p test_evaluation_reporting.py -v
python -B scripts/verify_wp5_bridge.py --wp5-root .
python -B scripts/run_wp5_regressions.py
python -B shared_protocol.py --verify
git diff --check
```

Initial implementation outcomes before the P2 corrections: original metrics **6/6 passed** (0.121 s); reporting focused
**30/30 passed** (7.168 s); synthetic
WP1 bridge **passed** with seen Top-1=1, unseen Top-1=0.5, H@1=2/3, H@5=1.
The final applicable suite, including all final immutable-type assertions,
reported `Ran 62 tests in 37.064s` and `OK` (exit 0).

Only these existing optimizer-update tests are excluded by the no-training runner:

- `test_independent_baseline.IndependentTests.test_separate_parameters_and_gradient_updates`
- `test_independent_baseline.IndependentTests.test_synthetic_training_records_and_inference_adapter`

All existing tests stay unchanged. The unreconciled-protocol guard test remains:
it calls the guard but exits before training. Other model tests use random
initialization / synthetic images and no pretrained downloads. Legacy metadata
contract tests still read official CSVs; they perform no model evaluation or new
development analysis from official labels. New reporting/reconstruction tests
use synthetic data or the explicit course training source only.

The first full run from stdin had a Windows worker-spawn harness error. Running
from the new guarded script fixed that invocation error without modifying or
excluding the worker test. The file-based suite passed 60/60, then 62/62 after
two more boundary/reconstruction tests were added. No remaining assertion failure
was observed in those runs. An initial slow ranking loop was also corrected to
reuse its candidate-ID tuple; score/ranking behavior did not change.

Synthetic golden counts (six noncontiguous candidate IDs):

| Fold | Expected/processed | Top-1 hits | Top-5 hits | Top-1 | Top-5 |
| --- | --- | --- | --- | --- | --- |
| Seen | 2/2 | 1 | 2 | 0.5 | 1.0 |
| Unseen | 2/2 | 1 | 1 | 0.5 | 0.5 |
| Combined micro | 4/4 | 2 | 3 | 0.5 | 0.75 |

H@1=0.5; H@5=2/3. These are fixture outcomes, not model results.
The final-sized synthetic fixture has 323 truth rows and 333 prediction columns:
seen matrix sum 2,000, diagonal/Top-1 hits 1,998; unseen 1,448/1,448 Top-1 hits.
It includes an unseen prediction column, an exact tie and a one-ULP lead.
It is visibly `synthetic=True` and establishes no final policy approval.

`shared_protocol.py --verify` passed with counts 4,868/1,238/1,894, 323 candidates,
WP1 candidate hash `8217be059e3a4a144f89081ee43b7cd042bb5a62c370197bf0e3fca1c6d28bf3`
and split hash `edc29a67dc9de9c454e5f548389b4bffd601ebc06423e63b42385be053b7ceff`.
The local artifact still explicitly says `pending_ordered_filename_reference`;
its contents were preserved. Issue #8's externally completed reconciliation is
not contradicted or rewritten by that historical local status.

## P2 validation corrections and verification

The publication audit confirmed two defects: direct/deserialized scores could
violate declared precision, and an empty mutable prompt hash could enter frozen
provenance. Both are corrected locally under the user's narrow authorization.
Only `evaluation/reporting.py`, `tests/test_evaluation_reporting.py` and this
document changed during that correction; no new files were added.

`_exact_score` captures an integer's exact value or a floating value's
`as_integer_ratio()` before any lossy conversion. A declared-dtype conversion
is used only as a validation probe: its finite exact ratio must match the
original. Only then is the accepted value stored as a Python float. This avoids
NumPy integer/float equality coercion and preserves exact ties, one-ULP leads,
signed zero, finite maxima, minimum normals and subnormals. Nonrepresentable
inputs raise `ValueError`, including huge integer overflows. No scorer, ranking
formula, metric API, candidate policy, matrix dtype rule or checkpoint selection
was changed.

`RunProvenance` now checks that `prompt_sha256` is a string before considering
whether it is empty. Existing optional empty strings and valid digest strings
remain supported. Lists, dictionaries, other containers, numbers, booleans and
null values are rejected before they can enter an immutable prediction record.

Nine added tests all passed (direct and JSON reconstruction paths are covered):

- `test_float32_declared_rejects_float64_ulp_at_both_boundaries`
- `test_exact_float32_values_and_ties_survive_both_boundaries`
- `test_float64_one_ulp_ranking_survives_both_boundaries`
- `test_inexact_integer_scores_rejected_at_both_boundaries`
- `test_finite_score_boundaries_are_exact_in_both_precisions`
- `test_nonfinite_overflow_and_underflow_scores_rejected_at_both_boundaries`
- `test_matrix_dtype_rules_remain_strict_for_exact_values`
- `test_prompt_sha256_rejects_non_string_provenance_at_both_boundaries`
- `test_prompt_sha256_accepts_only_valid_immutable_strings`

The regression tests were run before the fix and reproduced the validation
failures. After the fix, using the CPU-only environment above:

| Actual command | Result |
| --- | --- |
| `python -B -m unittest discover -s tests -p test_evaluation_reporting.py -v` | 39/39 passed, 8.379 s |
| `python -B -m unittest discover -s tests -p test_metrics.py -v` | 6/6 passed, 0.014 s |
| `python -B scripts/verify_wp5_bridge.py --wp5-root .` | Passed; H@1=2/3 and H@5=1 |
| `python -B scripts/run_wp5_regressions.py` | 71/71 applicable tests passed, 28.190 s |
| `python -B scripts/verify_csv_checkout.py` | Passed; all three source-byte hashes and scoped attributes preserved |
| `git diff --check` and `git diff --cached --check` | Passed; existing LF-to-CRLF conversion notices only |

The same two optimizer-update tests remain excluded solely under the no-training
constraint; their source and the regression runner are unchanged. Existing
canonical log-score/cosine, exact-tie, one-ULP, H@1/H@5 and rectangular-matrix
tests remain enabled and passed. Both reconstructed files were read without
regeneration; raw SHA-256 and canonical byte identity remain exactly:

```text
candidate ba17e94119e8b2b7a4b18603e25405d2f52bf4a93396973f80491800697eb62f
prompt    c9dd871d61d4141b33583ddd71961c309fd679a8edab088cb28ba20eccef88f2
```

Publication recommendation for these source corrections: **PASS for review**.
Both confirmed P2 defects are resolved; no known correctness blocker remains
within this scope. Final evaluation remains subject to the separate checks
below. No official-test images/predictions, training, staging, commit, push,
PR creation or merge occurred during the corrections.

## Remaining release checks

- **Resolved:** final candidate and prompt payload reconstruction/hash identity.
- **Pending approval:** explicit shared-333 final evaluation policy and review by
  Chia/Frederick with WP1/WP3 coordination. Matching payloads grant no approval.
- **Blocked on inputs:** approved official fold filename/truth manifests and
  ingress verification of the actual 2,000/1,448 identities. Only synthetic
  official-sized denominators were exercised; no actual test labels were used
  to construct these new manifests.
- **Blocked on inputs:** frozen checkpoint assets/hashes and exact inference
  transforms, prompt binding and run/budget declarations for every finalist,
  including the independently selected crop/disease pair. None were verified
  from actual model assets in this checkout.
- **Blocked locally:** accepted reconciled development manifest and immutable
  owner-approved development prediction records. The real M3/M6 paired health
  audit / saved-metric reproduction cannot be claimed from absent records.
- **Not rerun:** native pinned CLIP tokenization to [333,77]. The `clip` package
  is unavailable in this Windows environment; the referenced Katana interpreter
  `/home/user1/anaconda3/envs/as04_yolo_katana/bin/python` is also absent from
  the local Ubuntu WSL distribution (`test -x` exited 1). No package was
  installed. Exact prompt bytes are verified; tokenizer/embedding/model checks
  remain separate.
- **Not run:** draft WP2/WP3 integration suites requiring unpublished/not-merged
  dependency files. Their published tests were inspected; local synthetic
  canonical-score compatibility and the unchanged WP1 bridge were run instead.
- **Requires separate authorization:** official-test inference/evaluation and
  staging/commit/push/PR publication/merge. No real final metrics, trained-model
  claims, plots from real predictions, or checkpoint reselection were produced.

No outgoing coordination messages were sent. This handoff is ready for local
review; original experiment selection and results remain authoritative.
