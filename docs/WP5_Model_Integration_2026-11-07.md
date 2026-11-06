# WP5 selected-model output adapters — 7 November 2026

Local follow-up to draft [PR #18](https://github.com/FZhen2450/COS30082_ML_Project/pull/18),
published at `65d4fbfd47bfbe2c12ddde2ac2ec1a9426ec6984`.
Review requests were accepted for verified collaborators `Yoriichi-0u0` (WP2
PR #5 author, write access) and `FZhen2450` (repository owner, admin). Frederick's
GitHub identity was not verified, so no handle was guessed. No merge occurred.

This follow-up is prepared for the user-authorized update of the existing draft
PR after its final review and verification. It adds only:

- `evaluation/model_adapters.py`
- `tests/test_model_adapters.py`
- `docs/WP2_Review_2026-11-07.md`
- `docs/WP5_Model_Integration_2026-11-07.md`

Existing reporting/scoring code, source CSVs, attributes, frozen JSONs, experiments
and teammate branches remain unchanged. Source publication is authorized after
verification; final evaluation approval and merging are outside that authority.

## Requirement addressed

The initial reporter validates records but has no selected-model ingress that
binds their original producer metadata, historical hash namespaces, checkpoint,
preprocessing and explicit image identities. Anonymous WP2 logits cannot safely
be assigned filenames from counts or matching class labels. The new adapter
closes this gap without introducing another scorer or loading another model.

It adapts **prediction outputs**, not image inference. Real model loading and
owner verification of frozen assets remain prerequisites outside this module.
Successful synthetic ingestion does not demonstrate that any real checkpoint
has been loaded, evaluated or approved.

## Four explicit model roles

Selections follow [Issue #8 comment 6653018470](https://github.com/FZhen2450/COS30082_ML_Project/issues/8#issuecomment-6653018470).

| Model kind | Canonical score declaration | Additional declarations |
| --- | --- | --- |
| `wp1_independent_pair` | `sum_component_log_probabilities`; native declared float32/float64 | Both crop/disease checkpoint IDs, SHA-256 values and preprocessing recipes |
| `wp2_convnext_base` | `sum_log_softmax_v1`; float64 | `multitask_convnext_base`, epoch 1, actual input resolution 224 |
| `clip_vit_l14_frozen` | `float64_cpu_scaled_unit_feature_cosine_v1`; float64 | ViT-L/14, exact prompt identity; real-mode frozen weight digest pinned |
| `clip_vit_l14_adapter` | Same canonical direct cosine; float64 | ViT-L/14 base digest, epoch 1, training seed 45, residual cap 0.20, exact prompt identity |

These are contract checks on declarations. They do not authenticate the producer
or verify unavailable asset files. The independent checkpoint-pair selection and
real adapter checkpoint hash must come from their owners; no hashes are invented.
Base CLIP and adapter checkpoint digests must be distinct, preventing an
accidental reuse of the frozen base asset as the selected trained adapter.

## API and packet contract

```python
adapter = ModelOutputAdapter(approved_manifest, frozen_run_provenance, model_kind)
records = adapter.adapt(owner_verified_packet)
report = build_report(approved_manifest, all_batch_records)
paired = compare_models(approved_manifest, model_a_records, model_b_records)
export_report(report, new_output_directory)
```

For non-synthetic final records, construction also requires an explicit
`final_evaluation_reference`. This attests separate owner authorization for
record ingestion, in addition to the existing final candidate/prompt/policy
requirements. It grants no inference permission by itself and starts no job.

Required packet fields:

| Fields | Checks |
| --- | --- |
| `schema="wp5-model-output-v1"`, `model_kind` | Exact selected-model binding |
| `model_id`, `run_id`, `checkpoint_id`, `checkpoint_sha256`, `preprocessing`, `code_version`, `environment`, `score_precision`, `score_semantics` | Match the frozen `RunProvenance`; matrix precision also checked independently |
| `manifest_sha256`, `source_sha256` | Match the approved complete identity manifest and source |
| `candidate_ids`, `candidate_sha256`, `candidate_hash_namespace` | Exact global-ID column alignment and correct source serialization |
| `split_sha256`, `split_hash_namespace` | Exact source fold filename ordering / approved full-manifest identity |
| `selection` | Model-specific epoch/architecture/resolution/asset declarations above |
| `image_names`, `ground_truth_ids`, `fold` | Nonempty unique original filenames, exact approved truths/fold, aligned row count |
| Exactly one of `scores` or `ranked_ids` | Complete canonical score columns, or unique allowed ranked global IDs |
| `available_k` for ranked-only packets | At least `min(5,K)` and exact matrix width |
| `prompt_sha256` for both CLIP roles | Valid digest matching frozen run and, in real mode, the selected task |

Optional `independent_crop_ids` / `independent_disease_ids` preserve original
head decisions and feed separately labeled component accuracy. The adapter never
infers these from selected composition components.

The producer must emit/verify these fields against its own immutable run
evidence. Do not fill metadata with the expected WP5 values merely to make a
check pass. For WP2 use the accepted notebook canonical scorer and the selected
checkpoint's actual preprocessing. For CLIP use its accepted native image/text
processing and canonical direct cosine matrix. No NumPy reconstruction of
logits/cosines or probability-product substitution occurs here.

`adapt` copies values into immutable `PredictionRecord` objects through the
existing exact-precision validator. The original preprocessing description and
verified source hashes/selection are retained in a canonical JSON envelope in
record provenance. This makes the different source hash formats visible in
exports without replacing the original checkpoint/score declarations.

Partial batches are supported; `build_report` and `compare_models` still require
complete seen/unseen identity coverage and reject duplicates across batches.
No intersection analysis or denominator reduction is automatic.

JSON inputs retain their original scalar types until validation. Mixed Boolean /
integer identity lists are rejected before NumPy could turn `True` into `1`.
Serialized score lists pass directly through the exact-precision record boundary
before any numeric conversion; valid float32 JSON values are accepted, while
integer precision loss, overflow, nonfinite values and float64-only distinctions
declared as float32 are rejected. Native NumPy/Torch matrices retain the existing
strict dtype rules. Filename mappings are rejected rather than treated as arrays.

`import_model_packets(manifest, provenances, packets)` requires exactly all four
model roles, validates their complete batch collections, and returns four model
reports plus all six strict paired reports. Missing models/rows, duplicates or
incompatible comparison budgets reject the import before any reports are
returned or written. It performs no file access or inference. Non-synthetic
final imports also require `final_evaluation_reference` explicitly.

## Hash namespace bridge and task separation

`output_candidate_digest(spec, namespace)` supports these explicitly named
formats. The legacy namespace names below are WP5 labels for the documented
historical serializations, not renamed historical artifacts.

- WP1: `wp1-canonical-json-candidate-records-v1`, all six class fields.
- Historical WP2/WP3: `wp2-json-global-id-crop-disease-triples-v1`, ordered
  `[[composition_id,crop_id,disease_id],...]`.
- Final: `cos30082-final-candidates-v1`, the separately frozen schema/rows payload.

`output_split_digest(manifest, namespace)` bridges WP1's original fold keys and
the historical `fit`/`val_seen`/`val_unseen` keys, preserving each filename order.
Development packets additionally bind the complete WP5 identity-manifest hash.
Final packets must use the final candidate namespace and
`wp5-identity-manifest-v1` for their full fitting/seen/unseen identity binding.
Development split hashes cannot be reused as final identity evidence.

Real development remains K=323 and 4,868 / 1,238 / 1,894. The proposed final task
remains K=333 and 2,000 / 1,448. Synthetic tests also exercise tiny universes and
explicitly mark synthetic outputs; no fake images or identities are presented as
actual official membership. The real-mode deny-gate test uses public class-only
payloads with invented filenames solely to prove missing authorization fails.

## Actual checks

CPU only; CUDA disabled; OMP/MKL/OpenBLAS threads 2; Python `-B`;
existing Windows environment and packages unchanged.

| Actual command | Result |
| --- | --- |
| `python -B -m unittest discover -s tests -p test_model_adapters.py -v` | 26/26 passed, 2.293 s |
| `python -B -m unittest discover -s tests -p test_evaluation_reporting.py -q` | 39/39 passed, 9.671 s |
| `python -B -m unittest discover -s tests -p test_metrics.py -q` | 6/6 passed, 0.013 s |
| `python -B scripts/verify_wp5_bridge.py --wp5-root .` | Passed; synthetic H@1=2/3, H@5=1 |
| `python -B scripts/run_wp5_regressions.py` | 97/97 applicable regressions passed, 29.601 s |
| `python -B scripts/verify_csv_checkout.py` | Passed; all three CSV byte hashes/attributes unchanged |

The same two optimizer-update tests remain excluded, with no test-source changes
to them. Complete draft WP2/WP3 training suites were not invoked or copied here.

New coverage includes all four adapters, all six model pairs, canonical value
copies, exact/one-ULP ties, explicit independent-head decisions, source-namespace
differences, checkpoint/transform/prompt mismatches, missing or duplicate rows,
rank-only exports, task separation and real-final missing-authorization denial.
All four synthetic model reports exported JSON/CSV tables successfully.

All original 19 adapter tests remain enabled. Final review added seven regressions:

- `test_serialized_scores_validate_original_values_before_conversion`
- `test_exact_float32_json_import_keeps_native_matrix_dtype_rules`
- `test_float64_json_one_ulp_and_signed_zero_survive_import`
- `test_mixed_boolean_ids_are_rejected_before_numpy_coercion`
- `test_adapter_and_base_clip_checkpoint_identities_must_be_distinct`
- `test_four_model_import_validates_complete_reports_before_export`
- `test_four_model_import_rejects_partial_duplicate_and_incompatible_runs`

The training-source-only 323-candidate test independently reproduces both legacy
hashes below and passes noncontiguous global columns / one-ULP scores through all
four adapters. It reads no official-test records or images and changes no source
or reconciliation artifact.

```text
legacy candidate 4cb0a24a96cf647f32311817fa7b082a46d3cf00c20ae61f4287e83cc898dfaf
legacy split     f0c7ba212449fd68f1d19c0f84cc91bf4cc1af29ae405128e5198bce89f25f85
final candidate  ba17e94119e8b2b7a4b18603e25405d2f52bf4a93396973f80491800697eb62f
final prompt     c9dd871d61d4141b33583ddd71961c309fd679a8edab088cb28ba20eccef88f2
```

The final payload files were read for integrity/semantic checks and never
regenerated. Their complete SHA-256 values remain unchanged.

## External handoff and defensible comparison figures

### Minimum handoff for reporting existing approved development predictions

No raw images or model weight files are required if the owners provide immutable
predictions already produced by the selected frozen models. The minimum is:

- The accepted reconciled development identity manifest, original filename /
  fold / global truth membership, and both its raw file SHA-256 and WP5 semantic
  `manifest.sha256`. Those digests use different payloads and must not be aliased.
- One approved frozen run/provenance declaration for each model role: original
  run/checkpoint IDs and hashes, exact transforms, canonical scoring/precision,
  code/environment versions, prompt identity where applicable, selection fields
  above and an explicit common comparison plan/budget. Independent crop/disease
  identities and adapter/base identities remain separate in those declarations.
- Identified seen/unseen prediction packets for all four roles, with original
  filenames, truths, full ordered candidate columns/hashes and source split
  declarations. Either full canonical scores or ranked global IDs through Top-5
  suffice for the requested composition/health/paired metrics. Independent-head
  decisions are optional and must be supplied to claim independent-head accuracy.
- Owner-approved raw SHA-256 digests for every packet file and the handoff bundle,
  plus the approval reference. An existing multi-epoch CSV needs an explicit
  owner-frozen epoch/filter specification and provenance-preserving conversion;
  WP5 must not select another epoch or infer filenames during conversion.

For this development handoff use the 323-candidate task only. Do not send or open
official-test prediction files under the current authorization. Final-task
payload hashes are known, but policy, actual identities, frozen inputs and
separate evaluation/ingestion authorization remain required.

### Validated import workflow prepared for that handoff

The owner supplies an approved `wp5-owner-handoff-v1` JSON bundle with
`owner_approval_reference`, `identity_manifest: {path, file_sha256}`,
`manifest_sha256`, and `models` keyed by exactly `MODEL_KINDS`. Each model entry
contains a `provenance` dictionary accepted by `RunProvenance` and a
`packet_files` list of `{path, file_sha256}` entries. Paths identify only the
explicit approved files; there is no directory scan or inferred data source.
File hashes must be confirmed through the owner handoff, not trusted merely
because the same unverified file declares them.

The following development-only recipe is prepared; no real owner files were
available or opened. The pure import step and export sequence were exercised
end-to-end using JSON-decoded synthetic packets in the tests above.

```python
import hashlib
import json
from pathlib import Path
from evaluation.model_adapters import MODEL_KINDS, import_model_packets
from evaluation.reporting import EvaluationManifest, RunProvenance, export_report

def checked_json(entry):
    raw = Path(entry["path"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != entry["file_sha256"]:
        raise ValueError("Owner-approved input file digest mismatch")
    return json.loads(raw)  # Preserve JSON integers; do not pre-cast score/ID arrays.

# handoff_file is an explicit {path, file_sha256} supplied by the owner.
bundle = checked_json(handoff_file)
if bundle.get("schema") != "wp5-owner-handoff-v1":
    raise ValueError("Unknown owner handoff schema")
approval = bundle.get("owner_approval_reference")
if not isinstance(approval, str) or not approval.strip():
    raise ValueError("Owner approval reference required")
manifest = EvaluationManifest.from_dict(checked_json(bundle["identity_manifest"]))
if manifest.spec.task_mode != "development" or manifest.spec.synthetic:
    raise ValueError("This real import recipe accepts approved development-323 only")
if manifest.sha256 != bundle["manifest_sha256"]:
    raise ValueError("Approved semantic identity-manifest digest mismatch")
if set(bundle["models"]) != set(MODEL_KINDS):
    raise ValueError("Require all four frozen models")
provenances = {kind: RunProvenance(**entry["provenance"])
               for kind, entry in bundle["models"].items()}
packets = {kind: [checked_json(file) for file in entry["packet_files"]]
           for kind, entry in bundle["models"].items()}
imported = import_model_packets(manifest, provenances, packets)

# All model/coverage/paired checks succeeded before creating output files.
output = Path(new_output_directory)  # Explicit new path; never a historical run.
output.mkdir(parents=True, exist_ok=False)
receipt = {"owner_approval_reference": approval,
           "manifest_sha256": manifest.sha256,
           "candidate_sha256": manifest.spec.candidate_sha256,
           "handoff_file": handoff_file, "verified_inputs": bundle}
with (output / "import_receipt.json").open("x", encoding="utf-8") as stream:
    json.dump(receipt, stream, indent=2, allow_nan=False)
for kind, report in imported["models"].items():
    export_report(report, output / kind)
for pair, report in imported["paired"].items():
    export_report(report, output / "paired" / pair)
```

This workflow verifies declared file integrity and exact record alignment; it
does not authenticate unsupported checkpoint claims, grant final policy approval,
run models, reselect epochs or create real performance evidence from fixtures.

### Additional assets only if new frozen-model inference is authorized later

1. WP1/Fa Zhen: supply the selected crop/disease asset identities, both SHA-256
   values, exact separate transforms, frozen pair-manifest identity and immutable
   development output rows. Architecture selection is not inferred by WP5.
2. WP2/Chia: supply the selected epoch-1 ConvNeXt-Base asset/hash and 224 × 224
   transform evidence, canonical log scores or ranks, and original filenames.
   The generic B0/Tiny loader is not the selected Base loader; anonymous epoch
   outputs alone are insufficient for strict paired analysis.
3. CLIP/Chia: supply verified local ViT-L/14 assets and native preprocessing /
   tokenizer evidence. The weight identity is known; its actual file and native
   [333,77] tokenizer check were not verified in this environment.
4. Adapter/Chia: supply the matching seed-45, cap-0.20, epoch-1 adapter hash,
   base-CLIP binding, frozen inference code and identified canonical output rows.
   Public PR #17 is M5 replication preparation; it does not itself supply this
   M6 asset/output handoff. Historical M4 selected epochs must not be substituted.
5. WP1/owners: supply the accepted reconciled development artifact. Its ordered
   membership hashes match locally, but the checked-in pending status was not
   silently rewritten or approved.
6. Team: approve shared-333 policy, frozen model/prompt/preprocessing inputs and
   actual official fold identity manifests before any separately authorized
   official evaluation. No such evaluation was run or enabled by this work.

Once approved immutable development records are supplied, the adapters and
existing reports can produce shared-denominator comparison tables and all six
paired analyses. Final comparison figures must use actual approved records,
identify the task/candidate hash, show seen/unseen Top-1/5 and H@1/H@5, expose
healthy/diseased denominators and false-healthy errors, and retain disease-safety
failures. Missing records must block the primary figure. Paired deltas do not
establish significance or causation; seed uncertainty needs completed replicated
runs. Synthetic metrics here test interfaces and are not report findings.

The WP2 CPU-buffer bug and filename-export gap are documented separately in
`WP2_Review_2026-11-07.md`. No teammate code or historical experiment was fixed
in place. Source publication updates the existing draft PR under the user's
authorization. No Gmail message, training, official-test prediction access,
final-policy approval or merge is part of this follow-up.
