# Full-8,000 WP2 to WP5 handoff

This extension targets draft PR #24 at `d2149c9d1b73883a9962991316276837d6c2c50a`, not an unqualified current-main merge. It reuses the existing final manifest/inference-order infrastructure, `ModelOutputAdapter`, truth-attachment boundary and report/metric functions. The Base scorer and rank functions are the previously verified implementation, promoted unchanged. No training, image reading, model construction or inference is provided by the new path.

## Protocol selection

`EvaluationManifest` now recognizes exactly `cos30082-official-shared333 / full8000-20261112-v1`. That profile requires 8,000 unique fitting identities covering the 323 seen compositions, the original ordered fitting-filename fingerprint, 2,000 Seen and 1,448 Unseen rows, and all 333 ascending global candidate IDs. These checks apply even to synthetic fixtures. Unknown versions of the reserved protocol ID fail. Historical real protocols retain the 4,868 guard; the historical final builder and checkpoint pins are unchanged.

`load_full8000_manifest` loads the supplied serialized payload and uses the existing `EvaluationManifest.sha256 = fingerprint(asdict(manifest))` definition. It does not reconstruct an original artifact or define a new serialization namespace. Real mode requires semantic identity `41e4b1d499e1c45097fd8b74d917f06e0efc324772d9459e811259746f0f3a6c`; the raw JSON file SHA-256 is independently checked before decoding. Synthetic fixtures require explicit opt-in and their own externally supplied semantic pin.

## Owner interface: label-free outputs

WP5 issues `full8000_inference_order(manifest)`. It contains ordered filenames, the permitted candidate vocabulary and contract identities, but no per-image evaluation truths. Model owners may still know labels from course filenames/CSVs: this is label-free transport, not a blind evaluation.

```python
from evaluation.reporting import fingerprint
from evaluation.wp2_handoff import BaseOwnerHandoff

# order and its expected fingerprint are independently communicated by WP5.
# run and owner_metadata come from authenticated, reviewed release evidence.
collector = BaseOwnerHandoff(order, run, owner_metadata,
                            expected_order_sha256=approved_order_sha256)
for fold, outputs, actual_names, actual_ids, actual_pairs in collected_batches:
    collector.add_batch(fold, outputs, actual_names, actual_ids, actual_pairs)
completion_sha256 = collector.export(new_directory)
```

`outputs` is the evidenced mapping with floating `crop_logits[N,55]` and `disease_logits[N,175]`; optional features must be `[N,1024]`. The existing scorer converts components to FP64 before separate log-softmax operations, then adds the mapped valid-pair log-probabilities. Stable descending ranking follows original ascending global IDs. It does not multiply probabilities, round scores, use a 333-way replacement head or substitute Tiny.

The owner loader remains unavailable until its private schema/source is evidenced. `load_private_final_checkpoint` fails clearly without attempting deserialization; no pretrained downloads occur. The unchanged `validation_transform` supplies 224-square bilinear antialiased resizing, FP32 scaling and ImageNet normalization; the owner must preserve the preceding RGB conversion and record actual forward precision/runtime. This code does not prove that those image operations ran.

`owner_metadata` must contain exactly `OWNER_FIELDS` from `wp2_handoff.py`: architecture/epoch/resolution, explicit full-data training count/source/ordered fitting digest, initialization, selection rule, structured training and inference transforms, structured runtime, scorer-source identity, training-recipe identity and release-receipt identity. `RunProvenance.preprocessing` and `environment` must equal canonical JSON of the reviewed inference transform and runtime. Missing/pending declarations fail. The selected new WP2 checkpoint declaration is `27aba7779973b4caef68f98c627fb00c5d8afc9a9b7de58f9628954a90976b34`; PR #24's historical WP2 pin is not accepted by this path.

The previous `BaseHandoff` API remains **WP5-internal**, because it accepts truth-bearing manifest rows and exports metrics/confusions. Use `BaseOwnerHandoff` for owner transport; new intake rejects the old truth-bearing packet layout.

## Deterministic owner packet

The exporter refuses any existing destination, flushes a private partial receipt, then atomically publishes `COMPLETED.json` last using a same-filesystem hard link that cannot overwrite an existing marker. Filesystems without hard-link support fail closed. An interrupted completion write cannot expose a partial recognized marker. Repeated exports of identical synthetic inputs were byte-identical on the tested runtime. Cross-version/compression-library byte identity is not guaranteed; all consumers verify the actual externally pinned bytes.

| File | Contents |
|---|---|
| `seen_scores.npz`, `unseen_scores.npz` | Native FP64 `scores`, Unicode `filenames`, integer `candidate_ids`, stable integer `top5_ids`; no truths |
| `seen_packet.json`, `unseen_packet.json` | Strict `wp5-model-output-npz-v1` metadata and score-file reference; no truths or accuracy |
| `provenance.json` | Exact `run` and reviewed `owner` declarations |
| `COMPLETED.json` | Every companion raw hash and inference-order identity; independent acceptance and checkpoint-byte verification are false |

NPZ decoding uses `allow_pickle=False`, bounded ZIP member inspection and pre-allocation NPY header validation. Outer ZIP packets require a separate archive pin and are inspected before extraction. Traversal, absolute/drive paths, backslashes, Windows reserved names, case collisions, symlink/nonregular members, encryption, unsupported compression and oversized inputs fail. No archive code is executed.

## WP5 intake

Prepare a separately reviewed `wp5-full8000-wp2-import-v1` JSON handoff with exactly these fields:

- `protocol_id`, `protocol_version`, `manifest_sha256`, `inference_order_sha256`.
- `identity_manifest: {path, file_sha256}` relative to the handoff directory.
- `owner_packet: {format: "directory", path, receipt_sha256}`, or format `zip` with additional `archive_sha256`.
- `provenance`: the exact reviewed `RunProvenance` dictionary.
- `reviewed_owner_metadata`: independently reviewed owner declarations, not copied/approved automatically by the exporter.
- `final_policy_reference`, `evaluation_authorization_reference`, `owner_approval_reference`.

The handoff itself requires an independently communicated raw SHA-256. A self-hashed producer packet does not establish authenticity.

```powershell
python scripts/import_wp5_final.py --protocol full8000-20261112-v1 --handoff REVIEWED_HANDOFF.json --handoff-sha256 EXTERNAL_HANDOFF_SHA256 --output NEW_WP5_OUTPUT --registry COMMON_WP5_INTAKE_REGISTRY
```

All paths above are caller-supplied reviewed destinations. Use one shared registry for the evaluation; a different registry cannot establish global duplicate detection. `--allow-synthetic` is mandatory for fabricated fixtures. Do not pass historical official CSV arguments to this profile. Optional `--checkpoint-file PATH` streams and hashes explicitly supplied bytes without deserialization; omission records `checkpoint_bytes_verified=false`. No matching checkpoint-byte success was tested without the private asset.

Intake verifies identities and every hash, matches exact filename/fold/candidate order, checks native scores and saved stable ranks, attaches WP5-held truths via the existing `_attach_truths`, and invokes existing complete-record validation. Output contains `report.json` with labelled confusions, `summary.json` with integer hits and fraction-scale Overall/Seen/Unseen/H metrics, and `seen_validated_packet.json` / `unseen_validated_packet.json` using the existing `wp5-model-output-v1` schema. These WP5-internal packets preserve every native FP64 input value through JSON round-trip and can feed `ModelOutputAdapter` / the later four-role import without re-exporting model outputs. Retain the immutable original native NPZ input packet as evidence. A hash-covered `receipt.json` is written last and records input pins, verifier source hashes and verification environment. No labels or reports are sent back to owner packets.

The output and registry must be disjoint directories; overlap is rejected before any handoff input is read or claim is created. The registry atomically reserves one WP2 role per manifest across output paths. An interrupted export/import has no successful completion receipt; interrupted imports retain an incomplete registry claim for explicit review. Do not delete or overwrite evidence to force a retry. Review the failed destination/claim and authorize recovery separately.

This command validates one role only. It cannot certify the four-model comparison, approve historical frozen CLIP reuse, accept old M6 scores as full-data results, or prove which model generated saved scores.

## Synthetic CI demonstration

From an integrated repository:

```powershell
python -B -m unittest discover -s tests -p test_full8000_integration.py -v
```

All generated names, labels and tensors are fabricated; no actual ConvNeXt forward is run. Set `WP2_TEST_ARTIFACTS` to a dedicated writable directory. The tests remove temporary score packets and never populate empirical result tables. No additional runtime framework or dependency installation is introduced.

The release-candidate staging runner additionally supports the existing synthetic regression selection and the byte-identical external WP5 audit gate. Environment-dependent full-repository/Katana/private-asset tests remain separate gates.
