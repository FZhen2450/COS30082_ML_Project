# Executable WP5 development import — 7 November 2026

This increment completes the file import and report presentation steps around
the existing reporting and four-model adapters. It adds no scorer, model loader,
training run, checkpoint selection or official-test reader. PR #18 remains draft.

## What was already complete

`evaluation/reporting.py` validates immutable candidates, provenance and exact
declared score precision, complete filename/truth/fold joins, stable original-ID
ties, fold denominators, Top-1/5, H@1/5, health and class diagnostics, rectangular
confusions, strict paired comparisons and JSON/CSV exports.
`evaluation/model_adapters.py` binds the four selected model roles, their
checkpoint/transform/prompt declarations and historical hash namespaces. Its
`import_model_packets` API already produces four model reports and six pairs.
Those APIs and canonical WP1/WP2/CLIP score semantics are unchanged.

## Required owner handoff

Use the already documented `wp5-owner-handoff-v1` JSON schema:

- `schema`: `wp5-owner-handoff-v1`.
- `owner_approval_reference`: nonempty reference to the actual owner approval.
- `identity_manifest`: exactly `path` and `file_sha256` for the approved serialized
  `EvaluationManifest`. Include original fitting and validation filename/truth
  records and explicit healthy disease membership.
- `manifest_sha256`: the manifest's semantic `.sha256`, distinct from its raw
  file SHA-256.
- `models`: exactly `wp1_independent_pair`, `wp2_convnext_base`,
  `clip_vit_l14_frozen`, `clip_vit_l14_adapter`. Each entry contains `provenance`
  accepted by `RunProvenance` and nonempty `packet_files`, each exactly
  `{path, file_sha256}`. Files contain existing `wp5-model-output-v1` packets.

Every packet must carry the approved manifest/source identities, full ordered
323-column global-ID header, actual original filenames and ground truths,
fold, checkpoint/run/transform/code/environment/scoring/precision declarations,
model-specific selection and applicable prompt digest. All four runs require
the agreed comparison budget. Supply every seen/unseen image exactly once per
model, possibly split across packet files. The independent pair additionally
identifies both head assets and their actual separate preprocessing.

Full canonical score rows or at least five ranked global IDs are accepted.
JSON scores must retain their canonical numerical values: use round-trip safe
serialization of the original precision; never re-round or reduce decimal
places. Raw JSON integers stay integers until exact-precision validation.
Rank-only imports preserve upstream tie decisions but cannot independently
prove those decisions; the producer must verify stable original-ID ordering.
Independent head predictions are optional and needed only for head accuracies.

Owners must independently confirm the raw SHA-256 of the outer handoff bundle
and each file. An internally self-consistent bundle does not authenticate its
author, unavailable weights, producer correctness or approval. Anonymous logits
cannot be assigned manifest filenames from counts or matching labels alone.
No real prediction handoff is present in this workspace.

## Run the development-only command

From the repository root, substitute actual owner-supplied paths and digest:

```powershell
python -B scripts/import_wp5_development.py `
  --handoff C:\approved\development-handoff.json `
  --handoff-sha256 <OWNER_CONFIRMED_RAW_SHA256> `
  --output C:\new-results\development-comparison
```

Paths inside the handoff resolve relative to its directory; absolute paths are
also explicit. `--source` may identify another exact checkout of
`DPD_Small_train.csv`; it defaults to this repository's training CSV. There is
no glob, directory scan, anonymous-output fallback or official-test input mode.
`--allow-synthetic` enables explicitly synthetic **development-323** fixtures
only; every table/figure remains labeled as a fixture, never empirical evidence.

Before any result writes the command:

1. Rejects an existing output path, verifies the outer raw digest and requires
   schema/approval. Rejects duplicate JSON keys and nonfinite JSON constants.
2. Verifies the manifest's raw and semantic digests, development mode and 323
   candidates. Final manifests are rejected before prediction packets are opened.
3. For real manifests, reconstructs the unchanged seed-42 training-source
   protocol in memory. Verifies all candidate IDs/pairs/labels, exact ordered
   fitting/seen/unseen filenames, every filename/truth join, source bytes and
   WP1 split fingerprint. Real counts remain 4,868 / 1,238 / 1,894. No artifact
   is rewritten or granted historical reconciliation approval by this check.
4. Verifies every packet file's raw digest, preserves decoded score/ID scalar
   types, then uses `import_model_packets` through `VerifiedResults.from_packets`.
   Any missing, duplicate, incompatible or misidentified model row fails closed.
5. Creates an immutable result snapshot and prepares every presentation artifact,
   including the headless plot, before exclusively creating the output directory.

The original reports use `export_report`. The final `receipt.json` is a
completion marker; it records owner reference, raw input paths/hashes/sizes,
manifest/candidate/result identities, fold counts, Git commit/dirty state,
relevant source byte hashes, dependency versions and every other output file's
raw SHA-256. The receipt does not hash itself. There is no timestamp or random
run ID. Reproducibility requires the recorded sources, dependencies and inputs;
plot rendering may differ across platforms/fonts. An I/O failure after directory
creation may leave incomplete files without a receipt. Preserve that directory,
diagnose the failure, and retry with a new explicit path; never resume over it.

## Report artifacts and interpretation

`VerifiedResults.from_packets(...)` runs the existing complete-record validator
and freezes its JSON result. `presentation_files(verified_results)` accepts only
this type and returns filename/bytes artifacts in memory. Passing arbitrary
result dictionaries is rejected; mutating a returned dictionary cannot change
the snapshot. This type prevents accidental bypass, not hostile Python code.

- Four model directories: original JSON/CSV reports, labeled confusion axes and
  all existing diagnostics.
- Six directories under `paired/`: original paired reports and exports.
- `comparison.csv` and `comparison.md`: seen/unseen Top-1/5, H@1/5 and each fold's
  denominator, with manifest/candidate/result provenance.
- `comparison.png`: two panels for Top-1 and Top-5, each showing seen, unseen and
  harmonic mean. Uses the existing matplotlib dependency, without downloads.
- `health_errors.csv`: healthy/diseased composition errors, denominators and
  false-healthy errors.
- `paired_wins.csv`: A-only/B-only/both/neither counts for each pair, fold and k.
- `per_class_errors.csv`: composition errors grouped by crop, disease and
  composition truth, denominators, fitting support and low-support flags.
- `caption.txt`: scope, source identity, complete digests, denominators,
  interpretations and references to full run provenance in `report.json`.

All accuracies are fractions [0,1]. H@k combines fold accuracies rather than
pooling images. Zero-support groups have zero error counts with denominator zero;
they do not establish zero accuracy. Grouped composition error rates are not
independent-head accuracies. Wins are paired outcomes, without significance or
causal claims. Captions distinguish **training-derived development**, separately
authorized **official test**, and **synthetic fixtures**. Synthetic always takes
precedence, including final-sized adapter fixtures. This CLI has no final mode.

## Native CLIP class-only preflight

Blocked in this environment: `clip` (OpenAI native CLIP), `ftfy` and `regex` are
not installed. Matplotlib 3.11.2 is available. No tokenizer was substituted and
no packages, model weights or tokenizer assets were downloaded. Native
`clip.tokenize` reproducibility and shape `[333,77]` therefore remain unverified.
The existing prompt JSON's original bytes and canonical SHA-256 are checked
separately; matching payload hashes is not a tokenizer test or evaluation approval.

## Verification and remaining dependencies

Actual CPU verification on 7 November 2026:

| Command / check | Result |
| --- | --- |
| `python -B -m unittest discover -s tests -p test_development_import.py -v` | 28/28 passed, 16.359 s |
| `python -B scripts/run_wp5_regressions.py` | 125/125 passed, 49.872 s |
| Existing reporting / adapters / metrics within that full suite | 39/39, 26/26, 6/6 passed |
| `python -B scripts/verify_wp5_bridge.py --wp5-root .` | Passed, synthetic compatibility only |
| `python -B scripts/verify_csv_checkout.py` | Passed; all three CSVs and historical manifest hashes intact |
| Separate `python -B handoffs/wp2_cpu_snapshot/test_snapshot_patch.py -v` | 3/3 passed, 0.383 s; uncommitted owner handoff |
| Raw bytes and canonical `verify_final_payloads` checks | Both Issue #8 digests match; `approval_granted=False` |
| `git diff --check` | Passed; expected LF-to-CRLF notice, no whitespace errors |

The full suite excludes exactly the existing optimizer-update tests
`test_separate_parameters_and_gradient_updates` and
`test_synthetic_training_records_and_inference_adapter`; their code is unchanged.
CPU environment disables CUDA and sets OMP/MKL/OpenBLAS thread counts to two.
The new tests cover complete exports/receipts, CLI execution from another working
directory, corruption/missing-file/missing-model failures, identity/truth/fold and
budget mismatches, duplicate/missing rows, raw versus semantic hashes, exact
declared precision, output preservation, plot failures before writes, immutable
verified snapshots, chart/table values, repeatability, clear scope captions,
training-source reconciliation and rank-only 323/333 interfaces. Generated
fixture outputs live in temporary directories; no synthetic result is published
as empirical performance. The synthetic PNG was also visually inspected for
caption/denominator/provenance legibility.

Unchanged reconstructed digests:

```text
candidate ba17e94119e8b2b7a4b18603e25405d2f52bf4a93396973f80491800697eb62f
prompt    c9dd871d61d4141b33583ddd71961c309fd679a8edab088cb28ba20eccef88f2
```

No CSV, `.gitattributes`, frozen JSON, existing scorer, adapter or test file was
changed. The integration guide replaces its manual recipe with the executable
workflow link. Git's expected autocrlf notices do not change the scoped CSV
attributes or historical CRLF checkout bytes.

The minimum assets for **reporting already-approved development predictions**
are the approved reconciled identity manifest, frozen provenance/selection and
identified packets for all four models, raw file hashes and the external owner
approval reference. Images and physical weights are unnecessary for import.
If new inference is authorized separately, owners also need provide the two
independent head weights, selected WP2 Base checkpoint, native ViT-L/14 base,
selected adapter checkpoint, verified native code/tokenizer and exact transforms.
The CLIP base can be shared by its two model roles.

WP2 CPU snapshot correction is supplied separately under the local, uncommitted
`handoffs/wp2_cpu_snapshot/` directory for owner review. Its regression applies
the patch only to a temporary copy of the exact pinned notebook Git blob. No
teammate branch or historical experiment is edited. Filename-aware production
export and real checkpoint declarations remain owner responsibilities.

Shared-333 final policy approval, actual approved official identity manifests,
native tokenization, frozen final prediction inputs and separate authorization
remain external release gates. Matching reconstructed class-only JSON hashes
does not clear them. No official predictions/images are consumed or evaluated.
