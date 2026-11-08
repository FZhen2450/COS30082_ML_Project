"""Import owner-approved development-323 packets; no image/model access.

Paths in wp5-owner-handoff-v1 are relative to the handoff JSON, unless absolute.
The handoff's raw SHA-256 must be obtained independently from its owner.
"""
import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from collections import Counter
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.model_adapters import MODEL_KINDS, WP1_SPLIT, output_split_digest
from evaluation.reporting import EvaluationManifest, RunProvenance, canonical_json, export_report
from evaluation.result_artifacts import VerifiedResults, presentation_files


def _digest(raw):
    return hashlib.sha256(raw).hexdigest()


def _sha(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("Require an owner-approved lowercase SHA-256")
    return value


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field: {key}")
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError(f"Nonfinite JSON value: {value}")


def checked_json(path, expected_sha256, ledger, role):
    path = Path(path).resolve(strict=True)
    raw = path.read_bytes()
    if _digest(raw) != _sha(expected_sha256):
        raise ValueError(f"Owner-approved raw file SHA-256 mismatch: {path}")
    ledger.append({"role": role, "path": str(path), "file_sha256": expected_sha256, "bytes": len(raw)})
    # Preserve original integer/float scalars for the exact-precision boundary.
    return json.loads(raw, object_pairs_hook=_unique_object, parse_constant=_invalid_constant)


def validate_development_source(manifest, source_csv):
    """Verify exact seed-42 course-training candidates, fold order and truths.

    Reconstruct in memory only. Owner approval remains external; matching the
    deterministic source does not approve a historical reconciliation artifact.
    """
    from shared_protocol import FOLDS, build_protocol, validate_protocol
    path = Path(source_csv).resolve(strict=True)
    if path.name != "DPD_Small_train.csv":
        raise ValueError("Only explicit course training metadata DPD_Small_train.csv is permitted")
    protocol = build_protocol(path, seed=42)
    if manifest.spec.source_sha256 != protocol["source_sha256"]:
        raise ValueError("Training-source raw hash differs from the approved manifest")
    if [asdict(c) for c in manifest.spec.candidates] != protocol["candidates"]:
        raise ValueError("Development candidate identities/labels/order differ from training source")
    for key, fold in zip(FOLDS, ("fitting", "seen", "unseen")):
        if [s.sample_id for s in manifest.samples if s.fold == fold] != protocol[key]:
            raise ValueError(f"Development seed-42 ordered filename fold differs: {fold}")
    truths = {str(row.image_name): int(row.composition_id)
              for frame in validate_protocol(protocol, path) for row in frame.itertuples(index=False)}
    if any(truths.get(s.sample_id) != s.truth_id for s in manifest.samples):
        raise ValueError("Original filename/ground-truth join differs from training source")
    if (manifest.spec.split_hash_namespace != WP1_SPLIT
            or manifest.spec.split_sha256 != output_split_digest(manifest, WP1_SPLIT)):
        raise ValueError("Development spec split fingerprint/namespace differs from its ordered identities")
    return {"role": "training_source", "path": str(path), "file_sha256": protocol["source_sha256"],
            "bytes": path.stat().st_size}


def import_development(handoff_path, handoff_sha256, output_path, *,
                       source_csv=ROOT / "DPD_Small_train.csv", allow_synthetic=False):
    """Validate all inputs and prepare artifacts before exclusively creating output.

    receipt.json is written last and is the completion marker. An I/O failure may
    leave an incomplete new directory; it is preserved and cannot be reused.
    """
    requested_output = Path(output_path)
    if os.path.lexists(requested_output):
        raise FileExistsError(f"Preserving existing output: {requested_output}")
    output = requested_output.resolve()
    handoff = Path(handoff_path).resolve(strict=True)
    ledger = []
    bundle = checked_json(handoff, handoff_sha256, ledger, "owner_handoff")
    if not isinstance(bundle, dict) or bundle.get("schema") != "wp5-owner-handoff-v1":
        raise ValueError("Expected wp5-owner-handoff-v1")
    approval = bundle.get("owner_approval_reference")
    if not isinstance(approval, str) or not approval.strip():
        raise ValueError("Explicit owner approval reference required")

    def read_entry(entry, role):
        if not isinstance(entry, dict) or set(entry) != {"path", "file_sha256"}:
            raise ValueError("File entries require exactly path and file_sha256")
        if not isinstance(entry["path"], str) or not entry["path"].strip():
            raise ValueError("Explicit file path required")
        return checked_json(handoff.parent / entry["path"], entry["file_sha256"], ledger, role)

    manifest = EvaluationManifest.from_dict(read_entry(bundle["identity_manifest"], "identity_manifest"))
    # Reject final tasks before opening any prediction packets.
    if manifest.spec.task_mode != "development" or len(manifest.spec.ids) != 323:
        raise ValueError("This command accepts development-323 only; final evaluation remains separately gated")
    if manifest.spec.synthetic and not allow_synthetic:
        raise ValueError("Synthetic fixtures require explicit --allow-synthetic; never real performance")
    if manifest.sha256 != bundle.get("manifest_sha256"):
        raise ValueError("Owner-approved semantic manifest SHA-256 mismatch")
    if not manifest.spec.synthetic:
        ledger.append(validate_development_source(manifest, source_csv))
    models = bundle.get("models")
    if not isinstance(models, dict) or set(models) != set(MODEL_KINDS):
        raise ValueError("Require exactly all four selected models")
    provenances, packets = {}, {}
    for kind in MODEL_KINDS:
        entry = models[kind]
        if not isinstance(entry, dict) or set(entry) != {"provenance", "packet_files"}:
            raise ValueError("Each model requires provenance and packet_files")
        files = entry["packet_files"]
        if not isinstance(files, list) or not files:
            raise ValueError("Each model requires approved identified prediction packet files")
        provenances[kind] = RunProvenance(**entry["provenance"])
        packets[kind] = [read_entry(file, kind + f":packet:{i}") for i, file in enumerate(files)]
    verified = VerifiedResults.from_packets(manifest, provenances, packets)
    imported = verified.to_dict()
    presentation = presentation_files(verified)
    code = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain"], text=True))
    receipt = {"schema": "wp5-development-import-receipt-v1", "owner_approval_reference": approval,
               "task_mode": "development", "synthetic": manifest.spec.synthetic,
               "manifest_sha256": manifest.sha256, "candidate_sha256": manifest.spec.candidate_sha256,
               "results_sha256": verified.sha256, "fold_counts": dict(Counter(s.fold for s in manifest.samples)),
               "verified_inputs": ledger, "code_commit": code, "working_tree_dirty": dirty,
               "code_source_sha256": {name: _digest((ROOT / name).read_bytes()) for name in
                   ("scripts/import_wp5_development.py", "evaluation/result_artifacts.py",
                    "evaluation/model_adapters.py", "evaluation/reporting.py", "evaluation/metrics.py",
                    "shared_protocol.py", "data_utils.py")},
               "environment": {"python": platform.python_version(),
                               **{name: version(name) for name in ("numpy", "torch", "pandas", "matplotlib")}}}
    # Every semantic validation and all plotting completed before any result write.
    output.mkdir(parents=True, exist_ok=False)
    for kind, report in imported["models"].items():
        export_report(report, output / kind)
    for pair, report in imported["paired"].items():
        export_report(report, output / "paired" / pair)
    for name, raw in presentation.items():
        with (output / name).open("xb") as stream:
            stream.write(raw)
    receipt["outputs"] = [{"path": path.relative_to(output).as_posix(), "file_sha256": _digest(path.read_bytes())}
                          for path in sorted(output.rglob("*")) if path.is_file()]
    with (output / "receipt.json").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(canonical_json(receipt) + "\n")
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff", type=Path, required=True)
    parser.add_argument("--handoff-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=ROOT / "DPD_Small_train.csv")
    parser.add_argument("--allow-synthetic", action="store_true", help="Explicitly labeled test fixtures only")
    args = parser.parse_args(argv)
    try:
        receipt = import_development(args.handoff, args.handoff_sha256, args.output,
                                     source_csv=args.source, allow_synthetic=args.allow_synthetic)
    except (OSError, ValueError, TypeError, KeyError, subprocess.SubprocessError) as error:
        parser.exit(1, f"WP5 import failed: {error}\n")
    print(f"WP5 import complete; synthetic={receipt['synthetic']}; receipt={args.output / 'receipt.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
