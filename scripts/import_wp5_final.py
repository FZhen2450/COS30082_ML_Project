"""Import owner packets for the final-333 official evaluation. Separately gated.

Needs three separate references in the handoff:
- the final policy approval (Issue #8);
- the official-evaluation authorisation;
- the owner handoff approval.

Before any packet is opened, it rebuilds the final manifest from hash-checked
inputs (development manifest, Issue #8 payloads, official metadata CSVs) and
requires an identical manifest SHA-256. Final packets must be blind exports from
``inference_order``: a packet that carries ``ground_truth_ids`` is rejected,
because it shows that official labels reached a model owner. WP5 attaches truths
from the manifest by filename, and the existing adapter then validates every
identity and requires each fold to be complete. No image or model access.

Protocol status: the shared-333 candidate universe is the team-proposed protocol
recorded in Issue #8, not a confirmed lecturer requirement. Its status is carried
by ``final_policy_reference``; this code does not assert it.
"""
import argparse
import os
import platform
import subprocess
import sys
from collections import Counter
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.confusion_artifacts import confusion_files
from evaluation.final_manifest import final_manifest
from evaluation.model_adapters import MODEL_KINDS
from evaluation.reporting import EvaluationManifest, RunProvenance, canonical_json, export_report
from evaluation.result_artifacts import VerifiedResults, presentation_files
from scripts.import_wp5_development import _digest, checked_json

REFERENCES = ("final_policy_reference", "evaluation_authorization_reference", "owner_approval_reference")


def _attach_truths(packet, truths):
    """Attach WP5-held truths to a blind packet. Errors never echo names or labels."""
    if not isinstance(packet, dict):
        raise ValueError("Final prediction packet must be a JSON object")
    if "ground_truth_ids" in packet:
        raise ValueError("Final packets must be blind: ground_truth_ids present, so official labels "
                         "reached a model owner. Stop and report before any import")
    names = packet.get("image_names")
    if not isinstance(names, list) or any(not isinstance(n, str) or n not in truths for n in names):
        raise ValueError("Blind packet names are missing or outside the final manifest")
    return {**packet, "ground_truth_ids": [truths[n] for n in names]}, True


def import_final(handoff_path, handoff_sha256, output_path, *, seen_csv, unseen_csv, allow_synthetic=False):
    requested_output = Path(output_path)
    if os.path.lexists(requested_output):
        raise FileExistsError(f"Preserving existing output: {requested_output}")
    output = requested_output.resolve()
    handoff = Path(handoff_path).resolve(strict=True)
    ledger = []
    bundle = checked_json(handoff, handoff_sha256, ledger, "final_owner_handoff")
    if not isinstance(bundle, dict) or bundle.get("schema") != "wp5-final-owner-handoff-v1":
        raise ValueError("Expected wp5-final-owner-handoff-v1")
    for key in REFERENCES:
        if not isinstance(bundle.get(key), str) or not bundle[key].strip():
            raise ValueError(f"Explicit {key} required")

    def read_entry(name):
        entry = bundle.get(name)
        if not isinstance(entry, dict) or set(entry) != {"path", "file_sha256"}:
            raise ValueError(f"{name} requires exactly path and file_sha256")
        return checked_json(handoff.parent / entry["path"], entry["file_sha256"], ledger, name)

    manifest = EvaluationManifest.from_dict(read_entry("identity_manifest"))
    if manifest.spec.task_mode != "final" or len(manifest.spec.ids) != 333:
        raise ValueError("This command accepts final-333 only")
    if manifest.spec.synthetic and not allow_synthetic:
        raise ValueError("Synthetic fixtures require explicit --allow-synthetic; never real performance")
    if manifest.sha256 != bundle.get("manifest_sha256"):
        raise ValueError("Owner-approved semantic manifest SHA-256 mismatch")
    official_sha = bundle.get("official_csv_sha256")
    rebuilt = final_manifest(EvaluationManifest.from_dict(read_entry("development_manifest")), seen_csv, unseen_csv,
                             candidate_payload=read_entry("final_candidates"),
                             prompt_payload=read_entry("final_prompts"),
                             expected_sha256=official_sha, approval_reference=manifest.spec.approval_reference,
                             evaluation_authorization=bundle["evaluation_authorization_reference"],
                             synthetic=manifest.spec.synthetic)
    if rebuilt.sha256 != manifest.sha256:
        raise ValueError("Final manifest does not reproduce from its hash-checked sources")
    ledger += [{"role": "official_" + fold, "file_sha256": official_sha[fold]} for fold in ("seen", "unseen")]

    truths = {s.sample_id: s.truth_id for s in manifest.samples if s.fold in ("seen", "unseen")}
    models = bundle.get("models")
    if not isinstance(models, dict) or set(models) != set(MODEL_KINDS):
        raise ValueError("Require exactly all four selected models")
    provenances, packets, blind = {}, {}, Counter()
    for kind in MODEL_KINDS:
        entry = models[kind]
        if not isinstance(entry, dict) or set(entry) != {"provenance", "packet_files"}:
            raise ValueError("Each model requires provenance and packet_files")
        files = entry["packet_files"]
        if not isinstance(files, list) or not files:
            raise ValueError("Each model requires identified prediction packet files")
        provenances[kind] = RunProvenance(**entry["provenance"])
        packets[kind] = []
        for i, file in enumerate(files):
            if not isinstance(file, dict) or set(file) != {"path", "file_sha256"}:
                raise ValueError("File entries require exactly path and file_sha256")
            raw = checked_json(handoff.parent / file["path"], file["file_sha256"], ledger, f"{kind}:packet:{i}")
            packet, attached = _attach_truths(raw, truths)
            blind[kind] += attached
            packets[kind].append(packet)
    verified = VerifiedResults.from_packets(manifest, provenances, packets,
                                            final_evaluation_reference=bundle["evaluation_authorization_reference"])
    imported = verified.to_dict()
    presentation = presentation_files(verified)
    for kind in MODEL_KINDS:
        for fold in ("seen", "unseen"):
            for name, raw in confusion_files(imported["models"][kind], fold=fold,
                                             title=f"{kind}: official {fold} test, Top-1").items():
                presentation[f"confusion/{kind}/{name}"] = raw
    code = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain"], text=True))
    receipt = {"schema": "wp5-final-import-receipt-v1", **{key: bundle[key] for key in REFERENCES},
               "task_mode": "final", "synthetic": manifest.spec.synthetic,
               "manifest_sha256": manifest.sha256, "candidate_sha256": manifest.spec.candidate_sha256,
               "results_sha256": verified.sha256, "fold_counts": dict(Counter(s.fold for s in manifest.samples)),
               "blind_packets_truth_attached": dict(blind), "verified_inputs": ledger,
               "code_commit": code, "working_tree_dirty": dirty,
               "code_source_sha256": {name: _digest((ROOT / name).read_bytes()) for name in
                   ("scripts/import_wp5_final.py", "evaluation/final_manifest.py", "evaluation/confusion_artifacts.py",
                    "evaluation/result_artifacts.py", "evaluation/model_adapters.py", "evaluation/reporting.py",
                    "evaluation/metrics.py")},
               "environment": {"python": platform.python_version(),
                               **{name: version(name) for name in ("numpy", "matplotlib")}}}
    output.mkdir(parents=True, exist_ok=False)
    for kind, report in imported["models"].items():
        export_report(report, output / kind)
    for pair, report in imported["paired"].items():
        export_report(report, output / "paired" / pair)
    for name, raw in presentation.items():
        (output / name).parent.mkdir(parents=True, exist_ok=True)
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
    parser.add_argument("--official-seen", type=Path, required=True)
    parser.add_argument("--official-unseen", type=Path, required=True)
    parser.add_argument("--allow-synthetic", action="store_true", help="Explicitly labeled test fixtures only")
    args = parser.parse_args(argv)
    try:
        receipt = import_final(args.handoff, args.handoff_sha256, args.output, seen_csv=args.official_seen,
                               unseen_csv=args.official_unseen, allow_synthetic=args.allow_synthetic)
    except (OSError, ValueError, TypeError, KeyError, subprocess.SubprocessError) as error:
        parser.exit(1, f"WP5 final import failed: {error}\n")
    print(f"WP5 final import complete; synthetic={receipt['synthetic']}; receipt={args.output / 'receipt.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
