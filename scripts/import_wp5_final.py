"""Import owner packets for the final-333 official evaluation. Separately gated.

Needs three separate references in the handoff:
- the final policy approval (Issue #8);
- the official-evaluation authorisation;
- the owner handoff approval.

Before any packet is opened, it rebuilds the final manifest from hash-checked
inputs (development manifest, Issue #8 payloads, official metadata CSVs) and
requires an identical manifest SHA-256. For real (non-synthetic) imports it then
checks every model's *declared* checkpoint identity against the frozen pins below,
still before any packet is opened. Final packets must be label-free exports from
``inference_order``: exactly the full-score packet fields, no truths, no extra
fields and no rank-only rows. WP5 attaches truths from the manifest by filename,
and the existing adapter then validates every identity and requires each fold to
be complete. No image or model access.

Labels: the prediction packets are label-free, but this is not a label-blind
evaluation. Model owners can know official labels: DPD filenames embed crop and
disease names, and the course-supplied official CSVs are tracked in Git.

Checkpoints: the pins compare *declared provenance* (packet and handoff metadata)
with the frozen identities. They cannot prove which weights produced the scores.
Two separate requirements remain outside this importer: (1) trusted SHA-256
verification of the checkpoint files on the inference machine, recorded by the
owner before export, and (2) a post-run replay audit in which a second member
recomputes a pre-committed subset from the frozen files and compares scores.

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
from evaluation.model_adapters import (CLIP_ADAPTER, CLIP_FROZEN, CLIP_L14_SHA256, MODEL_KINDS, WP1_INDEPENDENT,
                                       WP2_CONVNEXT)
from evaluation.reporting import EvaluationManifest, RunProvenance, canonical_json, export_report
from evaluation.result_artifacts import VerifiedResults, presentation_files
from scripts.import_wp5_development import _digest, checked_json

REFERENCES = ("final_policy_reference", "evaluation_authorization_reference", "owner_approval_reference")
# Exactly the fields written by final333_predictors.export_blind_packet, a label-free (truth-free) export.
# prompt_sha256 is present for the CLIP models only.
PACKET_FIELDS = frozenset({
    "schema", "model_kind", "model_id", "run_id", "checkpoint_id", "checkpoint_sha256", "preprocessing",
    "code_version", "environment", "score_precision", "score_semantics", "manifest_sha256", "source_sha256",
    "candidate_ids", "candidate_hash_namespace", "candidate_sha256", "split_hash_namespace", "split_sha256",
    "selection", "fold", "image_names", "scores", "prompt_sha256"})
# Frozen selected identities (Issue #8 D5), equal to the verified WP5 development import provenance.
# WP2, CLIP and M6: raw SHA-256 of the checkpoint file, as pinned by the existing loaders
# (notebooks/wp2_selected_development_export.ipynb, wp4/clip_predictor.py).
# WP1: a WP5 fingerprint of the selected pair identity (scripts/export_wp1_development.py), not a file
# hash. The two head files are pinned separately in FROZEN_WP1_SELECTION by raw SHA-256.
FROZEN_CHECKPOINTS = {
    WP1_INDEPENDENT: "dfde18f00dbd001183598dca1552cfd345a35768c40b1a6d0ab4830a84ba27c8",
    WP2_CONVNEXT: "7a0460b7d03bc08d19b68ca06cdd266746953322149a3edad177cf29992d7c04",
    CLIP_FROZEN: CLIP_L14_SHA256,
    CLIP_ADAPTER: "8835a18b9eb933ae224bd17866f18bc56406e24d8612743f4e7c88848a8135cc",
}
FROZEN_WP1_SELECTION = {
    "crop_checkpoint_id": "wp1-release-20261106-final/crop/best.pt",
    "crop_checkpoint_sha256": "f0c82ca78153f492078a3a39a11eac1a847cc85eb496cf3b3d4d74db752c7d14",
    "crop_selected_epoch": 2,
    "disease_checkpoint_id": "wp1-release-20261106-final/disease/best.pt",
    "disease_checkpoint_sha256": "f497e19da62b68dfbcfbae58b9a5318a26b1989ba59b7a486223cad0d56e31a5",
    "disease_selected_epoch": 3,
}
CHECKPOINT_IDENTITY_SCOPE = ("declared provenance matched the frozen pins; checkpoint bytes are not verified by "
                             "this importer (owner on-machine SHA-256 log and replay audit are separate gates)")


def check_frozen_identity(kind, provenance, packets):
    """Compare *declared* identities with the frozen pins (real final imports only).

    This proves that the owner declared the selected checkpoints. It does not
    prove which weights produced the scores: that needs the owner's on-machine
    file-hash record and the replay audit.
    """
    if provenance.checkpoint_sha256 != FROZEN_CHECKPOINTS[kind]:
        raise ValueError(f"{kind}: declared checkpoint differs from the frozen selected model")
    if kind == WP1_INDEPENDENT:
        for packet in packets:
            selection = packet.get("selection")
            if (not isinstance(selection, dict)
                    or any(selection.get(k) != v for k, v in FROZEN_WP1_SELECTION.items())):
                raise ValueError(f"{kind}: declared crop/disease checkpoints differ from the frozen selected heads")


def _attach_truths(packet, truths):
    """Attach WP5-held truths to a label-free packet. Errors never echo class labels."""
    if not isinstance(packet, dict):
        raise ValueError("Final prediction packet must be a JSON object")
    if "ground_truth_ids" in packet:
        raise ValueError("Final packets must be label-free: ground_truth_ids present. Stop and report before import")
    unexpected = sorted(set(packet) - PACKET_FIELDS)
    if unexpected or "scores" not in packet:
        raise ValueError(f"Final packets carry exactly the full-score export fields; unexpected: {unexpected}")
    names = packet.get("image_names")
    if not isinstance(names, list) or any(not isinstance(n, str) or n not in truths for n in names):
        raise ValueError("Packet image names are missing or outside the final manifest")
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
    provenances, packets, attached_counts = {}, {}, Counter()
    for kind in MODEL_KINDS:  # identities first: no packet is opened for a substituted model
        entry = models[kind]
        if not isinstance(entry, dict) or set(entry) != {"provenance", "packet_files"}:
            raise ValueError("Each model requires provenance and packet_files")
        if not isinstance(entry["packet_files"], list) or not entry["packet_files"]:
            raise ValueError("Each model requires identified prediction packet files")
        provenances[kind] = RunProvenance(**entry["provenance"])
        if not manifest.spec.synthetic:
            check_frozen_identity(kind, provenances[kind], [])
    for kind in MODEL_KINDS:
        packets[kind] = []
        for i, file in enumerate(models[kind]["packet_files"]):
            if not isinstance(file, dict) or set(file) != {"path", "file_sha256"}:
                raise ValueError("File entries require exactly path and file_sha256")
            raw = checked_json(handoff.parent / file["path"], file["file_sha256"], ledger, f"{kind}:packet:{i}")
            packet, attached = _attach_truths(raw, truths)
            attached_counts[kind] += attached
            packets[kind].append(packet)
        if not manifest.spec.synthetic:
            check_frozen_identity(kind, provenances[kind], packets[kind])
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
               "label_free_packets_truth_attached": dict(attached_counts),
               "label_handling": "label-free prediction packets; not a label-blind evaluation",
               "checkpoint_identity_scope": CHECKPOINT_IDENTITY_SCOPE if not manifest.spec.synthetic
               else "synthetic fixture; frozen pins not applied", "verified_inputs": ledger,
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


def import_full8000_wp2(handoff_path, handoff_sha256, output_path, *, registry_path,
                        allow_synthetic=False, checkpoint_file=None):
    """Versioned WP2-only intake; historical four-model pins remain unchanged."""
    from evaluation.full8000_packets import import_full8000_wp2 as implementation
    return implementation(handoff_path, handoff_sha256, output_path, registry_path=registry_path,
                          allow_synthetic=allow_synthetic, checkpoint_file=checkpoint_file)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff", type=Path, required=True)
    parser.add_argument("--handoff-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", choices=("historical-final333-v1", "full8000-20261112-v1"),
                        default="historical-final333-v1", help="Explicit new protocol; historical default preserved")
    parser.add_argument("--official-seen", type=Path)
    parser.add_argument("--official-unseen", type=Path)
    parser.add_argument("--registry", type=Path, help="Required for full-8000 duplicate-import protection")
    parser.add_argument("--checkpoint-file", type=Path, help="Optional explicit checkpoint byte hashing; never loaded")
    parser.add_argument("--allow-synthetic", action="store_true", help="Explicitly labeled test fixtures only")
    args = parser.parse_args(argv)
    try:
        if args.protocol == "full8000-20261112-v1":
            if args.registry is None:
                parser.error("--registry is required for full-8000 intake")
            if args.official_seen is not None or args.official_unseen is not None:
                parser.error("Full-8000 intake uses the authenticated common manifest, not historical CSV rebuilding")
            receipt = import_full8000_wp2(args.handoff, args.handoff_sha256, args.output, registry_path=args.registry,
                                          allow_synthetic=args.allow_synthetic, checkpoint_file=args.checkpoint_file)
        else:
            if args.official_seen is None or args.official_unseen is None or args.checkpoint_file is not None:
                parser.error("Historical intake requires both official CSV paths; optional byte hashing is full-8000 only")
            receipt = import_final(args.handoff, args.handoff_sha256, args.output, seen_csv=args.official_seen,
                                   unseen_csv=args.official_unseen, allow_synthetic=args.allow_synthetic)
    except (OSError, ValueError, TypeError, KeyError, subprocess.SubprocessError) as error:
        parser.exit(1, f"WP5 final import failed: {error}\n")
    if args.protocol == "full8000-20261112-v1":
        print(f"WP2 saved-score intake validated; four-model acceptance pending; "
              f"synthetic={receipt['synthetic']}; receipt={args.output / 'receipt.json'}")
    else:
        print(f"WP5 final import complete; synthetic={receipt['synthetic']}; receipt={args.output / 'receipt.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
