"""Export the selected WP1 pair's named development predictions for WP5.

Inference only. Keeps the historical loader/version guard and float32 scorer.
Use an explicit WP5 checkout and a new output directory. No official inputs.
"""
import argparse
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wp5-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--release", type=Path, default=ROOT / "runs/wp1-release-20261106-final")
    parser.add_argument("--protocol", type=Path, default=ROOT / "development_seed42_reconciled.json")
    parser.add_argument("--images", type=Path, default=ROOT / "DPD_Small_dataset")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserving existing outputs; select a new directory")
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(args.wp5_root.resolve(strict=True)))
    import torch
    import torchvision
    from PIL import Image
    from independent_baseline import IndependentPredictor
    from shared_protocol import validate_protocol
    from evaluation.model_adapters import ModelOutputAdapter, WP1_INDEPENDENT, WP1_SPLIT
    from evaluation.reporting import (RunProvenance, build_report, canonical_json,
                                      development_manifest, export_report, fingerprint)

    torch.set_num_threads(4)
    source = ROOT / "DPD_Small_train.csv"
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    original = json.loads((args.release / "handoff.json").read_text(encoding="utf-8"))
    frames = validate_protocol(protocol, source)
    manifest = development_manifest(protocol, source, healthy_disease_names=("healthy",))
    source_hashes = {}
    selected = {}
    for task, epoch in (("crop", 2), ("disease", 3)):
        entry = original["checkpoints"][task]
        checkpoint = args.release / task / "best.pt"
        if digest(checkpoint) != entry["sha256"] or entry["metadata"]["selected_epoch"] != epoch:
            raise ValueError("Selected checkpoint identity differs from original release")
        for name, expected in entry["metadata"]["code_sha256"].items():
            if digest(ROOT / name) != expected:
                raise ValueError("Release loader source bytes differ: " + name)
            source_hashes[name] = expected
        selected[task] = {"checkpoint_id": f"wp1-release-20261106-final/{task}/best.pt",
                          "checkpoint_sha256": entry["sha256"], "selected_epoch": epoch,
                          "preprocessing": entry["metadata"]["preprocessing"]}
    predictor = IndependentPredictor(args.release / "crop/best.pt", args.release / "disease/best.pt",
                                     args.protocol, source, args.device)
    runtime = {"python": platform.python_version(), "torch": str(torch.__version__),
               "torchvision": str(torchvision.__version__), "device": args.device,
               "hardware": torch.cuda.get_device_name() if args.device == "cuda" else platform.processor()}
    pair_identity = {task: {k: v for k, v in item.items() if k != "preprocessing"}
                     for task, item in selected.items()}
    provenance = RunProvenance(
        model_id=WP1_INDEPENDENT, run_id="wp1-release-20261106-final-named-export-20261107",
        checkpoint_id="wp1-release-20261106-final/crop-epoch2+disease-epoch3",
        checkpoint_sha256=fingerprint(pair_identity),
        preprocessing=canonical_json({task: item["preprocessing"] for task, item in selected.items()}),
        code_version=canonical_json({"release_commit": original["checkpoints"]["crop"]["metadata"]["code_commit"],
                                     "release_code_dirty": True, "source_sha256": source_hashes,
                                     "exporter_sha256": digest(__file__)}),
        environment=canonical_json(runtime), score_semantics="sum_component_log_probabilities",
        score_precision="float32",
        comparison_budget="Complete reconciled development folds; one frozen selected run per model; no reselection")
    selection = {}
    for task, item in selected.items():
        for key in ("checkpoint_id", "checkpoint_sha256"):
            selection[task + "_" + key] = item[key]
        selection[task + "_preprocessing"] = canonical_json(item["preprocessing"])
        selection[task + "_selected_epoch"] = item["selected_epoch"]
    adapter = ModelOutputAdapter(manifest, provenance, WP1_INDEPENDENT)
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "identity_manifest.json", manifest.to_dict())
    write_json(args.output / "provenance.json", asdict(provenance))
    write_json(args.output / "selected_pair_identity.json", pair_identity)
    all_records, summaries = [], {}
    for fold, frame in zip(("seen", "unseen"), frames[1:]):
        matrices, names, truths, ranks = [], [], [], []
        started = time.perf_counter()
        for start in range(0, len(frame), 2):
            rows = frame.iloc[start:start + 2]
            images = []
            # Names and truths travel with the exact batch actually decoded.
            for row in rows.itertuples(index=False):
                with Image.open(args.images / row.image_name) as image:
                    images.append(image.convert("RGB"))
                names.append(str(row.image_name))
                truths.append(int(row.composition_id))
            output = predictor.predict(images, topk=5)
            scores = output["scores"].detach().clone()
            if scores.dtype != torch.float32 or not torch.isfinite(scores).all():
                raise ValueError("Unexpected canonical score dtype/value")
            matrices.append(scores)
            ranks.extend(output["topk_composition_ids"].tolist())
            if (start + 2) % 200 == 0:
                print(f"{fold}: {min(start + 2, len(frame))}/{len(frame)}", flush=True)
        matrix = torch.cat(matrices)
        ids = output["candidate_ids"]
        if ids.tolist() != list(manifest.spec.ids):
            raise ValueError("Candidate columns differ from manifest")
        if ids[matrix.argsort(dim=1, descending=True, stable=True)[:, :5]].tolist() != ranks:
            raise ValueError("Canonical score and predictor rank mismatch")
        packet = {"schema": "wp5-model-output-v1", "model_kind": WP1_INDEPENDENT,
                  **{k: getattr(provenance, k) for k in ("model_id", "run_id", "checkpoint_id", "checkpoint_sha256",
                      "preprocessing", "code_version", "environment", "score_precision", "score_semantics", "prompt_sha256")},
                  "manifest_sha256": manifest.sha256, "source_sha256": protocol["source_sha256"],
                  "candidate_ids": ids.tolist(), "candidate_sha256": protocol["candidate_sha256"],
                  "candidate_hash_namespace": manifest.spec.candidate_hash_namespace,
                  "split_sha256": protocol["split_sha256"], "split_hash_namespace": WP1_SPLIT,
                  "selection": selection, "fold": fold, "image_names": names,
                  "ground_truth_ids": truths, "scores": matrix.tolist()}
        path = args.output / (fold + "_packet.json")
        write_json(path, packet)
        # Validate decoded JSON, not merely the in-memory tensors.
        records = adapter.adapt(json.loads(path.read_text(encoding="utf-8")))
        if [list(r.ranking()[:5]) for r in records] != ranks:
            raise ValueError("JSON/WP5 ranking round trip changed results")
        all_records.extend(records)
        with (args.output / (fold + "_predictions.csv")).open("x", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["image_name", "fold", "ground_truth_id", "rank1", "rank2", "rank3", "rank4", "rank5"])
            writer.writerows([name, fold, truth, *rank] for name, truth, rank in zip(names, truths, ranks))
        summaries[fold] = {"count": len(names), "seconds": time.perf_counter() - started}
        print(f"{fold}: complete ({len(names)} rows)", flush=True)
    report = build_report(manifest, all_records)
    for fold in ("seen", "unseen"):
        old = original["development_results"][fold]
        if any(report[fold][key] != old[key] for key in ("count", "top1", "top5")):
            raise ValueError("Fresh named inference differs from historical aggregate; investigate before release")
    export_report(report, args.output / "wp5_report")
    write_json(args.output / "run_details.json", {"schema": "wp1-named-development-export-v1",
        "runtime": runtime, "folds": summaries, "selected_pair": selected,
        "checkpoint_pair_digest_semantics": "WP5 fingerprint of selected_pair_identity.json semantic object; not a weight-file hash",
        "manifest_sha256": manifest.sha256, "historical_metrics_exact_match": True,
        "official_test_evaluated": False, "training_run": False,
        "comparison_budget_status": "Producer proposal; coordinate same declaration with other model owners before four-model import",
        "portability": "Original strict torchvision 0.22.1+cu118 guard retained. Use matching isolated environment; no Katana 0.20.1 compatibility claimed."})
    snapshot = args.output / "source"
    snapshot.mkdir()
    for name in (*source_hashes, "evaluate_independent.py", "scripts/export_wp1_development.py",
                 "scripts/run_wp1_release.py", "configs/wp1/requirements_cuda118.txt", "configs/wp1/resnet50_rtx3050.json"):
        target = snapshot / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    shutil.copyfile(args.protocol, args.output / "development_seed42_reconciled.json")
    for name in ("handoff.json", "development.json"):
        shutil.copyfile(args.release / name, args.output / ("original_" + name))
    for task in selected:
        if digest(args.release / task / "best.pt") != selected[task]["checkpoint_sha256"]:
            raise ValueError("Checkpoint changed during export")
    for name, expected in source_hashes.items():
        if digest(ROOT / name) != expected or digest(snapshot / name) != expected:
            raise ValueError("Source changed during export")
    files = {str(path.relative_to(args.output)).replace("\\", "/"): digest(path)
             for path in sorted(args.output.rglob("*")) if path.is_file()}
    write_json(args.output / "receipt.json", {"complete": True, "files_sha256": files,
               "wp5_source_sha256": {str(p.relative_to(args.wp5_root)): digest(p) for p in
                   (args.wp5_root / "evaluation/reporting.py", args.wp5_root / "evaluation/model_adapters.py")},
               "official_test_evaluated": False})
    print(json.dumps({"output": str(args.output), "rows": len(all_records),
                      "H1": report["harmonic_mean_top1"], "H5": report["harmonic_mean_top5"]}), flush=True)


if __name__ == "__main__":
    main()
