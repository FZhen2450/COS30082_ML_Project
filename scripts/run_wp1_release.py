"""Train a declared independent baseline pair and write its private handoff.

Uses only the reconciled training-derived protocol, never official test CSVs.
Choose a new output directory for every invocation; existing runs are preserved.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from PIL import Image

from independent_baseline import IndependentPredictor, train
from evaluate_independent import evaluate
from shared_protocol import validate_protocol


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2) + "\n")


def release(config_path, protocol_path, source, images, output, device, image_preflight=None, completed_crop=None):
    config = json.loads(config_path.read_text(encoding="utf-8"))
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    folds = validate_protocol(protocol, source)
    if protocol["historical_reconciliation"] != "ordered_filenames_match_supplied_reference":
        raise ValueError("Reconcile ordered development filenames first")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "plan.json", {
        "config": config, "protocol": str(protocol_path.resolve()),
        "counts": protocol["counts"], "candidate_count": len(protocol["candidates"]),
        "architecture_policy": "Single predeclared backbone; no four-backbone search claimed",
        "checkpoint_selection": "seen-validation component Top-1, earliest epoch on ties",
        "official_test_evaluated": False,
        "completed_crop_reused_from": str(completed_crop.resolve()) if completed_crop else None,
    })
    # Decode only the original training-source images, before any model fitting.
    if image_preflight:
        audit = json.loads(image_preflight.read_text(encoding="utf-8"))
        if (audit.get("decoded") != sum(len(frame) for frame in folds)
                or audit.get("failures") != []
                or audit.get("source_sha256") != protocol["source_sha256"]
                or audit.get("official_test_images_read") is not False):
            raise ValueError("Previous image preflight does not match the training source")
        if audit.get("images_directory", str(images.resolve())) != str(images.resolve()):
            raise ValueError("Previous image preflight used another image directory")
        # Only reuse a same-session audit while the image directory is unchanged.
        # Every training/evaluation image is still decoded when consumed.
        audit = {**audit, "reused_from": str(image_preflight.resolve())}
        decoded = audit["decoded"]
    else:
        decoded = 0
        for frame in folds:
            for name in frame.image_name:
                with Image.open(images / name) as image:
                    image.convert("RGB").load()
                decoded += 1
                if decoded % 1000 == 0:
                    print(f"Image preflight: {decoded} decoded", flush=True)
        audit = {"decoded": decoded, "failures": [], "source_sha256": protocol["source_sha256"],
                 "images_directory": str(images.resolve()), "official_test_images_read": False}
    write_json(output / "image_preflight.json", audit)
    print(f"Preflight passed: all {decoded} development-source images decoded", flush=True)
    for task in ("crop", "disease"):
        if task == "crop" and completed_crop:
            previous = json.loads((completed_crop / "summary.json").read_text(encoding="utf-8"))
            history = json.loads((completed_crop / "history.json").read_text(encoding="utf-8"))
            payload = torch.load(completed_crop / "best.pt", map_location="cpu", weights_only=True)
            metadata = payload["metadata"]
            if (previous["epochs_completed"] != config["epochs"] or len(history) != config["epochs"]
                    or metadata["task"] != "crop" or metadata["config"] != config):
                raise ValueError("Previous crop run is incomplete or uses another configuration")
            for key in ("candidate_sha256", "split_sha256", "source_sha256"):
                if metadata[key] != protocol[key]:
                    raise ValueError("Previous crop run uses another development protocol")
            selected = max(history, key=lambda row: (row["seen_validation"]["top1"], -row["epoch"]))
            if selected["epoch"] != metadata["selected_epoch"]:
                raise ValueError("Previous crop checkpoint does not match its selection history")
            del payload
            (output / task).mkdir()
            for name in ("best.pt", "history.json", "run.json", "summary.json"):
                shutil.copy2(completed_crop / name, output / task / name)
            print("Reused verified completed crop run", flush=True)
            continue
        print(f"Training independent {task} model", flush=True)
        train(config, task, protocol_path, source, images, output / task, device)
    predictor = IndependentPredictor(output / "crop/best.pt", output / "disease/best.pt",
                                     protocol_path, source, device)
    results = evaluate(predictor, source, images, config["batch_size"])
    write_json(output / "development.json", results)
    artifacts = {}
    for task in ("crop", "disease"):
        checkpoint = output / task / "best.pt"
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        metadata = payload["metadata"]
        history = json.loads((output / task / "history.json").read_text(encoding="utf-8"))
        selected = max(history, key=lambda row: (row["seen_validation"]["top1"], -row["epoch"]))
        if metadata["selected_epoch"] != selected["epoch"]:
            raise ValueError("Checkpoint epoch does not match recorded selection")
        artifacts[task] = {
            "file": str(checkpoint.resolve()), "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "metadata": metadata,
        }
    handoff = {"schema": "wp1-private-handoff-v1", "checkpoints": artifacts,
               "development_results": results, "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
               "official_test_evaluated": False}
    write_json(output / "handoff.json", handoff)
    print(json.dumps({"handoff": str(output / "handoff.json"), "development": results}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/wp1/resnet50.json")
    parser.add_argument("--protocol", type=Path, default=ROOT / "development_seed42_reconciled.json")
    parser.add_argument("--source", type=Path, default=ROOT / "DPD_Small_train.csv")
    parser.add_argument("--images", type=Path, default=ROOT / "DPD_Small_dataset")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--image-preflight", type=Path,
                        help="Reuse a successful same-session audit while images are unchanged")
    parser.add_argument("--completed-crop", type=Path,
                        help="Reuse a completed matching crop run; disease still trains from scratch")
    args = parser.parse_args()
    torch.set_num_threads(4)
    release(args.config, args.protocol, args.source, args.images, args.output, args.device,
            args.image_preflight, args.completed_crop)


if __name__ == "__main__":
    main()
