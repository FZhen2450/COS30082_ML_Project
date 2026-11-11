"""Freeze, audit and optionally execute the independent full-8,000 release."""
import argparse
import gc
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from final_protocol import file_sha256, load_final_protocol, write_new_json


def release(protocol_path, config_path, source, images, pretrained, output, execute=False):
    import torch
    from PIL import Image
    from wp1_final import train_final_task
    protocol, frame = load_final_protocol(protocol_path, source)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    expected = {"backbone": "resnet50", "pretrained": True, "epochs_by_task": {"crop": 2, "disease": 3},
                "batch_size": 2, "lr": .0001, "weight_decay": .0001, "seed": 42, "num_workers": 0}
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("Recipe differs from the declared development-selected full-training release")
    if pretrained.name != "resnet50-11ad3fa6.pth" or not file_sha256(pretrained).startswith("11ad3fa6"):
        raise ValueError("Supply the existing official ResNet50 ImageNet V2 initialization")
    if execute and not torch.cuda.is_available():
        raise RuntimeError("The declared final run requires the compatible CUDA runtime")
    output.mkdir(parents=True, exist_ok=False)
    plan = {"schema": "wp1-full8000-plan-v1", "config": config,
            "source_sha256": protocol["source_sha256"],
            "training_manifest_sha256": protocol["training_manifest_sha256"],
            "candidate_sha256": protocol["candidate_sha256"], "candidate_count": 333,
            "training_count": 8000, "pretrained_sha256": file_sha256(pretrained),
            "selection": "fixed_crop_epoch_2_disease_epoch_3_no_validation_or_test_selection",
            "initialization": "fresh_ImageNet_V2_separately_seeded_per_task",
            "official_test_images_read": False,
            "source_code_sha256": {str(path.relative_to(ROOT)): file_sha256(path) for path in
                (ROOT/"wp1_final.py", ROOT/"final_protocol.py", Path(__file__).resolve(),
                 ROOT/"independent_baseline.py", ROOT/"dataset.py")}}
    write_new_json(output / "plan.json", plan)
    hashes = {}
    images = images.resolve()
    for name in frame.image_name:
        path = (images/name).resolve()
        if path.parent != images:
            raise ValueError("Training image resolves outside the declared flat dataset")
        hashes[name] = file_sha256(path)
        with Image.open(path) as image:
            image.convert("RGB").load()
        if len(hashes) % 1000 == 0:
            print(f"Full-training image audit: {len(hashes)}/8000", flush=True)
    write_new_json(output / "training_image_audit.json", {"count": len(hashes), "sha256": hashes,
                   "official_test_images_read": False})
    if not execute:
        print("Preparation complete. Training was not requested.", flush=True)
        return
    torch.set_num_threads(4)
    torch.hub.set_dir(str(pretrained.parent.parent))
    results = {}
    started = time.perf_counter()
    for task in ("crop", "disease"):
        print(f"Starting full-8,000 {task} training", flush=True)
        results[task] = train_final_task(config, task, protocol_path, source, images, output/task, "cuda")
    if any(file_sha256(images/name) != value for name, value in hashes.items()):
        raise ValueError("Training image bytes changed during the release")
    if any(file_sha256(ROOT/name) != value for name, value in plan["source_code_sha256"].items()):
        raise ValueError("Frozen training source changed while the release was running")
    release_record = {"schema": "wp1-full8000-release-v1", "training_scope": "full_8000",
                      "candidate_sha256": protocol["candidate_sha256"],
                      "training_manifest_sha256": protocol["training_manifest_sha256"],
                      "source_sha256": protocol["source_sha256"], "training_count": 8000,
                      "checkpoints": {task: {"path": (output/task/"final.pt").resolve().relative_to(ROOT).as_posix(),
                                            "sha256": results[task]["checkpoint_sha256"]} for task in results},
                      "results": results, "elapsed_seconds": time.perf_counter()-started,
                      "training_image_bytes_unchanged": True, "official_test_evaluated": False}
    write_new_json(output / "release.json", release_record)
    print(json.dumps(release_record, indent=2), flush=True)
    gc.collect()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT/"artifacts/data/final_full8000_333_v1.json")
    parser.add_argument("--config", type=Path, default=ROOT/"configs/wp1/resnet50_full8000.json")
    parser.add_argument("--source", type=Path, default=ROOT/"DPD_Small_train.csv")
    parser.add_argument("--images", type=Path, default=ROOT/"DPD_Small_dataset")
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute-training", action="store_true")
    args = parser.parse_args()
    release(args.protocol, args.config, args.source, args.images, args.pretrained,
            args.output, args.execute_training)


if __name__ == "__main__":
    main()
