"""Separate supervised crop/disease classifiers and checkpoint-safe inference.

Training reads only the original training CSV and the explicit development
protocol. Run once per task; the two models never share parameters or optimizers.
No DPD training, official-test performance or GPU compatibility is implied by
the synthetic CPU tests shipped with this implementation.
"""
import argparse
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path

import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models, transforms

from dataset import DPDataset, seed_everything, seed_worker
from shared_protocol import validate_protocol

BACKBONES = ("resnet50", "efficientnet_b3", "convnext_base", "vit_b_16")
TASK_CLASSES = {"crop": 55, "disease": 175}
CHECKPOINT_SCHEMA = "wp1-independent-v1"


def build_classifier(backbone, num_classes, pretrained=False):
    if backbone not in BACKBONES or num_classes <= 0:
        raise ValueError("Unsupported backbone or class count")
    weights = models.get_model_weights(backbone).DEFAULT if pretrained else None
    model = models.get_model(backbone, weights=weights)
    if backbone == "resnet50":
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif backbone == "vit_b_16":
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    else:
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    return model


def preprocessing(backbone, augment=False):
    """Use the selected torchvision weight recipe, including resolution/crop."""
    native = models.get_model_weights(backbone).DEFAULT.transforms()
    if not augment:
        return native
    return transforms.Compose([
        transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
        transforms.RandomAffine(18, translate=(.05, .05), scale=(.95, 1.05)), native,
    ])


def preprocessing_record(backbone):
    weights = models.get_model_weights(backbone).DEFAULT
    native = weights.transforms()
    return {"weights": str(weights), "crop_size": list(native.crop_size),
            "resize_size": list(native.resize_size), "mean": list(native.mean),
            "std": list(native.std), "interpolation": str(native.interpolation),
            "antialias": native.antialias, "rgb": True}


def run_epoch(model, loader, task, device, optimizer=None):
    if task not in TASK_CLASSES:
        raise ValueError("Task must be crop or disease")
    training = optimizer is not None
    model.train(training)
    count = correct1 = correct5 = 0
    total_loss = 0.
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        started = time.perf_counter()
    with torch.set_grad_enabled(training):
        for images, targets in loader:
            images, labels = images.to(device), targets[task].to(device)
            logits = model(images)
            loss = nn.functional.cross_entropy(logits, labels)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite loss")
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            n = labels.numel()
            ranking = logits.detach().argsort(dim=1, descending=True, stable=True)
            correct1 += int((ranking[:, 0] == labels).sum())
            correct5 += int((ranking[:, :min(5, logits.shape[1])] == labels[:, None]).any(dim=1).sum())
            count += n
            total_loss += float(loss.detach()) * n
    if not count:
        raise ValueError("Empty loader")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return {"count": count, "loss": total_loss / count, "top1": correct1 / count,
            "top5": correct5 / count, "wall_seconds": time.perf_counter() - started}


def save_checkpoint(path, model, metadata):
    payload = {"schema": CHECKPOINT_SCHEMA, "metadata": metadata,
               "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()}}
    torch.save(payload, path)


def load_checkpoint(path, device="cpu"):
    # Always rebuild without a pretrained download, then load tensors strictly.
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("Unsupported independent checkpoint schema")
    metadata = payload["metadata"]
    if metadata["task"] not in TASK_CLASSES or metadata["num_classes"] != TASK_CLASSES[metadata["task"]]:
        raise ValueError("Checkpoint task/class mismatch")
    if metadata["torchvision"] != torchvision.__version__:
        raise ValueError("Use the recorded torchvision version to reconstruct preprocessing")
    if metadata["preprocessing"] != preprocessing_record(metadata["backbone"]):
        raise ValueError("Checkpoint preprocessing no longer matches the installed weight recipe")
    model = build_classifier(metadata["backbone"], metadata["num_classes"], pretrained=False)
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.to(device).eval(), metadata


def candidate_log_scores(crop_logits, disease_logits, candidates):
    """Log P(crop|x) + log P(disease|x), ordered global candidate columns.

    Ranking is mathematically equivalent to WP5 PR #1's probability product.
    Values are ranking scores and must not be presented as calibrated confidence.
    """
    ids = [row["composition_id"] for row in candidates]
    pairs = [(row["crop_id"], row["disease_id"]) for row in candidates]
    if not ids or ids != sorted(set(ids)) or len(set(pairs)) != len(pairs):
        raise ValueError("Candidates require unique ascending global IDs and unique pairs")
    if crop_logits.ndim != 2 or disease_logits.ndim != 2 or crop_logits.shape[0] != disease_logits.shape[0]:
        raise ValueError("Expected matching [B,C] and [B,D] logits")
    if crop_logits.device != disease_logits.device:
        raise ValueError("Both logits must use the same device")
    if not torch.isfinite(crop_logits).all() or not torch.isfinite(disease_logits).all():
        raise ValueError("Logits contain NaN or infinity")
    ci, di = zip(*pairs)
    if min(ci) < 0 or max(ci) >= crop_logits.shape[1] or min(di) < 0 or max(di) >= disease_logits.shape[1]:
        raise ValueError("Candidate component outside classifier output")
    return crop_logits.log_softmax(1)[:, ci] + disease_logits.log_softmax(1)[:, di]


class IndependentPredictor:
    """WP4 adapter: PIL RGB images -> ordered global candidate scores/top-k.

    WP3 can expose the same output fields using its own native CLIP preprocessing
    and direct text similarities; never feed CLIP scores through the head formula.
    """
    def __init__(self, crop_checkpoint, disease_checkpoint, protocol_path, source_csv, device="cpu"):
        self.device = torch.device(device)
        self.protocol = json.loads(Path(protocol_path).read_text(encoding="utf-8"))
        validate_protocol(self.protocol, source_csv)
        self.crop, cm = load_checkpoint(crop_checkpoint, device)
        self.disease, dm = load_checkpoint(disease_checkpoint, device)
        for task, metadata in (("crop", cm), ("disease", dm)):
            if metadata["task"] != task:
                raise ValueError("Crop and disease checkpoint roles are reversed")
            for key in ("candidate_sha256", "split_sha256", "source_sha256"):
                if metadata[key] != self.protocol[key]:
                    raise ValueError(f"Checkpoint/protocol mismatch: {key}")
        self.crop_transform = preprocessing(cm["backbone"])
        self.disease_transform = preprocessing(dm["backbone"])

    @torch.inference_mode()
    def predict(self, images, topk=5):
        if not images or not 1 <= topk <= len(self.protocol["candidates"]):
            raise ValueError("Provide images and a valid topk")
        crop = self.crop(torch.stack([self.crop_transform(im.convert("RGB")) for im in images]).to(self.device))
        disease = self.disease(torch.stack([self.disease_transform(im.convert("RGB")) for im in images]).to(self.device))
        candidates = self.protocol["candidates"]
        scores = candidate_log_scores(crop, disease, candidates)
        order = scores.argsort(dim=1, descending=True, stable=True)[:, :topk]
        ids = torch.tensor([row["composition_id"] for row in candidates], device=self.device)
        return {"candidate_ids": ids.cpu(), "scores": scores.cpu(),
                "topk_composition_ids": ids[order].cpu(),
                "score_semantics": "sum_component_log_probabilities",
                "candidate_sha256": self.protocol["candidate_sha256"]}


def train(config, task, protocol_path, source_csv, img_dir, output_dir, device="cpu"):
    if task not in TASK_CLASSES:
        raise ValueError("Unknown task")
    protocol = json.loads(Path(protocol_path).read_text(encoding="utf-8"))
    fitting, seen, _ = validate_protocol(protocol, source_csv)
    if protocol["historical_reconciliation"] != "ordered_filenames_match_supplied_reference":
        raise ValueError("Reconcile the accepted WP2/WP3 ordered filenames before DPD training")
    if config["epochs"] < 1 or config["batch_size"] < 1 or config["lr"] <= 0 or config["num_workers"] < 0:
        raise ValueError("Invalid training budget or optimizer settings")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    seed_everything(config["seed"])
    device = torch.device(device)
    model = build_classifier(config["backbone"], TASK_CLASSES[task], config["pretrained"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    loaders = []
    for i, frame in enumerate((fitting, seen)):
        dataset = DPDataset(frame, img_dir, transform=preprocessing(config["backbone"], augment=i == 0))
        loaders.append(DataLoader(dataset, batch_size=config["batch_size"], shuffle=i == 0,
                                  num_workers=config["num_workers"], worker_init_fn=seed_worker,
                                  generator=torch.Generator().manual_seed(config["seed"] + i)))
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = "unavailable", True
    source_files = ("independent_baseline.py", "shared_protocol.py", "dataset.py", "data_utils.py")
    code_hashes = {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest() for name in source_files}
    metadata = {"task": task, "num_classes": TASK_CLASSES[task], "backbone": config["backbone"],
                "torch": str(torch.__version__), "torchvision": str(torchvision.__version__),
                "python": platform.python_version(), "device": str(device),
                "hardware": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
                "code_commit": commit, "code_dirty": dirty, "code_sha256": code_hashes,
                "config": config, "preprocessing": preprocessing_record(config["backbone"]),
                "parameter_count": sum(p.numel() for p in model.parameters()),
                "selection": "seen_validation_component_top1; earliest epoch wins ties",
                **{key: protocol[key] for key in ("candidate_sha256", "split_sha256", "source_sha256")}}
    (output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    best = -1.
    history = []
    for epoch in range(1, config["epochs"] + 1):
        record = {"epoch": epoch, "train": run_epoch(model, loaders[0], task, device, optimizer),
                  "seen_validation": run_epoch(model, loaders[1], task, device)}
        history.append(record)
        if record["seen_validation"]["top1"] > best:
            best = record["seen_validation"]["top1"]
            save_checkpoint(output / "best.pt", model, {**metadata, "selected_epoch": epoch})
        (output / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(record), flush=True)
    summary = {"best_seen_component_top1": best, "epochs_completed": len(history),
               "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
               "official_test_evaluated": False,
               "remaining": "Pair-level development H@1/H@5, latency and WP5 integration required before architecture selection"}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--task", choices=tuple(TASK_CLASSES), required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=Path("DPD_Small_train.csv"))
    parser.add_argument("--images", type=Path, default=Path("DPD_Small_dataset"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    train(json.loads(args.config.read_text(encoding="utf-8")), args.task, args.protocol,
          args.source, args.images, args.output, args.device)


if __name__ == "__main__":
    main()
