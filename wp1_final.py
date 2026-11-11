"""Independent full-8,000 training and final-333 inference; no validation selection."""
import gc
import json
import platform
from pathlib import Path

import torch
import torchvision
from torch.utils.data import DataLoader

from dataset import DPDataset, seed_everything, seed_worker
from final_protocol import file_sha256, load_final_protocol, write_new_json
from independent_baseline import (TASK_CLASSES, build_classifier, candidate_log_scores,
                                  load_checkpoint, preprocessing, preprocessing_record,
                                  run_epoch, save_checkpoint)


def train_final_task(config, task, protocol_path, source, images, output, device="cuda"):
    protocol, frame = load_final_protocol(protocol_path, source)
    if task not in TASK_CLASSES or config["epochs_by_task"].get(task) not in (2, 3):
        raise ValueError("Select a declared final task and fixed training budget")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    seed_everything(config["seed"])
    device = torch.device(device)
    model = build_classifier(config["backbone"], TASK_CLASSES[task], pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    data = DPDataset(frame, images, transform=preprocessing(config["backbone"], augment=True))
    loader = DataLoader(data, batch_size=config["batch_size"], shuffle=True,
                        num_workers=config["num_workers"], worker_init_fn=seed_worker,
                        generator=torch.Generator().manual_seed(config["seed"]))
    metadata = {"task": task, "num_classes": TASK_CLASSES[task], "backbone": config["backbone"],
                "torch": str(torch.__version__), "torchvision": str(torchvision.__version__),
                "python": platform.python_version(), "device": str(device),
                "hardware": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
                "config": config, "preprocessing": preprocessing_record(config["backbone"]),
                "selection": "fixed_development_selected_epoch_budget_no_validation_or_test_selection",
                "training_scope": "full_8000", "training_count": 8000,
                "training_manifest_sha256": protocol["training_manifest_sha256"],
                "source_sha256": protocol["source_sha256"],
                "candidate_sha256": protocol["candidate_sha256"],
                "prompt_sha256": protocol["prompt_sha256"],
                "source_code_sha256": {name: file_sha256(Path(__file__).parent / name) for name in
                    ("wp1_final.py", "final_protocol.py", "independent_baseline.py", "dataset.py")}}
    write_new_json(output / "run.json", metadata)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    history = []
    for epoch in range(1, config["epochs_by_task"][task]+1):
        record = {"epoch": epoch, "train": run_epoch(model, loader, task, device, optimizer)}
        if record["train"]["count"] != 8000:
            raise ValueError("Training epoch did not consume all 8,000 records")
        history.append(record)
        (output / "history.json").write_text(json.dumps(history, indent=2)+"\n", encoding="utf-8")
        print(json.dumps({"task": task, **record}), flush=True)
    checkpoint = output / "final.pt"
    save_checkpoint(checkpoint, model, {**metadata, "selected_epoch": len(history)})
    summary = {"task": task, "epochs_completed": len(history), "training_images_per_epoch": 8000,
               "checkpoint_sha256": file_sha256(checkpoint), "official_test_evaluated": False,
               "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None}
    write_new_json(output / "summary.json", summary)
    del model, optimizer, loader
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary


class FinalIndependentPredictor:
    """Load only full-training checkpoints that match the declared final contract."""
    def __init__(self, crop, disease, protocol_path, source, device="cpu"):
        self.protocol, _ = load_final_protocol(protocol_path, source)
        self.device = torch.device(device)
        self.crop, crop_meta = load_checkpoint(crop, device)
        self.disease, disease_meta = load_checkpoint(disease, device)
        for task, metadata in (("crop", crop_meta), ("disease", disease_meta)):
            if metadata.get("task") != task or metadata.get("training_scope") != "full_8000":
                raise ValueError("Final inference requires correctly assigned full-8,000 checkpoints")
            for key in ("source_sha256", "training_manifest_sha256", "candidate_sha256", "prompt_sha256", "training_count"):
                if metadata.get(key) != self.protocol[key]:
                    raise ValueError(f"Final checkpoint/protocol mismatch: {key}")
            if metadata.get("selected_epoch") != metadata["config"]["epochs_by_task"][task]:
                raise ValueError("Final checkpoint violates the fixed training budget")
        self.crop_transform = preprocessing(crop_meta["backbone"])
        self.disease_transform = preprocessing(disease_meta["backbone"])

    @torch.inference_mode()
    def predict(self, images, topk=5):
        if not images or type(topk) is not int or not 1 <= topk <= 333:
            raise ValueError("Supply images and a valid final top-k")
        crop = self.crop(torch.stack([self.crop_transform(im.convert("RGB")) for im in images]).to(self.device))
        disease = self.disease(torch.stack([self.disease_transform(im.convert("RGB")) for im in images]).to(self.device))
        scores = candidate_log_scores(crop, disease, self.protocol["candidates"]).cpu()
        ids = torch.tensor([row["composition_id"] for row in self.protocol["candidates"]], dtype=torch.int64)
        order = scores.argsort(dim=1, descending=True, stable=True)[:, :topk]
        return {"candidate_ids": ids, "scores": scores, "topk_composition_ids": ids[order],
                "candidate_sha256": self.protocol["candidate_sha256"],
                "score_semantics": "native_float32_sum_component_log_probabilities"}
