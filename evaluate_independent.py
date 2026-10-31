"""Evaluate independent checkpoints on development folds using WP5 metrics.

WP5 PR #1 must be available as evaluation.metrics. This module intentionally
does not fork that implementation, select official candidates or read tests.
"""
import argparse
import json
import time
from pathlib import Path

from PIL import Image
import torch

from independent_baseline import IndependentPredictor
from shared_protocol import validate_protocol


def evaluate(predictor, source_csv, img_dir, batch_size=8):
    from evaluation.metrics import evaluate_compositions, harmonic_mean, summarize_seen_unseen
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    _, seen, unseen = validate_protocol(predictor.protocol, source_csv)
    results = []
    for frame in (seen, unseen):
        batches, truths = [], []
        seconds = 0.
        for start in range(0, len(frame), batch_size):
            rows = frame.iloc[start:start + batch_size]
            started = time.perf_counter()
            images = []
            for name in rows.image_name:
                with Image.open(Path(img_dir) / name) as image:
                    images.append(image.convert("RGB"))
            output = predictor.predict(images, topk=min(5, len(predictor.protocol["candidates"])))
            # predict() transfers scores to CPU, synchronizing GPU work.
            seconds += time.perf_counter() - started
            batches.append(output["scores"])
            truths.extend(map(int, rows.composition_id))
        scores = torch.cat(batches)
        ids = output["candidate_ids"].tolist()
        result = evaluate_compositions(scores, truths, ids)
        result["decode_preprocess_inference_seconds"] = seconds
        result["milliseconds_per_image"] = 1000 * seconds / len(frame)
        per_pair = []
        top_ids = output["candidate_ids"][scores.argsort(dim=1, descending=True, stable=True)[:, :min(5, len(ids))]]
        for comp_id in sorted(set(truths)):
            mask = torch.tensor([truth == comp_id for truth in truths])
            ranking = top_ids[mask]
            per_pair.append({"composition_id": comp_id, "count": int(mask.sum()),
                             "correct_top1": int((ranking[:, 0] == comp_id).sum()),
                             "correct_top5": int((ranking == comp_id).any(dim=1).sum())})
        result["per_pair"] = per_pair
        results.append(result)
    summary = summarize_seen_unseen(*results)
    summary["harmonic_mean_top5"] = harmonic_mean(results[0]["top5"], results[1]["top5"])
    summary.update({"candidate_sha256": predictor.protocol["candidate_sha256"],
                    "split_sha256": predictor.protocol["split_sha256"],
                    "official_test_evaluated": False, "accuracy_scale": "fraction_0_to_1",
                    "selection_metric": "development_harmonic_mean_top1"})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crop", type=Path, required=True)
    parser.add_argument("--disease", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=Path("DPD_Small_train.csv"))
    parser.add_argument("--images", type=Path, default=Path("DPD_Small_dataset"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserving existing evaluation output; choose a new path")
    predictor = IndependentPredictor(args.crop, args.disease, args.protocol, args.source, args.device)
    if predictor.protocol["historical_reconciliation"] != "ordered_filenames_match_supplied_reference":
        raise ValueError("Reconcile the accepted WP2/WP3 filenames before model comparison")
    summary = evaluate(predictor, args.source, args.images, args.batch_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(summary, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
