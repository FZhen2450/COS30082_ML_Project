"""Synthetic compatibility check against an existing WP5 metrics checkout.

Usage: python scripts/verify_wp5_bridge.py --wp5-root PATH
No WP5 code is copied into WP1, and no DPD experiment is run.
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wp5-root", type=Path, required=True)
    args = parser.parse_args()
    if not (args.wp5_root / "evaluation/metrics.py").is_file():
        raise FileNotFoundError("Provide a checkout containing evaluation/metrics.py")
    sys.path.insert(0, str(args.wp5_root.resolve()))
    import pandas as pd
    import torch
    from PIL import Image
    from evaluation.metrics import composition_scores_from_logits, evaluate_compositions
    from independent_baseline import candidate_log_scores
    from evaluate_independent import evaluate

    candidates = [{"composition_id": i, "crop_id": c, "disease_id": d}
                  for i, c, d in [(10, 0, 0), (75, 0, 1), (332, 1, 1)]]
    crop = torch.tensor([[4., 0.], [0., 4.], [0., 0.]])
    disease = torch.tensor([[0., 4.], [0., 4.], [0., 0.]])
    log_scores = candidate_log_scores(crop, disease, candidates)
    wp5_scores = composition_scores_from_logits(crop, disease, [(0, 0), (0, 1), (1, 1)])
    torch.testing.assert_close(log_scores.exp(), wp5_scores)
    torch.testing.assert_close(log_scores.argsort(descending=True, stable=True),
                               wp5_scores.argsort(descending=True, stable=True))
    ids = [10, 75, 332]
    assert evaluate_compositions(log_scores, [75, 332, 10], ids)["top1"] == 1.

    class SyntheticPredictor:
        protocol = {"candidates": candidates, "candidate_sha256": "synthetic-candidates",
                    "split_sha256": "synthetic-split"}
        def predict(self, images, topk=3):
            all_scores = [[3., 0., 0.], [0., 3., 0.], [0., 0., 3.], [0., 3., 0.]]
            scores = torch.tensor([all_scores[im.getpixel((0, 0))[0]] for im in images])
            return {"candidate_ids": torch.tensor(ids), "scores": scores}

    with tempfile.TemporaryDirectory(prefix="wp1-wp5-") as folder:
        root = Path(folder)
        for i in range(4):
            Image.new("RGB", (8, 8), (i, 0, 0)).save(root / f"{i}.png")
        seen = pd.DataFrame({"image_name": ["0.png", "1.png"], "composition_id": [10, 75]})
        unseen = pd.DataFrame({"image_name": ["2.png", "3.png"], "composition_id": [332, 332]})
        with patch("evaluate_independent.validate_protocol", return_value=(None, seen, unseen)):
            summary = evaluate(SyntheticPredictor(), "unused-synthetic-source", root, batch_size=1)
    assert summary["seen"]["top1"] == 1.
    assert summary["unseen"]["top1"] == .5
    assert abs(summary["harmonic_mean_top1"] - 2 / 3) < 1e-12
    assert summary["harmonic_mean_top5"] == 1.
    assert summary["unseen"]["per_pair"][0] == {
        "composition_id": 332, "count": 2, "correct_top1": 1, "correct_top5": 2}
    print(json.dumps({"status": "passed", "scope": "synthetic WP5 compatibility only",
                      "seen_top1": 1., "unseen_top1": .5, "h1": summary["harmonic_mean_top1"],
                      "h5": 1., "noncontiguous_ids": ids, "official_test_evaluated": False}, indent=2))


if __name__ == "__main__":
    main()
