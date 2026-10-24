import unittest
import numpy as np
import torch
from evaluation.metrics import (topk_accuracy, harmonic_mean,
    evaluate_compositions, summarize_seen_unseen, composition_scores_from_logits)


class TestMetrics(unittest.TestCase):
    def test_noncontiguous_global_ids(self):
        scores = [[0.1, 0.9, 0.5], [0.4, 0.2, 0.7]]
        self.assertEqual(topk_accuracy(scores, [332, 10], [10, 332, 75], k=1), 0.5)
        self.assertEqual(topk_accuracy(scores, [332, 10], [10, 332, 75], k=3), 1.0)

    def test_perfect(self):
        x = evaluate_compositions(np.eye(5), list(range(5)), list(range(5)))
        self.assertEqual(x["top1"], 1.0)
        self.assertEqual(x["top5"], 1.0)

    def test_ranked_fifth(self):
        self.assertEqual(topk_accuracy([[5, 4, 3, 2, 1, 0]], [4], list(range(6)), k=1), 0)
        self.assertEqual(topk_accuracy([[5, 4, 3, 2, 1, 0]], [4], list(range(6)), k=5), 1)

    def test_harmonic(self):
        self.assertAlmostEqual(harmonic_mean(0.8, 0.6), 0.685714285714)
        self.assertEqual(harmonic_mean(0, 0), 0)
        self.assertAlmostEqual(summarize_seen_unseen(
            {"top1": 0.8, "top5": 0.9, "count": 10},
            {"top1": 0.6, "top5": 0.7, "count": 10})["harmonic_mean_top1"], 0.685714285714)

    def test_invalid_candidates(self):
        for candidate_ids in [[4,4], [3]]:
            with self.assertRaises(ValueError):
                topk_accuracy([[0.7, 0.3]], [4], candidate_ids)
        with self.assertRaises(ValueError):
            topk_accuracy([[0.7, 0.3]], [999], [4,5])
        with self.assertRaises(ValueError):
            topk_accuracy([[np.nan, 0.3]], [4], [4,5])

    def test_composition_pair_scores(self):
        crop = torch.tensor([[4., 0.]])
        disease = torch.tensor([[0., 4.]])
        scores = composition_scores_from_logits(crop, disease, [(0,0),(0,1),(1,1)])
        self.assertEqual(scores.shape, (1, 3))
        self.assertEqual(int(scores.argmax(1).item()), 1)


if __name__ == "__main__":
    unittest.main()
