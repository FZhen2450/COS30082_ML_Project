"""Synthetic CPU tests; no pretrained downloads, raw DPD images or benchmarks."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torchvision
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from independent_baseline import (
    BACKBONES, IndependentPredictor, build_classifier, candidate_log_scores,
    load_checkpoint, preprocessing, preprocessing_record, run_epoch, save_checkpoint, train,
)
from shared_protocol import FOLDS, build_protocol

ROOT = Path(__file__).resolve().parents[1]


class DictionaryDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.images = torch.rand(4, 3, 8, 8)
    def __len__(self):
        return 4
    def __getitem__(self, index):
        return self.images[index], {"crop": torch.tensor(index % 2), "disease": torch.tensor(index % 3)}


def tiny_model(classes):
    return nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, classes))


class IndependentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_separate_parameters_and_gradient_updates(self):
        crop, disease = tiny_model(55), tiny_model(175)
        before_crop = crop[-1].weight.detach().clone()
        before_disease = disease[-1].weight.detach().clone()
        loader = DataLoader(DictionaryDataset(), batch_size=2)
        result = run_epoch(crop, loader, "crop", torch.device("cpu"), torch.optim.SGD(crop.parameters(), lr=.1))
        self.assertEqual(result["count"], 4)
        self.assertFalse(torch.equal(before_crop, crop[-1].weight))
        torch.testing.assert_close(before_disease, disease[-1].weight)
        for p in disease.parameters():
            self.assertIsNone(p.grad)

    def test_validation_does_not_update_parameters(self):
        model = tiny_model(55)
        before = model[-1].weight.detach().clone()
        result = run_epoch(model, DataLoader(DictionaryDataset(), batch_size=2), "crop", torch.device("cpu"))
        torch.testing.assert_close(before, model[-1].weight)
        self.assertEqual(result["count"], 4)
        self.assertFalse(model.training)

    def test_all_weight_recipes_produce_native_rgb_resolution(self):
        for backbone in BACKBONES:
            with self.subTest(backbone=backbone):
                tensor = preprocessing(backbone)(Image.new("RGB", (320, 290), "green"))
                recipe = preprocessing_record(backbone)
                self.assertEqual(tensor.shape[0], 3)
                self.assertEqual(list(tensor.shape[1:]), recipe["crop_size"] * 2 if len(recipe["crop_size"]) == 1 else recipe["crop_size"])
                self.assertEqual(tensor.dtype, torch.float32)
                self.assertTrue(torch.isfinite(tensor).all())

    def test_all_backbone_builders_and_output_shapes_without_downloads(self):
        import gc
        for index, backbone in enumerate(BACKBONES):
            with self.subTest(backbone=backbone):
                classes = 55 if index % 2 == 0 else 175
                model = build_classifier(backbone, classes, pretrained=False).eval()
                size = preprocessing_record(backbone)["crop_size"][0]
                with torch.inference_mode():
                    output = model(torch.zeros(1, 3, size, size))
                self.assertEqual(tuple(output.shape), (1, classes))
                self.assertTrue(torch.isfinite(output).all())
                del model, output
                gc.collect()

    def test_real_resnet_checkpoint_reconstruction_without_download(self):
        torch.manual_seed(7)
        model = build_classifier("resnet50", 55).eval()
        metadata = {"task": "crop", "num_classes": 55, "backbone": "resnet50",
                    "torchvision": str(torchvision.__version__), "preprocessing": preprocessing_record("resnet50")}
        image = torch.rand(1, 3, 32, 32)
        with torch.inference_mode():
            expected = model(image)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.pt"
            save_checkpoint(path, model, metadata)
            with patch("torch.hub.download_url_to_file", side_effect=AssertionError("Unexpected download")):
                restored, meta = load_checkpoint(path)
            with torch.inference_mode():
                torch.testing.assert_close(expected, restored(image))
            self.assertEqual(meta, metadata)

    def test_score_ranking_equivalent_to_probability_product(self):
        candidates = [{"composition_id": i, "crop_id": c, "disease_id": d}
                      for i, c, d in [(10, 0, 0), (75, 0, 1), (332, 1, 1)]]
        crop, disease = torch.tensor([[4., 0.]]), torch.tensor([[0., 4.]])
        scores = candidate_log_scores(crop, disease, candidates)
        product = crop.softmax(1)[:, [0, 0, 1]] * disease.softmax(1)[:, [0, 1, 1]]
        torch.testing.assert_close(scores.exp(), product)
        self.assertEqual(scores.argmax(1).item(), 1)
        ties = candidate_log_scores(torch.zeros_like(crop), torch.zeros_like(disease), candidates)
        self.assertEqual(ties.argsort(descending=True, stable=True)[0].tolist(), [0, 1, 2])
        with self.assertRaises(ValueError):
            candidate_log_scores(crop, disease, candidates[::-1])
        with self.assertRaises(ValueError):
            candidate_log_scores(crop * float("nan"), disease, candidates)

    def test_unreconciled_protocol_blocks_training_before_output(self):
        protocol = build_protocol(ROOT / "DPD_Small_train.csv")
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "protocol.json"
            path.write_text(json.dumps(protocol), encoding="utf-8")
            output = Path(folder) / "run"
            with self.assertRaisesRegex(ValueError, "Reconcile"):
                train({}, "crop", path, ROOT / "DPD_Small_train.csv", folder, output)
            self.assertFalse(output.exists())

    def test_synthetic_training_records_and_inference_adapter(self):
        # Build a tiny balanced source with held-out cross-pairs. This is not DPD.
        import pandas as pd
        from data_utils import COLUMN_NAMES
        from shared_protocol import digest, write_new_protocol
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            records = []
            for i, (crop, disease) in enumerate([(0, 0), (1, 1), (0, 0), (1, 1), (0, 1), (1, 0)]):
                name = f"{i}.png"
                Image.new("RGB", (16, 16), (i * 30, 90, 120)).save(root / name)
                comp = crop * 2 + disease
                records.append([name, crop, disease, comp, f"c{crop}", f"d{disease}", f"p{comp}"])
            source = root / "train.csv"
            pd.DataFrame(records, columns=COLUMN_NAMES).to_csv(source, header=False, index=False)
            from shared_protocol import candidate_table
            from data_utils import load_metadata
            import hashlib
            p = {"schema": "wp1-development-v1", "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                 "historical_reconciliation": "ordered_filenames_match_supplied_reference",
                 "candidates": candidate_table(load_metadata(source)), "counts": [2, 2, 2],
                 FOLDS[0]: ["0.png", "1.png"], FOLDS[1]: ["2.png", "3.png"], FOLDS[2]: ["4.png", "5.png"]}
            p["candidate_sha256"] = digest(p["candidates"])
            p["split_sha256"] = digest({key: p[key] for key in FOLDS})
            protocol_path = root / "protocol.json"
            write_new_protocol(p, protocol_path)
            config = {"backbone": "resnet50", "pretrained": False, "epochs": 1, "batch_size": 2,
                      "lr": .001, "weight_decay": .0001, "seed": 42, "num_workers": 0}
            with patch("independent_baseline.build_classifier", side_effect=lambda b, n, pretrained=False: tiny_model(n)):
                for task in ("crop", "disease"):
                    summary = train(config, task, protocol_path, source, root, root / task)
                    self.assertFalse(summary["official_test_evaluated"])
                    history = json.loads((root / task / "history.json").read_text())
                    self.assertEqual(history[0]["train"]["count"], 2)
                    self.assertTrue((root / task / "best.pt").exists())
                predictor = IndependentPredictor(root / "crop/best.pt", root / "disease/best.pt", protocol_path, source)
                result = predictor.predict([Image.new("RGB", (20, 20), "green")], topk=2)
                self.assertEqual(tuple(result["scores"].shape), (1, 4))
                self.assertEqual(tuple(result["topk_composition_ids"].shape), (1, 2))
                self.assertEqual(result["candidate_ids"].tolist(), [0, 1, 2, 3])
                with self.assertRaises(ValueError):
                    IndependentPredictor(root / "disease/best.pt", root / "crop/best.pt", protocol_path, source)


if __name__ == "__main__":
    unittest.main()
