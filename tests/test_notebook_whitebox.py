"""M4C visible workflow tests: AST extraction and bounded synthetic CPU checks."""

import ast
import copy
import hashlib
import io
import json
import math
import random
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader

from test_notebook import code_sources, helper_namespace, literal_assignments, load_notebook

ROOT = Path(__file__).resolve().parents[1]


def small_candidates():
    return {"ids": (10, 75, 100, 150, 200, 332),
            "pairs": tuple((i, i) for i in range(6))}


def synthetic_rows():
    rows = []
    for composition, (crop, disease) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
        for index in range(8):
            rows.append([f"{composition}_{index}.png", crop, disease, composition + 10,
                         f"crop{crop}", f"disease{disease}", f"pair{composition}"])
    return pd.DataFrame(rows)


class WhiteBoxStructureTests(unittest.TestCase):
    def test_every_essential_operation_is_defined_in_the_notebook(self):
        names = {node.name for source in code_sources().values()
                 for node in ast.parse(source).body
                 if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
        required = {"load_metadata", "make_validation_split", "build_development_split",
                    "get_transforms", "DPDataset", "get_development_dataloaders",
                    "MultiTaskBackbone", "create_multitask_model", "compute_multitask_loss",
                    "train_one_epoch", "validate_one_epoch", "composition_scores",
                    "composition_topk", "harmonic_mean", "evaluate_development",
                    "checkpoint_selection_key", "save_checkpoint", "load_checkpoint",
                    "comparison_record", "fairness_signature"}
        self.assertTrue(required <= names, sorted(required - names))

    def test_no_hidden_project_execution_or_imports(self):
        project_modules = {"dataset", "data_utils", "development_data", "multitask_model",
                           "multitask_backbones", "train_multitask", "eda"}
        for source in code_sources().values():
            self.assertNotIn("%run", source)
            for node in ast.walk(ast.parse(source)):
                if isinstance(node, ast.ImportFrom):
                    self.assertNotIn((node.module or "").split(".")[0], project_modules)
                if isinstance(node, ast.Import):
                    self.assertFalse(any(item.name.split(".")[0] in project_modules
                                         or item.name == "importlib" for item in node.names))
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    self.assertNotIn(node.func.id, {"exec", "eval", "compile", "__import__"})
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    if isinstance(node.func.value, ast.Name) and node.func.value.id == "subprocess":
                        # The only process launched in notebook code is read-only Git inspection.
                        self.assertEqual(node.func.attr, "run")
                        self.assertIsInstance(node.args[0], ast.List)
                        self.assertEqual(node.args[0].elts[0].value, "git")

    def test_all_image_download_and_execution_controls_default_false(self):
        values = literal_assignments()
        flags = ("PREPARE_RUN", "RUN_MODEL_DIAGNOSTICS", "RUN_TRAINING",
                 "RUN_EXPENSIVE_EVALUATION", "SHOW_DATA_IMAGES", "SHOW_AUGMENTATIONS",
                 "RUN_IMAGE_AUDIT", "RUN_BATCH_INSPECTION", "RUN_PRETRAINED_ACQUISITION",
                 "RUN_METRIC_EXAMPLES", "READ_RECORDED_RESULTS")
        for name in flags:
            self.assertIs(values[name], False, name)
        cell = ast.parse(code_sources()["weight-selection"])
        acquisitions = [n for n in cell.body if isinstance(n, ast.If)
                        and isinstance(n.test, ast.Name) and n.test.id == "RUN_PRETRAINED_ACQUISITION"]
        self.assertEqual(len(acquisitions), 1)
        self.assertTrue(any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                            and n.func.id == "acquire_official_weights"
                            for n in ast.walk(acquisitions[0])))

    def test_comparison_and_visualization_never_invent_experiment_results(self):
        sources = code_sources()
        self.assertIn('if history_frame.empty:', sources["plots"])
        self.assertIn('No completed experiment results', sources["comparison"])
        self.assertIn('compare_planned_runs', sources["bounded-tuning"])
        self.assertIn('no automatic controlled winner', sources["bounded-tuning"])
        self.assertIn('peak_gpu_reserved_mib', sources["smoke-and-plot-functions"])
        self.assertIn('metric_implementation', sources["recovery-functions"])
        self.assertIn('split_sha256', sources["recovery-functions"])


class WhiteBoxDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = helper_namespace()

    def test_exact_reference_split_rows_mappings_and_hashes(self):
        from development_data import build_development_split
        expected = build_development_split(ROOT / "DPD_Small_train.csv")
        actual = self.ns["build_development_split"](ROOT / "DPD_Small_train.csv")
        for name in ("fit", "val_seen", "val_unseen"):
            pd.testing.assert_frame_equal(getattr(actual, name), getattr(expected, name))
        self.assertEqual(tuple(map(len, (actual.fit, actual.val_seen, actual.val_unseen))),
                         (4868, 1238, 1894))
        for name in ("mappings", "candidate_ids", "candidate_pairs", "candidate_sha256", "source_sha256"):
            self.assertEqual(getattr(actual, name), getattr(expected, name))
        self.assertEqual(len(actual.candidate_ids), 323)
        self.assertEqual(set(actual.fit.crop_id), set(range(55)))
        self.assertEqual(set(actual.fit.disease_id), set(range(175)))

    def test_synthetic_split_matches_reference_and_is_disjoint(self):
        from development_data import build_development_split
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic_train.csv"
            synthetic_rows().to_csv(path, index=False, header=False)
            expected = build_development_split(path)
            first = self.ns["build_development_split"](path)
            second = self.ns["build_development_split"](path)
            for name in ("fit", "val_seen", "val_unseen"):
                pd.testing.assert_frame_equal(getattr(first, name), getattr(expected, name))
                pd.testing.assert_frame_equal(getattr(first, name), getattr(second, name))
            parts = [set(getattr(first, name).image_name) for name in ("fit", "val_seen", "val_unseen")]
            self.assertEqual(len(set.union(*parts)), 32)
            self.assertTrue(all(not parts[i] & parts[j] for i in range(3) for j in range(i + 1, 3)))

    def test_metadata_rejects_unsafe_missing_duplicate_and_inconsistent_rows(self):
        loader, validator = self.ns["load_metadata"], self.ns["validate_metadata"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.csv"
            for mutate in (
                lambda frame: frame.iloc[:, :6],
                lambda frame: frame.replace("0_0.png", "../escape.png"),
                lambda frame: frame.replace("crop0", None),
            ):
                frame = mutate(synthetic_rows())
                frame.to_csv(path, index=False, header=False)
                with self.assertRaises(ValueError):
                    loader(path)
            good = synthetic_rows()
            good.to_csv(path, index=False, header=False)
            frame = loader(path)
            for bad in (pd.concat([frame, frame.iloc[:1]], ignore_index=True),
                        frame.assign(crop_name=["wrong"] + frame.crop_name.tolist()[1:])):
                with self.assertRaises(ValueError):
                    validator(bad, bad.iloc[:0], bad.iloc[:0])

    def test_dataset_decodes_rgb_tensor_and_preserves_global_id(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("L", (8, 9), color=120).save(root / "leaf.png")
            frame = pd.DataFrame([["leaf.png", 0, 0, 75, "crop", "disease", "pair"]],
                                 columns=self.ns["COLUMN_NAMES"])
            dataset = self.ns["DPDataset"](frame, root, augment=False)
            image, target = dataset[0]
            self.assertEqual(tuple(image.shape), (3, 224, 224))
            self.assertEqual(image.dtype, torch.float32)
            self.assertTrue(torch.isfinite(image).all())
            self.assertEqual(target["composition"].item(), 75)
            self.assertTrue(all(value.dtype == torch.int64 for value in target.values()))
            (root / "leaf.png").write_bytes(b"bad image")
            with self.assertRaisesRegex(RuntimeError, "leaf.png"):
                dataset[0]

    def test_visible_transforms_match_reference_including_seeded_augmentation(self):
        from dataset import get_transforms
        image = Image.fromarray(np.arange(12 * 13 * 3, dtype=np.uint8).reshape(12, 13, 3))
        for augment in (False, True):
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(42)
                expected = get_transforms(224, augment)(image)
                torch.manual_seed(42)
                actual = self.ns["get_transforms"](224, augment)(image)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_local_dataloaders_use_synthetic_images_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "synthetic_train.csv"
            frame = synthetic_rows()
            frame.to_csv(path, header=False, index=False)
            for name in frame[0]:
                Image.new("RGB", (6, 7), color=(100, 150, 80)).save(root / name)
            split = self.ns["build_development_split"](path)
            fit, seen, unseen = self.ns["get_development_dataloaders"](
                split, root, batch_size=2, augment_fit=False)
            self.assertEqual([len(loader.dataset) for loader in (fit, seen, unseen)], [18, 6, 8])
            image, targets = next(iter(seen))
            self.assertEqual(tuple(image.shape), (2, 3, 224, 224))
            self.assertTrue(set(targets["composition"].tolist()) <= set(split.candidate_ids))
            self.assertEqual(targets["composition"].dtype, torch.int64)


class WhiteBoxMetricTests(unittest.TestCase):
    def setUp(self):
        self.ns = helper_namespace()
        self.spec = small_candidates()

    def test_log_scoring_matches_independent_logsumexp_calculation(self):
        crop = torch.arange(110, dtype=torch.float64).reshape(2, 55) / 10
        disease = torch.arange(350, dtype=torch.float64).reshape(2, 175) / 20
        actual = self.ns["composition_scores"](crop, disease, self.spec)
        c = crop - torch.logsumexp(crop, dim=1, keepdim=True)
        d = disease - torch.logsumexp(disease, dim=1, keepdim=True)
        expected = torch.stack([c[:, i] + d[:, i] for i in range(6)], dim=1)
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)

    def test_extreme_float32_logits_have_finite_stable_scores(self):
        maximum = torch.finfo(torch.float32).max
        crop = torch.full((2, 55), -maximum)
        disease = torch.full((2, 175), -maximum)
        crop[:, 0], disease[:, 0] = maximum, maximum
        score = self.ns["composition_scores"](crop, disease, self.spec)
        self.assertEqual(score.dtype, torch.float64)
        self.assertTrue(torch.isfinite(score).all())
        self.assertTrue((score[:, 0] > score[:, 1]).all())
        truth = torch.tensor([10, 10])
        self.assertEqual(self.ns["composition_topk"](score, truth, self.spec, 1), 1.0)

    def test_exact_ties_rank_ascending_global_ids(self):
        scores = torch.zeros(2, 6)
        truths = torch.tensor([10, 332])
        for _ in range(2):
            self.assertEqual(self.ns["composition_topk"](scores, truths, self.spec, 1), 0.5)
            self.assertEqual(self.ns["composition_topk"](scores, truths, self.spec, 5), 0.5)

    def test_candidate_hash_and_pair_mismatch_rejected(self):
        validate = self.ns["validate_candidate_spec"]
        _, _, checksum = validate(self.spec)
        validate({**self.spec, "sha256": checksum})
        for spec in (
            {**self.spec, "sha256": "0" * 64},
            {**self.spec, "ids": tuple(reversed(self.spec["ids"]))},
            {**self.spec, "pairs": ((0, 0),) * 6},
            {**self.spec, "pairs": ((55, 0),) + self.spec["pairs"][1:]},
        ):
            with self.assertRaises(ValueError):
                validate(spec)

    def test_unknown_truths_invalid_k_shapes_and_nonfinite_inputs_rejected(self):
        score = torch.zeros(1, 6)
        for truth, k in ((torch.tensor([1]), 1), (torch.tensor([10.0]), 1),
                         (torch.tensor([10]), 0), (torch.tensor([10]), 7)):
            with self.assertRaises(ValueError):
                self.ns["composition_topk"](score, truth, self.spec, k)
        with self.assertRaises(ValueError):
            self.ns["composition_scores"](torch.full((1, 55), float("nan")),
                                          torch.zeros(1, 175), self.spec)
        with self.assertRaises(ValueError):
            self.ns["composition_topk"](torch.zeros(0, 6), torch.tensor([], dtype=torch.long),
                                        self.spec, 1)

    def test_harmonic_mean_zero_and_small_values(self):
        harmonic = self.ns["harmonic_mean"]
        self.assertEqual(harmonic(0, 0), 0)
        self.assertEqual(harmonic(1, 0), 0)
        self.assertAlmostEqual(harmonic(0.5, 0.25), 1 / 3)
        self.assertEqual(harmonic(1, 1), 1)
        self.assertAlmostEqual(harmonic(1e-300, 1e-300) / 1e-300, 1)
        for bad in (-1, 2, float("nan"), True):
            with self.assertRaises(ValueError):
                harmonic(bad, 0.5)

    def test_complete_wp5_metric_contract_and_sample_count_validation(self):
        seen = {"sample_count": 2, "crop_logits": torch.zeros(2, 55),
                "disease_logits": torch.zeros(2, 175), "composition_ids": torch.tensor([10, 332])}
        unseen = {**seen, "composition_ids": torch.tensor([75, 200])}
        result = self.ns["evaluate_development"](seen, unseen, self.spec)
        expected = {"seen_top1": 0.5, "unseen_top1": 0.0, "seen_top5": 0.5,
                    "unseen_top5": 1.0, "top1_harmonic_mean": 0.0, "top5_harmonic_mean": 2 / 3}
        self.assertEqual(result, expected)
        with self.assertRaises(ValueError):
            self.ns["evaluate_development"]({**seen, "sample_count": 3}, unseen, self.spec)


class WhiteBoxModelAndLoopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = helper_namespace()

    def test_model_contract_shared_heads_and_finite_synthetic_gradient(self):
        counts = {"multitask_efficientnet_b0": 4302178, "multitask_convnext_tiny": 27996998}
        for architecture, count in counts.items():
            with self.subTest(architecture=architecture), torch.random.fork_rng(devices=[]):
                torch.manual_seed(9)
                model = self.ns["create_multitask_model"](architecture).cpu().eval()
                self.assertEqual(sum(p.numel() for p in model.parameters()), count)
                seen = []
                with model.backbone.register_forward_hook(lambda m, a, out: seen.append(out)):
                    outputs = model(torch.randn(2, 3, 32, 32))
                self.assertEqual(len(seen), 1)
                self.assertIs(seen[0], outputs["features"])
                self.assertEqual(tuple(outputs["crop_logits"].shape), (2, 55))
                self.assertEqual(tuple(outputs["disease_logits"].shape), (2, 175))
                self.assertEqual(tuple(outputs["features"].shape), (2, model.feature_dim))
                self.assertTrue(all(torch.isfinite(value).all() for value in outputs.values()))
                self.assertTrue({id(p) for p in model.crop_head.parameters()}.isdisjoint(
                    {id(p) for p in model.disease_head.parameters()}))
                loss = self.ns["compute_multitask_loss"](
                    outputs, {"crop": torch.tensor([0, 54]), "disease": torch.tensor([0, 174])})
                loss["total"].backward()
                for module in (model.backbone, model.crop_head, model.disease_head):
                    grads = [p.grad for p in module.parameters()]
                    self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in grads))
                    self.assertGreater(sum(g.abs().sum().item() for g in grads), 0)

    def test_constructor_and_acquisition_never_download_by_default(self):
        factory = self.ns["create_multitask_model"]
        with patch("torch.hub.download_url_to_file", side_effect=AssertionError("download")):
            for name in ("multitask_efficientnet_b0", "multitask_convnext_tiny"):
                model = factory(name)
                self.assertEqual(model.architecture, name)
        cfg = {**self.ns["CONFIG"], "initialization": "random"}
        with patch("torch.hub.download_url_to_file", side_effect=AssertionError("download")):
            with self.assertRaises(ValueError):
                self.ns["acquire_official_weights"](cfg)

    def test_visible_loss_matches_reference_and_weighted_formula(self):
        from train_multitask import compute_multitask_loss
        outputs = {"crop_logits": torch.randn(3, 55), "disease_logits": torch.randn(3, 175)}
        targets = {"crop": torch.tensor([0, 5, 54]), "disease": torch.tensor([0, 8, 174])}
        expected = compute_multitask_loss(outputs, targets, crop_weight=0.25, disease_weight=0.75)
        actual = self.ns["compute_multitask_loss"](outputs, targets, crop_weight=0.25, disease_weight=0.75)
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)

    def test_bounded_toy_training_updates_parameters_and_preserves_sample_counts(self):
        from test_training import TinyMultiTask, SyntheticPairs
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(43)
            model = TinyMultiTask()
            loader = DataLoader(SyntheticPairs(torch.randn(3, 3, 4, 4)), batch_size=2)
        before = model.crop_head.weight.detach().clone()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        with redirect_stdout(io.StringIO()), patch.object(optimizer, "step", wraps=optimizer.step) as step:
            result = self.ns["train_one_epoch"](model, loader, optimizer, "cpu")
        self.assertEqual(step.call_count, 2)
        self.assertEqual((result["sample_count"], result["batch_count"]), (3, 2))
        self.assertFalse(torch.equal(before, model.crop_head.weight))
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in model.parameters()))

    def test_validation_is_sample_weighted_and_preserves_state(self):
        from test_training import LookupMultiTask, SyntheticPairs
        model = LookupMultiTask()
        loader = DataLoader(SyntheticPairs(torch.arange(3, dtype=torch.float32)[:, None]), batch_size=2)
        before = {key: value.clone() for key, value in model.state_dict().items()}
        actual = self.ns["validate_one_epoch"](model, loader, "cpu")
        expected = self.ns["compute_multitask_loss"](
            model(torch.arange(3, dtype=torch.float32)[:, None]),
            {"crop": torch.zeros(3, dtype=torch.long), "disease": torch.zeros(3, dtype=torch.long)})
        for key in expected:
            self.assertAlmostEqual(actual["losses"][key], expected[key].item(), places=6)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)
        self.assertEqual(actual["composition_ids"].tolist(), [10, 75, 332])

    def test_checkpoint_both_directions_with_reference_modules(self):
        from multitask_backbones import create_multitask_model
        from train_multitask import load_checkpoint, save_checkpoint
        for architecture in ("multitask_efficientnet_b0", "multitask_convnext_tiny"):
            with self.subTest(architecture=architecture), tempfile.TemporaryDirectory() as directory:
                original = self.ns["create_multitask_model"](architecture).cpu().eval()
                for module in original.modules():
                    if isinstance(module, nn.BatchNorm2d):
                        module.running_mean.fill_(0.125)
                        module.num_batches_tracked.fill_(7)
                path = Path(directory) / "notebook.pt"
                self.ns["save_checkpoint"](original, path, metadata={"epoch": 1})
                reference, info = load_checkpoint(path, expected_architecture=architecture)
                self.assertEqual(info["format_version"], 1)
                for key, value in original.state_dict().items():
                    torch.testing.assert_close(reference.state_dict()[key], value, rtol=0, atol=0)
                reverse = Path(directory) / "reference.pt"
                save_checkpoint(reference, reverse)
                loaded, _ = self.ns["load_checkpoint"](reverse, expected_architecture=architecture)
                with torch.inference_mode():
                    image = torch.randn(1, 3, 32, 32)
                    expected, actual = original(image), loaded(image)
                for key in expected:
                    torch.testing.assert_close(actual[key], expected[key], rtol=1e-5, atol=1e-6)
                with self.assertRaises(FileExistsError):
                    self.ns["save_checkpoint"](original, path)

    def test_checkpoint_mapping_preprocessing_and_architecture_checks(self):
        from multitask_backbones import create_multitask_model
        model = create_multitask_model("multitask_efficientnet_b0").eval()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reference.pt"
            from train_multitask import save_checkpoint
            mappings = {"crop_id_to_name": {i: f"c{i}" for i in range(55)},
                        "disease_id_to_name": {i: f"d{i}" for i in range(175)}}
            save_checkpoint(model, path, class_mappings=mappings, preprocessing={"size": 224})
            swapped = copy.deepcopy(mappings)
            swapped["crop_id_to_name"][0], swapped["crop_id_to_name"][1] = "c1", "c0"
            for kwargs in ({}, {"expected_class_mappings": swapped},
                           {"expected_class_mappings": mappings, "expected_preprocessing": {"size": 256}},
                           {"expected_class_mappings": mappings,
                            "expected_architecture": "multitask_convnext_tiny"}):
                with self.assertRaises(ValueError):
                    self.ns["load_checkpoint"](path, **kwargs)


if __name__ == "__main__":
    unittest.main()
