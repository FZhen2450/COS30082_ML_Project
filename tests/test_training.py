"""M3B epoch utilities: tiny synthetic CPU models, at most two steps per run."""

import copy
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from development_data import build_development_split, get_development_dataloaders
from test_development_data import synthetic_training_csv
from train_multitask import compute_multitask_loss, train_one_epoch, validate_one_epoch


ROOT = Path(__file__).resolve().parents[1]


class SyntheticPairs(Dataset):
    def __init__(self, images, tuple_targets=False, include_composition=True):
        self.images = images
        self.tuple_targets = tuple_targets
        self.targets = {
            "crop": torch.zeros(len(images), dtype=torch.int64),
            "disease": torch.zeros(len(images), dtype=torch.int64),
        }
        if include_composition:
            self.targets["composition"] = torch.tensor(
                [10, 75, 332][:len(images)], dtype=torch.int64
            )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        targets = {key: value[index] for key, value in self.targets.items()}
        if self.tuple_targets:
            targets = tuple(targets.values())
        return self.images[index], targets


class TinyMultiTask(nn.Module):
    def __init__(self, include_features=True):
        super().__init__()
        self.shared = nn.Conv2d(3, 4, 1)
        self.bn = nn.BatchNorm2d(4)
        self.crop_head = nn.Linear(4, 55)
        self.disease_head = nn.Linear(4, 175)
        self.include_features = include_features
        self.forward_flags = []

    def forward(self, images):
        self.forward_flags.append((self.training, torch.is_grad_enabled()))
        features = self.bn(self.shared(images)).relu().mean(dim=(2, 3))
        outputs = {
            "crop_logits": self.crop_head(features),
            "disease_logits": self.disease_head(features),
        }
        if self.include_features:
            outputs["features"] = features
        return outputs


class LookupMultiTask(nn.Module):
    """Stable row-specific logits make partial-batch weighting independently checkable."""
    def __init__(self):
        super().__init__()
        crop = torch.zeros(3, 55)
        disease = torch.zeros(3, 175)
        crop[:, 0] = torch.tensor([6.0, 6.0, -6.0])
        disease[:, 0] = torch.tensor([5.0, -5.0, 1.0])
        self.crop = nn.Parameter(crop)
        self.disease = nn.Parameter(disease)

    def forward(self, inputs):
        indices = inputs[:, 0].long()
        return {"crop_logits": self.crop[indices], "disease_logits": self.disease[indices]}


class EpochTrainingTests(unittest.TestCase):
    def setUp(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            self.model = TinyMultiTask()
            self.images = torch.randn(3, 3, 4, 4)
        self.loader = DataLoader(SyntheticPairs(self.images), batch_size=2, shuffle=False)

    def test_explicit_training_sets_mode_and_updates_with_two_steps(self):
        self.model.eval()
        before = {name: value.detach().clone() for name, value in self.model.named_parameters()}
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
        with patch.object(optimizer, "step", wraps=optimizer.step) as step:
            result = train_one_epoch(self.model, self.loader, optimizer, "cpu")
        self.assertEqual(step.call_count, 2)
        self.assertEqual((result["sample_count"], result["batch_count"]), (3, 2))
        self.assertTrue(self.model.training)
        self.assertEqual(self.model.forward_flags, [(True, True), (True, True)])
        self.assertTrue(any(not torch.equal(before[name], value)
                            for name, value in self.model.named_parameters()))
        for parameter in self.model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all().item())
        self.assertEqual(set(result["losses"]), {"total", "crop", "disease"})
        self.assertTrue(all(isinstance(value, float) for value in result["losses"].values()))

    def test_training_uses_sample_weighting_for_final_partial_batch(self):
        model = LookupMultiTask()
        data = SyntheticPairs(torch.arange(3, dtype=torch.float32)[:, None])
        loader = DataLoader(data, batch_size=2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
        with torch.no_grad():
            expected = compute_multitask_loss(
                model(data.images), data.targets, crop_weight=0.25, disease_weight=0.75
            )
            first = compute_multitask_loss(
                model(data.images[:2]), {key: value[:2] for key, value in data.targets.items()},
                crop_weight=0.25, disease_weight=0.75,
            )
            last = compute_multitask_loss(
                model(data.images[2:]), {key: value[2:] for key, value in data.targets.items()},
                crop_weight=0.25, disease_weight=0.75,
            )
        result = train_one_epoch(
            model, loader, optimizer, "cpu", crop_weight=0.25, disease_weight=0.75
        )
        for key in expected:
            self.assertAlmostEqual(result["losses"][key], expected[key].item(), places=6)
            self.assertNotAlmostEqual(
                result["losses"][key], (first[key].item() + last[key].item()) / 2, places=4
            )

    def test_training_supports_no_features_and_tuple_targets(self):
        model = TinyMultiTask(include_features=False)
        loader = DataLoader(SyntheticPairs(self.images[:2], tuple_targets=True), batch_size=2)
        result = train_one_epoch(model, loader, torch.optim.SGD(model.parameters(), lr=0.01), "cpu")
        self.assertEqual(result["sample_count"], 2)

    def test_training_does_not_require_composition_targets(self):
        loader = DataLoader(SyntheticPairs(self.images[:2], include_composition=False), batch_size=2)
        result = train_one_epoch(
            self.model, loader, torch.optim.SGD(self.model.parameters(), lr=0.01), "cpu"
        )
        self.assertEqual(result["sample_count"], 2)

    def test_nonfinite_gradients_are_rejected_before_optimizer_step(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
        before = {name: value.detach().clone() for name, value in self.model.named_parameters()}
        handle = self.model.crop_head.weight.register_hook(
            lambda gradient: torch.full_like(gradient, float("nan"))
        )
        try:
            with patch.object(optimizer, "step", wraps=optimizer.step) as step:
                with self.assertRaises(ValueError):
                    train_one_epoch(self.model, self.loader, optimizer, "cpu")
                step.assert_not_called()
        finally:
            handle.remove()
        for name, value in self.model.named_parameters():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)

    def test_unrelated_optimizer_parameters_are_rejected(self):
        other = TinyMultiTask()
        optimizer = torch.optim.SGD(other.parameters(), lr=0.1)
        with self.assertRaises(ValueError):
            train_one_epoch(self.model, self.loader, optimizer, "cpu")


    def test_explicit_training_enables_grad_inside_no_grad(self):
        loader = DataLoader(SyntheticPairs(self.images[:2]), batch_size=2)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        with torch.no_grad():
            result = train_one_epoch(self.model, loader, optimizer, "cpu")
        self.assertEqual(result["sample_count"], 2)
        self.assertEqual(self.model.forward_flags, [(True, True)])

    def test_training_rejects_outer_inference_mode_without_stepping(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        with torch.inference_mode(), patch.object(
            optimizer, "step", wraps=optimizer.step
        ) as step:
            with self.assertRaises(ValueError):
                train_one_epoch(self.model, self.loader, optimizer, "cpu")
            step.assert_not_called()



class EpochValidationTests(unittest.TestCase):
    def setUp(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(7)
            self.model = TinyMultiTask()
            self.images = torch.randn(3, 3, 4, 4)
        self.loader = DataLoader(SyntheticPairs(self.images), batch_size=2, shuffle=False)

    def test_validation_is_inference_only_and_preserves_parameters_buffers_and_gradients(self):
        self.model.train()
        with torch.no_grad():
            self.model.bn.running_mean.fill_(0.125)
            self.model.bn.running_var.fill_(1.5)
            self.model.bn.num_batches_tracked.fill_(9)
        for parameter in self.model.parameters():
            parameter.grad = torch.ones_like(parameter)
        before = copy.deepcopy(self.model.state_dict())
        before_grads = [parameter.grad.clone() for parameter in self.model.parameters()]
        result = validate_one_epoch(self.model, self.loader, "cpu")
        self.assertFalse(self.model.training)
        self.assertEqual(self.model.forward_flags, [(False, False), (False, False)])
        for name, value in self.model.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)
        for parameter, gradient in zip(self.model.parameters(), before_grads):
            torch.testing.assert_close(parameter.grad, gradient, rtol=0, atol=0)
        self.assertEqual((result["sample_count"], result["batch_count"]), (3, 2))
        for name, shape in (
            ("crop_logits", (3, 55)), ("disease_logits", (3, 175)), ("composition_ids", (3,))
        ):
            value = result[name]
            self.assertEqual(tuple(value.shape), shape)
            self.assertEqual(value.device.type, "cpu")
            self.assertFalse(value.requires_grad)
            self.assertIsNone(value.grad_fn)
        self.assertEqual(result["composition_ids"].dtype, torch.int64)
        self.assertEqual(result["composition_ids"].tolist(), [10, 75, 332])

    def test_validation_losses_match_full_sample_mean(self):
        model = LookupMultiTask()
        data = SyntheticPairs(torch.arange(3, dtype=torch.float32)[:, None])
        loader = DataLoader(data, batch_size=2)
        with torch.no_grad():
            expected = compute_multitask_loss(model(data.images), data.targets)
        result = validate_one_epoch(model, loader, "cpu")
        for key, value in expected.items():
            self.assertAlmostEqual(result["losses"][key], value.item(), places=6)
        torch.testing.assert_close(result["crop_logits"], model.crop.detach())
        torch.testing.assert_close(result["disease_logits"], model.disease.detach())
        self.assertEqual(result["composition_ids"].tolist(), [10, 75, 332])

    def test_validation_supports_tuple_targets_without_features(self):
        model = TinyMultiTask(include_features=False)
        loader = DataLoader(SyntheticPairs(self.images, tuple_targets=True), batch_size=2)
        result = validate_one_epoch(model, loader, "cpu:0")
        self.assertEqual(result["composition_ids"].tolist(), [10, 75, 332])

    def test_validation_works_with_development_loaders_using_only_synthetic_images(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            csv = synthetic_training_csv(path, write_images=True)
            split = build_development_split(csv, heldout_fraction=0.25)
            _, seen, unseen = get_development_dataloaders(
                split, path, batch_size=4, image_size=16, augment_fit=False
            )
            for loader, frame in ((seen, split.val_seen), (unseen, split.val_unseen)):
                result = validate_one_epoch(self.model, loader, "cpu")
                self.assertEqual(result["sample_count"], len(frame))
                self.assertEqual(result["composition_ids"].tolist(), frame.composition_id.tolist())
                self.assertEqual(tuple(result["crop_logits"].shape), (len(frame), 55))
                self.assertEqual(tuple(result["disease_logits"].shape), (len(frame), 175))

    def test_validation_snapshots_reused_output_storage(self):
        class ReusedStorageModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.crop = torch.zeros(1, 55)
                self.disease = torch.zeros(1, 175)

            def forward(self, inputs):
                self.crop.fill_(float(inputs[0, 0]))
                self.disease.fill_(float(inputs[0, 0]))
                return {"crop_logits": self.crop, "disease_logits": self.disease}

        model = ReusedStorageModel()
        data = SyntheticPairs(torch.arange(3, dtype=torch.float32)[:, None])
        result = validate_one_epoch(model, DataLoader(data, batch_size=1), "cpu")
        self.assertEqual(result["crop_logits"][:, 0].tolist(), [0.0, 1.0, 2.0])
        self.assertEqual(result["disease_logits"][:, 0].tolist(), [0.0, 1.0, 2.0])


    def test_nonfinite_epoch_loss_aggregation_is_rejected(self):
        class LargeFiniteLoss(nn.Module):
            def forward(self, inputs):
                crop = torch.zeros(len(inputs), 55, dtype=torch.float64)
                disease = torch.zeros(len(inputs), 175, dtype=torch.float64)
                crop[:, 0] = -1e308
                disease[:, 0] = -1e308
                return {"crop_logits": crop, "disease_logits": disease}

        loader = DataLoader(
            SyntheticPairs(torch.arange(3, dtype=torch.float32)[:, None]), batch_size=1
        )
        model = LargeFiniteLoss()
        inputs, targets = next(iter(loader))
        individual = compute_multitask_loss(model(inputs), targets)
        self.assertTrue(all(torch.isfinite(value).item() for value in individual.values()))
        with self.assertRaises(ValueError):
            validate_one_epoch(model, loader, "cpu")



class EpochInputValidationTests(unittest.TestCase):
    def setUp(self):
        self.model = TinyMultiTask()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
        self.images = torch.zeros(2, 3, 4, 4)
        self.targets = {
            "crop": torch.zeros(2, dtype=torch.int64),
            "disease": torch.zeros(2, dtype=torch.int64),
            "composition": torch.tensor([10, 332]),
        }

    def assert_both_reject(self, loader, error=(TypeError, ValueError)):
        with self.assertRaises(error):
            train_one_epoch(self.model, loader, self.optimizer, "cpu")
        with self.assertRaises(error):
            validate_one_epoch(self.model, loader, "cpu")

    def test_empty_loaders_and_drop_last_without_batches_are_rejected(self):
        for size, drop_last in ((0, False), (1, True)):
            with self.subTest(size=size, drop_last=drop_last):
                loader = DataLoader(
                    SyntheticPairs(self.images[:size]), batch_size=2, drop_last=drop_last
                )
                with patch.object(self.optimizer, "step", wraps=self.optimizer.step) as step:
                    self.assert_both_reject(loader)
                    step.assert_not_called()

    def test_invalid_batch_and_target_contracts_are_rejected(self):
        cases = [
            None, self.images, (self.images,),
            ("not a tensor", self.targets),
            (self.images.long(), self.targets),
            (torch.tensor(1.0), self.targets),
            (torch.full_like(self.images, float("nan")), self.targets),
            (self.images, {"crop": self.targets["crop"]}),
            (self.images, {**self.targets, "crop": self.targets["crop"].float()}),
            (self.images, {**self.targets, "disease": self.targets["disease"][:1]}),
            (self.images, {**self.targets, "composition": torch.tensor([10.0, 332.0])}),
            (self.images, {**self.targets, "composition": torch.tensor([10, -1])}),
            (self.images, (self.targets["crop"],)),
        ]
        for index, batch in enumerate(cases):
            with self.subTest(case=index):
                loader = DataLoader([0], batch_size=1, collate_fn=lambda values: batch)
                self.assert_both_reject(loader)

    def test_validation_requires_global_composition_truths(self):
        loader = DataLoader(SyntheticPairs(self.images, include_composition=False), batch_size=2)
        with self.assertRaises(ValueError):
            validate_one_epoch(self.model, loader, "cpu")

    def test_nonfinite_and_malformed_model_outputs_are_rejected_before_step(self):
        loader = DataLoader(SyntheticPairs(self.images), batch_size=2)
        for mode in ("nan", "width", "not_mapping"):
            class BadOutput(TinyMultiTask):
                def forward(self, inputs):
                    outputs = super().forward(inputs)
                    if mode == "nan":
                        outputs["crop_logits"] = outputs["crop_logits"] * float("nan")
                    elif mode == "width":
                        outputs["crop_logits"] = outputs["crop_logits"][:, :54]
                    else:
                        return outputs["crop_logits"]
                    return outputs

            with self.subTest(mode=mode):
                model = BadOutput()
                optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
                with patch.object(optimizer, "step", wraps=optimizer.step) as step:
                    with self.assertRaises((TypeError, ValueError)):
                        train_one_epoch(model, loader, optimizer, "cpu")
                    step.assert_not_called()
                with self.assertRaises((TypeError, ValueError)):
                    validate_one_epoch(model, loader, "cpu")

    def test_invalid_weights_are_rejected_before_iteration(self):
        loader = DataLoader(SyntheticPairs(self.images), batch_size=2)
        for kwargs in ({"crop_weight": -1}, {"disease_weight": float("inf")},
                       {"crop_weight": 0, "disease_weight": 0}):
            with self.subTest(kwargs=kwargs), patch.object(
                type(loader), "__iter__", side_effect=AssertionError("Should not iterate")
            ):
                with self.assertRaises(ValueError):
                    train_one_epoch(self.model, loader, self.optimizer, "cpu", **kwargs)
                with self.assertRaises(ValueError):
                    validate_one_epoch(self.model, loader, "cpu", **kwargs)

    def test_invalid_models_loaders_optimizers_and_devices_are_rejected(self):
        loader = DataLoader(SyntheticPairs(self.images), batch_size=2)
        with self.assertRaises(TypeError):
            train_one_epoch(object(), loader, self.optimizer, "cpu")
        with self.assertRaises(TypeError):
            validate_one_epoch(self.model, [], "cpu")
        with self.assertRaises(TypeError):
            train_one_epoch(self.model, loader, object(), "cpu")
        with self.assertRaises(ValueError):
            validate_one_epoch(self.model, loader, "meta")
        with self.assertRaises((TypeError, RuntimeError, ValueError)):
            validate_one_epoch(self.model, loader, "invalid-device")


class EpochImportSafetyTests(unittest.TestCase):
    def test_import_never_iterates_steps_backpropagates_or_writes(self):
        script = r"""
import importlib
import os
import sys
from unittest.mock import patch
import torch
import torchvision

def audit(event, args):
    if event == "open":
        _, mode, flags = args
        if ((isinstance(mode, str) and any(c in mode for c in "wax+")) or
            (isinstance(flags, int) and flags & (
                os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))):
            raise AssertionError("File write on import")
    if event.startswith("socket.") or event in {
        "os.mkdir", "os.remove", "os.rename", "os.rmdir", "os.link", "os.symlink",
        "subprocess.Popen", "os.system", "urllib.Request"
    }:
        raise AssertionError(f"Unexpected side effect: {event}")

sys.addaudithook(audit)
with (
    patch.object(torch.nn.Module, "__init__", side_effect=AssertionError("Model creation")),
    patch.object(torch.utils.data.DataLoader, "__iter__", side_effect=AssertionError("Iteration")),
    patch.object(torch.optim.SGD, "step", side_effect=AssertionError("Optimizer step")),
    patch.object(torch.optim.AdamW, "step", side_effect=AssertionError("Optimizer step")),
    patch.object(torch.autograd, "backward", side_effect=AssertionError("Backward")),
    patch.object(torch.hub, "download_url_to_file", side_effect=AssertionError("Download")),
):
    importlib.import_module("train_multitask")
print("IMPORT_SAFE")
"""
        result = subprocess.run(
            [sys.executable, "-B", "-c", script], cwd=ROOT,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1"},
            capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(result.stdout.strip(), "IMPORT_SAFE")


if __name__ == "__main__":
    unittest.main()
