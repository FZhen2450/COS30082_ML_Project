"""CPU-only M1/M2 tests using synthetic tensors and temporary checkpoints."""

import copy
import hashlib
import json
import os
import pickle
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
from torchvision.models.resnet import ResNet

from multitask_model import MultiTaskResNet18
from train_multitask import compute_multitask_loss, load_checkpoint, save_checkpoint


ROOT = Path(__file__).resolve().parents[1]


class MultiTaskResNet18Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            cls.model = MultiTaskResNet18().cpu().eval()
            cls.images = torch.randn(2, 3, 224, 224)

    def test_forward_shapes_and_finite_outputs(self):
        expected_widths = {"crop_logits": 55, "disease_logits": 175, "features": 512}
        for batch_size in (1, 2):
            with self.subTest(batch_size=batch_size), torch.inference_mode():
                outputs = self.model(self.images[:batch_size])
                self.assertEqual(set(outputs), set(expected_widths))
                for name, width in expected_widths.items():
                    self.assertEqual(tuple(outputs[name].shape), (batch_size, width))
                    self.assertEqual(outputs[name].dtype, torch.float32)
                    self.assertEqual(outputs[name].device.type, "cpu")
                    self.assertTrue(torch.isfinite(outputs[name]).all().item())

    def test_both_heads_receive_the_same_backbone_representation(self):
        captured = {"backbone": [], "crop": [], "disease": []}
        with (
            self.model.backbone.register_forward_hook(
                lambda module, args, output: captured["backbone"].append(output)
            ),
            self.model.crop_head.register_forward_pre_hook(
                lambda module, args: captured["crop"].append(args[0])
            ),
            self.model.disease_head.register_forward_pre_hook(
                lambda module, args: captured["disease"].append(args[0])
            ),
            torch.inference_mode(),
        ):
            outputs = self.model(self.images[:1])
        for values in captured.values():
            self.assertEqual(len(values), 1)
            self.assertIs(values[0], outputs["features"])

    def test_backbone_is_one_resnet18_with_global_average_pooling(self):
        backbones = [module for module in self.model.modules() if isinstance(module, ResNet)]
        self.assertEqual(len(backbones), 1)
        self.assertIs(backbones[0], self.model.backbone)
        self.assertEqual(
            [len(getattr(self.model.backbone, f"layer{i}")) for i in range(1, 5)],
            [2, 2, 2, 2],
        )
        self.assertIsInstance(self.model.backbone.fc, nn.Identity)
        self.assertIsInstance(self.model.backbone.avgpool, nn.AdaptiveAvgPool2d)
        self.assertEqual(self.model.backbone.avgpool.output_size, (1, 1))

    def test_heads_have_required_dimensions_and_independent_parameters(self):
        self.assertIsInstance(self.model.crop_head, nn.Linear)
        self.assertIsInstance(self.model.disease_head, nn.Linear)
        self.assertEqual(self.model.crop_head.in_features, 512)
        self.assertEqual(self.model.crop_head.out_features, 55)
        self.assertEqual(self.model.disease_head.in_features, 512)
        self.assertEqual(self.model.disease_head.out_features, 175)
        groups = [
            {id(parameter) for parameter in module.parameters()}
            for module in (self.model.backbone, self.model.crop_head, self.model.disease_head)
        ]
        for index, group in enumerate(groups):
            self.assertTrue(group)
            for other in groups[index + 1:]:
                self.assertTrue(group.isdisjoint(other))

    def test_outputs_are_raw_linear_logits(self):
        with torch.inference_mode():
            outputs = self.model(self.images[:1])
            for name, head in (
                ("crop_logits", self.model.crop_head),
                ("disease_logits", self.model.disease_head),
            ):
                expected = nn.functional.linear(outputs["features"], head.weight, head.bias)
                torch.testing.assert_close(outputs[name], expected)

    def test_evaluation_is_repeatable_and_preserves_buffers(self):
        self.assertTrue(all(not module.training for module in self.model.modules()))
        before = {name: value.clone() for name, value in self.model.named_buffers()}
        with torch.inference_mode():
            first = self.model(self.images[:1])
            second = self.model(self.images[:1])
        for name in first:
            torch.testing.assert_close(first[name], second[name])
        for name, value in self.model.named_buffers():
            torch.testing.assert_close(value, before[name])

    def test_import_and_construction_have_no_external_side_effects(self):
        # Isolate the project import after dependencies load. The audit hook is
        # process-local and rejects writes, networking and subprocess launches.
        script = r"""
import importlib
import json
import os
import sys
from unittest.mock import patch

import torch
import torchvision.models as models

write_flags = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
mutations = {
    "os.mkdir", "os.remove", "os.rename", "os.rmdir", "os.truncate",
    "os.link", "os.symlink", "os.chmod", "os.chown", "os.utime",
}

def reject_side_effects(event, args):
    if event == "open":
        _, mode, flags = args
        if (
            isinstance(mode, str) and any(char in mode for char in "wax+")
        ) or (
            isinstance(flags, int) and flags & write_flags
        ):
            raise AssertionError(f"Unexpected file write: {args[0]}")
    if (
        event in mutations
        or event.startswith("socket.")
        or event in {"urllib.Request", "subprocess.Popen", "os.system"}
    ):
        raise AssertionError(f"Unexpected external side effect: {event}")

sys.addaudithook(reject_side_effects)
with (
    patch.object(models, "resnet18", wraps=models.resnet18) as builder,
    patch.object(torch.hub, "download_url_to_file",
                 side_effect=AssertionError("Unexpected weight download")),
    patch.object(torch.nn.Module, "_call_impl",
                 side_effect=AssertionError("Unexpected forward pass")),
    patch.object(torch.autograd, "backward",
                 side_effect=AssertionError("Unexpected backward pass")),
):
    with patch.object(
        torch.nn.Module, "__init__",
        side_effect=AssertionError("Import constructed a model"),
    ):
        module = importlib.import_module("multitask_model")
        importlib.import_module("train_multitask")
    builder.assert_not_called()
    model = module.MultiTaskResNet18()
    builder.assert_called_once_with(weights=None)
    assert all(parameter.device.type == "cpu" for parameter in model.parameters())
print(json.dumps({"import_and_construction": "passed"}))
"""
        result = subprocess.run(
            [sys.executable, "-B", "-c", script],
            cwd=ROOT,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": ""},
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(
            json.loads(result.stdout), {"import_and_construction": "passed"}
        )


class MultiTaskLossTests(unittest.TestCase):
    def setUp(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(7)
            self.outputs = {
                "crop_logits": torch.randn(3, 55, dtype=torch.float64),
                "disease_logits": torch.randn(3, 175, dtype=torch.float64),
            }
        self.targets = {
            "crop": torch.tensor([0, 12, 54], dtype=torch.int64),
            "disease": torch.tensor([0, 82, 174], dtype=torch.int64),
        }

    def test_losses_are_finite_and_match_weighted_cross_entropy(self):
        expected_crop = nn.functional.cross_entropy(
            self.outputs["crop_logits"], self.targets["crop"]
        )
        expected_disease = nn.functional.cross_entropy(
            self.outputs["disease_logits"], self.targets["disease"]
        )
        for weights in ({}, {"crop_weight": 0.25, "disease_weight": 0.75},
                        {"crop_weight": 2.0, "disease_weight": 3.0}):
            with self.subTest(weights=weights):
                losses = compute_multitask_loss(self.outputs, self.targets, **weights)
                self.assertEqual(set(losses), {"total", "crop", "disease"})
                for value in losses.values():
                    self.assertEqual(value.ndim, 0)
                    self.assertTrue(torch.isfinite(value).item())
                torch.testing.assert_close(losses["crop"], expected_crop)
                torch.testing.assert_close(losses["disease"], expected_disease)
                expected = (weights.get("crop_weight", 0.5) * expected_crop
                            + weights.get("disease_weight", 0.5) * expected_disease)
                torch.testing.assert_close(losses["total"], expected)

    def test_invalid_loss_weights_are_rejected(self):
        for crop, disease in [
            (-1, 0.5), (0.5, -1), (0, 0), (float("nan"), 0.5),
            (float("inf"), 0.5), (True, 0.5), ("0.5", 0.5),
        ]:
            with self.subTest(crop=crop, disease=disease):
                with self.assertRaises((TypeError, ValueError)):
                    compute_multitask_loss(
                        self.outputs, self.targets, crop_weight=crop, disease_weight=disease
                    )

    def test_incompatible_labels_and_logits_are_rejected(self):
        cases = [
            (self.outputs, {**self.targets, "crop": self.targets["crop"].float()}),
            (self.outputs, {**self.targets, "crop": torch.tensor([0, -1, 54])}),
            (self.outputs, {**self.targets, "disease": torch.tensor([0, 82, 175])}),
            (self.outputs, {**self.targets, "crop": self.targets["crop"][:2]}),
            (self.outputs, {**self.targets, "crop": self.targets["crop"][:, None]}),
            ({**self.outputs, "crop_logits": torch.zeros(3, 54)}, self.targets),
            ({**self.outputs, "disease_logits": torch.zeros(2, 175)}, self.targets),
            ({**self.outputs, "crop_logits": torch.zeros(3, 55, dtype=torch.int64)},
             self.targets),
            ({**self.outputs, "crop_logits": torch.full((3, 55), float("nan"))},
             self.targets),
            ({"crop_logits": self.outputs["crop_logits"]}, self.targets),
            (self.outputs, {"crop": self.targets["crop"]}),
            ({"crop_logits": torch.empty(0, 55), "disease_logits": torch.empty(0, 175)},
             {"crop": torch.empty(0, dtype=torch.long),
              "disease": torch.empty(0, dtype=torch.long)}),
        ]
        for index, (outputs, targets) in enumerate(cases):
            with self.subTest(case=index), self.assertRaises((TypeError, ValueError)):
                compute_multitask_loss(outputs, targets)

    def test_zero_task_weight_blocks_its_logit_gradients(self):
        for active, inactive in (("crop", "disease"), ("disease", "crop")):
            with self.subTest(active=active):
                outputs = {key: value.clone().requires_grad_()
                           for key, value in self.outputs.items()}
                losses = compute_multitask_loss(
                    outputs, self.targets,
                    crop_weight=float(active == "crop"),
                    disease_weight=float(active == "disease"),
                )
                losses["total"].backward()
                self.assertGreater(outputs[f"{active}_logits"].grad.abs().sum().item(), 0)
                self.assertEqual(outputs[f"{inactive}_logits"].grad.abs().sum().item(), 0)


class MultiTaskGradientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(43)
            cls.model = MultiTaskResNet18().cpu().eval()
            cls.images = torch.randn(2, 3, 224, 224)
        cls.targets = {
            "crop": torch.tensor([0, 54], dtype=torch.int64),
            "disease": torch.tensor([0, 174], dtype=torch.int64),
        }

    def setUp(self):
        self.model.zero_grad(set_to_none=True)

    def assert_finite_nonzero_gradients(self, module):
        gradients = [parameter.grad for parameter in module.parameters()]
        self.assertTrue(gradients)
        self.assertTrue(all(gradient is not None for gradient in gradients))
        self.assertTrue(all(torch.isfinite(gradient).all().item() for gradient in gradients))
        self.assertGreater(sum(gradient.abs().sum().item() for gradient in gradients), 0)

    def test_total_loss_reaches_backbone_and_both_heads(self):
        losses = compute_multitask_loss(self.model(self.images), self.targets)
        losses["total"].backward()
        for module in (self.model.backbone, self.model.crop_head, self.model.disease_head):
            self.assert_finite_nonzero_gradients(module)

    def test_crop_component_reaches_only_crop_head_and_backbone(self):
        losses = compute_multitask_loss(self.model(self.images), self.targets)
        losses["crop"].backward()
        self.assert_finite_nonzero_gradients(self.model.backbone)
        self.assert_finite_nonzero_gradients(self.model.crop_head)
        self.assertTrue(all(p.grad is None for p in self.model.disease_head.parameters()))

    def test_disease_component_reaches_only_disease_head_and_backbone(self):
        losses = compute_multitask_loss(self.model(self.images), self.targets)
        losses["disease"].backward()
        self.assert_finite_nonzero_gradients(self.model.backbone)
        self.assert_finite_nonzero_gradients(self.model.disease_head)
        self.assertTrue(all(p.grad is None for p in self.model.crop_head.parameters()))


class MultiTaskCheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(11)
            cls.model = MultiTaskResNet18().cpu().eval()
            cls.images = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            for module in cls.model.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.running_mean.fill_(0.125)
                    module.running_var.fill_(1.5)
                    module.num_batches_tracked.fill_(7)
        cls.mappings = {
            "crop_id_to_name": {i: f"crop_{i}" for i in range(55)},
            "disease_id_to_name": {i: f"disease_{i}" for i in range(175)},
        }
        cls.preprocessing = {
            "image_size": 224, "resize": "square", "normalize_imagenet": True,
            "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
        }

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "model.pt"

    def tearDown(self):
        self.temp.cleanup()

    def test_round_trip_preserves_state_metadata_and_evaluation_outputs(self):
        original_mode = self.model.training
        saved_path = save_checkpoint(
            self.model, self.path, class_mappings=self.mappings,
            preprocessing=self.preprocessing, metadata={"seed": 11, "purpose": "synthetic"},
        )
        self.assertEqual(saved_path, self.path)
        self.assertEqual(self.model.training, original_mode)
        # Dictionary insertion order is irrelevant; IDs determine class identity.
        expected_mappings = {
            key: dict(reversed(list(values.items()))) for key, values in self.mappings.items()
        }
        with (
            patch("torch.hub.download_url_to_file",
                  side_effect=AssertionError("Unexpected weight download")),
            patch("train_multitask.torch.load", wraps=torch.load) as reader,
        ):
            loaded, info = load_checkpoint(
                self.path, expected_class_mappings=expected_mappings,
                expected_preprocessing=self.preprocessing,
            )
            reader.assert_called_once_with(
                self.path, map_location="cpu", weights_only=True
            )
        self.assertEqual(info["format_version"], 1)
        self.assertEqual(info["architecture"], "multitask_resnet18")
        self.assertEqual((info["num_crops"], info["num_diseases"], info["feature_dim"]),
                         (55, 175, 512))
        self.assertEqual(info["class_mappings"], self.mappings)
        self.assertEqual(info["preprocessing"], self.preprocessing)
        self.assertEqual(info["metadata"], {"seed": 11, "purpose": "synthetic"})
        self.assertIn("torch", info["runtime"])
        self.assertNotIn("model_state_dict", info)
        self.assertTrue(all(not module.training for module in loaded.modules()))
        self.assertEqual(set(loaded.state_dict()), set(self.model.state_dict()))
        for name, tensor in self.model.state_dict().items():
            torch.testing.assert_close(loaded.state_dict()[name], tensor, rtol=0, atol=0)
        with torch.inference_mode():
            expected = self.model(self.images)
            actual = loaded(self.images)
        for name in expected:
            torch.testing.assert_close(actual[name], expected[name], rtol=1e-5, atol=1e-6)

    def test_existing_checkpoint_requires_explicit_overwrite(self):
        save_checkpoint(self.model, self.path, metadata={"revision": 1})
        before = hashlib.sha256(self.path.read_bytes()).hexdigest()
        with self.assertRaises(FileExistsError):
            save_checkpoint(self.model, self.path, metadata={"revision": 2})
        self.assertEqual(hashlib.sha256(self.path.read_bytes()).hexdigest(), before)
        save_checkpoint(self.model, self.path, metadata={"revision": 2}, overwrite=True)
        _, info = load_checkpoint(self.path)
        self.assertEqual(info["metadata"]["revision"], 2)

    def test_save_rejects_incompatible_head_dimensions(self):
        with patch.object(self.model, "crop_head", nn.Linear(512, 54)):
            with self.assertRaises(ValueError):
                save_checkpoint(self.model, self.path)
        self.assertFalse(self.path.exists())

    def test_load_rejects_incompatible_header(self):
        header = {
            "format_version": 1, "architecture": "multitask_resnet18",
            "num_crops": 55, "num_diseases": 175, "feature_dim": 512,
            "model_state_dict": {}, "model_dtype": "torch.float32",
            "class_mappings": None, "preprocessing": None,
            "metadata": {}, "runtime": {},
        }
        for key, value in [
            ("format_version", 99), ("architecture", "resnet50"),
            ("num_crops", 54), ("num_diseases", 176), ("feature_dim", 1024),
        ]:
            with self.subTest(field=key):
                torch.save({**header, key: value}, self.path)
                with self.assertRaises(ValueError):
                    load_checkpoint(self.path)

    def test_state_dict_is_loaded_strictly(self):
        save_checkpoint(self.model, self.path)
        payload = torch.load(self.path, map_location="cpu", weights_only=True)
        del payload["model_state_dict"]["crop_head.bias"]
        torch.save(payload, self.path)
        with self.assertRaises(RuntimeError):
            load_checkpoint(self.path)

    def test_class_mappings_cannot_be_silently_reordered_or_omitted(self):
        save_checkpoint(self.model, self.path, class_mappings=self.mappings)
        swapped = copy.deepcopy(self.mappings)
        swapped["crop_id_to_name"][0], swapped["crop_id_to_name"][1] = (
            swapped["crop_id_to_name"][1], swapped["crop_id_to_name"][0]
        )
        with self.assertRaises(ValueError):
            load_checkpoint(self.path, expected_class_mappings=swapped)
        with self.assertRaises(ValueError):
            load_checkpoint(self.path)
        save_checkpoint(self.model, self.path, overwrite=True)
        with self.assertRaises(ValueError):
            load_checkpoint(self.path, expected_class_mappings=self.mappings)

    def test_invalid_class_mappings_are_rejected_before_writing(self):
        missing = copy.deepcopy(self.mappings)
        del missing["crop_id_to_name"][0]
        noncontiguous = copy.deepcopy(self.mappings)
        noncontiguous["crop_id_to_name"][55] = noncontiguous["crop_id_to_name"].pop(0)
        duplicate = copy.deepcopy(self.mappings)
        duplicate["disease_id_to_name"][1] = duplicate["disease_id_to_name"][0]
        for mappings in (missing, noncontiguous, duplicate):
            with self.subTest(mappings=mappings["crop_id_to_name"]):
                with self.assertRaises((TypeError, ValueError)):
                    save_checkpoint(self.model, self.path, class_mappings=mappings)
                self.assertFalse(self.path.exists())

    def test_preprocessing_mismatch_is_rejected(self):
        save_checkpoint(self.model, self.path, preprocessing=self.preprocessing)
        with self.assertRaises(ValueError):
            load_checkpoint(
                self.path, expected_preprocessing={**self.preprocessing, "image_size": 256}
            )
        save_checkpoint(self.model, self.path, overwrite=True)
        with self.assertRaises(ValueError):
            load_checkpoint(self.path, expected_preprocessing=self.preprocessing)

    def test_metadata_requires_finite_plain_data(self):
        for metadata in ({"tensor": torch.tensor(1)}, {"bad": float("nan")}):
            with self.subTest(metadata=metadata):
                with self.assertRaises((TypeError, ValueError)):
                    save_checkpoint(self.model, self.path, metadata=metadata)
                self.assertFalse(self.path.exists())

    def test_failed_save_preserves_existing_file_and_removes_temporary_file(self):
        save_checkpoint(self.model, self.path)
        before = hashlib.sha256(self.path.read_bytes()).hexdigest()
        with patch("train_multitask.torch.save", side_effect=OSError("synthetic write failure")):
            with self.assertRaises(OSError):
                save_checkpoint(self.model, self.path, overwrite=True)
        self.assertEqual(hashlib.sha256(self.path.read_bytes()).hexdigest(), before)
        self.assertEqual(list(self.path.parent.iterdir()), [self.path])

    def test_safe_load_never_falls_back_to_unrestricted_pickle(self):
        with patch(
            "train_multitask.torch.load", side_effect=pickle.UnpicklingError("unsupported")
        ) as reader:
            with self.assertRaises(pickle.UnpicklingError):
                load_checkpoint(self.path)
        reader.assert_called_once_with(self.path, map_location="cpu", weights_only=True)

    def test_float64_checkpoint_preserves_state_dtype_and_outputs(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(17)
            original = MultiTaskResNet18().double().eval()
        save_checkpoint(original, self.path)
        loaded, info = load_checkpoint(self.path)
        self.assertEqual(info["model_dtype"], "torch.float64")
        self.assertEqual(next(loaded.parameters()).dtype, torch.float64)
        for name, value in original.state_dict().items():
            torch.testing.assert_close(loaded.state_dict()[name], value, rtol=0, atol=0)
        with torch.inference_mode():
            expected = original(self.images.double())
            actual = loaded(self.images.double())
        for name in expected:
            torch.testing.assert_close(actual[name], expected[name], rtol=1e-10, atol=1e-12)

    def test_class_names_are_normalized_to_safe_plain_strings(self):
        class ClassName(str):
            pass

        mappings = {
            key: {i: ClassName(name) for i, name in values.items()}
            for key, values in self.mappings.items()
        }
        save_checkpoint(self.model, self.path, class_mappings=mappings)
        _, info = load_checkpoint(self.path, expected_class_mappings=self.mappings)
        self.assertEqual(info["class_mappings"], self.mappings)
        self.assertTrue(all(type(name) is str
                            for values in info["class_mappings"].values()
                            for name in values.values()))


if __name__ == "__main__":
    unittest.main()
