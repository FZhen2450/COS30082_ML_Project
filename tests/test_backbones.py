"""M4A CPU architecture/checkpoint tests; synthetic inputs and bounded updates."""

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
from torch.utils.data import DataLoader
from torchvision.models import convnext_tiny, efficientnet_b0
from torchvision.models.convnext import LayerNorm2d

from multitask_backbones import MultiTaskBackbone, create_multitask_model
from multitask_model import MultiTaskResNet18
from train_multitask import (
    compute_multitask_loss, load_checkpoint, save_checkpoint,
    train_one_epoch, validate_one_epoch,
)


ROOT = Path(__file__).resolve().parents[1]
ARCHITECTURES = {
    "multitask_efficientnet_b0": (1280, 4302178),
    "multitask_convnext_tiny": (768, 27996998),
}


class BackboneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(81)
            cls.models = {name: MultiTaskBackbone(name).cpu().eval()
                          for name in ARCHITECTURES}
            cls.images = torch.randn(2, 3, 64, 64)

    def test_shapes_finiteness_and_parameter_counts(self):
        for name, model in self.models.items():
            dim, count = ARCHITECTURES[name]
            self.assertEqual(sum(p.numel() for p in model.parameters()), count)
            self.assertEqual(model.feature_dim, dim)
            self.assertEqual(model.architecture, name)
            for batch_size in (1, 2):
                with self.subTest(architecture=name, batch=batch_size), torch.inference_mode():
                    outputs = model(self.images[:batch_size])
                    self.assertEqual(set(outputs), {"crop_logits", "disease_logits", "features"})
                    for key, width in (("crop_logits", 55), ("disease_logits", 175),
                                       ("features", dim)):
                        self.assertEqual(tuple(outputs[key].shape), (batch_size, width))
                        self.assertTrue(torch.isfinite(outputs[key]).all().item())

    def test_assignment_input_size_224(self):
        images = torch.zeros(1, 3, 224, 224)
        for name, model in self.models.items():
            with self.subTest(architecture=name), torch.inference_mode():
                outputs = model(images)
                self.assertEqual(tuple(outputs["features"].shape), (1, ARCHITECTURES[name][0]))
                self.assertEqual(tuple(outputs["crop_logits"].shape), (1, 55))
                self.assertEqual(tuple(outputs["disease_logits"].shape), (1, 175))

    def test_single_shared_forward_and_separate_raw_linear_heads(self):
        for name, model in self.models.items():
            captured = {"backbone": [], "crop": [], "disease": []}
            with (
                self.subTest(architecture=name),
                model.backbone.register_forward_hook(
                    lambda m, a, output: captured["backbone"].append(output)),
                model.crop_head.register_forward_pre_hook(
                    lambda m, args: captured["crop"].append(args[0])),
                model.disease_head.register_forward_pre_hook(
                    lambda m, args: captured["disease"].append(args[0])),
                torch.inference_mode(),
            ):
                outputs = model(self.images[:1])
                for values in captured.values():
                    self.assertEqual(len(values), 1)
                    self.assertIs(values[0], outputs["features"])
                groups = [{id(p) for p in module.parameters()} for module in
                          (model.backbone, model.crop_head, model.disease_head)]
                for index, group in enumerate(groups):
                    self.assertTrue(group)
                    for other in groups[index + 1:]:
                        self.assertTrue(group.isdisjoint(other))
                for task in ("crop", "disease"):
                    head = getattr(model, task + "_head")
                    torch.testing.assert_close(
                        outputs[task + "_logits"],
                        nn.functional.linear(outputs["features"], head.weight, head.bias),
                    )

    def test_pooling_dropout_removal_and_convnext_final_normalization(self):
        efficient = self.models["multitask_efficientnet_b0"]
        convnext = self.models["multitask_convnext_tiny"]
        self.assertIsInstance(efficient.backbone.classifier, nn.Identity)
        self.assertFalse(any(isinstance(m, nn.Dropout) for m in efficient.backbone.modules()))
        self.assertIsInstance(convnext.backbone.classifier[0], LayerNorm2d)
        self.assertIsInstance(convnext.backbone.classifier[1], nn.Flatten)
        self.assertIsInstance(convnext.backbone.classifier[2], nn.Identity)
        for model in self.models.values():
            self.assertIsInstance(model.backbone.avgpool, nn.AdaptiveAvgPool2d)
            with torch.inference_mode():
                pooled = model.backbone.avgpool(model.backbone.features(self.images[:1]))
                expected = (pooled.flatten(1) if model is efficient
                            else convnext.backbone.classifier[:2](pooled))
                torch.testing.assert_close(model(self.images[:1])["features"], expected)

    def test_loss_and_bounded_synthetic_gradient_flow(self):
        targets = {"crop": torch.tensor([0, 54]), "disease": torch.tensor([0, 174])}
        for name, model in self.models.items():
            with self.subTest(architecture=name):
                model.zero_grad(set_to_none=True)
                outputs = model(self.images[:, :, :32, :32])
                losses = compute_multitask_loss(outputs, targets)
                self.assertTrue(all(torch.isfinite(v).item() for v in losses.values()))
                losses["total"].backward()
                for module in (model.backbone, model.crop_head, model.disease_head):
                    grads = [p.grad for p in module.parameters() if p.requires_grad]
                    self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in grads))
                    self.assertGreater(sum(g.abs().sum().item() for g in grads), 0)
                model.zero_grad(set_to_none=True)

    def test_existing_epoch_functions_accept_each_backbone(self):
        # Keep a 2x2 final feature map for EfficientNet BatchNorm during the
        # single update; 32x32 inputs collapse it to 1x1 and are unstable here.
        images = self.images
        pairs = [(image, (torch.tensor(0), torch.tensor(0), torch.tensor(cid)))
                 for image, cid in zip(images, (10, 332))]
        loader = DataLoader(pairs, batch_size=2)
        for name in ARCHITECTURES:
            with self.subTest(architecture=name), torch.random.fork_rng(devices=[]):
                torch.manual_seed(82)
                model = MultiTaskBackbone(name).cpu()
                optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
                before = model.crop_head.weight.detach().clone()
                with patch.object(optimizer, "step", wraps=optimizer.step) as step:
                    result = train_one_epoch(model, loader, optimizer, "cpu")
                self.assertEqual(step.call_count, 1)
                self.assertTrue(model.training)
                self.assertEqual(result["sample_count"], 2)
                self.assertEqual(result["batch_count"], 1)
                self.assertFalse(torch.equal(before, model.crop_head.weight))
                # Parameter tensor versions detect in-place state writes without
                # retaining a second full backbone copy.
                versions = {key: value._version for key, value in model.state_dict().items()}
                result = validate_one_epoch(model, loader, "cpu")
                self.assertFalse(model.training)
                self.assertEqual(result["composition_ids"].tolist(), [10, 332])
                self.assertEqual(tuple(result["crop_logits"].shape), (2, 55))
                self.assertEqual(tuple(result["disease_logits"].shape), (2, 175))
                self.assertEqual(result["sample_count"], 2)
                self.assertEqual(versions, {k: v._version for k, v in model.state_dict().items()})

    def test_explicit_local_torchvision_state_is_loaded_before_classifier_removal(self):
        for name, builder in (("multitask_efficientnet_b0", efficientnet_b0),
                              ("multitask_convnext_tiny", convnext_tiny)):
            with self.subTest(architecture=name), torch.random.fork_rng(devices=[]):
                torch.manual_seed(83)
                stock = builder(weights=None).eval()
                with torch.no_grad():
                    # Non-default persistent state verifies it is copied.
                    next(stock.parameters()).fill_(0.0123)
                    if name.endswith("b0"):
                        stock.features[0][1].running_mean.fill_(0.25)
                expected = stock.state_dict()
                model = MultiTaskBackbone(name, backbone_state_dict=expected).eval()
                for key, value in model.backbone.state_dict().items():
                    torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
                missing = dict(expected)
                del missing[next(iter(missing))]
                with self.assertRaises(RuntimeError):
                    MultiTaskBackbone(name, backbone_state_dict=missing)

    def test_unknown_architectures_and_resnet_local_state_are_rejected(self):
        for name in ("efficientnet_b7", "multitask_resnet50", "", None):
            with self.subTest(architecture=name), self.assertRaises((TypeError, ValueError)):
                create_multitask_model(name)
        with self.assertRaises(ValueError):
            create_multitask_model("multitask_resnet18", backbone_state_dict={})
        reference = create_multitask_model("multitask_resnet18")
        self.assertIsInstance(reference, MultiTaskResNet18)


class BackboneCheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(84)
            cls.models = {name: MultiTaskBackbone(name).cpu().eval()
                          for name in ARCHITECTURES}
            cls.images = torch.randn(1, 3, 32, 32)
            for model in cls.models.values():
                for module in model.modules():
                    if isinstance(module, nn.BatchNorm2d):
                        module.running_mean.fill_(0.125)
                        module.running_var.fill_(1.25)
                        module.num_batches_tracked.fill_(7)
        cls.mappings = {
            "crop_id_to_name": {i: f"crop-{i}" for i in range(55)},
            "disease_id_to_name": {i: f"disease-{i}" for i in range(175)},
        }
        cls.preprocessing = {"image_size": 224, "normalization": "imagenet"}

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "model.pt"

    def test_round_trip_reconstructs_all_state_and_outputs_without_downloads(self):
        for name, model in self.models.items():
            with self.subTest(architecture=name):
                save_checkpoint(model, self.path, overwrite=self.path.exists(),
                                class_mappings=self.mappings, preprocessing=self.preprocessing)
                with patch("torch.hub.download_url_to_file", side_effect=AssertionError("download")):
                    loaded, info = load_checkpoint(
                        self.path, expected_architecture=name,
                        expected_class_mappings=self.mappings,
                        expected_preprocessing=self.preprocessing,
                    )
                self.assertIsInstance(loaded, MultiTaskBackbone)
                self.assertEqual(loaded.architecture, name)
                self.assertEqual(info["architecture"], name)
                self.assertEqual(info["format_version"], 1)
                self.assertEqual(info["feature_dim"], ARCHITECTURES[name][0])
                self.assertEqual((info["num_crops"], info["num_diseases"]), (55, 175))
                self.assertFalse(loaded.training)
                for key, value in model.state_dict().items():
                    torch.testing.assert_close(loaded.state_dict()[key], value, rtol=0, atol=0)
                with torch.inference_mode():
                    expected, actual = model(self.images), loaded(self.images)
                for key in expected:
                    torch.testing.assert_close(actual[key], expected[key], rtol=1e-5, atol=1e-6)

    def test_mapping_preprocessing_and_expected_architecture_mismatches_rejected(self):
        reordered = copy.deepcopy(self.mappings)
        reordered["crop_id_to_name"][0], reordered["crop_id_to_name"][1] = (
            reordered["crop_id_to_name"][1], reordered["crop_id_to_name"][0])
        for name, model in self.models.items():
            save_checkpoint(model, self.path, overwrite=self.path.exists(),
                            class_mappings=self.mappings, preprocessing=self.preprocessing)
            cases = [
                {"expected_class_mappings": reordered},
                {},
                {"expected_class_mappings": self.mappings,
                 "expected_preprocessing": {"image_size": 256}},
                {"expected_class_mappings": self.mappings,
                 "expected_architecture": "multitask_resnet18"},
                {"expected_class_mappings": self.mappings,
                 "expected_architecture": "unrecognized"},
            ]
            for case in cases:
                with self.subTest(architecture=name, arguments=case), self.assertRaises(ValueError):
                    load_checkpoint(self.path, **case)

    def test_wrong_architecture_header_cannot_relabel_another_backbone_state(self):
        model = self.models["multitask_efficientnet_b0"]
        save_checkpoint(model, self.path)
        payload = torch.load(self.path, map_location="cpu", weights_only=True)
        payload.update(architecture="multitask_convnext_tiny", feature_dim=768)
        torch.save(payload, self.path)
        with self.assertRaises(RuntimeError):
            load_checkpoint(self.path)

    def test_incorrect_counts_dimensions_and_unsupported_architecture_rejected(self):
        header = {
            "format_version": 1, "architecture": "multitask_efficientnet_b0",
            "num_crops": 55, "num_diseases": 175, "feature_dim": 1280,
        }
        for field, value in (("num_crops", 54), ("num_diseases", 174),
                             ("feature_dim", 768), ("format_version", 2),
                             ("architecture", "efficientnet_b7"), ("architecture", [])):
            with self.subTest(field=field), self.assertRaises((TypeError, ValueError)):
                torch.save({**header, field: value}, self.path)
                load_checkpoint(self.path)

    def test_save_rejects_incompatible_new_heads_and_protects_existing_file(self):
        for name, model in self.models.items():
            with self.subTest(architecture=name):
                save_checkpoint(model, self.path, overwrite=self.path.exists())
                before = self.path.stat().st_size
                with self.assertRaises(FileExistsError):
                    save_checkpoint(model, self.path)
                self.assertEqual(self.path.stat().st_size, before)
                with patch.object(model, "disease_head", nn.Linear(model.feature_dim, 174)):
                    with self.assertRaises(ValueError):
                        save_checkpoint(model, self.path, overwrite=True)

    def test_legacy_version_one_resnet_checkpoint_reconstructs_unchanged(self):
        # Build the accepted v1 payload explicitly, independently of the new saver.
        model = MultiTaskResNet18().cpu().eval()
        payload = {
            "format_version": 1, "architecture": "multitask_resnet18",
            "num_crops": 55, "num_diseases": 175, "feature_dim": 512,
            "model_dtype": "torch.float32", "model_state_dict": model.state_dict(),
            "class_mappings": None, "preprocessing": None, "metadata": {"legacy": True},
            "runtime": {},
        }
        torch.save(payload, self.path)
        loaded, info = load_checkpoint(self.path, expected_architecture="multitask_resnet18")
        self.assertIsInstance(loaded, MultiTaskResNet18)
        self.assertTrue(info["metadata"]["legacy"])
        self.assertEqual(set(model.state_dict()), set(loaded.state_dict()))
        with torch.inference_mode():
            expected, actual = model(self.images), loaded(self.images)
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key])


class BackboneImportSafetyTests(unittest.TestCase):
    def test_import_and_construction_never_download_train_or_write(self):
        script = r"""
import importlib
import os
import sys
from unittest.mock import patch
import torch
import torchvision.models as models

def audit(event, args):
    if event == "open":
        _, mode, flags = args
        if ((isinstance(mode, str) and any(c in mode for c in "wax+"))
                or (isinstance(flags, int) and flags &
                    (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))):
            raise AssertionError("file write")
    if (event.startswith("socket.") or event in {
            "urllib.Request", "subprocess.Popen", "os.system", "os.mkdir", "os.remove",
            "os.rename", "os.rmdir", "os.link", "os.symlink"}):
        raise AssertionError(event)

sys.addaudithook(audit)
with (
    patch.object(torch.hub, "download_url_to_file", side_effect=AssertionError("download")),
    patch.object(torch.nn.Module, "_call_impl", side_effect=AssertionError("forward")),
    patch.object(torch.autograd, "backward", side_effect=AssertionError("backward")),
    patch.object(torch.optim.SGD, "step", side_effect=AssertionError("step")),
    patch.object(models, "efficientnet_b0", wraps=models.efficientnet_b0) as efficient,
    patch.object(models, "convnext_tiny", wraps=models.convnext_tiny) as convnext,
):
    with patch.object(torch.nn.Module, "__init__", side_effect=AssertionError("construction")):
        module = importlib.import_module("multitask_backbones")
        importlib.import_module("train_multitask")
    efficient.assert_not_called()
    convnext.assert_not_called()
    module.MultiTaskBackbone("multitask_efficientnet_b0")
    module.MultiTaskBackbone("multitask_convnext_tiny")
    efficient.assert_called_once_with(weights=None)
    convnext.assert_called_once_with(weights=None)
print("SAFE")
"""
        result = subprocess.run(
            [sys.executable, "-B", "-c", script], cwd=ROOT,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1"},
            capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(result.stdout.strip(), "SAFE")


if __name__ == "__main__":
    unittest.main()
