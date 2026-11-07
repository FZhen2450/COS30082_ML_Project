"""Notebook safety regressions, updated for the authorized M4C visible contract."""

import ast
import copy
import hashlib
import inspect
import json
import math
import uuid
from datetime import datetime, timezone
import gc
import itertools
from contextlib import nullcontext
import os
import platform
import random
import tempfile as tempfile_module
import time
from collections import Counter
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from PIL import Image
import torchvision
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from torchvision.models import efficientnet_b0, convnext_tiny, EfficientNet_B0_Weights, ConvNeXt_Tiny_Weights, efficientnet_b3, convnext_base, EfficientNet_B3_Weights, ConvNeXt_Base_Weights
from torch.utils.checkpoint import checkpoint as activation_checkpoint
import re
import tempfile
import unittest
from collections.abc import Mapping
from pathlib import Path
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/wp2_backbone_comparison.ipynb"


def load_notebook():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def code_sources():
    return {cell["id"]: "".join(cell["source"]) for cell in load_notebook()["cells"]
            if cell["cell_type"] == "code"}


def literal_assignments():
    values = {}
    for source in code_sources().values():
        for node in ast.parse(source).body:
            if isinstance(node, ast.Assign):
                try:
                    value = ast.literal_eval(node.value)
                except (ValueError, TypeError):
                    continue
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        values[target.id] = value
    return values


def helper_namespace():
    # Compile visible function/class definitions only. No notebook cell setup,
    # dataset iteration, training orchestration, download or plotting executes.
    namespace = {
        "__name__": __name__, "Path": Path, "ast": ast, "hashlib": hashlib, "torch": torch,
        "torchvision": torchvision, "Mapping": Mapping, "Tensor": Tensor, "nn": nn,
        "F": F, "Dataset": Dataset, "DataLoader": DataLoader, "v2": v2,
        "math": math, "uuid": uuid, "datetime": datetime, "timezone": timezone, "gc": gc, "itertools": itertools, "nullcontext": nullcontext,
        "activation_checkpoint": activation_checkpoint,
        "efficientnet_b3": efficientnet_b3, "convnext_base": convnext_base, "re": re, "json": json, "copy": copy, "os": os,
        "platform": platform, "tempfile": tempfile_module, "time": time,
        "Counter": Counter, "dataclass": dataclass, "Integral": Integral, "Real": Real,
        "Any": Any, "random": random, "np": np, "pd": pd, "Image": Image,
        "urlparse": urlparse, "efficientnet_b0": efficientnet_b0, "convnext_tiny": convnext_tiny,
        **literal_assignments(),
    }
    namespace["_FLOAT_DTYPES"] = {str(dtype): dtype for dtype in
                                  (torch.float16, torch.bfloat16, torch.float32, torch.float64)}
    namespace["WEIGHT_ENUMS"] = {
        "multitask_efficientnet_b0": EfficientNet_B0_Weights.IMAGENET1K_V1,
        "multitask_convnext_tiny": ConvNeXt_Tiny_Weights.IMAGENET1K_V1,
        "multitask_efficientnet_b3": EfficientNet_B3_Weights.IMAGENET1K_V1,
        "multitask_convnext_base": ConvNeXt_Base_Weights.IMAGENET1K_V1,
    }
    definitions = []
    for source in code_sources().values():
        definitions.extend(node for node in ast.parse(source).body
                           if isinstance(node, (ast.FunctionDef, ast.ClassDef)))
    module = ast.Module(body=definitions, type_ignores=[])
    exec(compile(module, "<isolated visible notebook definitions>", "exec"), namespace)
    return namespace


class NotebookStructureTests(unittest.TestCase):
    def test_notebook_exists_is_unexecuted_and_has_sixteen_sections(self):
        notebook = load_notebook()
        self.assertEqual((notebook["nbformat"], notebook["nbformat_minor"]), (4, 5))
        ids = [cell["id"] for cell in notebook["cells"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue(all(re.fullmatch(r"[A-Za-z0-9_-]{1,64}", key) for key in ids))
        markdown = "".join("".join(cell["source"]) for cell in notebook["cells"]
                           if cell["cell_type"] == "markdown")
        self.assertEqual(re.findall(r"^## (\d+)\.", markdown, re.M),
                         [str(i) for i in range(1, 17)])
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                self.assertIsNone(cell["execution_count"])
                self.assertEqual(cell["outputs"], [])
        self.assertFalse(notebook["metadata"]["cos30082"]["training_executed"])

    def test_all_python_cells_compile_without_execution(self):
        for key, source in code_sources().items():
            with self.subTest(cell=key):
                compile(source, f"notebook:{key}", "exec")

    def test_default_flags_are_false_and_pretrained_is_explicit(self):
        values = literal_assignments()
        for key in ("PREPARE_RUN", "RUN_MODEL_DIAGNOSTICS", "RUN_TRAINING",
                    "RUN_EXPENSIVE_EVALUATION", "SAVE_FINAL_SNAPSHOT",
                    "VERIFY_SAVED_CHECKPOINT"):
            self.assertIs(values[key], False, key)
        self.assertEqual(values["CONFIG"]["initialization"], "local_pretrained")
        self.assertTrue(Path(values["CONFIG"]["local_weight_path"]).is_absolute())
        self.assertRegex(values["CONFIG"]["expected_weight_sha256"], r"^[0-9a-f]{64}$")
        self.assertIs(values["RUN_PLANNED_EXPERIMENTS"], False)
        self.assertIs(values["RUN_GPU_TUNING"], False)
        self.assertEqual(values["SCORING_ID"], "sum_log_softmax_v1")
        self.assertEqual(values["METRIC_IMPLEMENTATION"], "notebook_wp5_spec_v1")
        self.assertIs(values["RUN_PRETRAINED_ACQUISITION"], False)

    def test_epoch_calls_exist_only_inside_both_flag_guard(self):
        sources=code_sources()
        guard=ast.parse(sources["training"]).body[1]
        self.assertIsInstance(guard,ast.If)
        self.assertEqual(ast.unparse(guard.test),"not (RUN_TRAINING and RUN_EXPENSIVE_EVALUATION)")
        self.assertFalse(any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name)
                             and n.func.id=="run_experiment_sequence"
                             for node in guard.body for n in ast.walk(node)))
        self.assertTrue(any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name)
                            and n.func.id=="run_experiment_sequence"
                            for node in guard.orelse for n in ast.walk(node)))
        self.assertIn("RUN_PLANNED_EXPERIMENTS",sources["training"])
        self.assertIn("for epoch in range(",sources["recovery-functions"])
        self.assertIn('cfg["warmup_epochs"] + cfg["finetune_epochs"] + 1',sources["recovery-functions"])

    def test_model_creation_and_checkpoint_actions_are_guarded(self):
        sources=code_sources()
        diagnostic=ast.parse(sources["diagnostics"]).body[0]
        self.assertIsInstance(diagnostic,ast.If)
        self.assertEqual(diagnostic.test.id,"RUN_MODEL_DIAGNOSTICS")
        self.assertNotIn("create_multitask_model(",sources["setup"])
        self.assertIn("create_multitask_model(",sources["recovery-functions"])
        self.assertIn("if PREPARE_RUN:",sources["setup"])
        self.assertIn("last_completed.pt",sources["checkpoint-actions"])
        self.assertIn("save_recovery_checkpoint(context",sources["recovery-functions"])

    def test_visible_processing_replaces_imports_without_hidden_execution(self):
        all_source = "\n".join(code_sources().values())
        tree = ast.parse(all_source)
        names = {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        required = {"create_multitask_model", "build_development_split", "get_development_dataloaders",
                    "train_one_epoch", "validate_one_epoch", "save_checkpoint", "load_checkpoint",
                    "compute_multitask_loss", "composition_scores", "composition_topk", "harmonic_mean"}
        self.assertTrue(required <= names)
        forbidden = {"dataset", "data_utils", "development_data", "multitask_backbones",
                     "multitask_model", "train_multitask", "importlib"}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                self.assertNotIn((node.module or "").split(".")[0], forbidden)
            if isinstance(node, ast.Import):
                self.assertFalse(any(item.name.split(".")[0] in forbidden for item in node.names))
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                self.assertNotIn(node.func.id, {"exec", "eval", "__import__"})
        self.assertIn("F.log_softmax", all_source)
        self.assertIn("stable=True", all_source)
        loads = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                 and isinstance(n.func.value, ast.Name) and n.func.value.id == "torch" and n.func.attr == "load"]
        self.assertGreaterEqual(len(loads), 2)
        for load in loads:
            self.assertTrue(any(k.arg == "weights_only" and isinstance(k.value, ast.Constant)
                                and k.value.value is True for k in load.keywords))
        self.assertNotIn("get_dataloaders(", all_source)
        self.assertNotIn("%run", all_source)

    def test_provenance_logging_and_snapshot_labels_are_present(self):
        sources=code_sources()
        self.assertIn("seed=42",sources["split"])
        self.assertIn("(4868, 1238, 1894)",sources["split"])
        self.assertIn("len(development.candidate_ids) != 323",sources["split"])
        recovery=sources["recovery-functions"]
        for key in ("candidate_sha256","source_sha256","configuration","initialization",
                    "preprocessing","environment","code","selection_policy","split_sha256",
                    "epoch_seconds","peak_gpu_allocated_mib","combined_validation_loss","selected_epoch"):
            self.assertIn(key,recovery)
        self.assertIn('context["completed"]=stop_reason != "owner_stop_file"',recovery)
        self.assertIn("last_completed.pt",recovery)
        self.assertIn("torch.save",sources["checkpoint-functions"])
        self.assertIn("os.link",sources["checkpoint-functions"])


    def test_snapshot_requires_normal_completion_and_partial_runs_are_labelled(self):
        recovery=code_sources()["recovery-functions"]
        self.assertIn("save_recovery_checkpoint(context,epoch",recovery)
        self.assertLess(recovery.index('context["history"].append(row)'),
                        recovery.index("save_recovery_checkpoint(context,epoch"))
        self.assertIn("except KeyboardInterrupt:",recovery)
        self.assertIn("interrupted_resume_last_completed",recovery)
        self.assertIn("partial_or_interrupted",recovery)
        self.assertIn("atomic_torch_save(payload",recovery)
        self.assertIn("completed_epoch",recovery)


class NotebookHelperTests(unittest.TestCase):
    def setUp(self):
        self.ns = helper_namespace()
        self.config = copy.deepcopy(self.ns["CONFIG"])

    def test_configuration_accepts_only_the_two_selected_backbones(self):
        for architecture in self.ns["APPROVED_ARCHITECTURES"]:
            self.config["architecture"] = architecture
            result = self.ns["validate_config"](self.config)
            self.assertEqual(result, self.config)
            self.assertIsNot(result, self.config)
        for architecture in ("multitask_resnet18", "resnet50", "", None):
            with self.subTest(architecture=architecture), self.assertRaises(ValueError):
                self.ns["validate_config"]({**self.config, "architecture": architecture})

    def test_invalid_configuration_is_rejected(self):
        for field, value in (
            ("initialization", None), ("batch_size", 0), ("epochs", True),
            ("seed", -1), ("seed", 2**32), ("num_workers", -1),
            ("learning_rate", 0), ("learning_rate", float("nan")),
            ("weight_decay", -1), ("crop_weight", True), ("optimizer", "unknown"),
            ("augment_fit", 1), ("deterministic_algorithms", 1),
            ("preprocessing_policy", "native_preset"), ("run_id", "../overwrite"),
            ("output_directory", "relative/path"), ("device", "meta"),
        ):
            with self.subTest(field=field, value=value), self.assertRaises((TypeError, ValueError)):
                self.ns["validate_config"]({**self.config, field: value})
        with self.assertRaises(ValueError):
            self.ns["validate_config"]({**self.config, "crop_weight": 0, "disease_weight": 0})

    def test_root_discovery_from_repository_notebook_and_override(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "DPD_Small_train.csv").touch()
            (root / "notebooks").mkdir()
            (root / "notebooks/wp2_backbone_comparison.ipynb").touch()
            nested = root / "notebooks" / "nested"
            nested.mkdir(parents=True)
            find = self.ns["find_project_root"]
            self.assertEqual(find(nested), root.resolve())
            self.assertEqual(find(Path("/"), str(root)), root.resolve())
            with self.assertRaises(FileNotFoundError):
                find(root, str(root / "notebooks"))

    def test_random_initialization_is_explicit_and_never_loads_a_file(self):
        config = {**self.config, "initialization": "random"}
        with patch.object(torch, "load", side_effect=AssertionError("unexpected load")):
            state, info = self.ns["verified_backbone_state"](config)
        self.assertIsNone(state)
        self.assertEqual(info["mode"], "random")

    def test_missing_or_mismatched_pretrained_weights_fail_without_fallback(self):
        verify = self.ns["verified_backbone_state"]
        with self.assertRaises(ValueError):
            verify({**self.config, "local_weight_path": None, "expected_weight_sha256": None})
        with self.assertRaises(FileNotFoundError):
            verify({**self.config, "local_weight_path": "/definitely_missing_m4b_weights.pt",
                    "expected_weight_sha256": "a" * 64})
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "synthetic.pt"
            path.write_bytes(b"synthetic invalid file")
            with patch.object(torch, "load", side_effect=AssertionError("must reject before load")):
                with self.assertRaises(ValueError):
                    verify({**self.config, "local_weight_path": str(path),
                            "expected_weight_sha256": "b" * 64})

    def test_verified_local_tensor_state_loads_safely_and_records_source(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "synthetic_stock_state.pt"
            torch.save({"synthetic.weight": torch.ones(2)}, path)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            config = {**self.config, "local_weight_path": str(path),
                      "expected_weight_sha256": digest.upper()}
            with patch.object(torch, "load", wraps=torch.load) as loader:
                state, info = self.ns["verified_backbone_state"](config)
            self.assertTrue(torch.equal(state["synthetic.weight"], torch.ones(2)))
            self.assertTrue(loader.call_args.kwargs["weights_only"])
            self.assertEqual(loader.call_args.kwargs["map_location"], "cpu")
            self.assertEqual(info["sha256"], digest)
            self.assertEqual(info["identifier"], self.ns["PRETRAINED_IDS"][config["architecture"]])

    def test_hash_verification_detects_changed_file_and_invalid_payloads(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "synthetic.pt"
            torch.save({"weight": torch.ones(1)}, path)
            config = {**self.config, "local_weight_path": str(path),
                      "expected_weight_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            # Model a file changing after its first verified hash.
            calls = iter([config["expected_weight_sha256"], "0" * 64])
            with patch.dict(self.ns, {"stream_sha256": lambda stream: next(calls)}):
                with self.assertRaises(ValueError):
                    self.ns["verified_backbone_state"](config)
            for payload in ({}, {"not_tensor": "value"}, [torch.ones(1)]):
                torch.save(payload, path)
                config["expected_weight_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
                with self.assertRaises(ValueError):
                    self.ns["verified_backbone_state"](config)

    def test_unique_run_directory_and_json_never_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = self.ns["reserve_run_directory"](folder, "run_001")
            with self.assertRaises(FileExistsError):
                self.ns["reserve_run_directory"](folder, "run_001")
            with self.assertRaises(ValueError):
                self.ns["reserve_run_directory"](folder, "../escape")
            path = directory / "record.json"
            self.ns["write_new_json"](path, {"epoch": 1})
            before = path.read_bytes()
            with self.assertRaises(FileExistsError):
                self.ns["write_new_json"](path, {"epoch": 2})
            self.assertEqual(before, path.read_bytes())

    def test_fingerprint_changes_when_configuration_changes(self):
        fingerprint = self.ns["config_fingerprint"]
        self.assertEqual(fingerprint(self.config), fingerprint(copy.deepcopy(self.config)))
        self.assertNotEqual(fingerprint(self.config), fingerprint({**self.config, "seed": 7}))

    def test_preprocessing_metadata_matches_preserved_wp1_pipeline(self):
        from dataset import get_transforms
        for augment in (True, False):
            config = {**self.config, "augment_fit": augment, "active_resolution": 224}
            result = self.ns["preprocessing_record"](config)
            self.assertEqual(result["fit_transform"], repr(get_transforms(224, augment, True)))
            self.assertEqual(result["validation_transform"], repr(get_transforms(224, False, True)))
            self.assertEqual(result["resize"]["interpolation"], "bilinear")
            self.assertEqual(result["augment_fit"], augment)
            self.assertEqual(result["image_size"], [224, 224])

    def test_local_metrics_require_valid_global_truths_and_candidates(self):
        spec = {"ids": (10, 75, 100, 150, 200, 332), "pairs": tuple((i, i) for i in range(6))}
        fold = {"sample_count": 1, "crop_logits": torch.zeros(1, 55),
                "disease_logits": torch.zeros(1, 175), "composition_ids": torch.tensor([10])}
        result = self.ns["evaluate_development"](fold, fold, spec)
        self.assertEqual(result["top1_harmonic_mean"], 1.0)
        with self.assertRaises(ValueError):
            self.ns["evaluate_development"]({**fold, "composition_ids": torch.tensor([1])}, fold, spec)
        with self.assertRaises(ValueError):
            self.ns["evaluate_development"](fold, fold, {**spec, "sha256": "0" * 64})

    def test_local_metrics_preserve_wp5_output_contract_and_zero_rule(self):
        spec = {"ids": (10, 75, 100, 150, 200, 332), "pairs": tuple((i, i) for i in range(6))}
        fold = {"sample_count": 1, "crop_logits": torch.zeros(1, 55),
                "disease_logits": torch.zeros(1, 175), "composition_ids": torch.tensor([332])}
        result = self.ns["evaluate_development"](fold, fold, spec)
        self.assertEqual(set(result), {"seen_top1", "unseen_top1", "seen_top5", "unseen_top5",
                                      "top1_harmonic_mean", "top5_harmonic_mean"})
        self.assertTrue(all(value == 0.0 for value in result.values()))
        with self.assertRaises(ValueError):
            self.ns["harmonic_mean"](float("nan"), 0.5)

    def test_selection_uses_exact_h1_then_weighted_loss_then_earliest_epoch(self):
        key = self.ns["checkpoint_selection_key"]
        seen = {"sample_count": 2, "losses": {"total": 2.0}}
        unseen = {"sample_count": 1, "losses": {"total": 8.0}}
        metric = {"top1_harmonic_mean": 0.5}
        self.assertEqual(key(metric, seen, unseen, 4), (-0.5, 4.0, 4))
        high_h1 = key({"top1_harmonic_mean": 0.6}, seen, unseen, 9)
        self.assertLess(high_h1, key(metric, seen, unseen, 1))
        low_loss = key(metric, seen, {"sample_count": 1, "losses": {"total": 5.0}}, 9)
        self.assertLess(low_loss, key(metric, seen, unseen, 1))
        self.assertLess(key(metric, seen, unseen, 1), key(metric, seen, unseen, 2))
        # An arbitrarily small H@1 improvement wins even with worse loss/epoch.
        exact = key({"top1_harmonic_mean": math.nextafter(0.5, 1.0)}, seen, unseen, 99)
        self.assertLess(exact, key(metric, seen, {"sample_count": 1, "losses": {"total": 0.0}}, 1))
        with self.assertRaises(RuntimeError):
            key(None, seen, unseen, 1)
        for bad in ({"sample_count": 0, "losses": {"total": 1.0}},
                    {"sample_count": 2, "losses": {"total": float("inf")}}):
            with self.assertRaises(ValueError):
                key(metric, bad, unseen, 1)


if __name__ == "__main__":
    unittest.main()
