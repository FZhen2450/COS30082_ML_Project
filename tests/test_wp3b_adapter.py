"""M2 notebook contracts. Synthetic CPU features/images only; no real CLIP weights."""
import ast
import copy
import hashlib
import importlib.metadata as metadata
import io
import json
import math
import platform
import random
import re
import sys
import tempfile
import time
import unicodedata
import unittest
import uuid
from collections import Counter
from contextlib import redirect_stdout
from dataclasses import dataclass
from datetime import datetime, timezone
from numbers import Integral, Real
from pathlib import Path
from typing import Any
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/wp3b_trainable_adapter.ipynb"

def notebook():
    return json.loads(NOTEBOOK.read_text())

def namespace():
    nodes, values = [], {}
    for cell in notebook()["cells"]:
        if cell["cell_type"] != "code":
            continue
        for node in ast.parse("".join(cell["source"])).body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                nodes.append(node)
            elif isinstance(node, ast.Assign):
                try:
                    value = ast.literal_eval(node.value)
                except (ValueError, TypeError):
                    continue
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        values[target.id] = value
    ns = {**globals(), **values, "ROOT": ROOT, "__name__": __name__}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "<visible M2 definitions>", "exec"), ns)
    return ns

def workspace_temporary_directory():
    # Synthetic checkpoint fixtures follow the same ignored output policy as M2.
    base = Path('/home/user1/COS30082 Applied Machine Learning/00 Assigments/AS02 ML Project/workspace/experiments/_verification/wp3b-m2')
    base.mkdir(parents=True, exist_ok=True)
    return tempfile.TemporaryDirectory(dir=base)

class StructureTests(unittest.TestCase):
    def test_json_syntax_unique_cells_and_sanitization(self):
        book = notebook()
        self.assertEqual(book["nbformat"], 4)
        self.assertEqual(len(book["cells"]), len({c["id"] for c in book["cells"]}))
        for c in book["cells"]:
            self.assertFalse(c.get("outputs"))
            self.assertIsNone(c.get("execution_count"))
            self.assertFalse(c.get("attachments"))
            if c["cell_type"] == "code":
                compile("".join(c["source"]), c["id"], "exec")

    def test_default_controls_and_no_hidden_pipeline(self):
        ns = namespace()
        for name in ("PREDECLARE_RUN", "LOAD_CLIP_MODEL", "RUN_TEXT_ENCODING", "BUILD_ADAPTER",
                     "RUN_DEVELOPMENT_EVALUATION", "RUN_TRAINING", "SAVE_CHECKPOINTS", "EXPORT_RESULTS"):
            self.assertIs(ns[name], False)
        forbidden = {"dataset", "data_utils", "development_data", "multitask_model",
                     "multitask_backbones", "train_multitask", "evaluation"}
        for c in notebook()["cells"]:
            if c["cell_type"] != "code":
                continue
            for n in ast.walk(ast.parse("".join(c["source"]))):
                if isinstance(n, ast.Import):
                    self.assertFalse(any(a.name.split(".")[0] in forbidden for a in n.names))
                if isinstance(n, ast.ImportFrom):
                    self.assertNotIn((n.module or "").split(".")[0], forbidden)
                if isinstance(n, ast.Call):
                    self.assertNotIn(ast.unparse(n.func), ("exec", "eval", "compile", "importlib.import_module"))
        self.assertNotIn("%run", NOTEBOOK.read_text())
        self.assertNotIn("DPD_Small_test_", NOTEBOOK.read_text())

    def test_cold_default_run_all_has_no_model_image_or_output_access(self):
        # Execute actual cell order with all controls off, catching the M1 ordering gap.
        ns = {"__name__": __name__, "display": lambda *a, **k: None}
        with patch("clip.load", side_effect=AssertionError("model load")), \
             patch("PIL.Image.open", side_effect=AssertionError("image decode")), \
             patch("torch.save", side_effect=AssertionError("checkpoint write")), \
             patch.object(Path, "write_text", side_effect=AssertionError("file write")), \
             patch.object(pd.DataFrame, "to_csv", side_effect=AssertionError("CSV write")), \
             redirect_stdout(io.StringIO()):
            for c in notebook()["cells"]:
                if c["cell_type"] == "code":
                    exec(compile("".join(c["source"]), c["id"], "exec"), ns)
        self.assertIsNone(ns["FROZEN_CLIP"])
        self.assertIsNone(ns["ADAPTER"])
        self.assertIsNone(ns["RUN_DIRECTORY"])
        self.assertEqual([len(ns["SPLIT"].fit), len(ns["SPLIT"].val_seen), len(ns["SPLIT"].val_unseen)],
                         [4868, 1238, 1894])

    def test_reference_scoring_and_prompt_functions_unchanged(self):
        def definitions(book):
            return {n.name: ast.dump(n, include_attributes=False)
                    for c in book["cells"] if c["cell_type"] == "code"
                    for n in ast.parse("".join(c["source"])).body
                    if isinstance(n, ast.FunctionDef)}
        actual = definitions(notebook())
        reference = definitions(json.loads((ROOT / "notebooks/wp3_clip_zero_shot.ipynb").read_text()))
        for name in ("normalize_label", "build_prompt_table", "unit_features", "cosine_candidate_scores",
                     "rank_global_ids", "topk_accuracy", "harmonic_mean", "load_frozen_clip",
                     "make_validation_split", "build_development_split"):
            self.assertEqual(actual[name], reference[name], name)

class AdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = namespace()
        cls.split = cls.ns["build_development_split"](ROOT / "DPD_Small_train.csv")
        cls.table = cls.ns["build_prompt_table"](cls.split)
        cls.contract = cls.ns["make_m2_contract"](cls.split, cls.table, cls.ns["CONFIG"])
        g = torch.Generator().manual_seed(42)
        cls.direct = cls.ns["unit_features"](torch.randn(323, 768, generator=g))
        cls.prototypes = cls.ns["component_prototypes"](cls.direct, cls.table, cls.contract["fitting_composition_ids"])
        cls.state = cls.ns["seal_text_state"](cls.direct, cls.prototypes, cls.contract)
    def setUp(self):
        torch.manual_seed(7)
        self.adapter = self.ns["ResidualVisualAdapter"]()
        self.x = torch.randn(2, 768)
        self.crop = torch.tensor([0, 1])
        self.disease = torch.tensor([0, 1])

    def test_identity_exact_and_parameter_count(self):
        self.assertEqual(sum(p.numel() for p in self.adapter.parameters()), 99136)
        self.assertEqual(len(set(id(p) for p in self.adapter.parameters())), 4)
        torch.testing.assert_close(self.adapter(self.x), self.x, rtol=0, atol=0)
        self.assertTrue(torch.equal(self.adapter.up.weight, torch.zeros_like(self.adapter.up.weight)))

    def test_identity_direct_scores_and_ranking_exact(self):
        x = self.x.half()
        expected = self.ns["cosine_candidate_scores"](x, self.direct)
        actual = self.ns["cosine_candidate_scores"](self.adapter(x), self.direct)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertTrue(torch.equal(self.ns["rank_global_ids"](actual, self.split.candidate_ids),
                                    self.ns["rank_global_ids"](expected, self.split.candidate_ids)))

    def test_bad_feature_shapes_and_nonfinite_rejected(self):
        for x in (torch.ones(2, 10), torch.zeros(2, 768), torch.full((2, 768), float("nan"))):
            with self.assertRaises(ValueError):
                self.adapter(x)

    def test_context_averaged_prototypes_correct_and_frozen(self):
        for kind in ("crop", "disease"):
            ids = self.prototypes[kind + "_ids"]
            self.assertEqual(tuple(ids), tuple(sorted(self.table[kind + "_id"].unique())))
            self.assertFalse(self.prototypes[kind].requires_grad)
            mask = self.table.global_composition_id.isin(self.contract["fitting_composition_ids"]).to_numpy()
            vectors = self.ns["unit_features"](self.direct)[mask]
            eligible = self.table.loc[mask].reset_index(drop=True)
            expected = self.ns["unit_features"](torch.stack([
                vectors[eligible[kind + "_id"].to_numpy() == key].mean(0) for key in ids])).float()
            expected = torch.nn.functional.normalize(expected, dim=1)
            torch.testing.assert_close(self.prototypes[kind], expected, rtol=0, atol=0)

    def test_loss_exact_weighting_and_identity_initial_zero(self):
        y = self.adapter(self.x)
        loss = self.ns["adapter_objective"](y, self.x, self.crop, self.disease, self.prototypes)
        unit = torch.nn.functional.normalize(y, dim=1)
        expected_c = torch.nn.functional.cross_entropy(unit @ self.prototypes["crop"].T / .07, self.crop)
        expected_d = torch.nn.functional.cross_entropy(unit @ self.prototypes["disease"].T / .07, self.disease)
        torch.testing.assert_close(loss["crop_loss"], expected_c)
        torch.testing.assert_close(loss["disease_loss"], expected_d)
        self.assertEqual(loss["identity_loss"].item(), 0.)
        torch.testing.assert_close(loss["loss"], .5 * expected_c + .5 * expected_d)
        self.assertTrue(torch.isfinite(loss["loss"]))

    def test_identity_penalty_is_per_sample_squared_l2(self):
        y = self.x.clone()
        y[:, :10] += .2
        loss = self.ns["adapter_objective"](y, self.x, self.crop, self.disease, self.prototypes)
        expected = ((torch.nn.functional.normalize(y, dim=1)
                     - torch.nn.functional.normalize(self.x, dim=1)) ** 2).sum(1).mean()
        torch.testing.assert_close(loss["identity_loss"], expected)
        torch.testing.assert_close(loss["loss"], .5 * loss["crop_loss"] + .5 * loss["disease_loss"] + expected)

    def test_labels_and_trainable_prototypes_rejected(self):
        for labels in (self.crop.float(), torch.tensor([999, 0]), self.crop[:1]):
            with self.assertRaises(ValueError):
                self.ns["adapter_objective"](self.adapter(self.x), self.x, labels, self.disease, self.prototypes)
        bad = dict(self.prototypes)
        bad["crop"] = bad["crop"].clone().requires_grad_(True)
        with self.assertRaises(ValueError):
            self.ns["adapter_objective"](self.adapter(self.x), self.x, self.crop, self.disease, bad)

    def test_gradient_first_up_then_down_and_input_detached(self):
        x = self.x.requires_grad_(True)
        optimizer = self.ns["make_adapter_optimizer"](self.adapter, self.ns["CONFIG"])
        loss = self.ns["adapter_objective"](self.adapter(x), x, self.crop, self.disease, self.prototypes)["loss"]
        loss.backward()
        self.assertIsNone(x.grad)
        self.assertGreater(self.adapter.up.weight.grad.abs().sum().item(), 0)
        self.assertEqual(self.adapter.down.weight.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in self.adapter.parameters()))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        self.ns["adapter_objective"](self.adapter(x), x, self.crop, self.disease, self.prototypes)["loss"].backward()
        self.assertGreater(self.adapter.down.weight.grad.abs().sum().item(), 0)

    def test_optimizer_only_owns_adapter(self):
        optimizer = self.ns["make_adapter_optimizer"](self.adapter, self.ns["CONFIG"])
        self.assertEqual({id(p) for g in optimizer.param_groups for p in g["params"]},
                         {id(p) for p in self.adapter.parameters()})

    def test_contract_preserves_source_split_candidates_prompts(self):
        c = self.contract
        self.assertEqual(c["split"]["source_sha256"], "9f1b729ab08f120c9e8d3a5681aa1ccbdf1d9210f9aaa6143959363dcb106a7d")
        self.assertEqual(c["split"]["split_sha256"], "f0c7ba212449fd68f1d19c0f84cc91bf4cc1af29ae405128e5198bce89f25f85")
        self.assertEqual(c["split"]["candidate_sha256"], "4cb0a24a96cf647f32311817fa7b082a46d3cf00c20ae61f4287e83cc898dfaf")
        self.assertEqual(c["direct_prompt_sha256"], "753672c7964bcb20a1346145f21a72b40a19f55590a006b5783823f23c13f537")
        self.assertEqual(c["candidate_ids"], list(self.split.candidate_ids))
        self.assertNotEqual(c["candidate_ids"], list(range(323)))

    def test_selection_full_precision_ties_and_epoch_zero(self):
        best = {"epoch": 0, "h1": .08, "combined_validation_loss": 5.}
        self.assertTrue(self.ns["better_selection"]({"epoch": 2, "h1": .08, "combined_validation_loss": 4.}, best))
        self.assertFalse(self.ns["better_selection"]({"epoch": 1, "h1": .08, "combined_validation_loss": 5.}, best))
        higher = float(np.nextafter(.08, 1.))
        self.assertTrue(self.ns["better_selection"]({"epoch": 2, "h1": higher, "combined_validation_loss": 100.}, best))

    def test_selection_uses_processed_sample_weighting(self):
        seen = {"sample_count": 2, "loss": 1., "metrics": {"top1": .5, "top5": 1.}}
        unseen = {"sample_count": 3, "loss": 4., "metrics": {"top1": 1/3, "top5": 2/3}}
        record = self.ns["selection_record"](1, seen, unseen)
        self.assertEqual(record["combined_validation_loss"], 14/5)
        self.assertEqual(record["h1"], self.ns["harmonic_mean"](.5, 1/3))

    def test_checkpoint_roundtrip_and_no_overwrite(self):
        with torch.no_grad():
            self.adapter.up.weight.normal_(std=.001)
        record = {"epoch": 0, "h1": .08, "combined_validation_loss": 3.}
        with workspace_temporary_directory() as directory:
            path = Path(directory) / "adapter.pt"
            self.ns["save_adapter_checkpoint"](path, self.adapter, self.state, self.contract, record)
            restored, text, meta = self.ns["load_adapter_checkpoint"](path, self.contract)
            self.adapter.eval()
            torch.testing.assert_close(restored(self.x), self.adapter(self.x), rtol=0, atol=0)
            torch.testing.assert_close(text["direct"], self.state["direct"], rtol=0, atol=0)
            self.assertEqual(meta, record)
            with self.assertRaises(FileExistsError):
                self.ns["save_adapter_checkpoint"](path, self.adapter, self.state, self.contract, record)

    def test_checkpoint_reordered_mapping_and_model_identity_rejected(self):
        with workspace_temporary_directory() as directory:
            path = Path(directory) / "adapter.pt"
            self.ns["save_adapter_checkpoint"](path, self.adapter, self.state, self.contract, {"epoch": 0})
            for field in ("candidate_ids", "model"):
                changed = copy.deepcopy(self.contract)
                if field == "candidate_ids":
                    changed[field][:2] = reversed(changed[field][:2])
                else:
                    changed[field]["sha256"] = "0" * 64
                with self.assertRaises(ValueError):
                    self.ns["load_adapter_checkpoint"](path, changed)

    def test_checkpoint_tampered_architecture_and_text_rejected(self):
        with workspace_temporary_directory() as directory:
            path = Path(directory) / "adapter.pt"
            self.ns["save_adapter_checkpoint"](path, self.adapter, self.state, self.contract, {"epoch": 0})
            payload = torch.load(path, weights_only=True)
            payload["adapter_spec"]["bottleneck"] = 128
            other = Path(directory) / "bad.pt"
            torch.save(payload, other)
            with self.assertRaises(ValueError):
                self.ns["load_adapter_checkpoint"](other, self.contract)
            bad = copy.deepcopy(self.state)
            bad["direct"][0, 0] += .1
            with self.assertRaises(ValueError):
                self.ns["save_adapter_checkpoint"](Path(directory) / "stale.pt", self.adapter, bad, self.contract, {})

    def synthetic_loader(self, role="fit"):
        frame = getattr(self.split, role).iloc[:3].copy()
        class Tiny(Dataset):
            def __init__(self):
                self.role = role
                self.frame = frame
            def __len__(self):
                return len(frame)
            def __getitem__(self, i):
                r = frame.iloc[i]
                return torch.ones(3, 4, 4) * (i + 1), {"image_name": r.image_name,
                    "crop": int(r.crop_id), "disease": int(r.disease_id), "composition": int(r.composition_id)}
        return DataLoader(Tiny(), batch_size=2, shuffle=False), frame

    def fake_clip(self):
        class TinyCLIP(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = torch.nn.Linear(3, 768)
                self.bn = torch.nn.BatchNorm1d(768)
            def encode_image(self, images):
                assert not torch.is_grad_enabled()
                return self.bn(self.proj(images.mean((2, 3))))
        return TinyCLIP().eval().requires_grad_(False)

    def test_bounded_fit_updates_only_adapter_and_finite_gradients(self):
        loader, frame = self.synthetic_loader()
        model = self.fake_clip()
        before = {k: v.clone() for k, v in model.state_dict().items()}
        optimizer = self.ns["make_adapter_optimizer"](self.adapter, self.ns["CONFIG"])
        initial = self.adapter.up.weight.detach().clone()
        result = self.ns["train_adapter_epoch"](model, self.adapter, loader, optimizer, self.state, frame, "cpu", False)
        self.assertEqual(result["sample_count"], 3)
        self.assertTrue(math.isfinite(result["loss"]))
        self.assertFalse(torch.equal(initial, self.adapter.up.weight))
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        self.assertTrue(all(torch.equal(before[k], v) for k, v in model.state_dict().items()))

    def test_validation_no_parameter_or_buffer_updates_identity_control(self):
        loader, frame = self.synthetic_loader("val_seen")
        model = self.fake_clip()
        before = {k: v.clone() for k, v in model.state_dict().items()}
        adapter_before = {k: v.clone() for k, v in self.adapter.state_dict().items()}
        result = self.ns["validate_adapter"](model, self.adapter, loader, self.state, frame,
                                              self.split.candidate_ids, "cpu", False)
        self.assertEqual(result["sample_count"], 3)
        self.assertEqual(result["ranking"].shape, (3, 323))
        self.assertTrue(torch.equal(result["ranking"], result["control_ranking"]))
        self.assertTrue(all(torch.equal(before[k], v) for k, v in model.state_dict().items()))
        self.assertTrue(all(torch.equal(adapter_before[k], v) for k, v in self.adapter.state_dict().items()))

    def test_validation_loader_cannot_be_used_for_fitting(self):
        loader, frame = self.synthetic_loader("val_seen")
        with self.assertRaises(ValueError):
            self.ns["train_adapter_epoch"](self.fake_clip(), self.adapter, loader,
                self.ns["make_adapter_optimizer"](self.adapter, self.ns["CONFIG"]), self.state,
                self.split.fit, "cpu", False)

    def test_unfrozen_or_train_mode_clip_rejected(self):
        model = self.fake_clip()
        self.ns["require_frozen_clip"](model)
        model.train()
        with self.assertRaises(ValueError):
            self.ns["require_frozen_clip"](model)
        model.eval().requires_grad_(True)
        with self.assertRaises(ValueError):
            self.ns["require_frozen_clip"](model)

    def test_empty_and_nonfinite_inputs_rejected(self):
        loader, frame = self.synthetic_loader()
        class Empty(Dataset):
            role = "fit"
            def __len__(self): return 0
            def __getitem__(self, index): raise IndexError
        with self.assertRaises(ValueError):
            self.ns["train_adapter_epoch"](self.fake_clip(), self.adapter, DataLoader(Empty()),
                self.ns["make_adapter_optimizer"](self.adapter, self.ns["CONFIG"]), self.state, frame, "cpu", False)

    def test_invalid_first_selection_is_rejected(self):
        for record in ({"epoch": 0, "h1": float("nan"), "combined_validation_loss": 1.},
                       {"epoch": -1, "h1": .1, "combined_validation_loss": 1.}):
            with self.assertRaises(ValueError):
                self.ns["better_selection"](record, None)

    def test_validation_rows_disguised_as_fitting_rejected_before_forward(self):
        loader, _ = self.synthetic_loader("val_seen")
        loader.dataset.role = "fit"
        model = self.fake_clip()
        optimizer = self.ns["make_adapter_optimizer"](self.adapter, self.ns["CONFIG"])
        with patch.object(model, "encode_image", side_effect=AssertionError("must reject first")), \
             patch.object(optimizer, "step", side_effect=AssertionError("must not optimize")):
            with self.assertRaises(ValueError):
                self.ns["train_adapter_epoch"](model, self.adapter, loader, optimizer, self.state,
                                                 self.split.fit, "cpu", False)

    def test_validation_partial_batches_are_sample_weighted(self):
        loader, frame = self.synthetic_loader("val_seen")
        model = self.fake_clip()
        result = self.ns["validate_adapter"](model, self.adapter, loader, self.state, frame,
                                             self.split.candidate_ids, "cpu", False)
        images = torch.stack([loader.dataset[i][0] for i in range(3)])
        with torch.no_grad():
            original = model.encode_image(images)
            expected = self.ns["adapter_objective"](self.adapter(original), original,
                torch.tensor(frame.crop_id.to_numpy()), torch.tensor(frame.disease_id.to_numpy()), self.state)
        for key in ("loss", "crop_loss", "disease_loss", "identity_loss"):
            self.assertAlmostEqual(result[key], expected[key].item(), delta=1e-6)

    def test_checkpoint_load_is_safe_offline_and_backbone_free(self):
        with workspace_temporary_directory() as directory:
            path = Path(directory) / "adapter.pt"
            self.ns["save_adapter_checkpoint"](path, self.adapter, self.state, self.contract, {"epoch": 0})
            payload = torch.load(path, weights_only=True)
            self.assertEqual(set(payload["adapter_state"]), {"down.weight", "down.bias", "up.weight", "up.bias"})
            self.assertNotIn("optimizer", payload)
            with patch("clip.load", side_effect=AssertionError("no CLIP load")), \
                 patch("torch.hub.load_state_dict_from_url", side_effect=AssertionError("no download")), \
                 patch("torch.load", wraps=torch.load) as safe_load:
                self.ns["load_adapter_checkpoint"](path, self.contract)
                self.assertIs(safe_load.call_args.kwargs["weights_only"], True)
                self.assertEqual(safe_load.call_args.kwargs["map_location"], "cpu")

    def test_real_text_preparation_boundary_uses_normal_frozen_tensors(self):
        class FakeText(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.ones(1), requires_grad=False)
            def encode_text(self, tokens):
                assert not torch.is_grad_enabled()
                return tokens.sum(1).float()[:, None] + torch.arange(768).float()[None, :] + 1
        config = {**self.ns["CONFIG"], "device": "cpu"}
        contract = self.ns["make_m2_contract"](self.split, self.table, config)
        model = FakeText().eval()
        state, provenance = self.ns["prepare_text_state"](model, self.table, config, contract)
        for key in ("direct", "crop", "disease"):
            self.assertFalse(torch.is_inference(state[key]), key)
            self.assertFalse(state[key].requires_grad)
        self.assertEqual(provenance["features_shape"], [323, 768])
        loss = self.ns["adapter_objective"](self.adapter(self.x), self.x, self.crop, self.disease, state)
        loss["loss"].backward()
        self.assertTrue(torch.isfinite(self.adapter.up.weight.grad).all())
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_component_loss_supports_explicit_noncontiguous_ids(self):
        g = torch.Generator().manual_seed(1)
        prototypes = {"crop": torch.nn.functional.normalize(torch.randn(2, 768, generator=g), dim=1),
                      "crop_ids": [10, 50],
                      "disease": torch.nn.functional.normalize(torch.randn(3, 768, generator=g), dim=1),
                      "disease_ids": [7, 9, 31]}
        y = self.adapter(self.x)
        loss = self.ns["adapter_objective"](y, self.x, torch.tensor([10, 50]), torch.tensor([31, 9]), prototypes)
        unit = torch.nn.functional.normalize(y, dim=1)
        expected = .5 * torch.nn.functional.cross_entropy(unit @ prototypes["crop"].T / .07, torch.tensor([0, 1]))
        expected += .5 * torch.nn.functional.cross_entropy(unit @ prototypes["disease"].T / .07, torch.tensor([2, 1]))
        torch.testing.assert_close(loss["loss"], expected)

    def test_nonfinite_fitting_images_rejected_before_forward(self):
        _, frame = self.synthetic_loader()
        class Bad(Dataset):
            role = "fit"
            def __len__(self): return 1
            def __getitem__(self, index):
                r = frame.iloc[0]
                return torch.full((3, 4, 4), float("nan")), {"image_name": r.image_name,
                    "crop": int(r.crop_id), "disease": int(r.disease_id), "composition": int(r.composition_id)}
        model = self.fake_clip()
        optimizer = self.ns["make_adapter_optimizer"](self.adapter, self.ns["CONFIG"])
        with patch.object(model, "encode_image", side_effect=AssertionError("no encoder")), \
             patch.object(optimizer, "step", side_effect=AssertionError("no update")):
            with self.assertRaises(ValueError):
                self.ns["train_adapter_epoch"](model, self.adapter, DataLoader(Bad(), batch_size=1),
                                                optimizer, self.state, frame, "cpu", False)

    def test_heldout_composition_text_cannot_affect_training_anchors(self):
        pool = self.contract["fitting_composition_ids"]
        self.assertEqual(len(pool), 258)
        changed = self.direct.clone()
        heldout = ~self.table.global_composition_id.isin(pool).to_numpy()
        changed[heldout] = torch.randn(int(heldout.sum()), 768, dtype=torch.float64)
        actual = self.ns["component_prototypes"](changed, self.table, pool)
        for kind in ("crop", "disease"):
            torch.testing.assert_close(actual[kind], self.prototypes[kind], rtol=0, atol=0)
        for bad in (None, list(reversed(pool)), [99999]):
            with self.assertRaises(ValueError):
                self.ns["component_prototypes"](self.direct, self.table, bad)


class WorkspacePathTests(unittest.TestCase):
    def test_configuration_uses_canonical_workspace(self):
        ns = namespace()
        base = Path('/home/user1/COS30082 Applied Machine Learning/00 Assigments/AS02 ML Project/workspace')
        self.assertEqual(Path(ns['WORKSPACE_DIRECTORY']), base)
        self.assertEqual(Path(ns['CONFIG']['output_directory']), base / 'experiments/wp3b-m2')
        self.assertEqual(Path(ns['CONFIG']['weight_directory']), base / 'pretrained_weights/clip')
        self.assertEqual(Path(ns['CONFIG']['image_directory']), base / 'datasets/DPD_Small_dataset')
        self.assertEqual(Path(ns['HANDOFF_DIRECTORY']), base / 'handoffs')

    def test_canonical_output_allowed_and_external_paths_rejected(self):
        ns = namespace()
        base = Path('/home/user1/COS30082 Applied Machine Learning/00 Assigments/AS02 ML Project/workspace/experiments')
        for path in (base, base / 'wp3b-m2/future/checkpoint.pt'):
            self.assertEqual(ns['require_external_path'](path), path)
        for path in ('relative.pt', '/tmp/checkpoint.pt', ROOT / 'checkpoint.pt', base.parent / 'handoffs/checkpoint.pt'):
            with self.assertRaises(ValueError):
                ns['require_external_path'](path)

    def test_output_symlink_escape_rejected(self):
        ns = namespace()
        with workspace_temporary_directory() as directory:
            link = Path(directory) / 'escape'
            link.symlink_to('/tmp', target_is_directory=True)
            with self.assertRaises(ValueError):
                ns['require_external_path'](link / 'checkpoint.pt')
            link.unlink()  # Test-created alias only.

    def test_nested_git_checkout_below_experiments_rejected(self):
        ns = namespace()
        with workspace_temporary_directory() as directory:
            (Path(directory) / '.git').mkdir()
            with self.assertRaises(ValueError):
                ns['require_external_path'](Path(directory) / 'checkpoint.pt')

if __name__ == "__main__":
    unittest.main()
