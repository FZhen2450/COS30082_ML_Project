"""WP3A notebook contracts; synthetic CPU checks and native tokenizer only.

Definitions are extracted for unit testing; no notebook execution cells run.
Real CLIP weights, image datasets and GPU inference are not needed by this suite.
"""
import ast
import hashlib
import importlib.metadata as metadata
import json
import math
import random
import re
import sys
import tempfile
import time
import unicodedata
import unittest
import uuid
from datetime import datetime, timezone
from collections import Counter
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/wp3_clip_zero_shot.ipynb"


def notebook():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def sources():
    return ["".join(c["source"]) for c in notebook()["cells"] if c["cell_type"] == "code"]


def literals():
    values = {}
    for source in sources():
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
    namespace = {**globals(), **literals(), "__name__": __name__, "ROOT": ROOT,
                 "DEFAULT_CSVS": (str(ROOT / "DPD_Small_train.csv"),)}
    definitions = [n for source in sources() for n in ast.parse(source).body
                   if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
    exec(compile(ast.Module(body=definitions, type_ignores=[]),
                 "<visible WP3 notebook definitions>", "exec"), namespace)
    return namespace


class WP3StructureTests(unittest.TestCase):
    def test_valid_json_unique_cells_and_unexecuted(self):
        nb = notebook()
        self.assertEqual((nb["nbformat"], nb["nbformat_minor"]), (4, 5))
        ids = [c["id"] for c in nb["cells"]]
        self.assertEqual(len(ids), len(set(ids)))
        for cell in nb["cells"]:
            if cell["cell_type"] == "code":
                self.assertIsNone(cell["execution_count"])
                self.assertEqual(cell["outputs"], [])
        self.assertFalse(nb["metadata"]["cos30082"]["full_development_inference_executed"])

    def test_all_cells_compile(self):
        for i, source in enumerate(sources()):
            compile(source, f"wp3-cell-{i}", "exec")

    def test_sixteen_numbered_sections(self):
        markdown = "\n".join("".join(c["source"]) for c in notebook()["cells"] if c["cell_type"] == "markdown")
        self.assertEqual(re.findall(r"^## (\d+)\.", markdown, re.M), [str(i) for i in range(1, 17)])

    def test_all_expensive_flags_default_false(self):
        values = literals()
        for key in ("ACQUIRE_PRETRAINED_WEIGHTS", "LOAD_CLIP_MODEL", "RUN_TEXT_ENCODING",
                    "RUN_IMAGE_SMOKE", "RUN_DEVELOPMENT_EVALUATION", "SHOW_IMAGES",
                    "READ_WP2_RESULTS", "SAVE_RESULTS", "RUN_TRAINING"):
            self.assertIs(values[key], False, key)

    def test_no_project_ml_imports_or_dynamic_pipeline(self):
        forbidden = {"dataset", "data_utils", "development_data", "train_multitask",
                     "multitask_model", "multitask_backbones", "evaluation"}
        tree = ast.parse("\n".join(sources()))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                self.assertFalse(any(a.name.split(".")[0] in forbidden for a in node.names))
            if isinstance(node, ast.ImportFrom):
                self.assertNotIn((node.module or "").split(".")[0], forbidden)
            if isinstance(node, ast.Call):
                self.assertNotIn(ast.unparse(node.func), ("exec", "eval", "compile", "importlib.import_module"))
        self.assertNotIn("%run", "\n".join(sources()))

    def test_required_steps_are_visible(self):
        namespace = helper_namespace()
        for name in ("load_metadata", "make_validation_split", "build_development_split",
                     "build_prompt_table", "CLIPDevelopmentDataset", "load_frozen_clip",
                     "tokenize_candidates", "unit_features", "cosine_candidate_scores",
                     "rank_global_ids", "topk_accuracy", "harmonic_mean", "infer_images",
                     "prediction_diagnostics", "export_results"):
            self.assertIn(name, namespace)

    def test_model_and_inference_calls_are_guarded(self):
        guarded = {"load_frozen_clip": "LOAD_CLIP_MODEL", "acquire_official_weight": "ACQUIRE_PRETRAINED_WEIGHTS",
                   "encode_candidate_texts": "RUN_TEXT_ENCODING", "export_results": "SAVE_RESULTS"}
        def walk(nodes, conditions=()):
            for node in nodes:
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    continue
                if isinstance(node, ast.If):
                    walk(node.body, conditions + (ast.unparse(node.test),))
                    walk(node.orelse, conditions + ("NOT " + ast.unparse(node.test),))
                else:
                    for part in ast.walk(node):
                        if isinstance(part, ast.Call) and isinstance(part.func, ast.Name):
                            name = part.func.id
                            if name in guarded:
                                self.assertIn(guarded[name], conditions)
                            if name == "infer_images":
                                self.assertTrue(any(c in ("RUN_IMAGE_SMOKE", "RUN_DEVELOPMENT_EVALUATION") for c in conditions))
        for source in sources():
            walk(ast.parse(source).body)

    def test_no_optimizer_training_or_official_csv_paths(self):
        code = "\n".join(sources())
        for token in (".backward(", "torch.optim.", "DPD_Small_test_seen.csv", "DPD_Small_test_unseen.csv"):
            self.assertNotIn(token, code)

    def test_source_metadata_gate_rejects_other_csv_before_read(self):
        ns = helper_namespace()
        with patch.object(pd, "read_csv", side_effect=AssertionError("must not read")):
            with self.assertRaises(ValueError):
                ns["build_development_split"]("unapproved.csv")

    def test_native_loading_uses_verified_path_and_freezes(self):
        ns = helper_namespace()
        model = torch.nn.Linear(3, 2)
        model.visual = type("Visual", (), {"input_resolution": 224})()
        model.context_length = 77
        model.logit_scale = torch.nn.Parameter(torch.tensor(1.0))
        import clip
        config = {"model": "ViT-B/32", "device": "cpu", "weight_directory": "/not/a/download"}
        artifact = {"path": "/verified/local.pt", "sha256": ns["MODEL_REGISTRY"]["ViT-B/32"]["sha256"], "bytes": 1}
        ns["verify_weight"] = lambda *args: artifact
        with patch.object(clip, "load", return_value=(model, "native-transform")) as loader:
            result, transform, provenance = ns["load_frozen_clip"](config)
        loader.assert_called_once_with("/verified/local.pt", device=torch.device("cpu"), jit=False)
        self.assertFalse(result.training)
        self.assertFalse(any(p.requires_grad for p in result.parameters()))
        self.assertEqual(transform, "native-transform")
        self.assertEqual(provenance["trainable_parameters"], 0)


class WP3MappingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = helper_namespace()
        cls.split = cls.ns["build_development_split"](ROOT / "DPD_Small_train.csv")
        cls.table = cls.ns["build_prompt_table"](cls.split)

    def test_exact_split_counts_and_component_coverage(self):
        s = self.split
        self.assertEqual(tuple(map(len, (s.fit, s.val_seen, s.val_unseen))), (4868, 1238, 1894))
        self.assertEqual((s.fit.crop_id.nunique(), s.fit.disease_id.nunique()), (55, 175))
        self.assertEqual((s.fit.composition_id.nunique(), s.val_unseen.composition_id.nunique()), (258, 65))

    def test_disjoint_full_coverage_and_composition_isolation(self):
        s = self.split
        frames = [s.fit, s.val_seen, s.val_unseen]
        self.assertEqual(len(set(pd.concat(frames).image_name)), 8000)
        self.assertEqual(sum(map(len, frames)), 8000)
        self.assertFalse(set(s.fit.composition_id) & set(s.val_unseen.composition_id))
        self.assertTrue(set(s.val_seen.composition_id) <= set(s.fit.composition_id))

    def test_matches_accepted_reference_split_and_hashes(self):
        from development_data import build_development_split
        reference = build_development_split(ROOT / "DPD_Small_train.csv")
        self.assertEqual(self.split.candidate_ids, reference.candidate_ids)
        self.assertEqual(self.split.candidate_pairs, reference.candidate_pairs)
        self.assertEqual(self.split.candidate_sha256, reference.candidate_sha256)
        for key in ("fit", "val_seen", "val_unseen"):
            pd.testing.assert_frame_equal(getattr(self.split, key), getattr(reference, key))
        identity = self.ns["split_identity"](self.split)
        self.assertEqual(identity["candidate_sha256"], "4cb0a24a96cf647f32311817fa7b082a46d3cf00c20ae61f4287e83cc898dfaf")
        self.assertEqual(identity["split_sha256"], "f0c7ba212449fd68f1d19c0f84cc91bf4cc1af29ae405128e5198bce89f25f85")

    def test_323_prompts_align_with_original_noncontiguous_ids(self):
        t = self.table
        self.assertEqual(len(t), 323)
        self.assertEqual(t.global_composition_id.tolist(), list(self.split.candidate_ids))
        self.assertEqual(t.score_column.tolist(), list(range(323)))
        self.assertNotEqual(t.global_composition_id.tolist(), list(range(323)))
        self.assertEqual(list(zip(t.crop_id, t.disease_id)), list(self.split.candidate_pairs))

    def test_healthy_and_diseased_templates(self):
        vowel_crops = {"apple", "apricot", "eggplant", "olive"}
        self.assertEqual(self.ns["PROMPT_POLICY"]["healthy"], "a photo of a healthy {crop} plant")
        for row in self.table.itertuples():
            article = "an" if row.crop in vowel_crops else "a"
            expected = (f"a photo of a healthy {row.crop} plant" if row.disease == "healthy"
                        else f"a photo of {article} {row.crop} plant affected by {row.disease}")
            self.assertEqual(row.prompt, expected)
        self.assertFalse(self.table.prompt.duplicated().any())

    def test_label_only_article_rule(self):
        for crop, expected in (("apple", "an"), ("Apricot", "an"), (" eggplant ", "an"),
                               ("olive", "an"), ("banana", "a"), ("PEPPER_BELL", "a")):
            with self.subTest(crop=crop):
                self.assertEqual(self.ns["crop_indefinite_article"](crop), expected)
        with self.assertRaises(ValueError):
            self.ns["crop_indefinite_article"]("   ")

    def test_exactly_23_diseased_prompts_change_and_healthy_stays_identical(self):
        changed = Counter()
        healthy_count = 0
        for row in self.table.itertuples():
            old = (f"a photo of a healthy {row.crop} plant" if row.disease == "healthy"
                   else f"a photo of a {row.crop} plant affected by {row.disease}")
            if row.disease == "healthy":
                healthy_count += 1
                self.assertEqual(row.prompt, old)
            elif row.prompt != old:
                changed[row.crop] += 1
                self.assertEqual(row.prompt, old.replace(f"a {row.crop} plant", f"an {row.crop} plant", 1))
        self.assertEqual(changed, {"apple": 16, "apricot": 2, "eggplant": 3, "olive": 2})
        self.assertEqual(sum(changed.values()), 23)
        self.assertEqual(healthy_count, 37)

    def test_prompt_policy_version_is_incremented(self):
        self.assertEqual(self.ns["PROMPT_POLICY"]["version"], "label_only_single_template_v2")
        self.assertIn("article_rule", self.ns["PROMPT_POLICY"])
        digest = self.ns["json_hash"]({"policy": self.ns["PROMPT_POLICY"], "rows": self.table.to_dict("records")})
        self.assertEqual(digest, "753672c7964bcb20a1346145f21a72b40a19f55590a006b5783823f23c13f537")
        self.assertEqual(notebook()["metadata"]["cos30082"]["prompt_sha256"], digest)

    def test_both_backbones_share_the_final_prompt_table_and_tokens(self):
        ns = helper_namespace()
        records, digests, token_tables = [], [], []
        import clip
        with patch.object(clip, "load", side_effect=AssertionError("model loading forbidden")):
            for name in ("ViT-B/32", "ViT-L/14"):
                ns["CONFIG"] = {"model": name}
                table = ns["build_prompt_table"](self.split)
                records.append(table.to_dict("records"))
                digests.append(ns["json_hash"]({"policy": ns["PROMPT_POLICY"], "rows": records[-1]}))
                token_tables.append(ns["tokenize_candidates"](table))
        self.assertEqual(records[0], records[1])
        self.assertEqual(digests[0], digests[1])
        self.assertTrue(torch.equal(token_tables[0], token_tables[1]))
        self.assertEqual(tuple(token_tables[0].shape), (323, 77))

    def test_normalization_preserves_label_words(self):
        self.assertEqual(self.ns["normalize_label"]("  Late__Blight\t"), "late blight")
        self.assertEqual(self.ns["normalize_label"]("Ｂｌｉｇｈｔ"), "blight")
        with self.assertRaises(ValueError):
            self.ns["normalize_label"]("   ")

    def test_native_tokens_and_no_truncation(self):
        tokens = self.ns["tokenize_candidates"](self.table)
        self.assertEqual(tokens.shape, (323, 77))
        self.assertEqual(tokens.dtype, torch.int32)
        import clip
        self.assertTrue(torch.equal(tokens, clip.tokenize(self.table.prompt.tolist(), truncate=False)))
        long = self.table.iloc[:1].copy()
        long["prompt"] = "plant " * 200
        with self.assertRaises(RuntimeError):
            self.ns["tokenize_candidates"](long)

    def test_duplicate_prompts_rejected(self):
        table = pd.concat([self.table.iloc[:1], self.table.iloc[:1]], ignore_index=True)
        with self.assertRaises(ValueError):
            self.ns["tokenize_candidates"](table)

    def test_source_hash_and_candidate_hash_are_recomputed(self):
        self.assertEqual(self.ns["sha256_file"](self.split.source_csv), self.split.source_sha256)
        rows = [[key, *pair] for key, pair in zip(self.split.candidate_ids, self.split.candidate_pairs)]
        self.assertEqual(hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest(),
                         self.split.candidate_sha256)


class WP3MetricTests(unittest.TestCase):
    def setUp(self):
        self.ns = helper_namespace()
        self.ids = (2, 7, 19, 23, 90, 140)

    def test_unit_features_are_float64_and_normalized(self):
        unit = self.ns["unit_features"](torch.tensor([[3., 4.], [1., -1.]]))
        self.assertEqual(unit.dtype, torch.float64)
        self.assertTrue(torch.allclose(torch.linalg.vector_norm(unit, dim=1), torch.ones(2, dtype=torch.float64)))

    def test_zero_nonfinite_and_invalid_features_rejected(self):
        for value in (torch.zeros(1, 2), torch.tensor([[float("nan"), 1.]]), torch.empty(0, 2), torch.ones(2, dtype=torch.int64)):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    self.ns["unit_features"](value)

    def test_similarity_is_normalized_dot_product(self):
        score = self.ns["cosine_candidate_scores"](torch.tensor([[3., 4.]]), torch.tensor([[3., 4.], [-3., -4.], [4., -3.]]))
        torch.testing.assert_close(score, torch.tensor([[1., -1., 0.]], dtype=torch.float64))

    def test_extreme_finite_features_are_stable(self):
        score = self.ns["cosine_candidate_scores"](torch.tensor([[1e30, -1e30]]), torch.tensor([[1e30, -1e30]]))
        self.assertTrue(torch.isfinite(score).all())
        self.assertAlmostEqual(score.item(), 1.)

    def test_shape_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            self.ns["cosine_candidate_scores"](torch.ones(2, 3), torch.ones(2, 4))

    def test_exact_ties_use_ascending_global_id(self):
        ranks = self.ns["rank_global_ids"](torch.ones(2, 6), self.ids)
        self.assertEqual(ranks.tolist(), [list(self.ids), list(self.ids)])

    def test_near_equal_scores_are_not_rounded_or_treated_as_ties(self):
        x = torch.ones(1, 6, dtype=torch.float64)
        x[0, 1] = torch.nextafter(x[0, 1], torch.tensor(float("inf"), dtype=torch.float64))
        self.assertEqual(self.ns["rank_global_ids"](x, self.ids)[0, 0].item(), 7)

    def test_noncontiguous_truth_ids_top1_top5(self):
        scores = torch.tensor([[6., 5., 4., 3., 2., 1.], [1., 2., 3., 4., 5., 6.]])
        metrics, ranks = self.ns["fold_metrics"](scores, torch.tensor([2, 7]), self.ids)
        self.assertEqual(metrics, {"sample_count": 2, "top1": .5, "top5": 1.})
        self.assertEqual(ranks[1, 0].item(), 140)

    def test_missing_truth_or_invalid_k_rejected(self):
        ranks = self.ns["rank_global_ids"](torch.ones(1, 6), self.ids)
        with self.assertRaises(ValueError):
            self.ns["topk_accuracy"](ranks, torch.tensor([999]), 1)
        with self.assertRaises(ValueError):
            self.ns["topk_accuracy"](ranks, torch.tensor([2]), 7)

    def test_unsorted_duplicate_or_nonfinite_candidates_rejected(self):
        for ids in ((7, 2), (2, 2), (2, -1)):
            with self.assertRaises(ValueError):
                self.ns["rank_global_ids"](torch.ones(1, 2), ids)
        with self.assertRaises(ValueError):
            self.ns["rank_global_ids"](torch.full((1, 6), float("inf")), self.ids)

    def test_harmonic_mean_zero_and_known_values(self):
        h = self.ns["harmonic_mean"]
        self.assertEqual(h(0., 0.), 0.)
        self.assertEqual(h(1., 0.), 0.)
        self.assertAlmostEqual(h(.5, .25), 1/3)
        for invalid in (float("nan"), -0.1, 1.1):
            with self.assertRaises(ValueError):
                h(invalid, .5)

    def test_development_metrics_keep_both_counts_and_k_values(self):
        result = self.ns["development_metrics"]({"sample_count": 3, "top1": .5, "top5": 1.},
                                                 {"sample_count": 7, "top1": .25, "top5": .5})
        self.assertEqual((result["seen_count"], result["unseen_count"]), (3, 7))
        self.assertAlmostEqual(result["h1"], 1/3)
        self.assertAlmostEqual(result["h5"], 2/3)


class WP3DataAndSafetyTests(unittest.TestCase):
    def setUp(self):
        self.ns = helper_namespace()

    def test_native_preprocess_has_clip_constants_and_geometry(self):
        from clip.clip import _transform
        transform = _transform(224)
        image = Image.new("RGB", (80, 160), (255, 0, 0))
        value = transform(image)
        self.assertEqual(value.shape, (3, 224, 224))
        self.assertAlmostEqual(value[0, 0, 0].item(), (1 - .48145466) / .26862954, places=5)
        self.assertIn("BICUBIC", repr(transform).upper())
        self.assertIn("CenterCrop", repr(transform))

    def test_dataset_decodes_only_allowed_flat_training_filenames(self):
        from clip.clip import _transform
        with tempfile.TemporaryDirectory() as directory:
            Image.new("RGB", (64, 48), "green").save(Path(directory) / "fit.png")
            frame = pd.DataFrame([["fit.png", 1, 2, 90]], columns=["image_name", "crop_id", "disease_id", "composition_id"])
            ds = self.ns["CLIPDevelopmentDataset"](frame, directory, _transform(224), {"fit.png"})
            image, target = ds[0]
            self.assertEqual(image.shape, (3, 224, 224))
            self.assertEqual(target, {"crop": 1, "disease": 2, "composition": 90, "image_name": "fit.png"})
            with self.assertRaises(ValueError):
                self.ns["CLIPDevelopmentDataset"](frame, directory, _transform(224), {"other.png"})

    def test_dataset_rejects_path_escape(self):
        frame = pd.DataFrame([["../bad.png", 1, 2, 90]], columns=["image_name", "crop_id", "disease_id", "composition_id"])
        with self.assertRaises(ValueError):
            self.ns["CLIPDevelopmentDataset"](frame, "/tmp", lambda x: x, {"../bad.png"})

    def test_weight_hash_mismatch_and_missing_file_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.pt"
            with self.assertRaises(FileNotFoundError):
                self.ns["verify_weight"](path, "0" * 64)
            path.write_bytes(b"synthetic")
            with self.assertRaises(ValueError):
                self.ns["verify_weight"](path, "0" * 64)
            good = self.ns["verify_weight"](path, hashlib.sha256(b"synthetic").hexdigest())
            self.assertEqual(good["bytes"], 9)

    def test_existing_weight_acquisition_never_downloads_or_overwrites(self):
        from urllib.request import urlopen
        ns = self.ns
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ViT-B-32.pt"
            path.write_bytes(b"synthetic")
            ns["MODEL_REGISTRY"] = {"ViT-B/32": {"filename": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}}
            with patch("urllib.request.urlopen", side_effect=AssertionError("network forbidden")):
                self.assertEqual(ns["acquire_official_weight"]("ViT-B/32", directory)["bytes"], 9)
            self.assertEqual(path.read_bytes(), b"synthetic")

    def test_no_export_without_actual_complete_metrics(self):
        with self.assertRaises(ValueError):
            self.ns["export_results"]({}, None, None, None, None, None, None, None, {})

    def test_synthetic_inference_retains_ids_counts_and_no_gradients(self):
        class TinyCLIP(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.eye(2), requires_grad=False)
            def encode_image(self, x):
                return x @ self.weight
        model = TinyCLIP().eval()
        images = torch.tensor([[1., 0.], [0., 1.], [1., 0.]])
        targets = {"crop": torch.tensor([0, 1, 0]), "disease": torch.tensor([0, 1, 0]),
                   "composition": torch.tensor([2, 7, 2]), "image_name": ["a", "b", "c"]}
        result = self.ns["infer_images"](model, [(images, targets)], torch.tensor([[1., 0.], [0., 1.], [-1., 0.], [0., -1.], [1., 1.]]),
                                         (2, 7, 19, 23, 90), "cpu", progress=False)
        self.assertEqual(result["metrics"]["sample_count"], 3)
        self.assertEqual(result["metrics"]["top1"], 1.)
        self.assertEqual(result["timing"]["score_shape"], [3, 5])
        self.assertEqual(result["truths"].tolist(), [2, 7, 2])
        self.assertIsNone(model.weight.grad)

    def test_empty_or_unfrozen_inference_rejected(self):
        model = torch.nn.Linear(2, 2)
        with self.assertRaises(ValueError):
            self.ns["infer_images"](model, [], torch.ones(5, 2), (2, 7, 19, 23, 90), "cpu", False)
        model.eval().requires_grad_(False)
        with self.assertRaises(ValueError):
            self.ns["infer_images"](model, [], torch.ones(5, 2), (2, 7, 19, 23, 90), "cpu", False)


    def test_extreme_float64_normalization_does_not_overflow(self):
        unit = self.ns["unit_features"](torch.tensor([[1e300, -1e300], [1e-300, 1e-300]], dtype=torch.float64))
        self.assertTrue(torch.isfinite(unit).all())
        torch.testing.assert_close(torch.linalg.vector_norm(unit, dim=1), torch.ones(2, dtype=torch.float64))

    def test_top5_cutoff_tie_uses_smallest_five_ids(self):
        ranks = self.ns["rank_global_ids"](torch.ones(1, 6), (2, 7, 19, 23, 90, 140))
        self.assertEqual(self.ns["topk_accuracy"](ranks, torch.tensor([140]), 5), 0.)

    def test_prompt_mutation_or_model_mismatch_blocks_evaluation(self):
        ns = self.ns
        split = ns["build_development_split"](ROOT / "DPD_Small_train.csv")
        table = ns["build_prompt_table"](split)
        text = {"prompt_sha256": ns["json_hash"]({"policy": ns["PROMPT_POLICY"], "rows": table.to_dict("records")})}
        provenance = {"model": "ViT-B/32", "revision": ns["CLIP_REVISION"],
                      "artifact": {"sha256": ns["MODEL_REGISTRY"]["ViT-B/32"]["sha256"]}}
        digest = ns["verify_evaluation_contract"](split, table, text, provenance, {"model": "ViT-B/32"})
        self.assertEqual(digest, text["prompt_sha256"])
        changed = table.copy()
        changed.loc[0, "prompt"] += " new words"
        with self.assertRaises(ValueError):
            ns["verify_evaluation_contract"](split, changed, text, provenance, {"model": "ViT-B/32"})
        with self.assertRaises(ValueError):
            ns["verify_evaluation_contract"](split, table, text, provenance, {"model": "ViT-L/14"})

    def test_synthetic_inference_handles_final_partial_batch(self):
        class TinyCLIP(torch.nn.Module):
            def encode_image(self, x):
                return x
        model = TinyCLIP().eval().requires_grad_(False)
        batches = []
        for i, size in enumerate((2, 1)):
            batches.append((torch.tensor([[1., 0.]]).repeat(size, 1),
                {"crop": torch.zeros(size, dtype=torch.int64), "disease": torch.zeros(size, dtype=torch.int64),
                 "composition": torch.full((size,), 90, dtype=torch.int64),
                 "image_name": [f"part{i}_{j}" for j in range(size)]}))
        result = self.ns["infer_images"](model, batches, torch.tensor([[0., 1.], [-1., 0.], [0., -1.], [1., 1.], [1., 0.]]),
                                         (2, 7, 19, 23, 90), "cpu", progress=False)
        self.assertEqual(result["metrics"], {"sample_count": 3, "top1": 1., "top5": 1.})

    def test_repeated_filenames_in_inference_are_rejected(self):
        class TinyCLIP(torch.nn.Module):
            def encode_image(self, x):
                return x
        model = TinyCLIP().eval()
        targets = {"crop": torch.tensor([0, 0]), "disease": torch.tensor([0, 0]),
                   "composition": torch.tensor([2, 2]), "image_name": ["same", "same"]}
        with self.assertRaises(ValueError):
            self.ns["infer_images"](model, [(torch.ones(2, 2), targets)], torch.ones(5, 2),
                                     (2, 7, 19, 23, 90), "cpu", progress=False)

    def test_export_uses_unique_directories_and_preserves_candidate_table(self):
        ns = self.ns
        fixture_split = type("Split", (), {"mappings": {}, "candidate_ids": (2, 7), "candidate_pairs": ((0, 0), (0, 1))})()
        table = pd.DataFrame({"global_composition_id": [2, 7], "prompt": ["synthetic A", "synthetic B"]})
        diagnostic = {"summary": {"sample_count": 1}, "per_pair": pd.DataFrame({"global_composition_id": [2]}),
                      "representative_errors": []}
        ns["verify_evaluation_contract"] = lambda *args: "synthetic"
        ns["code_identity"] = lambda: {"source": "synthetic unit test"}
        ns["split_identity"] = lambda *args: {"source": "synthetic unit test"}
        ns["ENVIRONMENT"] = {"purpose": "synthetic unit test"}
        with tempfile.TemporaryDirectory() as directory:
            config = {"output_directory": directory, "model": "ViT-B/32"}
            args = (config, {"seen_count": 1238, "unseen_count": 1894}, diagnostic, diagnostic, {}, {},
                    fixture_split, table, {"purpose": "synthetic unit test"})
            first = ns["export_results"](*args)
            before = {p.name: p.read_bytes() for p in first.iterdir()}
            second = ns["export_results"](*args)
            self.assertNotEqual(first, second)
            self.assertEqual(before, {p.name: p.read_bytes() for p in first.iterdir()})
            self.assertFalse(any(p.suffix in (".pt", ".pth") for p in first.iterdir()))
            saved = pd.read_csv(first / "candidate_prompts.csv")
            self.assertEqual(saved.global_composition_id.tolist(), [2, 7])


if __name__ == "__main__":
    unittest.main()
