"""Synthetic GUI inference contracts; no CLIP installation/weights/images needed."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

from PIL import Image
import torch
from torch import nn

from shared_protocol import FOLDS, build_protocol
from wp4.service import Catalog, present_prediction
from wp4 import clip_predictor as cp

ROOT = Path(__file__).resolve().parents[1]


def make_catalog(directory):
    source = ROOT / "DPD_Small_train.csv"
    proposal = build_protocol(source)
    reference = {key: proposal[key] for key in FOLDS}
    # Anchor the independently reconstructed order to the recorded historical digest.
    assert cp.json_hash({name: reference[key] for name, key in
                        zip(("fit", "val_seen", "val_unseen"), FOLDS)}) == cp.SPLIT_SHA256
    protocol = build_protocol(source, reference=reference)
    target = directory / "protocol.json"
    target.write_text(json.dumps(protocol), encoding="utf-8")
    return Catalog(target, source, directory)


class FakeCLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))
        generator = torch.Generator().manual_seed(91)
        self.register_buffer("texts", torch.randn(323, 768, generator=generator))
        self.register_buffer("image", torch.randn(768, generator=generator))
        self.visual = types.SimpleNamespace(input_resolution=224)
        self.context_length = 77
        self.eval().requires_grad_(False)

    def encode_text(self, tokens):
        return self.texts[tokens[:, 0].long()]

    def encode_image(self, batch):
        # Depends on the actual supplied image, rather than a cached image feature.
        return self.image[None, :].expand(len(batch), -1) + batch.mean((1, 2, 3))[:, None]


def fake_preprocess(image):
    assert image.mode == "RGB"
    return torch.full((3, 224, 224), sum(image.getpixel((0, 0))) / 765)


def fake_tokenize(prompts, context_length, truncate):
    assert context_length == 77 and truncate is False and len(prompts) == 323
    tokens = torch.zeros(323, 77, dtype=torch.int32)
    tokens[:, 0] = torch.arange(323)
    return tokens


def fixture_payload(expected):
    contract = {"schema": "synthetic-inference-fixture", "adapter": dict(cp.ADAPTER_SPEC),
                "preprocessing": copy.deepcopy(cp.NATIVE_PREPROCESSING),
                "candidate_ids": list(expected["ids"]), "candidate_pairs": copy.deepcopy(expected["pairs"]),
                "component_names": copy.deepcopy(expected["names"]),
                "direct_prompt_records": copy.deepcopy(expected["records"]),
                "direct_prompt_policy": dict(cp.PROMPT_POLICY), "direct_prompt_sha256": cp.PROMPT_SHA256,
                "split": {"source_sha256": cp.SOURCE_SHA256, "candidate_sha256": cp.CANDIDATE_SHA256,
                          "split_sha256": cp.SPLIT_SHA256},
                "model": {"name": "ViT-L/14", "revision": cp.CLIP_REVISION,
                          "sha256": cp.CLIP_SHA256, "embedding_dimension": 768, "input_resolution": 224},
                "hyperparameters": {"training_seed": 45, "residual_cap": .2}}
    direct = cp.unit_features(torch.randn(323, 768, generator=torch.Generator().manual_seed(8)))
    adapter = cp.ResidualVisualAdapter()
    with torch.no_grad():
        for parameter in adapter.parameters():
            parameter.zero_()
    checksum = cp.json_hash(contract)
    return {"schema": "wp3b-visual-adapter-m4-v1", "adapter_spec": dict(cp.ADAPTER_SPEC),
            "adapter_state": adapter.state_dict(), "contract": contract, "contract_sha256": checksum,
            "selection": {"epoch": 1}, "text_state": {"direct": direct,
                "candidate_ids": list(expected["ids"]), "contract_sha256": checksum,
                "hashes": {"direct": cp.tensor_hash(direct)}}}


class CLIPPredictorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.temp = tempfile.TemporaryDirectory()
        cls.directory = Path(cls.temp.name)
        cls.catalog = make_catalog(cls.directory)
        cls.expected = cp.catalog_contract(cls.catalog)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def saved_payload(self, value):
        path = self.directory / "synthetic-adapter.pt"
        torch.save(value, path)
        return path, cp.file_hash(path)

    def predictor(self, payload=None):
        model = FakeCLIP()
        native = patch.object(cp, "load_native_clip", return_value=(model, fake_preprocess, fake_tokenize))
        self.addCleanup(native.stop)
        native.start()
        if payload is None:
            return cp.CLIPPredictor(self.catalog, "synthetic-clip-file")
        path, checksum = self.saved_payload(payload)
        pinned = patch.object(cp, "M6_SHA256", checksum)
        self.addCleanup(pinned.stop)
        pinned.start()
        return cp.CLIPPredictor(self.catalog, "synthetic-clip-file", path)

    def test_frozen_native_image_path_scores_match_independent_cosine(self):
        predictor = self.predictor()
        images = [Image.new("RGB", (18, 27), color) for color in ((100, 20, 30), (10, 90, 200))]
        result = predictor.predict(images)
        raw = predictor.model.encode_image(torch.stack([fake_preprocess(x) for x in images]))
        text = cp.unit_features(predictor.model.texts)
        reference = cp.unit_features(raw) @ cp.unit_features(text).T
        torch.testing.assert_close(result["scores"], reference, rtol=0, atol=0)
        self.assertEqual(result["scores"].dtype, torch.float64)
        self.assertFalse(torch.equal(result["scores"][0], result["scores"][1]))
        self.assertEqual(result["candidate_sha256"], cp.CATALOG_SHA256)
        self.assertEqual(result["model_candidate_sha256"], cp.CANDIDATE_SHA256)
        self.assertNotEqual(result["candidate_sha256"], result["model_candidate_sha256"])

    def test_m6_roundtrip_uses_saved_direct_bank_and_presenter(self):
        payload = fixture_payload(self.expected)
        predictor = self.predictor(payload)
        image = Image.new("RGB", (16, 16), "green")
        result = predictor.predict([image])
        original = predictor.model.encode_image(fake_preprocess(image)[None])
        canonical = cp.unit_features(original) @ cp.unit_features(payload["text_state"]["direct"]).T
        torch.testing.assert_close(result["scores"], canonical, rtol=0, atol=0)
        rows = present_prediction(result, self.catalog)
        self.assertEqual([row["composition_id"] for row in rows], result["topk_composition_ids"][0].tolist())
        bridge = result["model_provenance"]["protocol_bridge"]
        self.assertEqual(bridge["model_candidate"]["sha256"], cp.CANDIDATE_SHA256)
        self.assertEqual(bridge["catalog_candidate"]["sha256"], cp.CATALOG_SHA256)

    def test_exact_and_one_ulp_sensitive_ranks_keep_global_ids(self):
        payload = fixture_payload(self.expected)
        direct = torch.zeros(323, 768, dtype=torch.float64)
        direct[:, 0] = -1
        direct[0, :2] = torch.tensor([1., 1.], dtype=torch.float64)
        direct[1, :2] = torch.tensor([1. + 4e-15, 1.], dtype=torch.float64)
        payload["text_state"]["direct"] = direct
        payload["text_state"]["hashes"]["direct"] = cp.tensor_hash(direct)
        predictor = self.predictor(payload)
        feature = torch.zeros(1, 768)
        feature[0, 0] = 1
        with patch.object(predictor.model, "encode_image", return_value=feature):
            result = predictor.predict([Image.new("RGB", (16, 16))])
        self.assertEqual(result["topk_composition_ids"][0, 0], self.expected["ids"][1])
        self.assertGreater(result["scores"][0, 1], result["scores"][0, 0])
        predictor.direct = torch.ones(323, 768, dtype=torch.float64)
        result = predictor.predict([Image.new("RGB", (16, 16))])
        self.assertEqual(result["topk_composition_ids"][0].tolist(), self.expected["ids"][:5])

    def test_wrong_hash_rejected_before_deserialization_or_backbone(self):
        path = self.directory / "invalid.pt"
        path.write_bytes(b"not a model")
        with patch.object(torch, "load", side_effect=AssertionError("must not deserialize")), \
                patch.object(cp, "load_native_clip", side_effect=AssertionError("must not load CLIP")):
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                cp.CLIPPredictor(self.catalog, "missing-clip", path)

    def test_invalid_checkpoint_metadata_and_tensors_are_rejected(self):
        changes = {
            "schema": lambda p: p.update(schema="unknown"),
            "contract digest": lambda p: p.update(contract_sha256="wrong"),
            "candidate order": lambda p: p["contract"]["candidate_ids"].reverse(),
            "component pair": lambda p: p["contract"]["candidate_pairs"][0].__setitem__(0, 54),
            "component label": lambda p: p["contract"]["component_names"]["crop"][0].__setitem__(1, "wrong"),
            "prompt": lambda p: p["contract"]["direct_prompt_records"][0].update(prompt="wrong"),
            "preprocessing": lambda p: p["contract"]["preprocessing"].update(center_crop=320),
            "source": lambda p: p["contract"]["split"].update(source_sha256="wrong"),
            "model": lambda p: p["contract"]["model"].update(sha256="wrong"),
            "seed": lambda p: p["contract"]["hyperparameters"].update(training_seed=46),
            "epoch": lambda p: p["selection"].update(epoch=2),
            "cap": lambda p: p["contract"]["hyperparameters"].update(residual_cap=None),
            "text dtype": lambda p: p["text_state"].update(direct=p["text_state"]["direct"].float()),
            "text hash": lambda p: p["text_state"]["hashes"].update(direct="wrong"),
            "text columns": lambda p: p["text_state"]["candidate_ids"].reverse(),
            "text NaN": lambda p: p["text_state"]["direct"].__setitem__((0, 0), float("nan")),
            "adapter shape": lambda p: p["adapter_state"].update(up_bias=torch.zeros(2)),
            "adapter NaN": lambda p: p["adapter_state"]["up.bias"].__setitem__(0, float("nan")),
        }
        for name, change in changes.items():
            with self.subTest(name=name):
                value = fixture_payload(self.expected)
                change(value)
                if name != "contract digest":
                    value["contract_sha256"] = cp.json_hash(value["contract"])
                    value["text_state"]["contract_sha256"] = value["contract_sha256"]
                path, checksum = self.saved_payload(value)
                with patch.object(cp, "M6_SHA256", checksum), self.assertRaises((ValueError, RuntimeError)):
                    cp.load_m6_checkpoint(path, self.expected)

    def test_catalog_rejects_label_order_folds_and_unreconciled_status(self):
        for field in ("label", "order", "fold", "reconciliation"):
            with self.subTest(field=field):
                catalog = copy.deepcopy(self.catalog)
                if field == "label":
                    next(iter(catalog.candidates.values()))["crop_name"] = "wrong"
                elif field == "order":
                    catalog.candidates = dict(reversed(list(catalog.candidates.items())))
                elif field == "fold":
                    catalog.protocol[FOLDS[0]].reverse()
                else:
                    catalog.protocol["historical_reconciliation"] = "pending_ordered_filename_reference"
                with self.assertRaises(ValueError):
                    cp.catalog_contract(catalog)

    def test_residual_norm_bound_and_identity_match_published_arithmetic(self):
        adapter = cp.ResidualVisualAdapter()
        x = torch.randn(4, 768)
        torch.testing.assert_close(adapter(x), x, rtol=0, atol=0)
        with torch.no_grad():
            adapter.up.bias.fill_(100)
        y = adapter(x)
        ratio = torch.linalg.vector_norm(y - x, dim=1) / torch.linalg.vector_norm(x, dim=1)
        self.assertTrue((ratio <= .200001).all())
        with self.assertRaises(ValueError):
            adapter(torch.zeros(1, 768))

    def test_invalid_image_features_and_unfrozen_model_fail_without_predictions(self):
        predictor = self.predictor()
        for feature in (torch.zeros(1, 768), torch.full((1, 768), float("nan")), torch.ones(1, 512)):
            with patch.object(predictor.model, "encode_image", return_value=feature), self.assertRaises(ValueError):
                predictor.predict([Image.new("RGB", (16, 16))])
        predictor.model.train()
        with self.assertRaises(ValueError):
            predictor.predict([Image.new("RGB", (16, 16))])

    def test_importing_picker_or_gui_does_not_import_clip_or_torch(self):
        code = "import sys; import wp4.model_loader, wp4.model_picker, wp4.gui; assert 'clip' not in sys.modules; assert 'torch' not in sys.modules"
        subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True, capture_output=True)


class NativeLoaderTests(unittest.TestCase):
    def test_unapproved_package_fails_without_loading_weights(self):
        distribution = types.SimpleNamespace(version="wrong", read_text=lambda name: "{}")
        with patch.object(cp.importlib.metadata, "distribution", return_value=distribution), \
                patch.object(torch.jit, "load") as load:
            with self.assertRaisesRegex(RuntimeError, "pinned"):
                cp.load_native_clip("missing.pt", torch.device("cpu"))
        load.assert_not_called()

    def test_verified_jit_only_build_freezes_native_model_without_pickle_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic-official-file.pt"
            path.write_bytes(b"explicit synthetic JIT fixture")
            model = FakeCLIP()
            scripted = Mock()
            scripted.eval.return_value = scripted
            scripted.state_dict.return_value = {"synthetic": torch.ones(1)}
            package = types.ModuleType("clip")
            package.__path__ = []
            package.tokenize = fake_tokenize
            model_module = types.ModuleType("clip.model")
            model_module.build_model = Mock(return_value=model)
            api_module = types.ModuleType("clip.clip")
            api_module._transform = Mock(return_value=fake_preprocess)
            distribution = types.SimpleNamespace(version="1.0", read_text=lambda name: json.dumps({
                "vcs_info": {"commit_id": cp.CLIP_REVISION}}))
            with patch.dict(sys.modules, {"clip": package, "clip.model": model_module, "clip.clip": api_module}), \
                    patch.object(cp.importlib.metadata, "distribution", return_value=distribution), \
                    patch.object(cp, "CLIP_SHA256", cp.file_hash(path)), \
                    patch.object(torch.jit, "load", return_value=scripted) as jit_load, \
                    patch.object(torch, "load", side_effect=AssertionError("pickle fallback forbidden")):
                loaded, preprocess, tokenize = cp.load_native_clip(path, torch.device("cpu"))
            self.assertIs(loaded, model)
            self.assertIs(preprocess, fake_preprocess)
            self.assertIs(tokenize, fake_tokenize)
            self.assertFalse(loaded.training)
            self.assertFalse(any(p.requires_grad for p in loaded.parameters()))
            self.assertEqual(jit_load.call_args.kwargs, {"map_location": "cpu"})


if __name__ == "__main__":
    unittest.main()
