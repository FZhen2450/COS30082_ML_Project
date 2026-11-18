"""J2 final-model GUI integration: synthetic checks that need no private model files.

Covers the inference-asset contract (fail closed), the frozen scoring arithmetic against an independent numpy
re-derivation, fixed seen-candidate calibration, stable Top-5 ranking through the existing presentation layer,
model registration and discovery pins, and refusal of a non-final catalog or wrong checkpoint without fallback.
Real-weight parity is checked separately by scripts/j2_verify_gui_parity.py (needs the private assets).
"""
from pathlib import Path
import copy
import hashlib
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from wp4 import j2_predictor as jp
from wp4.asset_discovery import ASSETS, FIELDS
from wp4.model_loader import MODEL_SPECS, load_model, validate_model_inputs
from wp4.model_picker import SHORT_LABELS
from wp4.service import Catalog, present_prediction

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / 'artifacts/data/final_full8000_333_v1.json'
SOURCE = ROOT / 'DPD_Small_train.csv'


def synthetic_asset(seed=0):
    g = torch.Generator().manual_seed(seed)
    nc, nd = 55, 61
    head = lambda n: {"weight": torch.randn(n, 768, generator=g) * 0.05, "bias": torch.randn(n, generator=g) * 0.1}  # noqa: E731
    return {"schema": jp.ASSET_SCHEMA, "checkpoint_sha256": jp.J2_CHECKPOINT_SHA256, "features_sha256": jp.J2_FEATURES_SHA256,
            "config_sha256": jp.J2_CONFIG_SHA256, "candidate_sha256": jp.FINAL_CANDIDATE_SHA256, "gamma": 8.75, "fuse": 1.0,
            "normalise": True, "candidate_ids": torch.arange(333, dtype=torch.int64),
            "cand_crop": torch.randint(0, nc, (333,), generator=g), "cand_dis": torch.randint(0, nd, (333,), generator=g),
            "seen_mask": torch.tensor([i not in jp.FINAL_UNSEEN_IDS for i in range(333)]),
            "crop_ids": torch.arange(nc), "disease_ids": torch.arange(nd),
            "mu": torch.randn(768, generator=g) * 0.01, "sd": torch.rand(768, generator=g) * 0.02 + 0.01,
            "heads": {"composition": head(333), "crop": head(nc), "disease": head(nd)}}


def numpy_reference(asset, feats):
    """Independent re-derivation of the D1 fused score in float64 numpy."""
    x = feats.double().numpy()
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)
    x = (x - asset["mu"].double().numpy()) / asset["sd"].double().numpy()
    def lsm(name):
        z = x @ asset["heads"][name]["weight"].double().numpy().T + asset["heads"][name]["bias"].double().numpy()
        z = z - z.max(1, keepdims=True)
        return z - np.log(np.exp(z).sum(1, keepdims=True))
    fac = lsm("crop")[:, asset["cand_crop"].numpy()] + lsm("disease")[:, asset["cand_dis"].numpy()]
    return (fac + lsm("composition")) / 2.0 - 8.75 * asset["seen_mask"].numpy()[None]


class AssetContract(unittest.TestCase):
    def test_valid_synthetic_asset_passes(self):
        jp._check_asset(synthetic_asset())

    def test_every_identity_or_structure_change_is_refused(self):
        def mut(fn):
            a = copy.deepcopy(synthetic_asset()); fn(a); return a
        cases = {
            "schema": lambda a: a.update(schema="other"),
            "checkpoint": lambda a: a.update(checkpoint_sha256="0" * 64),
            "features": lambda a: a.update(features_sha256="0" * 64),
            "config": lambda a: a.update(config_sha256="0" * 64),
            "candidates sha": lambda a: a.update(candidate_sha256="0" * 64),
            "gamma": lambda a: a.update(gamma=8.5),
            "fuse": lambda a: a.update(fuse=0.5),
            "candidate order": lambda a: a.update(candidate_ids=torch.arange(332, -1, -1)),
            "seen mask": lambda a: a["seen_mask"].__setitem__(39, True),
            "missing head": lambda a: a["heads"].pop("disease"),
            "head shape": lambda a: a["heads"]["composition"].update(weight=torch.zeros(332, 768)),
            "non-finite": lambda a: a["heads"]["crop"]["weight"].__setitem__((0, 0), float("nan")),
            "zero scale": lambda a: a["sd"].__setitem__(0, 0.0),
            "index range": lambda a: a["cand_dis"].__setitem__(0, 10_000),
        }
        for name, fn in cases.items():
            with self.subTest(name), self.assertRaises(ValueError):
                jp._check_asset(mut(fn))

    def test_load_asset_requires_the_pinned_hash(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "asset.pt"; torch.save(synthetic_asset(), p)
            digest = hashlib.sha256(p.read_bytes()).hexdigest()
            self.assertEqual(jp.load_asset(p, digest)["gamma"], 8.75)
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                jp.load_asset(p)  # the registered production hash is a different file
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                jp.load_asset(p, "0" * 64)


class Scoring(unittest.TestCase):
    def setUp(self):
        self.asset = synthetic_asset()
        self.feats = torch.randn(7, 768, generator=torch.Generator().manual_seed(1))

    def test_scores_match_an_independent_numpy_derivation(self):
        s = jp.score_features(self.asset, self.feats)
        self.assertEqual(s.shape, (7, 333)); self.assertEqual(s.dtype, torch.float64)
        self.assertLess(float(np.abs(s.numpy() - numpy_reference(self.asset, self.feats)).max()), 1e-5)
        self.assertTrue(torch.equal(s, jp.score_features(self.asset, self.feats)))  # deterministic

    def test_gamma_is_subtracted_exactly_on_the_323_seen_candidates(self):
        a0 = dict(self.asset, gamma=0.0)
        diff = (jp.score_features(a0, self.feats) - jp.score_features(self.asset, self.feats)).numpy()
        seen = self.asset["seen_mask"].numpy()
        self.assertTrue(np.allclose(diff[:, seen], 8.75)); self.assertTrue(np.allclose(diff[:, ~seen], 0.0))
        self.assertEqual(int(seen.sum()), 323)

    def test_bad_features_refused(self):
        for bad in (torch.randn(2, 512), torch.full((1, 768), float("nan"))):
            with self.subTest(shape=tuple(bad.shape)), self.assertRaises(ValueError):
                jp.score_features(self.asset, bad)


class InputOrientation(unittest.TestCase):
    def test_j2_reads_stored_pixels_without_exif_rotation(self):
        from PIL import Image
        from wp4.service import open_rgb
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "rotated.jpg"
            img = Image.new("RGB", (40, 20), (200, 30, 30))
            exif = img.getexif(); exif[0x0112] = 6  # "rotate 90 CW" orientation tag
            img.save(p, exif=exif)
            self.assertEqual(jp.J2Predictor.load_input(p).size, (40, 20))  # stored pixels, as J2 was trained/evaluated
            self.assertEqual(open_rgb(p).size, (20, 40))                    # GUI display path stays EXIF-corrected
        self.assertEqual(jp.J2Predictor.input_orientation, "stored_pixels")


class GUIContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.catalog = Catalog(PROTOCOL, SOURCE, cls.tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_final_protocol_is_the_contract_j2_was_evaluated_on(self):
        self.assertTrue(self.catalog.is_final)
        self.assertEqual(self.catalog.protocol["candidate_sha256"], jp.FINAL_CANDIDATE_SHA256)
        self.assertEqual(list(self.catalog.candidates), list(range(333)))

    def test_presentation_ranks_j2_scores_with_stable_ascending_id_ties(self):
        scores = torch.zeros(1, 333, dtype=torch.float64)
        scores[0, [7, 3, 300]] = 5.0; scores[0, 39] = 4.0; scores[0, 0] = 4.0
        top = present_prediction({"candidate_ids": torch.arange(333), "scores": scores,
                                  "candidate_sha256": jp.FINAL_CANDIDATE_SHA256}, self.catalog)
        self.assertEqual([t["composition_id"] for t in top], [3, 7, 300, 0, 39])
        self.assertEqual(top[4]["crop_name"], "pepper_bell")  # unseen id 39 resolves to readable names
        with self.assertRaises(ValueError):  # a wrong candidate fingerprint is refused
            present_prediction({"candidate_ids": torch.arange(333), "scores": scores, "candidate_sha256": "x"}, self.catalog)

    def test_registration_and_discovery_pins(self):
        self.assertEqual([k for k, _ in MODEL_SPECS["j2"]["fields"]], ["clip_weights", "j2_checkpoint", "j2_asset"])
        self.assertEqual(FIELDS["j2"], ("clip_weights", "j2_checkpoint", "j2_asset"))
        self.assertEqual(ASSETS["j2_checkpoint"][0], jp.J2_CHECKPOINT_SHA256)
        self.assertEqual(ASSETS["j2_asset"][0], jp.J2_ASSET_SHA256)
        self.assertIn("j2", SHORT_LABELS)
        for key in ("wp1", "frozen_clip", "m6"):  # existing models untouched
            self.assertIn(key, MODEL_SPECS)

    def test_model_picker_has_a_path_field_for_every_registered_model(self):
        import tkinter as tk
        try:
            root = tk.Tk()
        except tk.TclError:
            self.skipTest("no display for Tk")
        try:
            root.withdraw()
            from wp4.model_picker import ModelPanel
            panel = ModelPanel(root, on_load=lambda *a: None)
            for kind, spec in MODEL_SPECS.items():
                for key, _ in spec["fields"]:
                    self.assertIn(key, panel.paths, f"{kind}: {key}")
            panel.choice.current(panel.kinds.index("j2")); panel.render_fields()  # used to raise KeyError
            panel.destroy()
        finally:
            root.destroy()

    def test_missing_assets_and_wrong_files_fail_without_fallback(self):
        with self.assertRaisesRegex(ValueError, "Select an existing"):
            validate_model_inputs("j2", {"clip_weights": "/nope", "j2_checkpoint": "/nope", "j2_asset": "/nope"}, "cpu")
        with tempfile.TemporaryDirectory() as d:
            fake = Path(d) / "fake.pt"; fake.write_bytes(b"not a model")
            paths = {"clip_weights": str(fake), "j2_checkpoint": str(fake), "j2_asset": str(fake)}
            with self.assertRaisesRegex(ValueError, "J2 inference asset"):
                load_model(self.catalog, "j2", paths, "cpu")
            with self.assertRaisesRegex(ValueError, "frozen J2 checkpoint"):
                jp.load_j2_backbone(fake, fake, torch.device("cpu"))
            non_final = SimpleNamespace(is_final=False, protocol={})
            with self.assertRaisesRegex(ValueError, "final 333-candidate"):
                jp.J2Predictor(non_final, fake, fake, fake)


if __name__ == "__main__":
    unittest.main()
