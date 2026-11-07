"""Contract tests for frozen finalist robustness adapters. No owner weights, downloads or training."""
import contextlib
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import torchvision
from PIL import Image
from torch import nn

from evaluation.model_adapters import CLIP_ADAPTER, CLIP_FROZEN, WP1_INDEPENDENT, WP2_CONVNEXT
from evaluation.robustness import NATIVE_SUITE, Perturbation, decode_rgb, pixel_digest, run_native_suite
from evaluation.robustness_adapters import FINALIST_CONTRACTS, PredictorAdapter, WP2NotebookPredictor


def tiny_model(classes):
    torch.manual_seed(classes)
    return nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, classes)).eval()


def write_images(root, count=3):
    names = []
    for i in range(count):
        rng = np.random.default_rng(i)
        array = rng.integers(0, 256, size=(30 + 10 * i, 40, 3), dtype=np.uint8)
        name = f"{i}.png"
        Image.fromarray(array).save(root / name)
        names.append(name)
    return names


class WP1RealPredictorTests(unittest.TestCase):
    """The real IndependentPredictor, real torchvision preprocessing and real float32 scorer."""

    def test_real_wp1_predictor_through_v2_runner(self):
        import pandas as pd
        from data_utils import COLUMN_NAMES, load_metadata
        from independent_baseline import IndependentPredictor, preprocessing_record, save_checkpoint
        from shared_protocol import FOLDS, candidate_table, digest, write_new_protocol
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            records = [[f"{i}.png", c, d, c * 2 + d, f"c{c}", f"d{d}", f"p{c * 2 + d}"]
                       for i, (c, d) in enumerate([(0, 0), (1, 1), (0, 0), (1, 1), (0, 1), (1, 0)])]
            source = root / "train.csv"
            pd.DataFrame(records, columns=COLUMN_NAMES).to_csv(source, header=False, index=False)
            protocol = {"schema": "wp1-development-v1",
                        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                        "historical_reconciliation": "ordered_filenames_match_supplied_reference",
                        "candidates": candidate_table(load_metadata(source)), "counts": [2, 2, 2],
                        FOLDS[0]: ["0.png", "1.png"], FOLDS[1]: ["2.png", "3.png"], FOLDS[2]: ["4.png", "5.png"]}
            protocol["candidate_sha256"] = digest(protocol["candidates"])
            protocol["split_sha256"] = digest({key: protocol[key] for key in FOLDS})
            write_new_protocol(protocol, root / "protocol.json")
            with patch("independent_baseline.build_classifier",
                       side_effect=lambda b, n, pretrained=False: tiny_model(n)):
                for task, classes in (("crop", 55), ("disease", 175)):
                    metadata = {"task": task, "num_classes": classes, "backbone": "resnet50",
                                "torchvision": str(torchvision.__version__),
                                "preprocessing": preprocessing_record("resnet50"),
                                **{k: protocol[k] for k in ("candidate_sha256", "split_sha256", "source_sha256")}}
                    save_checkpoint(root / f"{task}.pt", tiny_model(classes), metadata)
                predictor = IndependentPredictor(root / "crop.pt", root / "disease.pt", root / "protocol.json", source)
            names = write_images(root)
            ids = [row["composition_id"] for row in protocol["candidates"]]
            adapter = PredictorAdapter(predictor, model_kind=WP1_INDEPENDENT, candidate_ids=ids)
            suite = (NATIVE_SUITE[0], Perturbation("darken", 3, 0.3))
            result = run_native_suite({"wp1": adapter}, names, load=lambda k: decode_rgb(root / k),
                                      suite=suite, batch_sizes={"wp1": 2})
            scores = result["wp1"]["clean"]["scores"]
            self.assertEqual(scores.dtype, np.float32)
            self.assertEqual(scores.shape, (3, 4))
            direct = []
            for name in names:  # the frozen export's own decode path, one image at a time
                with Image.open(root / name) as image:
                    direct.append(predictor.predict([image.convert("RGB")], topk=4)["scores"].numpy())
            np.testing.assert_allclose(scores, np.concatenate(direct), rtol=0, atol=1e-6)
            self.assertFalse(np.array_equal(scores, result["wp1"]["darken_s3"]["scores"]))
            self.assertEqual(result["pixel_sha256"]["clean"],
                             [pixel_digest(decode_rgb(root / n)) for n in names])


class FakeCLIPPredictor:
    """Reproduces the GUI CLIPPredictor output contract (gui: wp4/clip_predictor.py)."""

    def __init__(self, variant="frozen_clip", dtype=torch.float64, ids=(3, 7, 9), semantics=None, bad_topk=False):
        self.variant, self.dtype, self.ids = variant, dtype, torch.tensor(ids, dtype=torch.int64)
        self.semantics = semantics or "direct_float64_clip_cosine_similarity"
        self.bad_topk = bad_topk

    def predict(self, images, topk=5):
        means = torch.tensor([np.asarray(im).mean() for im in images], dtype=torch.float64)
        scores = torch.stack([means, -means, means / 2], dim=1).to(self.dtype)
        order = scores.argsort(dim=1, descending=True, stable=True)[:, :min(topk, 3)]
        if self.bad_topk:
            order = order.flip(1)
        return {"candidate_ids": self.ids, "scores": scores, "topk_composition_ids": self.ids[order],
                "score_semantics": self.semantics, "model_provenance": {"variant": self.variant}}


class ClipContractTests(unittest.TestCase):
    images = [Image.fromarray(np.full((8, 8, 3), v, np.uint8)) for v in (10, 200)]

    def test_frozen_and_m6_contracts_accept_matching_predictors(self):
        out = PredictorAdapter(FakeCLIPPredictor(), model_kind=CLIP_FROZEN, candidate_ids=[3, 7, 9]).predict(self.images)
        self.assertEqual(out["scores"].dtype, np.float64)
        PredictorAdapter(FakeCLIPPredictor("m6"), model_kind=CLIP_ADAPTER, candidate_ids=[3, 7, 9]).predict(self.images)

    def test_contract_violations_are_rejected(self):
        cases = [
            (FakeCLIPPredictor("m6"), CLIP_FROZEN, [3, 7, 9]),                      # wrong variant
            (FakeCLIPPredictor("frozen_clip"), CLIP_ADAPTER, [3, 7, 9]),            # wrong variant
            (FakeCLIPPredictor(dtype=torch.float32), CLIP_FROZEN, [3, 7, 9]),       # wrong precision
            (FakeCLIPPredictor(semantics="sum_log_softmax_v1"), CLIP_FROZEN, [3, 7, 9]),
            (FakeCLIPPredictor(ids=(3, 7, 10)), CLIP_FROZEN, [3, 7, 9]),            # candidate columns
            (FakeCLIPPredictor(bad_topk=True), CLIP_FROZEN, [3, 7, 9]),             # top-k vs scores
        ]
        for predictor, kind, ids in cases:
            with self.assertRaises(ValueError, msg=(kind, predictor.__dict__)):
                PredictorAdapter(predictor, model_kind=kind, candidate_ids=ids).predict(self.images)
        with self.assertRaises(ValueError):
            PredictorAdapter(FakeCLIPPredictor(), model_kind=CLIP_FROZEN, candidate_ids=[9, 7, 3])
        with self.assertRaises(ValueError):
            PredictorAdapter(FakeCLIPPredictor(), model_kind="unknown", candidate_ids=[3, 7, 9])

    def test_contracts_cover_all_four_finalists_and_reuse_packet_semantics(self):
        self.assertEqual(set(FINALIST_CONTRACTS), {WP1_INDEPENDENT, WP2_CONVNEXT, CLIP_FROZEN, CLIP_ADAPTER})
        self.assertEqual(FINALIST_CONTRACTS[WP2_CONVNEXT]["packet_score_semantics"], "sum_log_softmax_v1")
        self.assertEqual(FINALIST_CONTRACTS[WP2_CONVNEXT]["selection"]["input_resolution"], 224)
        with self.assertRaises(TypeError):
            FINALIST_CONTRACTS[WP1_INDEPENDENT]["score_dtype"] = "float64"

    def test_nested_contract_values_are_read_only_copies(self):
        from evaluation.model_adapters import SELECTION
        with self.assertRaises(TypeError):
            FINALIST_CONTRACTS[WP2_CONVNEXT]["selection"]["epoch"] = 2
        with self.assertRaises(TypeError):
            FINALIST_CONTRACTS[WP2_CONVNEXT]["packet_environment"]["precision"] = "fp32"
        self.assertEqual(SELECTION[WP2_CONVNEXT]["epoch"], 1)
        self.assertIsNot(FINALIST_CONTRACTS[WP2_CONVNEXT]["selection"], SELECTION[WP2_CONVNEXT])
        # Values recorded from the identified development packets.
        wp2 = FINALIST_CONTRACTS[WP2_CONVNEXT]
        self.assertEqual((wp2["historical_batch_size"], wp2["packet_environment"]["precision"],
                          wp2["packet_environment"]["channels_last"]), (16, "bf16", True))
        self.assertEqual(FINALIST_CONTRACTS[WP1_INDEPENDENT]["packet_environment"]["torch"], "2.7.1+cu118")


def notebook_composition_scores(crop_logits, disease_logits, candidate_spec):
    """Test fixture: the published WP2 notebook scorer (float64 log-softmax sum)."""
    pairs = torch.tensor(candidate_spec["pairs"], dtype=torch.int64)
    crop = torch.log_softmax(crop_logits.to(torch.float64), dim=1)
    disease = torch.log_softmax(disease_logits.to(torch.float64), dim=1)
    return crop[:, pairs[:, 0]] + disease[:, pairs[:, 1]]


def notebook_amp_context(device, precision):
    return contextlib.nullcontext() if precision == "fp32" else torch.autocast(
        device_type=device.type, dtype=torch.float16 if precision == "fp16" else torch.bfloat16)


class BufferReusingWP2(nn.Module):
    """Returns the SAME output buffers every call (the CPU aliasing hazard in d_collect_logits)."""

    def __init__(self):
        super().__init__()
        self.register_buffer("crop", torch.zeros(2, 55))
        self.register_buffer("disease", torch.zeros(2, 175))
        self.formats = []

    def forward(self, x):
        self.formats.append(x.is_contiguous(memory_format=torch.channels_last))
        value = x.mean(dim=(1, 2, 3))
        self.crop.zero_()
        self.disease.zero_()
        self.crop[: len(x), 0] = value
        self.disease[: len(x), 0] = -value
        return {"crop_logits": self.crop[: len(x)], "disease_logits": self.disease[: len(x)]}


class WP2NotebookPredictorTests(unittest.TestCase):
    spec = {"ids": [10, 11, 12], "pairs": [(0, 0), (1, 0), (0, 1)]}

    def transform(self, image):
        return torchvision.transforms.functional.to_tensor(image.resize((4, 4)))

    def test_owner_scorer_receives_independent_float32_snapshots_and_channels_last(self):
        model = BufferReusingWP2().eval()
        seen = []

        def scorer(crop, disease, spec):
            seen.append((crop.dtype, crop.device.type))
            return notebook_composition_scores(crop, disease, spec)

        predictor = WP2NotebookPredictor(model, self.transform, scorer, self.spec,
                                         amp_context=notebook_amp_context, precision="fp32", channels_last=True)
        batches = [[Image.fromarray(np.full((6, 6, 3), v, np.uint8)) for v in pair] for pair in ((0, 50), (100, 150))]
        outputs = [predictor.predict(b, topk=3) for b in batches]
        self.assertEqual(model.formats, [True, True])
        first = outputs[0]["crop_logits"][:, 0].tolist()
        self.assertAlmostEqual(first[0], 0.0)
        self.assertAlmostEqual(first[1], 50 / 255, places=5)  # not overwritten by the second batch
        self.assertTrue(all(d == torch.float32 and dev == "cpu" for d, dev in seen))
        self.assertEqual(outputs[0]["scores"].dtype, torch.float64)
        adapted = PredictorAdapter(predictor, model_kind=WP2_CONVNEXT, candidate_ids=self.spec["ids"])
        self.assertEqual(adapted.predict(batches[1])["scores"].dtype, np.float64)

    def test_undeclared_precision_or_training_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            WP2NotebookPredictor(BufferReusingWP2().eval(), self.transform, notebook_composition_scores, self.spec,
                                 amp_context=notebook_amp_context, precision="auto", channels_last=True)
        with self.assertRaises(TypeError):
            WP2NotebookPredictor(BufferReusingWP2().eval(), self.transform, notebook_composition_scores, self.spec,
                                 amp_context=notebook_amp_context, precision="fp32", channels_last=None)
        trained = WP2NotebookPredictor(BufferReusingWP2().train(), self.transform, notebook_composition_scores,
                                       self.spec, amp_context=notebook_amp_context, precision="fp32",
                                       channels_last=False)
        with self.assertRaises(ValueError):
            trained.predict([Image.new("RGB", (6, 6))], topk=3)
        eval_predictor = WP2NotebookPredictor(BufferReusingWP2().eval(), self.transform, notebook_composition_scores,
                                              self.spec, amp_context=notebook_amp_context, precision="fp32",
                                              channels_last=False)
        with self.assertRaises(ValueError):
            eval_predictor.predict([Image.new("RGB", (6, 6))], topk=4)  # more than 3 candidates


if __name__ == "__main__":
    unittest.main()
