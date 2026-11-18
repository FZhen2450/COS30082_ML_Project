"""Final-333 views over the REAL predictor classes, real Issue #8 tables and the real 323 catalog.

Weights and backbones are synthetic (fixtures reused from the WP4 and robustness tests).
No CLIP package, real image, official CSV or label is used.
"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torchvision
from PIL import Image
from torch import nn

from evaluation.final333_predictors import (Final333CLIP, Final333Independent, Restricted323,
                                            export_blind_packet, wp2_final_spec)
from evaluation.final_manifest import final_manifest, inference_order
from evaluation.model_adapters import (CLIP_ADAPTER, CLIP_FROZEN, SCORING, WP1_INDEPENDENT, WP2_CONVNEXT,
                                       ModelOutputAdapter)
from evaluation.reporting import (Candidate, CandidateSpec, EvaluationManifest, FINAL_PROMPT_SHA256, FINAL_UNSEEN_IDS,
                                  RunProvenance, Sample, WP1_NAMESPACE, candidate_digest, fingerprint)
from evaluation.robustness_adapters import PredictorAdapter, WP2NotebookPredictor
from scripts.import_wp5_final import _attach_truths
from test_final_readiness import write_csv
from test_robustness_adapters import notebook_amp_context, notebook_composition_scores, tiny_model
from test_wp4_clip_predictor import FakeCLIP, fake_preprocess, fake_tokenize, fixture_payload, make_catalog
from wp4 import clip_predictor as cp

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts/wp5/final_reconstruction_2026-11-07"
CANDIDATES = json.loads((ART / "verified_reconstruction_candidates.json").read_text(encoding="utf-8"))
PROMPTS = json.loads((ART / "verified_reconstruction_prompts.json").read_text(encoding="utf-8"))
PROTOCOL = ROOT / "artifacts/data/development_seed42_reconciled_v1.json"
SOURCE = ROOT / "DPD_Small_train.csv"
ALL_IDS = list(range(333))


def images():
    return [Image.new("RGB", (20 + i, 24), (30 * i, 90, 200 - 20 * i)) for i in range(3)]


def extra_tokenize(prompts, context_length, truncate):
    """Synthetic tokens selecting fake text rows 100.. for the 10 extra prompts."""
    assert context_length == 77 and truncate is False
    tokens = torch.zeros(len(prompts), 77, dtype=torch.int32)
    tokens[:, 0] = torch.arange(len(prompts)) + 100
    return tokens


def columns(ids):
    return [ALL_IDS.index(i) for i in ids]


def real_wp1_predictor(root):
    from independent_baseline import IndependentPredictor, preprocessing_record, save_checkpoint
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    with patch("independent_baseline.build_classifier", side_effect=lambda b, n, pretrained=False: tiny_model(n)):
        for task, classes in (("crop", 55), ("disease", 175)):
            metadata = {"task": task, "num_classes": classes, "backbone": "resnet50",
                        "torchvision": str(torchvision.__version__), "preprocessing": preprocessing_record("resnet50"),
                        **{k: protocol[k] for k in ("candidate_sha256", "split_sha256", "source_sha256")}}
            save_checkpoint(root / f"{task}.pt", tiny_model(classes), metadata)
        return IndependentPredictor(root / "crop.pt", root / "disease.pt", PROTOCOL, SOURCE)


class Final333CLIPTests(unittest.TestCase):
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

    def predictor(self, payload=None):
        native = patch.object(cp, "load_native_clip", return_value=(FakeCLIP(), fake_preprocess, fake_tokenize))
        native.start()
        self.addCleanup(native.stop)
        if payload is None:
            return cp.CLIPPredictor(self.catalog, "synthetic-clip-file")
        path = self.directory / f"adapter-{self.id()}.pt"
        torch.save(payload, path)
        pinned = patch.object(cp, "M6_SHA256", cp.file_hash(path))
        pinned.start()
        self.addCleanup(pinned.stop)
        return cp.CLIPPredictor(self.catalog, "synthetic-clip-file", path)

    def views(self):
        frozen, m6 = self.predictor(), self.predictor(fixture_payload(self.expected))
        f = Final333CLIP(frozen, CANDIDATES, PROMPTS, tokenize=extra_tokenize,
                         base_bank_sha256=cp.tensor_hash(frozen.direct))
        g = Final333CLIP(m6, CANDIDATES, PROMPTS, extra_bank=f.extra_bank, base_bank_sha256=cp.tensor_hash(m6.direct))
        return frozen, m6, f, g

    def test_real_tables_keep_historical_rows_and_share_the_10_extra_rows(self):
        frozen, m6, f, g = self.views()
        self.assertEqual(f.extra_ids, list(FINAL_UNSEEN_IDS))
        base_cols, extra_cols = columns(frozen.expected["ids"]), columns(FINAL_UNSEEN_IDS)
        self.assertTrue(torch.equal(f.bank[base_cols], frozen.direct))
        self.assertTrue(torch.equal(g.bank[base_cols], m6.direct))
        self.assertTrue(torch.equal(f.bank[extra_cols], g.bank[extra_cols]))
        self.assertTrue(torch.equal(f.bank[extra_cols], cp.unit_features(frozen.model.texts[100:110])))
        self.assertEqual((f.provenance["variant"], g.provenance["variant"]), ("frozen_clip", "m6"))
        self.assertEqual((frozen.direct.shape[0], len(frozen.expected["ids"])), (323, 323))  # base untouched

    def test_333_scores_preserve_historical_323_columns_and_pass_the_finalist_contract(self):
        frozen, m6, f, g = self.views()
        for base, view, kind in ((frozen, f, CLIP_FROZEN), (m6, g, CLIP_ADAPTER)):
            with self.subTest(kind):
                out, old = view.predict(images()), base.predict(images())
                self.assertEqual((tuple(out["scores"].shape), out["scores"].dtype), ((3, 333), torch.float64))
                restricted = out["scores"][:, columns(base.expected["ids"])]
                torch.testing.assert_close(restricted, old["scores"], rtol=0, atol=0)
                self.assertEqual(out["prompt_sha256"], FINAL_PROMPT_SHA256)
                self.assertNotIn("candidate_sha256", out)
                checked = PredictorAdapter(view, model_kind=kind, candidate_ids=ALL_IDS).predict(images())
                self.assertEqual(checked["topk_composition_ids"].shape, (3, 5))
                # Development rehearsal view: identical to the historical predictor's own output.
                rehearsal = Restricted323(view, base.expected["ids"]).predict(images())
                self.assertTrue(torch.equal(rehearsal["scores"], old["scores"]))
                self.assertTrue(torch.equal(rehearsal["topk_composition_ids"], old["topk_composition_ids"]))
                PredictorAdapter(Restricted323(view, base.expected["ids"]), model_kind=kind,
                                 candidate_ids=base.expected["ids"]).predict(images())

    def test_gates(self):
        frozen = self.predictor()
        with self.assertRaisesRegex(ValueError, "clean-parity bank"):
            Final333CLIP(frozen, CANDIDATES, PROMPTS, tokenize=extra_tokenize)  # fake bank != historical hash
        changed = {**PROMPTS, "rows": [dict(r) for r in PROMPTS["rows"]]}
        changed["rows"][0]["prompt"] += " (edited)"
        bank = cp.tensor_hash(frozen.direct)
        with self.assertRaisesRegex(ValueError, "Issue #8 final hash mismatch"):
            Final333CLIP(frozen, CANDIDATES, changed, tokenize=extra_tokenize, base_bank_sha256=bank)
        with self.assertRaisesRegex(ValueError, "Historical prompts differ"):
            Final333CLIP(frozen, CANDIDATES, changed, tokenize=extra_tokenize, base_bank_sha256=bank, synthetic=True)
        with self.assertRaisesRegex(ValueError, "tokenizer or a shared"):
            Final333CLIP(frozen, CANDIDATES, PROMPTS, base_bank_sha256=bank)
        with self.assertRaisesRegex(ValueError, "Extra text bank"):
            Final333CLIP(frozen, CANDIDATES, PROMPTS, extra_bank=torch.zeros(9, 768, dtype=torch.float64),
                         base_bank_sha256=bank)


class Final333CNNTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_wp1_real_predictor_333_scores_keep_its_323_columns_bit_exact(self):
        predictor = real_wp1_predictor(self.root)
        view = Final333Independent(predictor, CANDIDATES, PROMPTS)
        out, old = view.predict(images()), predictor.predict(images())
        self.assertEqual((tuple(out["scores"].shape), out["scores"].dtype), ((3, 333), torch.float32))
        dev_ids = [r["composition_id"] for r in predictor.protocol["candidates"]]
        self.assertTrue(torch.equal(out["scores"][:, columns(dev_ids)], old["scores"]))
        self.assertEqual(len(predictor.protocol["candidates"]), 323)  # base untouched
        PredictorAdapter(view, model_kind=WP1_INDEPENDENT, candidate_ids=ALL_IDS).predict(images())
        renamed = {**CANDIDATES, "rows": [dict(r) for r in CANDIDATES["rows"]]}
        renamed["rows"][0]["crop_name"] = "renamed"
        with self.assertRaisesRegex(ValueError, "Development candidates differ"):
            Final333Independent(predictor, renamed, synthetic=True)

    def test_wp2_existing_notebook_wrapper_accepts_the_333_spec(self):
        class Heads(nn.Module):
            def __init__(self):
                super().__init__()
                torch.manual_seed(5)
                self.crop, self.disease = nn.Linear(3, 55), nn.Linear(3, 175)

            def forward(self, x):
                pooled = x.mean(dim=(2, 3))
                return {"crop_logits": self.crop(pooled), "disease_logits": self.disease(pooled)}

        def transform(image):
            return torchvision.transforms.functional.to_tensor(image.resize((8, 8)))

        model = Heads().eval()

        def make(spec):
            return WP2NotebookPredictor(model, transform, notebook_composition_scores, spec,
                                        amp_context=notebook_amp_context, precision="fp32", channels_last=True)

        final = make(wp2_final_spec(CANDIDATES, PROMPTS))
        protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
        dev_spec = {"ids": [r["composition_id"] for r in protocol["candidates"]],
                    "pairs": [[r["crop_id"], r["disease_id"]] for r in protocol["candidates"]]}
        out, old = final.predict(images()), make(dev_spec).predict(images())
        self.assertEqual(tuple(out["scores"].shape), (3, 333))
        self.assertTrue(torch.equal(out["scores"][:, columns(dev_spec["ids"])], old["scores"]))
        PredictorAdapter(final, model_kind=WP2_CONVNEXT, candidate_ids=ALL_IDS).predict(images())


class BlindExportTests(unittest.TestCase):
    """inference_order -> real-class predictor view -> truth-free packet -> WP5 attach + adapter."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        rows = tuple(Candidate(**r) for r in json.loads(PROTOCOL.read_text(encoding="utf-8"))["candidates"])
        ids = [c.composition_id for c in rows]
        spec = CandidateSpec("synthetic-development", "1", "development", rows,
                             candidate_digest(rows, WP1_NAMESPACE), WP1_NAMESPACE, fingerprint("s"),
                             fingerprint("p"), "synthetic-split", "synthetic", (), synthetic=True)
        samples = tuple(Sample(f"fit{i}.jpg", "fitting", i) for i in ids[:258]) + (
            Sample("dev_s.jpg", "seen", ids[0]), Sample("dev_u.jpg", "unseen", ids[300]))
        development = EvaluationManifest(spec, samples, tuple(ids[:258]), tuple(ids[258:]), 1, 1, "synthetic")
        by_id = {r["composition_id"]: r for r in CANDIDATES["rows"]}

        def line(name, t):
            r = by_id[t]
            return [name, r["crop_id"], r["disease_id"], t, r["crop_name"], r["disease_name"], r["composition_name"]]

        seen, unseen = self.root / "DPD_Small_test_seen.csv", self.root / "DPD_Small_test_unseen.csv"
        digests = {"seen": write_csv(seen, [line(f"syn_s{n}.jpg", t) for n, t in enumerate((0, 1, 5))]),
                   "unseen": write_csv(unseen, [line(f"syn_u{n}.jpg", t) for n, t in enumerate((39, 303))])}
        self.manifest = final_manifest(development, seen, unseen, candidate_payload=CANDIDATES, prompt_payload=PROMPTS,
                                       expected_sha256=digests, approval_reference="SYNTHETIC",
                                       evaluation_authorization="SYNTHETIC", synthetic=True)
        self.order = inference_order(self.manifest)
        self.provenance = RunProvenance(WP1_INDEPENDENT, "synthetic-run", "synthetic-pair", fingerprint("ckpt"),
                                        "torchvision resnet50 DEFAULT", "synthetic-code", "synthetic-env",
                                        SCORING[WP1_INDEPENDENT], "float32", "final-333 frozen finalists")

    def test_wp1_blind_packet_imports_after_truth_attachment(self):
        view = Final333Independent(real_wp1_predictor(self.root), CANDIDATES, PROMPTS)
        selection = {t + s: v for t in ("crop", "disease") for s, v in (
            ("_checkpoint_id", "synthetic-" + t), ("_checkpoint_sha256", fingerprint(t)), ("_preprocessing", "rgb"))}

        def load(name):
            return Image.new("RGB", (24, 24), (len(name) * 9 % 256, 80, 120))

        records = []
        for fold in ("seen", "unseen"):
            packet = export_blind_packet(view, self.order, fold, load, model_kind=WP1_INDEPENDENT,
                                         provenance=self.provenance, selection=selection, batch_size=2)
            self.assertNotIn("ground_truth_ids", packet)
            truths = {s.sample_id: s.truth_id for s in self.manifest.samples if s.fold == fold}
            attached, blind = _attach_truths(json.loads(json.dumps(packet)), truths)
            self.assertTrue(blind)
            records += ModelOutputAdapter(self.manifest, self.provenance, WP1_INDEPENDENT).adapt(attached)
        self.assertEqual(len(records), 5)
        self.assertEqual(len(records[0].scores), 333)

    def test_export_rejects_non_canonical_semantics_and_missing_order(self):
        view = Final333Independent(real_wp1_predictor(self.root), CANDIDATES, PROMPTS)
        bad = RunProvenance(WP1_INDEPENDENT, "r", "c", fingerprint("c"), "p", "v", "e", "other", "float32", "b")
        with self.assertRaisesRegex(ValueError, "canonical name"):
            export_blind_packet(view, self.order, "seen", None, model_kind=WP1_INDEPENDENT, provenance=bad,
                                selection={}, batch_size=2)
        with self.assertRaisesRegex(ValueError, "inference order"):
            export_blind_packet(view, {"schema": "x"}, "seen", None, model_kind=WP1_INDEPENDENT,
                                provenance=self.provenance, selection={}, batch_size=2)


if __name__ == "__main__":
    unittest.main()
