"""Final-333 readiness regressions. Synthetic metadata only: no official CSV, image or model."""
import csv
import hashlib
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import numpy as np

from evaluation.confusion_artifacts import confusion_files
from evaluation.final_manifest import final_manifest, inference_order
from evaluation.model_adapters import MODEL_KINDS
from evaluation.reporting import (Candidate, CandidateSpec, EvaluationManifest, FINAL_NAMESPACE, RunProvenance,
                                  Sample, WP1_NAMESPACE, build_report, candidate_digest, fingerprint,
                                  records_from_scores)
from scripts.import_wp5_final import FROZEN_CHECKPOINTS, FROZEN_WP1_SELECTION, check_frozen_identity, import_final
from test_development_import import synthetic_inputs, write_json

# Invented classes: 5 crops, one disease per class, disease 0 "healthy". Not the frozen labels.
CANDIDATES = tuple(Candidate(i, i % 5, i, f"crop{i % 5}", "healthy" if i == 0 else f"disease{i}", f"pair{i}")
                   for i in range(333))
SEEN_TRUTHS = (0, 1, 1, 2, 7, 258, 300, 322)       # training compositions (incl. dev-unseen ones)
UNSEEN_TRUTHS = (323, 324, 330, 332, 332)          # the 10 extra synthetic classes only


def development():
    dev = CANDIDATES[:323]
    spec = CandidateSpec("synthetic-development", "1", "development", dev, candidate_digest(dev, WP1_NAMESPACE),
                         WP1_NAMESPACE, fingerprint("synthetic training source"), fingerprint("synthetic split"),
                         "synthetic-split", "synthetic labels", (0,), synthetic=True)
    samples = (tuple(Sample(f"fit{i}.jpg", "fitting", i) for i in range(258))
               + (Sample("dev_s0.jpg", "seen", 0), Sample("dev_u0.jpg", "unseen", 300)))
    return EvaluationManifest(spec, samples, tuple(range(258)), tuple(range(258, 323)), 1, 1, "synthetic")


def write_csv(path, rows, *, bom=False):
    with path.open("w", newline="", encoding="utf-8-sig" if bom else "utf-8") as stream:
        csv.writer(stream, lineterminator="\r\n").writerows(rows)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows_for(prefix, truths):
    by_id = {c.composition_id: c for c in CANDIDATES}
    return [[f"{prefix}{n}.jpg", by_id[t].crop_id, by_id[t].disease_id, t, by_id[t].crop_name,
             by_id[t].disease_name, by_id[t].composition_name] for n, t in enumerate(truths)]


class FinalManifestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.seen_rows, self.unseen_rows = rows_for("ts", SEEN_TRUTHS), rows_for("tu", UNSEEN_TRUTHS)
        self.payload = {"schema": FINAL_NAMESPACE, "rows": [asdict(c) for c in CANDIDATES]}
        self.prompts = {"policy": {"version": "synthetic"}, "rows": [{"score_column": i} for i in range(333)]}

    def build(self, *, seen=None, unseen=None, bom=False, sha=None, seen_path=None, **kwargs):
        default_seen, unseen_path = self.root / "DPD_Small_test_seen.csv", self.root / "DPD_Small_test_unseen.csv"
        digests = {"seen": write_csv(default_seen, self.seen_rows if seen is None else seen, bom=bom),
                   "unseen": write_csv(unseen_path, self.unseen_rows if unseen is None else unseen)}
        if seen_path is not None:
            seen_path.write_bytes(default_seen.read_bytes())
        options = {"candidate_payload": self.payload, "prompt_payload": self.prompts,
                   "expected_sha256": sha or digests, "approval_reference": "SYNTHETIC policy fixture",
                   "evaluation_authorization": "SYNTHETIC authorization fixture", "synthetic": True, **kwargs}
        return final_manifest(development(), seen_path or default_seen, unseen_path, **options)

    def test_manifest_has_323_seen_by_333_columns_and_label_free_order_has_no_truths(self):
        manifest = self.build()
        self.assertEqual((len(manifest.spec.ids), len(manifest.seen_truth_ids), manifest.unseen_truth_ids),
                         (333, 323, tuple(range(323, 333))))
        self.assertEqual((manifest.expected_seen, manifest.expected_unseen), (8, 5))
        self.assertEqual(sum(s.fold == "fitting" for s in manifest.samples), 258)
        order = inference_order(manifest)
        self.assertEqual(order["folds"]["seen"], [r[0] for r in self.seen_rows])
        self.assertEqual(order["split_sha256"], manifest.sha256)
        text = json.dumps(order)
        self.assertNotIn("truth_id", text)
        self.assertNotIn("pair1", text)  # no class labels either

    def test_rejections_report_counts_without_official_filenames_or_labels(self):
        bad_label = [row[:] for row in self.seen_rows]
        bad_label[2][5] = "renamed-disease"
        header = [["image", "crop_id", "disease_id", "composition_id", "crop", "disease", "pair"]] + self.seen_rows
        cases = {"inconsistent": dict(seen=bad_label), "malformed": dict(seen=header),
                 "duplicate": dict(seen=self.seen_rows + self.seen_rows[:1]),
                 "membership": dict(seen=rows_for("ts", (0, 325)))}
        for name, kwargs in cases.items():
            with self.subTest(name), self.assertRaises(ValueError) as caught:
                self.build(**kwargs)
            self.assertNotIn("ts", str(caught.exception))
            self.assertNotIn("renamed", str(caught.exception))
        for name in ("fit3.jpg", "dev_s0.jpg", "dev_u0.jpg"):  # fitting and both development validation folds
            with self.subTest(name), self.assertRaisesRegex(ValueError, "overlap the training-source"):
                self.build(seen=[[name, *self.seen_rows[0][1:]]])
        with self.assertRaisesRegex(ValueError, "raw SHA-256 differs"):
            self.build(sha={"seen": "0" * 64, "unseen": "0" * 64})
        with self.assertRaisesRegex(ValueError, "authorization"):
            self.build(evaluation_authorization=" ")
        with self.assertRaisesRegex(ValueError, "Synthetic flag"):
            self.build(synthetic=False)
        with self.assertRaisesRegex(ValueError, "Expected DPD_Small_test_seen.csv"):
            self.build(seen_path=self.root / "renamed.csv")

    def test_utf8_bom_is_stripped_and_recorded(self):
        manifest = self.build(bom=True)
        self.assertEqual(manifest.samples[258].sample_id, "ts0.jpg")
        self.assertTrue(json.loads(manifest.identity_source)["official_sources"][0]["utf8_bom_stripped"])

    def test_confusion_artifacts_keep_counts_and_allow_unseen_only_predictions(self):
        manifest = self.build()
        provenance = RunProvenance("m", "r", "c", fingerprint("c"), "p", "v", "e", "s", "float64", "budget")
        samples = [s for s in manifest.samples if s.fold != "fitting"]
        scores = np.zeros((len(samples), 333))
        for row, sample in enumerate(samples):
            scores[row, sample.truth_id] = 1.0
        scores[1, 1], scores[1, 330] = 0.0, 2.0     # seen truth 1 -> unseen-only class 330
        scores[2, 1], scores[2, 330] = 0.0, 2.0
        scores[4, 7], scores[4, 2] = 0.0, 2.0       # seen truth 7 -> class 2 (same crop)
        report = build_report(manifest, records_from_scores(manifest.spec, provenance, samples, scores,
                                                            manifest.spec.ids))
        files = confusion_files(report, fold="seen")
        self.assertTrue(files["seen_confusion_heatmap.png"].startswith(b"\x89PNG"))
        top = list(csv.DictReader(files["seen_confusion_top_pairs.csv"].decode().splitlines()))
        self.assertEqual([(r["truth_id"], r["predicted_id"], r["count"]) for r in top],
                         [("1", "330", "2"), ("7", "2", "1")])
        self.assertEqual((top[0]["prediction_in_truth_axis"], top[1]["same_crop"]), ("False", "True"))
        crop = list(csv.DictReader(files["seen_confusion_crop_level.csv"].decode().splitlines()))
        self.assertEqual(sum(int(r["count"]) for r in crop), len(SEEN_TRUTHS))


class FinalImportTests(unittest.TestCase):
    def setUp(self):
        helper = FinalManifestTests("test_utf8_bom_is_stripped_and_recorded")
        helper.setUp()
        self.addCleanup(helper.temp.cleanup)
        self.root, self.unseen_rows = helper.root, helper.unseen_rows
        self.manifest = helper.build()
        provenances, packets = synthetic_inputs(self.manifest)
        for kind in MODEL_KINDS:
            for packet in packets[kind]:
                del packet["ground_truth_ids"]  # label-free export (owners may still know labels)
        self.packets = packets
        self.bundle = {"schema": "wp5-final-owner-handoff-v1",
                       "final_policy_reference": "SYNTHETIC Issue #8 fixture",
                       "evaluation_authorization_reference": "SYNTHETIC authorization fixture",
                       "owner_approval_reference": "SYNTHETIC owner fixture",
                       "identity_manifest": write_json(self.root / "final_manifest.json", self.manifest.to_dict()),
                       "development_manifest": write_json(self.root / "dev_manifest.json", development().to_dict()),
                       "final_candidates": write_json(self.root / "candidates.json", helper.payload),
                       "final_prompts": write_json(self.root / "prompts.json", helper.prompts),
                       "official_csv_sha256": {f: hashlib.sha256((self.root / n).read_bytes()).hexdigest()
                                               for f, n in (("seen", "DPD_Small_test_seen.csv"),
                                                            ("unseen", "DPD_Small_test_unseen.csv"))},
                       "manifest_sha256": self.manifest.sha256,
                       "models": {kind: {"provenance": asdict(provenances[kind]),
                                         "packet_files": [write_json(self.root / f"{kind}-{i}.json", packet)
                                                          for i, packet in enumerate(packets[kind])]}
                                  for kind in MODEL_KINDS}}
        self.output = self.root / "final-results"

    def run_import(self, **kwargs):
        entry = write_json(self.root / "handoff.json", self.bundle)
        return import_final(self.root / "handoff.json", entry["file_sha256"], self.output,
                            seen_csv=self.root / "DPD_Small_test_seen.csv",
                            unseen_csv=self.root / "DPD_Small_test_unseen.csv", **{"allow_synthetic": True, **kwargs})

    def test_end_to_end_label_free_packets_reports_and_confusion_outputs(self):
        receipt = self.run_import()
        self.assertEqual(receipt["task_mode"], "final")
        self.assertEqual(receipt["label_free_packets_truth_attached"], {kind: 2 for kind in MODEL_KINDS})
        self.assertEqual(receipt["label_handling"], "label-free prediction packets; not a label-blind evaluation")
        self.assertIn("not applied", receipt["checkpoint_identity_scope"])  # synthetic fixtures are exempt
        for kind in MODEL_KINDS:
            report = json.loads((self.output / kind / "report.json").read_text(encoding="utf-8"))
            matrix = report["seen"]["confusion_matrix"]
            self.assertEqual((len(matrix["truth_ids"]), len(matrix["prediction_ids"]), matrix["count"]), (323, 333, 8))
            self.assertTrue((self.output / "confusion" / kind / "seen_confusion_heatmap.png").is_file())
        self.assertIn("SYNTHETIC", (self.output / "caption.txt").read_text(encoding="utf-8"))
        self.assertTrue((self.output / "receipt.json").is_file())

    def test_gates_reject_before_any_output(self):
        for key in ("evaluation_authorization_reference", "final_policy_reference"):
            with self.subTest(key):
                saved, self.bundle[key] = self.bundle[key], ""
                with self.assertRaisesRegex(ValueError, key):
                    self.run_import()
                self.bundle[key] = saved
        with self.assertRaisesRegex(ValueError, "allow-synthetic"):
            self.run_import(allow_synthetic=False)
        write_csv(self.root / "DPD_Small_test_unseen.csv", self.unseen_rows[:-1])  # tampered after approval
        with self.assertRaisesRegex(ValueError, "raw SHA-256 differs"):
            self.run_import()
        self.assertFalse(self.output.exists())

    def replace_packet(self, kind, index, packet):
        self.bundle["models"][kind]["packet_files"][index] = write_json(self.root / f"{kind}-{index}.json", packet)

    def assert_rejected(self, pattern):
        with self.assertRaisesRegex(ValueError, pattern) as caught:
            self.run_import()
        self.assertFalse(self.output.exists())
        # Synthetic class names (pairN, diseaseN, cropN) and truth lists never appear in errors.
        self.assertNotRegex(str(caught.exception), r"(pair|disease|crop)[0-9]|ground_truth_ids\s*[:=]")

    def test_packets_carrying_labels_are_rejected(self):
        kind = MODEL_KINDS[2]
        packet = dict(self.packets[kind][0])
        packet["ground_truth_ids"] = [s.truth_id for s in self.manifest.samples if s.fold == "seen"]
        self.replace_packet(kind, 0, packet)
        self.assert_rejected("must be label-free")

    def test_incomplete_or_mismatched_packets_are_rejected(self):
        kind = MODEL_KINDS[1]
        original = self.packets[kind][0]
        cases = {
            "missing image": ({**original, "image_names": original["image_names"][:-1],
                               "scores": original["scores"][:-1]}, "Missing 1 expected prediction row"),
            "outside manifest": ({**original, "image_names": ["other.jpg", *original["image_names"][1:]]},
                                 "outside the final manifest"),
            "duplicate image": ({**original,
                                 "image_names": [original["image_names"][0], *original["image_names"][:-1]]},
                                "duplicate"),
            "manifest digest": ({**original, "manifest_sha256": "0" * 64, "split_sha256": "0" * 64}, "manifest_sha256"),
            "candidate digest": ({**original, "candidate_sha256": "0" * 64}, "candidate digest"),
            "candidate order": ({**original, "candidate_ids": original["candidate_ids"][::-1]}, "candidate columns"),
        }
        for name, (packet, pattern) in cases.items():
            with self.subTest(name):
                self.replace_packet(kind, 0, packet)
                self.assert_rejected(pattern)
        self.replace_packet(kind, 0, original)
        self.bundle["models"][kind]["packet_files"] = self.bundle["models"][kind]["packet_files"][:1]  # no unseen
        self.assert_rejected("Missing 5 expected prediction rows")
        del self.bundle["models"][kind]
        self.assert_rejected("all four selected models")

    def test_extra_fields_and_rank_only_packets_are_rejected(self):
        kind = MODEL_KINDS[3]
        original = self.packets[kind][0]
        ranked = {key: value for key, value in original.items() if key != "scores"}
        ranked["ranked_ids"] = [original["candidate_ids"][:5] for _ in original["image_names"]]
        for name, packet in {"label-like field": {**original, "labels": ["x"] * len(original["image_names"])},
                             "note field": {**original, "notes": "anything"}, "rank-only": ranked}.items():
            with self.subTest(name):
                self.replace_packet(kind, 0, packet)
                self.assert_rejected("exactly the full-score export fields")

    def test_real_final_import_requires_the_four_frozen_checkpoints(self):
        for kind in MODEL_KINDS:
            packets = [{"selection": dict(FROZEN_WP1_SELECTION)}]
            frozen = RunProvenance("m", "r", "c", FROZEN_CHECKPOINTS[kind], "p", "v", "e", "s", "float64", "b")
            check_frozen_identity(kind, frozen, packets)  # the selected identity passes
            with self.subTest(kind), self.assertRaisesRegex(ValueError, "frozen selected model"):
                check_frozen_identity(kind, RunProvenance("m", "r", "c", "1" * 64, "p", "v", "e", "s", "float64",
                                                          "b"), packets)
        wp1 = RunProvenance("m", "r", "c", FROZEN_CHECKPOINTS[MODEL_KINDS[0]], "p", "v", "e", "s", "float32", "b")
        changed = {"crop_checkpoint_sha256": "2" * 64, "disease_checkpoint_sha256": "2" * 64,
                   "crop_checkpoint_id": "other.pt", "disease_selected_epoch": 1}
        for key, value in changed.items():
            with self.subTest(key), self.assertRaisesRegex(ValueError, "frozen selected heads"):
                check_frozen_identity(MODEL_KINDS[0], wp1, [{"selection": {**FROZEN_WP1_SELECTION, key: value}}])

    def test_frozen_pins_are_complete_and_trace_to_existing_sources(self):
        """Full 64-hex values that equal the hashes already pinned elsewhere in the repository."""
        root = Path(__file__).resolve().parents[1]
        values = [*FROZEN_CHECKPOINTS.values(), FROZEN_WP1_SELECTION["crop_checkpoint_sha256"],
                  FROZEN_WP1_SELECTION["disease_checkpoint_sha256"]]
        self.assertTrue(all(len(v) == 64 and set(v) <= set("0123456789abcdef") for v in values))
        from evaluation.model_adapters import CLIP_ADAPTER, CLIP_FROZEN, CLIP_L14_SHA256, WP2_CONVNEXT
        from wp4.clip_predictor import CLIP_SHA256, M6_SHA256
        self.assertEqual((FROZEN_CHECKPOINTS[CLIP_FROZEN], FROZEN_CHECKPOINTS[CLIP_ADAPTER]),
                         (CLIP_SHA256, M6_SHA256))
        self.assertEqual(CLIP_SHA256, CLIP_L14_SHA256)
        notebook = (root / "notebooks" / "wp2_selected_development_export.ipynb").read_text(encoding="utf-8")
        self.assertIn(f"CHECKPOINT_SHA256='{FROZEN_CHECKPOINTS[WP2_CONVNEXT]}'", notebook)
        # WP1 pair identity: the fingerprint that scripts/export_wp1_development.py computes from the two heads.
        pair = {task: {"checkpoint_id": FROZEN_WP1_SELECTION[task + "_checkpoint_id"],
                       "checkpoint_sha256": FROZEN_WP1_SELECTION[task + "_checkpoint_sha256"],
                       "selected_epoch": FROZEN_WP1_SELECTION[task + "_selected_epoch"]}
                for task in ("crop", "disease")}
        self.assertEqual(fingerprint(pair), FROZEN_CHECKPOINTS[MODEL_KINDS[0]])

    def test_development_manifest_is_not_accepted_as_final(self):
        self.bundle["identity_manifest"] = self.bundle["development_manifest"]
        self.bundle["manifest_sha256"] = development().sha256
        with self.assertRaisesRegex(ValueError, "final-333 only"):
            self.run_import()
        self.assertFalse(self.output.exists())

class RealModeImportTests(unittest.TestCase):
    """Real (non-synthetic) gates, exercised without any official file.

    The development manifest is the real one, built from the tracked training CSV.
    The two "official" CSVs are synthetic stand-ins: invented filenames, the real
    Issue #8 class rows and the frozen 2,000 / 1,448 denominators. Scores are zeros.
    """

    @classmethod
    def setUpClass(cls):
        from evaluation.reporting import development_manifest
        root = Path(__file__).resolve().parents[1]
        art = root / "artifacts" / "wp5" / "final_reconstruction_2026-11-07"
        cls.candidates = json.loads((art / "verified_reconstruction_candidates.json").read_text(encoding="utf-8"))
        cls.prompts = json.loads((art / "verified_reconstruction_prompts.json").read_text(encoding="utf-8"))
        protocol = json.loads((root / "artifacts" / "data" / "development_seed42_reconciled_v1.json")
                              .read_text(encoding="utf-8"))
        cls.development = development_manifest(protocol, root / "DPD_Small_train.csv",
                                               healthy_disease_names=("healthy",))

    def setUp(self):
        from evaluation.model_adapters import CLIP_ADAPTER, CLIP_FROZEN, CLIP_L14_SHA256, SCORING, SELECTION
        from evaluation.reporting import FINAL_PROMPT_SHA256
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        by_id = {r["composition_id"]: r for r in self.candidates["rows"]}
        seen_ids = list(self.development.spec.ids)
        extra = sorted(set(by_id) - set(seen_ids))

        def row(name, i):
            r = by_id[i]
            return [name, r["crop_id"], r["disease_id"], i, r["crop_name"], r["disease_name"], r["composition_name"]]
        seen_csv, unseen_csv = self.root / "DPD_Small_test_seen.csv", self.root / "DPD_Small_test_unseen.csv"
        self.digests = {"seen": write_csv(seen_csv, [row(f"synthetic_seen_{n}.jpg", seen_ids[n % 323])
                                                     for n in range(2000)]),
                        "unseen": write_csv(unseen_csv, [row(f"synthetic_unseen_{n}.jpg", extra[n % 10])
                                                         for n in range(1448)])}
        self.manifest = final_manifest(self.development, seen_csv, unseen_csv, candidate_payload=self.candidates,
                                       prompt_payload=self.prompts, expected_sha256=self.digests,
                                       approval_reference="TEST ONLY",
                                       evaluation_authorization="TEST ONLY")
        self.assertFalse(self.manifest.spec.synthetic)
        self.order = inference_order(self.manifest)
        self.models = {}
        for kind in MODEL_KINDS:
            clip = kind in (CLIP_FROZEN, CLIP_ADAPTER)
            provenance = RunProvenance(kind, f"run-{kind}", f"ckpt-{kind}", FROZEN_CHECKPOINTS[kind], "native",
                                       "test-code", "test-env", SCORING[kind],
                                       "float32" if kind == MODEL_KINDS[0] else "float64", "frozen",
                                       prompt_sha256=FINAL_PROMPT_SHA256 if clip else "")
            selection = dict(SELECTION.get(kind, {}))
            if kind == MODEL_KINDS[0]:
                selection.update(FROZEN_WP1_SELECTION)
                selection.update({f"{task}_preprocessing": "native" for task in ("crop", "disease")})
            if kind == CLIP_ADAPTER:
                selection["base_clip_sha256"] = CLIP_L14_SHA256
            self.models[kind] = (provenance, selection)

    def packet(self, kind, fold, **changes):
        provenance, selection = self.models[kind]
        names = self.order["folds"][fold]
        packet = {"schema": "wp5-model-output-v1", "model_kind": kind,
                  **{key: getattr(provenance, key) for key in (
                      "model_id", "run_id", "checkpoint_id", "checkpoint_sha256", "preprocessing", "code_version",
                      "environment", "score_precision", "score_semantics")},
                  **{key: self.order[key] for key in ("manifest_sha256", "source_sha256", "candidate_ids",
                                                      "candidate_hash_namespace", "candidate_sha256",
                                                      "split_hash_namespace", "split_sha256")},
                  "selection": selection, "fold": fold, "image_names": names,
                  "scores": [[0.0] * 333 for _ in names]}
        if provenance.prompt_sha256:
            packet["prompt_sha256"] = provenance.prompt_sha256
        return {**packet, **changes}

    def run_import(self, changes=None):
        changes = changes or {}
        bundle = {"schema": "wp5-final-owner-handoff-v1", "final_policy_reference": "TEST ONLY",
                  "evaluation_authorization_reference": "TEST ONLY", "owner_approval_reference": "TEST ONLY",
                  "identity_manifest": write_json(self.root / "final_manifest.json", self.manifest.to_dict()),
                  "development_manifest": write_json(self.root / "dev.json", self.development.to_dict()),
                  "final_candidates": write_json(self.root / "candidates.json", self.candidates),
                  "final_prompts": write_json(self.root / "prompts.json", self.prompts),
                  "official_csv_sha256": self.digests, "manifest_sha256": self.manifest.sha256, "models": {}}
        for kind in MODEL_KINDS:
            provenance = asdict(self.models[kind][0])
            provenance.update(changes.get(kind, {}).get("provenance", {}))
            files = [write_json(self.root / f"{kind}-{fold}.json",
                                self.packet(kind, fold, **changes.get(kind, {}).get("packet", {})))
                     for fold in ("seen", "unseen")]
            bundle["models"][kind] = {"provenance": provenance, "packet_files": files}
        entry = write_json(self.root / "handoff.json", bundle)
        self.output = self.root / "out"
        return import_final(self.root / "handoff.json", entry["file_sha256"], self.output,
                            seen_csv=self.root / "DPD_Small_test_seen.csv",
                            unseen_csv=self.root / "DPD_Small_test_unseen.csv")

    def test_real_mode_accepts_the_frozen_four_and_builds_323_by_333_seen_confusion(self):
        receipt = self.run_import()
        self.assertEqual((receipt["synthetic"], receipt["fold_counts"]),
                         (False, {"fitting": 4868, "seen": 2000, "unseen": 1448}))
        self.assertIn("checkpoint bytes are not verified", receipt["checkpoint_identity_scope"])
        for kind in MODEL_KINDS:
            report = json.loads((self.output / kind / "report.json").read_text(encoding="utf-8"))
            matrix = report["seen"]["confusion_matrix"]
            self.assertEqual((len(matrix["truth_ids"]), len(matrix["prediction_ids"]), matrix["count"]),
                             (323, 333, 2000))

    def test_real_mode_rejects_a_substituted_checkpoint_for_each_model(self):
        other = "3" * 64
        for kind in MODEL_KINDS:
            with self.subTest(kind), self.assertRaisesRegex(ValueError, "frozen selected model"):
                self.run_import({kind: {"provenance": {"checkpoint_sha256": other},
                                        "packet": {"checkpoint_sha256": other}}})
            self.assertFalse(self.output.exists())
        heads = {**self.models[MODEL_KINDS[0]][1], "disease_checkpoint_sha256": "4" * 64}
        with self.assertRaisesRegex(ValueError, "frozen selected heads"):
            self.run_import({MODEL_KINDS[0]: {"packet": {"selection": heads}}})
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
