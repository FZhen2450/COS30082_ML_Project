"""Synthetic WP5 regression fixtures; no training or official image inference."""
import builtins
import csv
import json
import tempfile
import unittest
import zipfile
import io
import hashlib
from dataclasses import FrozenInstanceError, asdict, replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from evaluation.metrics import evaluate_compositions, harmonic_mean, topk_accuracy
from evaluation.reporting import (
    Candidate, CandidateSpec, EvaluationManifest, PredictionRecord, RunProvenance,
    Sample, FINAL_NAMESPACE, WP1_NAMESPACE, build_report, candidate_digest,
    canonical_json, compare_models, confusion_matrix, development_manifest,
    export_report, fingerprint, records_from_scores, verify_final_payloads,
)


def fixture():
    # Non-contiguous global IDs; disease 9 is healthy by explicit label mapping.
    candidates = tuple(Candidate(i, c, d, f"crop{c}", "healthy" if d == 9 else f"disease{d}", f"pair{i}")
                       for i, c, d in ((10, 0, 9), (75, 0, 1), (99, 1, 9),
                                       (200, 1, 1), (250, 2, 1), (332, 2, 9)))
    spec = CandidateSpec("synthetic-development", "1", "development", candidates,
                         candidate_digest(candidates, WP1_NAMESPACE), WP1_NAMESPACE,
                         fingerprint("synthetic-source"), fingerprint("synthetic-split"),
                         "synthetic-folds-v1", "synthetic labels", (9,), synthetic=True)
    samples = (Sample("fit10", "fitting", 10), Sample("fit75", "fitting", 75),
               Sample("fit99", "fitting", 99), Sample("fit200", "fitting", 200),
               Sample("s1", "seen", 10), Sample("s2", "seen", 75),
               Sample("u1", "unseen", 250), Sample("u2", "unseen", 332))
    manifest = EvaluationManifest(spec, samples, (10, 75, 99, 200), (250, 332), 2, 2,
                                  "synthetic immutable truth fixture")
    provenance = RunProvenance("model-a", "run-a", "epoch-1", fingerprint("checkpoint-a"),
                               "synthetic RGB transform", "synthetic-code-v1", "synthetic CPU",
                               "canonical float64 log_softmax crop + disease", "float64", "one-epoch-control")
    scores = np.array([[6, 5, 4, 3, 2, 1],  # s1 correct top1
                       [6, 5, 4, 3, 2, 1],  # s2 correct top5
                       [1, 2, 3, 4, 6, 5],  # u1 correct top1
                       [6, 5, 4, 3, 2, 1]], dtype=np.float64)  # u2 rank6, wrong top5
    records = records_from_scores(spec, provenance, samples[4:], scores, spec.ids)
    return manifest, provenance, scores, records


class ReportingTests(unittest.TestCase):
    def setUp(self):
        self.manifest, self.provenance, self.scores, self.records = fixture()
        self.spec = self.manifest.spec

    def test_golden_counts_harmonics_and_shared_metric_compatibility(self):
        report = build_report(self.manifest, self.records)
        for fold, values in (("seen", (1, 2, .5, 1.)), ("unseen", (1, 1, .5, .5))):
            result = report[fold]
            self.assertEqual((result["correct_top1"], result["correct_top5"], result["top1"], result["top5"]), values)
            self.assertEqual((result["count"], result["processed"], result["expected"]), (2, 2, 2))
            index = slice(0, 2) if fold == "seen" else slice(2, 4)
            old = evaluate_compositions(self.scores[index], [r.sample.truth_id for r in self.records[index]], self.spec.ids)
            self.assertEqual((old["top1"], old["top5"]), values[2:])
        self.assertEqual(report["harmonic_mean_top1"], .5)
        self.assertEqual(report["harmonic_mean_top5"], 2 / 3)
        self.assertEqual(report["combined_micro"]["correct_top1"], 2)
        self.assertEqual(report["combined_micro"]["correct_top5"], 3)

    def test_exact_ties_rank1_rank5_and_one_ulp(self):
        scores = np.ones((4, 6), dtype=np.float64)
        rows = records_from_scores(self.spec, self.provenance, self.manifest.samples[4:], scores, self.spec.ids)
        self.assertEqual(rows[0].ranking(), self.spec.ids)
        self.assertEqual(topk_accuracy(scores, [10, 75, 250, 332], self.spec.ids, 5), .75)
        self.assertEqual(build_report(self.manifest, rows)["unseen"]["correct_top5"], 1)
        scores[1, 1] = np.nextafter(1., np.inf)
        scores[3, 5] = np.nextafter(1., np.inf)
        near = records_from_scores(self.spec, self.provenance, self.manifest.samples[4:], scores, self.spec.ids)
        self.assertEqual(near[1].ranking()[0], 75)
        self.assertEqual(near[3].ranking()[0], 332)
        scores[2, 4] = np.nextafter(1., -np.inf)
        cutoff = records_from_scores(self.spec, self.provenance, self.manifest.samples[4:], scores, self.spec.ids)
        self.assertEqual(cutoff[2].ranking()[-1], 250)

    def test_extreme_finite_scores_preserved_without_probability_product(self):
        scores = np.tile([-1e300, -1e200, -3000., -2999., -4000., -2000.], (4, 1))
        records = records_from_scores(self.spec, self.provenance, self.manifest.samples[4:], scores, self.spec.ids)
        self.assertEqual(records[0].scores, tuple(scores[0]))
        self.assertEqual(records[0].ranking()[0], 332)
        self.assertEqual(build_report(self.manifest, records)["unseen"]["top1"], .5)

    def test_tiny_candidates_use_min5(self):
        candidates = self.spec.candidates[:2]
        spec = replace(self.spec, candidates=candidates, candidate_sha256=candidate_digest(candidates, WP1_NAMESPACE))
        manifest = EvaluationManifest(spec, (Sample("f", "fitting", 10), Sample("s", "seen", 10),
                                            Sample("u", "unseen", 75)), (10,), (75,), 1, 1, "tiny synthetic")
        records = records_from_scores(spec, self.provenance, manifest.samples[1:], np.ones((2, 2)), spec.ids)
        report = build_report(manifest, records)
        self.assertEqual(report["seen"]["top5"], 1.)
        self.assertEqual(report["unseen"]["top5"], 1.)
        self.assertEqual(report["harmonic_mean_top1"], 0.)

    def test_perfect_and_zero_predictions(self):
        perfect = [replace(r, scores=(), ranked_ids=(r.sample.truth_id,) + tuple(i for i in self.spec.ids if i != r.sample.truth_id))
                   for r in self.records]
        report = build_report(self.manifest, perfect)
        self.assertEqual((report["harmonic_mean_top1"], report["harmonic_mean_top5"]), (1., 1.))
        zero = [replace(r, scores=(), ranked_ids=tuple(i for i in self.spec.ids if i != r.sample.truth_id) + (r.sample.truth_id,))
                for r in self.records]
        report = build_report(self.manifest, zero)
        self.assertEqual((report["harmonic_mean_top1"], report["harmonic_mean_top5"]), (0., 0.))

    def test_fraction_units_and_historical_tiny_h_boundary_unchanged(self):
        with self.assertRaises(ValueError):
            harmonic_mean(75, 50)
        self.assertEqual(harmonic_mean(1e-300, 1e-300), 0.)
        self.assertEqual(harmonic_mean(0, 0), 0.)

    def test_group_support_and_health_denominators(self):
        report = build_report(self.manifest, self.records)
        healthy = report["unseen"]["health"]["healthy"]
        diseased = report["unseen"]["health"]["diseased"]
        self.assertEqual((healthy["count"], healthy["correct_top1"], healthy["predicted_healthy_count"]), (1, 0, 1))
        self.assertEqual((diseased["count"], diseased["correct_top1"], diseased["predicted_healthy_frequency"]), (1, 1, 0.))
        zero = next(r for r in report["seen"]["per_composition"] if r["composition_id"] == 332)
        self.assertEqual(zero["count"], 0)
        self.assertIsNone(zero["top1"])
        self.assertTrue(zero["low_support"])
        self.assertEqual(zero["fitting_support"], 0)
        self.assertEqual(sum(r["count"] for r in report["seen"]["per_crop"]), 2)
        self.assertEqual(sum(r["correct_top5"] for r in report["seen"]["per_disease"]), 2)

    def test_transfer_support_comes_only_from_fitting(self):
        report = build_report(self.manifest, self.records)
        row = next(r for r in report["unseen"]["composition_component_consistency"] if r["composition_id"] == 332)
        self.assertEqual((row["fitting_crop_support"], row["fitting_disease_support"], row["fitting_composition_support"]), (0, 2, 0))
        self.assertEqual(row["disease_right_crop_wrong"], 1)
        self.assertEqual(row["predicted_fitting_composition"], 1)
        self.assertEqual(row["predicted_healthy"], 1)

    def test_all_component_error_categories_and_false_healthy(self):
        # u1 is crop2/disease1. Pick each kind of erroneous composition.
        for prediction, category in ((332, "crop_right_disease_wrong"), (75, "disease_right_crop_wrong"),
                                     (10, "both_wrong"), (250, "both_right")):
            ranking = (prediction,) + tuple(i for i in self.spec.ids if i != prediction)
            records = list(self.records)
            records[2] = replace(records[2], scores=(), ranked_ids=ranking)
            row = next(r for r in build_report(self.manifest, records)["unseen"]["composition_component_consistency"]
                       if r["composition_id"] == 250)
            self.assertEqual(row[category], 1)
            self.assertEqual(sum(row[k] for k in ("both_right", "both_wrong", "crop_right_disease_wrong", "disease_right_crop_wrong")), 1)

    def test_independent_component_accuracy_is_separate_and_partial(self):
        records = list(self.records)
        records[1] = replace(records[1], independent_crop_id=0, independent_disease_id=1)
        metrics = build_report(self.manifest, records)["seen"]["independent_component_accuracy"]
        self.assertEqual((metrics["crop"]["available"], metrics["crop"]["expected"], metrics["crop"]["accuracy"]), (1, 2, 1.))
        self.assertEqual(metrics["disease"]["correct"], 1)
        with self.assertRaises(ValueError):
            replace(records[1], independent_crop_id=999)

    def test_record_and_spec_serialization_are_immutable(self):
        data = json.loads(json.dumps(self.records[0].to_dict()))
        restored = PredictionRecord.from_dict(data)
        self.assertEqual(restored, self.records[0])
        with self.assertRaises(FrozenInstanceError):
            restored.available_k = 1
        with self.assertRaises(TypeError):
            restored.scores[0] = 0
        original = restored.scores
        data["scores"][0] = -100
        self.scores[0, 0] = -100
        self.assertEqual(restored.scores, original)
        self.assertEqual(CandidateSpec.from_dict(json.loads(json.dumps(self.spec.to_dict()))), self.spec)
        self.assertEqual(EvaluationManifest.from_dict(json.loads(json.dumps(self.manifest.to_dict()))), self.manifest)
        with self.assertRaises(ValueError):
            replace(self.spec, final_prompt_json={"mutable": True})
        with self.assertRaises(ValueError):
            replace(self.records[0], provenance=asdict(self.provenance))

    def test_strict_integer_ids(self):
        for value in (True, "10", 10., 10.9, -1, None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                replace(self.spec.candidates[0], composition_id=value)
            with self.subTest(value=value), self.assertRaises(ValueError):
                Sample("x", "seen", value)
        with self.assertRaises(ValueError):
            replace(self.records[0], candidate_ids=(10., *self.spec.ids[1:]))

    def test_bad_candidates_names_pairs_order_and_hashes(self):
        bad_tables = (self.spec.candidates[::-1], self.spec.candidates + self.spec.candidates[:1],
                      (self.spec.candidates[0], replace(self.spec.candidates[1], crop_id=0, disease_id=9)),
                      (self.spec.candidates[0], replace(self.spec.candidates[1], crop_name="wrong")),
                      (self.spec.candidates[0], replace(self.spec.candidates[1], disease_name="healthy")))
        for table in bad_tables:
            with self.subTest(table=table), self.assertRaises(ValueError):
                replace(self.spec, candidates=table, candidate_sha256=candidate_digest(table, WP1_NAMESPACE))
        with self.assertRaises(ValueError):
            replace(self.spec, candidate_sha256="0" * 64)
        with self.assertRaises(ValueError):
            replace(self.spec, healthy_disease_ids=(999,))
        with self.assertRaises(ValueError):
            replace(self.spec, tie_rule="round_to_3_decimals")
        with self.assertRaises(ValueError):
            replace(self.spec.candidates[0], composition_name="")

    def test_missing_empty_duplicate_and_unexpected_prediction_rows(self):
        for records in ((), self.records[:2], self.records[:-1], self.records + self.records[:1],
                        (replace(self.records[0], sample=Sample("unknown", "seen", 10)), *self.records[1:])):
            with self.subTest(records=len(records)), self.assertRaises(ValueError):
                build_report(self.manifest, records)

    def test_wrong_truth_fold_or_provenance_cannot_join(self):
        for changed in (replace(self.records[0], sample=Sample("s1", "seen", 75)),
                        replace(self.records[0], sample=Sample("s1", "unseen", 10)),
                        replace(self.records[0], provenance=replace(self.provenance, code_version="different")),
                        replace(self.records[0], spec=replace(self.spec, split_sha256=fingerprint("other")))):
            with self.subTest(changed=changed.sample), self.assertRaises(ValueError):
                build_report(self.manifest, (changed, *self.records[1:]))

    def test_empty_manifest_folds_duplicates_wrong_counts_and_membership(self):
        for updates in ({"expected_seen": 0}, {"expected_unseen": 3},
                        {"samples": self.manifest.samples[:-1]},
                        {"samples": self.manifest.samples + self.manifest.samples[:1]},
                        {"seen_truth_ids": (10, 75)}, {"unseen_truth_ids": (250, 999)}):
            with self.subTest(updates=updates.keys()), self.assertRaises(ValueError):
                replace(self.manifest, **updates)

    def test_bad_scores_k_rankings_columns_and_precision(self):
        for scores in (np.full((4, 6), np.nan), np.full((4, 6), np.inf),
                       np.ones((4, 5)), np.ones((0, 6)), np.ones((4, 6), dtype=np.float32)):
            with self.subTest(shape=scores.shape), self.assertRaises(ValueError):
                records_from_scores(self.spec, self.provenance, self.manifest.samples[4:], scores, self.spec.ids)
        with self.assertRaises(ValueError):
            records_from_scores(self.spec, self.provenance, self.manifest.samples[4:], self.scores, self.spec.ids[::-1])
        for updates in ({"available_k": 0}, {"available_k": True}, {"available_k": 7},
                        {"scores": ()}, {"ranked_ids": self.spec.ids},
                        {"scores": (), "ranked_ids": (10,) * 6},
                        {"scores": (), "ranked_ids": (10, 75, 99, 200, 250, 999)}):
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                replace(self.records[0], **updates)

    def _score_record_boundaries(self, scores, precision):
        provenance = replace(self.provenance, score_precision=precision)
        data = self.records[0].to_dict()
        data["provenance"]["score_precision"] = precision
        data["scores"] = list(scores)
        return (
            ("direct", lambda: replace(self.records[0], provenance=provenance, scores=scores)),
            ("json", lambda: PredictionRecord.from_dict(json.loads(json.dumps(data)))),
        )

    def test_float32_declared_rejects_float64_ulp_at_both_boundaries(self):
        scores = (1., float(np.nextafter(1., np.inf)), 0., 0., 0., 0.)
        for boundary, construct in self._score_record_boundaries(scores, "float32"):
            with self.subTest(boundary=boundary), self.assertRaisesRegex(ValueError, "exactly representable"):
                construct()

    def test_exact_float32_values_and_ties_survive_both_boundaries(self):
        for lead, top_id in ((1., 10), (float(np.nextafter(np.float32(1.), np.float32(np.inf))), 75)):
            scores = (1., lead, .5, -.5, 0., -0.)
            for boundary, construct in self._score_record_boundaries(scores, "float32"):
                with self.subTest(boundary=boundary, lead=lead):
                    record = construct()
                    self.assertEqual(record.scores, scores)
                    self.assertEqual(record.ranking()[0], top_id)
                    self.assertTrue(np.signbit(record.scores[-1]))
            native = replace(self.records[0], provenance=replace(self.provenance, score_precision="float32"),
                             scores=tuple(np.float32(s) for s in scores))
            self.assertEqual(native.scores, scores)

    def test_float64_one_ulp_ranking_survives_both_boundaries(self):
        scores = (1., float(np.nextafter(1., np.inf)), 0., 0., 0., 0.)
        for boundary, construct in self._score_record_boundaries(scores, "float64"):
            with self.subTest(boundary=boundary):
                record = construct()
                self.assertEqual(record.scores, scores)
                self.assertEqual(record.ranking()[:2], (75, 10))

    def test_inexact_integer_scores_rejected_at_both_boundaries(self):
        for precision, integer in (("float32", 2**24 + 1), ("float64", 2**53 + 1)):
            for value in (integer, -integer):
                scores = (value, 0, 0, 0, 0, 0)
                for boundary, construct in self._score_record_boundaries(scores, precision):
                    with self.subTest(precision=precision, value=value, boundary=boundary):
                        with self.assertRaisesRegex(ValueError, "exactly representable"):
                            construct()
        for precision, integer in (("float32", 2**24), ("float64", 2**53), ("float64", 2**100)):
            for boundary, construct in self._score_record_boundaries((integer, 0, 0, 0, 0, 0), precision):
                with self.subTest(precision=precision, integer=integer, boundary=boundary):
                    self.assertEqual(construct().scores[0].as_integer_ratio(), (integer, 1))

    def test_finite_score_boundaries_are_exact_in_both_precisions(self):
        for precision in ("float32", "float64"):
            dtype = np.dtype(precision).type
            info = np.finfo(dtype)
            smallest = float(np.nextafter(dtype(0.), dtype(1.)))
            scores = (float(info.max), -float(info.max), float(info.tiny), smallest, -smallest, -0.)
            for boundary, construct in self._score_record_boundaries(scores, precision):
                with self.subTest(precision=precision, boundary=boundary):
                    record = construct()
                    self.assertEqual(record.scores, scores)
                    self.assertEqual(record.ranking()[0], 10)
                    self.assertTrue(np.signbit(record.scores[-1]))

    def test_nonfinite_overflow_and_underflow_scores_rejected_at_both_boundaries(self):
        invalid = {
            "float32": (float("nan"), float("inf"), -float("inf"),
                        float(np.finfo(np.float32).max) * 2, 2**128, 2**-150, -2**-150),
            "float64": (float("nan"), float("inf"), -float("inf"), 2**1024, -(2**1024)),
        }
        for precision, values in invalid.items():
            for value in values:
                for boundary, construct in self._score_record_boundaries((value, 0, 0, 0, 0, 0), precision):
                    with self.subTest(precision=precision, value=value, boundary=boundary):
                        with self.assertRaises(ValueError):
                            construct()

    def test_matrix_dtype_rules_remain_strict_for_exact_values(self):
        for precision, matrix_dtype in (("float32", np.float64), ("float64", np.float32)):
            with self.subTest(precision=precision), self.assertRaisesRegex(ValueError, "dtype differs"):
                records_from_scores(self.spec, replace(self.provenance, score_precision=precision),
                                    self.manifest.samples[4:], np.ones((4, 6), dtype=matrix_dtype), self.spec.ids)

    def test_prompt_sha256_rejects_non_string_provenance_at_both_boundaries(self):
        for value in ([], {}, (), set(), bytearray(), b"", None, False, 0, 1, 0., ["0" * 64]):
            with self.subTest(value=repr(value)), self.assertRaises(ValueError):
                replace(self.provenance, prompt_sha256=value)
        for value in ([], {}, None, False, 0, 1, 0., ["0" * 64]):
            data = self.records[0].to_dict()
            data["provenance"]["prompt_sha256"] = value
            with self.subTest(json_value=value), self.assertRaises(ValueError):
                PredictionRecord.from_dict(json.loads(json.dumps(data)))

    def test_prompt_sha256_accepts_only_valid_immutable_strings(self):
        for value in ("", "0" * 64, fingerprint("synthetic prompt table")):
            provenance = replace(self.provenance, prompt_sha256=value)
            record = replace(self.records[0], provenance=provenance)
            restored = PredictionRecord.from_dict(json.loads(json.dumps(record.to_dict())))
            self.assertEqual(restored.provenance, provenance)
            with self.assertRaises(FrozenInstanceError):
                restored.provenance.prompt_sha256 = "changed"
        for value in (" ", "0" * 63, "0" * 65, "G" * 64, "A" * 64):
            with self.subTest(value=value), self.assertRaises(ValueError):
                replace(self.provenance, prompt_sha256=value)

    def test_float32_values_and_torch_canonical_matrices(self):
        provenance = replace(self.provenance, score_precision="float32", score_semantics="WP1 native log scores")
        scores = torch.tensor(self.scores, dtype=torch.float32)
        records = records_from_scores(self.spec, provenance, self.manifest.samples[4:], scores, self.spec.ids)
        self.assertEqual(build_report(self.manifest, records)["seen"]["top1"], .5)
        scores[0, 0] = -99
        self.assertEqual(records[0].scores[0], 6.)

    def test_log_scores_and_direct_cosine_consumed_unchanged(self):
        generator = torch.Generator().manual_seed(42)
        crop = torch.randn(4, 3, generator=generator, dtype=torch.float64)
        disease = torch.randn(4, 10, generator=generator, dtype=torch.float64)
        ci, di = zip(*((c.crop_id, c.disease_id) for c in self.spec.candidates))
        log_scores = crop.log_softmax(1)[:, ci] + disease.log_softmax(1)[:, di]
        images = torch.randn(4, 8, generator=generator, dtype=torch.float64)
        texts = torch.randn(6, 8, generator=generator, dtype=torch.float64)
        cosine = torch.nn.functional.normalize(images, dim=1) @ torch.nn.functional.normalize(texts, dim=1).T
        for matrix, semantics in ((log_scores, "WP2 float64 log-softmax sum"), (cosine, "WP3 direct float64 cosine")):
            provenance = replace(self.provenance, score_semantics=semantics)
            records = records_from_scores(self.spec, provenance, self.manifest.samples[4:], matrix, self.spec.ids)
            expected_order = matrix.argsort(dim=1, descending=True, stable=True)
            for i, record in enumerate(records):
                self.assertEqual(record.scores, tuple(matrix[i].tolist()))
                self.assertEqual(record.ranking(), tuple(self.spec.ids[j] for j in expected_order[i]))

    def test_rectangular_confusion_and_zero_support_rows(self):
        matrix = confusion_matrix(self.records[:2], self.spec.ids, self.spec.ids, candidates=self.spec.candidates)
        self.assertEqual(matrix["count"], 2)
        self.assertEqual(matrix["diagonal_hits"], 1)
        self.assertEqual(matrix["truth_support"][-1], 0)
        self.assertEqual(matrix["row_normalized"][-1], [None] * 6)
        matrix = build_report(self.manifest, self.records)["seen"]["confusion_matrix"]
        self.assertEqual((len(matrix["counts"]), len(matrix["counts"][0])), (4, 6))
        with self.assertRaises(ValueError):
            confusion_matrix(self.records[:2], (10,), self.spec.ids, candidates=self.spec.candidates)
        with self.assertRaises(ValueError):
            confusion_matrix(self.records[:2], self.spec.ids, (75,), candidates=self.spec.candidates)

    def test_paired_alignment_gains_losses_and_harmonic_deltas(self):
        provenance_b = replace(self.provenance, model_id="model-b", run_id="run-b",
                               checkpoint_sha256=fingerprint("b"), score_semantics="WP3 direct cosine")
        # A: s1,u1; B: s2,u2 => one A-only and one B-only per fold.
        scores_b = np.array([[1, 6, 5, 4, 3, 2], [1, 6, 5, 4, 3, 2],
                             [6, 5, 4, 3, 2, 1], [1, 2, 3, 4, 5, 6]], dtype=np.float64)
        b = records_from_scores(self.spec, provenance_b, self.manifest.samples[4:], scores_b, self.spec.ids)
        report = compare_models(self.manifest, self.records[::-1], b[::-1])
        for fold in ("seen", "unseen"):
            row = next(r for r in report["paired"] if r["fold"] == fold and r["composition_id"] is None and r["k"] == 1)
            self.assertEqual((row["both_correct"], row["a_only"], row["b_only"], row["neither"]), (0, 1, 1, 0))
            self.assertEqual(row["count"], 2)
        self.assertEqual(report["harmonic_mean_top1_b_minus_a"], 0.)
        pair = next(r for r in report["paired"] if r["composition_id"] == 75 and r["fold"] == "seen" and r["k"] == 1)
        self.assertEqual(pair["b_minus_a_hits"], 1)
        self.assertEqual(pair["b_minus_a_accuracy"], 1.)
        health = next(r for r in report["health_comparison"] if r["fold"] == "unseen" and r["truth_group"] == "diseased")
        self.assertEqual(health["count"], 1)
        self.assertEqual((health["a_correct_top1"], health["b_correct_top1"]), (1, 0))
        self.assertEqual((health["a_false_healthy_error_count"], health["b_false_healthy_error_count"]), (0, 1))

    def test_paired_both_correct_neither_and_incompatible_inputs(self):
        report = compare_models(self.manifest, self.records, self.records)
        row = next(r for r in report["paired"] if r["fold"] == "seen" and r["composition_id"] is None and r["k"] == 1)
        self.assertEqual((row["both_correct"], row["neither"]), (1, 1))
        for b in (self.records[:-1], self.records + self.records[:1],
                  tuple(replace(r, provenance=replace(self.provenance, comparison_budget="other-budget")) for r in self.records),
                  tuple(replace(r, spec=replace(self.spec, source_sha256=fingerprint("other-source"))) for r in self.records)):
            with self.subTest(length=len(b)), self.assertRaises(ValueError):
                compare_models(self.manifest, self.records, b)
        with self.assertRaises(ValueError):
            replace(self.provenance, metric_policy="percent")

    def test_final_policy_requires_payloads_and_explicit_approval(self):
        with self.assertRaises(ValueError):
            replace(self.spec, task_mode="final")
        candidates = tuple(Candidate(i, i, 0, f"c{i}", "d0", f"p{i}") for i in range(333))
        payload = {"schema": FINAL_NAMESPACE, "rows": [asdict(c) for c in candidates]}
        prompts = {"policy": {"synthetic": True}, "rows": []}
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            verify_final_payloads(payload, prompts)
        # Exercise the success branch with synthetic pinned hashes, never the
        # real Issue #8 digest or an invented claim of payload verification.
        with patch("evaluation.reporting.FINAL_CANDIDATE_SHA256", fingerprint(payload)), \
             patch("evaluation.reporting.FINAL_PROMPT_SHA256", fingerprint(prompts)):
            result = verify_final_payloads(payload, prompts)
        self.assertFalse(result["approval_granted"])
        self.assertEqual(result["verification"], "payload_hashes_match_issue_8")
        with self.assertRaises(ValueError):
            replace(self.spec, synthetic=False)

    def test_synthetic_final_333_columns_323_seen_rows_and_official_sized_denominators(self):
        # Test-only mappings. No candidate identity here is a frozen final label.
        candidates = tuple(Candidate(i, i, 0, f"synthetic-crop{i}", "synthetic-disease", f"synthetic-pair{i}")
                           for i in range(333))
        spec = replace(self.spec, task_mode="final", candidates=candidates,
                       candidate_hash_namespace=FINAL_NAMESPACE,
                       candidate_sha256=candidate_digest(candidates, FINAL_NAMESPACE), healthy_disease_ids=(),
                       approval_reference="synthetic fixture only; no final policy approval")
        fitting = tuple(Sample(f"synthetic-fit{i}", "fitting", i) for i in range(323))
        seen = tuple(Sample(f"synthetic-seen{i}", "seen", i % 323) for i in range(2000))
        unseen = tuple(Sample(f"synthetic-unseen{i}", "unseen", 323 + i % 10) for i in range(1448))
        manifest = EvaluationManifest(spec, fitting + seen + unseen, tuple(range(323)), tuple(range(323, 333)),
                                      2000, 1448, "all identities synthetic")
        scores = np.zeros((3448, 333), dtype=np.float64)
        for i, sample in enumerate(seen + unseen):
            scores[i, sample.truth_id] = 1.
        scores[0, 332] = 2.  # Retain an unseen prediction column for seen truth.
        scores[1, :] = 1.  # Exact tie => global ID 0.
        scores[2, :] = 1.
        scores[2, 2] = np.nextafter(1., np.inf)
        records = records_from_scores(spec, self.provenance, seen + unseen, scores, spec.ids)
        report = build_report(manifest, records)
        matrix = report["seen"]["confusion_matrix"]
        self.assertEqual((len(matrix["truth_ids"]), len(matrix["prediction_ids"])), (323, 333))
        self.assertEqual(matrix["counts"][0][332], 1)
        self.assertEqual(matrix["count"], 2000)
        self.assertEqual(matrix["diagonal_hits"], 1998)
        self.assertEqual(report["seen"]["correct_top1"], 1998)
        self.assertEqual(report["unseen"]["correct_top1"], 1448)
        self.assertTrue(report["candidate_spec"]["synthetic"])

    def test_hash_namespaces_and_utf8_are_explicit(self):
        value = {"name": "caf\u00e9"}
        self.assertNotEqual(fingerprint(value), fingerprint(value, ascii_only=True))
        self.assertNotIn(" ", canonical_json(value))
        self.assertNotEqual(candidate_digest(self.spec.candidates, WP1_NAMESPACE),
                            candidate_digest(self.spec.candidates, FINAL_NAMESPACE))

    def test_exports_roundtrip_csv_counts_nulls_and_preserve_existing(self):
        report = build_report(self.manifest, self.records)
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "new-report"
            export_report(report, output)
            saved = json.loads((output / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["harmonic_mean_top1"], .5)
            with (output / "seen_confusion.csv").open(newline="", encoding="utf-8") as stream:
                self.assertEqual(sum(int(r["count"]) for r in csv.DictReader(stream)), 2)
            with (output / "per_composition.csv").open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertTrue(any(r["count"] == "0" and r["top1"] == "" for r in rows))
            original = (output / "report.json").read_bytes()
            with self.assertRaises(FileExistsError):
                export_report(report, output)
            self.assertEqual((output / "report.json").read_bytes(), original)
            paired = compare_models(self.manifest, self.records, self.records)
            export_report(paired, Path(folder) / "paired")
            self.assertTrue((Path(folder) / "paired/paired.csv").exists())

    def test_reporting_entrypoints_do_not_open_files_or_images(self):
        with patch.object(builtins, "open", side_effect=AssertionError("unexpected file access")), \
             patch.object(Path, "open", side_effect=AssertionError("unexpected path access")), \
             patch("PIL.Image.open", side_effect=AssertionError("unexpected image access")), \
             patch("torch.hub.download_url_to_file", side_effect=AssertionError("unexpected model download")):
            records = records_from_scores(self.spec, self.provenance, self.manifest.samples[4:], self.scores, self.spec.ids)
            build_report(self.manifest, records)
            compare_models(self.manifest, records, records)

    def test_development_ingress_rejects_official_paths_before_open(self):
        with patch.object(Path, "open", side_effect=AssertionError("unexpected file access")):
            for name in ("DPD_Small_test_seen.csv", "DPD_Small_test_unseen.csv"):
                with self.assertRaisesRegex(ValueError, "training-source filename"):
                    development_manifest({}, name, healthy_disease_names=("healthy",))

    def test_reconstruction_helpers_use_only_public_training_member(self):
        from scripts import reconstruct_final_tables as reconstruction
        candidates = tuple(Candidate(i, i, 9, f"crop{i}", "healthy", f"pair{i}") for i in range(333))
        course = tuple(c for c in candidates if c.composition_id not in reconstruction.EXTRA_IDS)
        def csv_bytes(values):
            stream = io.StringIO(newline="")
            writer = csv.writer(stream)
            for index, c in enumerate(values):
                writer.writerow([f"synthetic{index}", c.crop_id, c.disease_id, c.composition_id,
                                 c.crop_name, c.disease_name, c.composition_name])
            return stream.getvalue().encode("utf-8")
        course_bytes = csv_bytes(tuple(course[i % 323] for i in range(8000)))
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, "w") as handle:
            handle.writestr(reconstruction.TRAIN_MEMBER, csv_bytes(candidates))
            handle.writestr("csv/default/test_seen.csv", b"MUST NOT BE READ")
        archive_bytes = archive.getvalue()
        payload = {"schema": FINAL_NAMESPACE, "rows": [asdict(c) for c in candidates]}
        prompts = reconstruction.prompt_payload(candidates)
        with patch.object(reconstruction, "COURSE_SOURCE_SHA256", hashlib.sha256(course_bytes).hexdigest()), \
             patch.object(reconstruction, "ARCHIVE_MD5", hashlib.md5(archive_bytes).hexdigest()), \
             patch.object(reconstruction, "DEVELOPMENT_PROMPT_SHA256", fingerprint(reconstruction.prompt_payload(course))), \
             patch("evaluation.reporting.FINAL_CANDIDATE_SHA256", fingerprint(payload)), \
             patch("evaluation.reporting.FINAL_PROMPT_SHA256", fingerprint(prompts)):
            actual_candidates, actual_prompts, provenance = reconstruction.reconstruct(course_bytes, archive_bytes)
        self.assertEqual(actual_candidates, payload)
        self.assertEqual(actual_prompts, prompts)
        self.assertEqual(provenance["preserved_development_prompt_strings"], 323)
        self.assertFalse(provenance["official_test_access"])
        self.assertFalse(provenance["approval_granted"])
        with self.assertRaises(ValueError):
            reconstruction.reconstruct(b"wrong source", archive_bytes)
        self.assertEqual(reconstruction.normalize_label("  APPLE__leaf "), "apple leaf")

    def test_wp1_adapter_refuses_unreconciled_local_artifact(self):
        root = Path(__file__).resolve().parents[1]
        protocol = json.loads((root / "artifacts/data/development_seed42_wp1_v1.json").read_text())
        with self.assertRaisesRegex(ValueError, "Reconciled"):
            development_manifest(protocol, root / "DPD_Small_train.csv", healthy_disease_names=("healthy",))


if __name__ == "__main__":
    unittest.main()
