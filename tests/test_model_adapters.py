"""Synthetic selected-model integration; no image, weight or prediction files."""
import builtins
import itertools
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from evaluation.model_adapters import (
    CLIP_ADAPTER, CLIP_FROZEN, IDENTITY_MANIFEST,
    LEGACY_CANDIDATES, LEGACY_SPLIT, MODEL_KINDS, SCORING, SELECTION,
    WP1_INDEPENDENT, WP1_SPLIT, WP2_CONVNEXT, ModelOutputAdapter,
    import_model_packets, output_candidate_digest, output_split_digest,
)
from evaluation.reporting import (
    Candidate, CandidateSpec, EvaluationManifest, Sample, FINAL_NAMESPACE, FINAL_UNSEEN_IDS, WP1_NAMESPACE,
    build_report, candidate_digest, compare_models, export_report, fingerprint,
)
from test_evaluation_reporting import fixture


def selected_packet(manifest, provenance, matrix, kind, fold="seen", *, namespace=None):
    """Test-only owner declarations and synthetic canonical scores."""
    provenance = replace(provenance, model_id=kind, run_id="synthetic-" + kind,
                         checkpoint_id="synthetic-checkpoint-" + kind,
                         checkpoint_sha256=fingerprint("synthetic-asset-" + kind),
                         score_semantics=SCORING[kind], score_precision="float64",
                         prompt_sha256=fingerprint("synthetic prompt policy") if kind in (CLIP_FROZEN, CLIP_ADAPTER) else "")
    final = manifest.spec.task_mode == "final"
    candidate_namespace = namespace or (FINAL_NAMESPACE if final else WP1_NAMESPACE if kind == WP1_INDEPENDENT else LEGACY_CANDIDATES)
    split_namespace = IDENTITY_MANIFEST if final else WP1_SPLIT if kind == WP1_INDEPENDENT else LEGACY_SPLIT
    samples = tuple(s for s in manifest.samples if s.fold == fold)
    selection = dict(SELECTION.get(kind, {}))
    if kind == WP1_INDEPENDENT:
        for component in ("crop", "disease"):
            selection.update({component + "_checkpoint_id": "synthetic-" + component,
                              component + "_checkpoint_sha256": fingerprint("synthetic-" + component),
                              component + "_preprocessing": "synthetic RGB transform"})
    if kind == CLIP_ADAPTER:
        selection["base_clip_sha256"] = fingerprint("synthetic CLIP base")
    packet = {"schema": "wp5-model-output-v1", "model_kind": kind,
              **{key: getattr(provenance, key) for key in ("model_id", "run_id", "checkpoint_id", "checkpoint_sha256",
                  "preprocessing", "code_version", "environment", "score_precision", "score_semantics", "prompt_sha256")},
              "manifest_sha256": manifest.sha256, "source_sha256": manifest.spec.source_sha256,
              "candidate_ids": torch.tensor(manifest.spec.ids),
              "candidate_hash_namespace": candidate_namespace,
              "candidate_sha256": output_candidate_digest(manifest.spec, candidate_namespace),
              "split_hash_namespace": split_namespace, "split_sha256": output_split_digest(manifest, split_namespace),
              "selection": selection, "fold": fold,
              "image_names": [s.sample_id for s in samples],
              "ground_truth_ids": torch.tensor([s.truth_id for s in samples]), "scores": matrix}
    return ModelOutputAdapter(manifest, provenance, kind), packet


def sized_manifest(count):
    """All identities/labels are invented; no final-policy approval is implied."""
    candidates = tuple(Candidate(i, i, 0, f"synthetic-crop{i}", "synthetic-disease", f"synthetic-pair{i}")
                       for i in range(count))
    final = count == 333
    namespace = FINAL_NAMESPACE if final else WP1_NAMESPACE
    spec = CandidateSpec("synthetic-final" if final else "synthetic-development", "1",
                         "final" if final else "development", candidates, candidate_digest(candidates, namespace),
                         namespace, fingerprint("synthetic source"), fingerprint("synthetic split"),
                         "synthetic-split", "synthetic class labels only", (), synthetic=True,
                         approval_reference="synthetic fixture only; no final policy approval" if final else "")
    fitting_ids = tuple(range(323 if final else 258))
    unseen_ids = tuple(range(323, 333)) if final else tuple(range(258, 323))
    seen = (Sample("s0", "seen", 0), Sample("s1", "seen", 1))
    unseen = (Sample("u0", "unseen", unseen_ids[0]), Sample("u1", "unseen", unseen_ids[-1]))
    fitting = tuple(Sample(f"fit{i}", "fitting", i) for i in fitting_ids)
    return EvaluationManifest(spec, fitting + seen + unseen, fitting_ids, unseen_ids, 2, 2, "synthetic identities")


def decoded_packet(packet):
    """Exercise actual JSON decoding without coercing Python integers first."""
    def encode(value):
        if hasattr(value, "detach"):
            return value.detach().cpu().tolist()
        if isinstance(value, np.ndarray):
            return value.tolist()
        raise TypeError("Unsupported JSON fixture value")
    return json.loads(json.dumps(packet, default=encode))


class ModelAdapterTests(unittest.TestCase):
    def setUp(self):
        self.manifest, self.provenance, self.matrix, _ = fixture()

    def records(self, kind, manifest=None, matrix=None):
        manifest = manifest or self.manifest
        matrix = self.matrix if matrix is None else matrix
        result = []
        for fold, part in (("seen", matrix[:2]), ("unseen", matrix[2:])):
            adapter, packet = selected_packet(manifest, self.provenance, part, kind, fold)
            result.extend(adapter.adapt(packet))
        return tuple(result)

    def test_all_four_adapters_keep_canonical_scores_and_report_counts(self):
        for kind in MODEL_KINDS:
            with self.subTest(kind=kind):
                records = self.records(kind)
                self.assertEqual([r.scores for r in records], [tuple(row) for row in self.matrix])
                report = build_report(self.manifest, records)
                self.assertEqual((report["harmonic_mean_top1"], report["harmonic_mean_top5"]), (.5, 2 / 3))
                source = json.loads(records[0].provenance.preprocessing)
                self.assertEqual(source["declared_preprocessing"], self.provenance.preprocessing)
                self.assertEqual(source["model_kind"], kind)
                self.assertEqual(report["seen"]["expected"], 2)

    def test_all_six_model_pairs_align_identical_original_identities(self):
        records = {kind: self.records(kind) for kind in MODEL_KINDS}
        for a, b in itertools.combinations(MODEL_KINDS, 2):
            report = compare_models(self.manifest, records[a][::-1], records[b])
            self.assertEqual(report["harmonic_mean_top1_b_minus_a"], 0.)
            for row in report["paired"]:
                self.assertEqual(sum(row[key] for key in ("both_correct", "a_only", "b_only", "neither")), row["count"])

    def test_namespaced_candidate_and_split_hashes_remain_distinct(self):
        self.assertNotEqual(output_candidate_digest(self.manifest.spec, WP1_NAMESPACE),
                            output_candidate_digest(self.manifest.spec, LEGACY_CANDIDATES))
        self.assertNotEqual(output_split_digest(self.manifest, WP1_SPLIT), output_split_digest(self.manifest, LEGACY_SPLIT))
        for namespace in (WP1_NAMESPACE, LEGACY_CANDIDATES):
            adapter, packet = selected_packet(self.manifest, self.provenance, self.matrix[:2], WP2_CONVNEXT, namespace=namespace)
            self.assertEqual(len(adapter.adapt(packet)), 2)
        packet["candidate_hash_namespace"] = WP1_NAMESPACE
        with self.assertRaisesRegex(ValueError, "candidate digest"):
            adapter.adapt(packet)

    def test_missing_original_filenames_cannot_be_inferred_from_matching_truths(self):
        adapter, packet = selected_packet(self.manifest, self.provenance, self.matrix[:2], WP2_CONVNEXT)
        del packet["image_names"]
        with self.assertRaisesRegex(ValueError, "Original image identities"):
            adapter.adapt(packet)
        with self.assertRaises(ValueError):
            adapter.adapt({"crop_logits": torch.zeros(2, 55), "disease_logits": torch.zeros(2, 175),
                           "composition_ids": torch.tensor([10, 75])})

    def test_rejects_wrong_model_checkpoint_preprocessing_and_task_headers(self):
        adapter, packet = selected_packet(self.manifest, self.provenance, self.matrix[:2], WP2_CONVNEXT)
        for key in ("model_kind", "model_id", "run_id", "checkpoint_id", "checkpoint_sha256", "preprocessing",
                    "code_version", "environment", "score_precision", "manifest_sha256", "source_sha256", "score_semantics",
                    "candidate_sha256", "split_sha256"):
            bad = dict(packet, **{key: "mismatch"})
            with self.subTest(key=key), self.assertRaises(ValueError):
                adapter.adapt(bad)

    def test_rejects_duplicate_unknown_wrong_truth_and_wrong_fold_rows(self):
        adapter, packet = selected_packet(self.manifest, self.provenance, self.matrix[:2], WP2_CONVNEXT)
        updates = ({"image_names": ["s1", "s1"]}, {"image_names": ["unknown", "s2"]},
                   {"image_names": [["s1"], "s2"]}, {"ground_truth_ids": [75, 75]},
                   {"image_names": {"s1": "untrusted", "s2": "untrusted"}},
                   {"ground_truth_ids": [10., 75.]}, {"fold": "unseen"})
        for update in updates:
            with self.subTest(update=update), self.assertRaises(ValueError):
                adapter.adapt(dict(packet, **update))

    def test_missing_batches_fail_complete_report_and_pairing(self):
        adapter, packet = selected_packet(self.manifest, self.provenance, self.matrix[:2], WP2_CONVNEXT)
        partial = adapter.adapt(packet)
        with self.assertRaisesRegex(ValueError, "Missing"):
            build_report(self.manifest, partial)
        with self.assertRaisesRegex(ValueError, "Missing"):
            compare_models(self.manifest, partial, self.records(CLIP_FROZEN))

    def test_rejects_reordered_columns_nonfinite_and_wrong_precision_scores(self):
        adapter, packet = selected_packet(self.manifest, self.provenance, self.matrix[:2], WP2_CONVNEXT)
        for update in ({"candidate_ids": self.manifest.spec.ids[::-1]},
                       {"candidate_ids": [float(i) for i in self.manifest.spec.ids]},
                       {"scores": np.ones((2, 6), dtype=np.float32)}, {"scores": np.full((2, 6), np.nan)}):
            with self.subTest(update=update.keys()), self.assertRaises(ValueError):
                adapter.adapt(dict(packet, **update))

    def test_wp1_retains_native_float32_without_substituting_wp2_precision(self):
        adapter, packet = selected_packet(self.manifest, self.provenance, self.matrix[:2], WP1_INDEPENDENT)
        provenance = replace(adapter.provenance, score_precision="float32")
        adapter = replace(adapter, provenance=provenance)
        packet.update(score_precision="float32", scores=torch.tensor(self.matrix[:2], dtype=torch.float32))
        self.assertEqual(adapter.adapt(packet)[0].scores, tuple(self.matrix[0]))
        with self.assertRaises(ValueError):
            ModelOutputAdapter(self.manifest, replace(provenance, score_semantics=SCORING[WP2_CONVNEXT]), WP2_CONVNEXT)

    def test_clip_requires_the_exact_declared_prompt_identity(self):
        for kind in (CLIP_FROZEN, CLIP_ADAPTER):
            adapter, packet = selected_packet(self.manifest, self.provenance, self.matrix[:2], kind)
            for value in (None, "", [], fingerprint("other prompts")):
                with self.subTest(kind=kind, prompt=value), self.assertRaises(ValueError):
                    adapter.adapt(dict(packet, prompt_sha256=value))

    def test_selected_checkpoint_epoch_resolution_seed_and_cap_are_explicit(self):
        for kind, keys in ((WP2_CONVNEXT, ("architecture", "epoch", "input_resolution")),
                           (CLIP_ADAPTER, ("architecture", "epoch", "training_seed", "residual_cap", "base_clip_sha256"))):
            adapter, packet = selected_packet(self.manifest, self.provenance, self.matrix[:2], kind)
            for key in keys:
                selection = dict(packet["selection"])
                selection[key] = "incorrect"
                with self.subTest(kind=kind, key=key), self.assertRaises(ValueError):
                    adapter.adapt(dict(packet, selection=selection))
        adapter, packet = selected_packet(self.manifest, self.provenance, self.matrix[:2], WP1_INDEPENDENT)
        del packet["selection"]["crop_preprocessing"]
        with self.assertRaises(ValueError):
            adapter.adapt(packet)

    def test_rank_only_owner_exports_obey_available_k_and_global_ids(self):
        adapter, packet = selected_packet(self.manifest, self.provenance, self.matrix[:2], WP2_CONVNEXT)
        scores = packet.pop("scores")
        packet["available_k"] = 5
        packet["ranked_ids"] = np.array(self.manifest.spec.ids)[np.argsort(-scores, axis=1, kind="stable")[:, :5]]
        ranks = adapter.adapt(packet)
        self.assertEqual(ranks[0].ranking(), self.manifest.spec.ids[:5])
        self.assertFalse(ranks[0].scores)
        for update in ({"available_k": 1}, {"ranked_ids": np.full((2, 5), 999)},
                       {"ranked_ids": np.ones((2, 5), dtype=np.int64)}, {"scores": scores}):
            with self.subTest(update=update.keys()), self.assertRaises(ValueError):
                adapter.adapt(dict(packet, **update))

    def test_independent_heads_remain_separate_from_selected_composition(self):
        adapter, packet = selected_packet(self.manifest, self.provenance, self.matrix[:2], WP2_CONVNEXT)
        packet.update(independent_crop_ids=[0, 0], independent_disease_ids=[9, 1])
        records = adapter.adapt(packet)
        self.assertEqual(records[1].independent_disease_id, 1)
        packet["independent_disease_ids"] = [999, 1]
        with self.assertRaises(ValueError):
            adapter.adapt(packet)

    def test_synthetic_323_and_333_tasks_work_for_all_four_models(self):
        for count in (323, 333):
            manifest = sized_manifest(count)
            samples = [s for s in manifest.samples if s.fold in ("seen", "unseen")]
            scores = np.zeros((4, count), dtype=np.float64)
            for i, sample in enumerate(samples):
                scores[i, sample.truth_id] = 1.
            scores[0, :] = 1.  # Tie remains original candidate-column order.
            scores[1, :] = 1.
            scores[1, 1] = np.nextafter(1., np.inf)
            for kind in MODEL_KINDS:
                with self.subTest(candidates=count, kind=kind):
                    records = self.records(kind, manifest, scores)
                    self.assertEqual(records[0].ranking(), manifest.spec.ids)
                    self.assertEqual(records[1].ranking()[0], 1)
                    report = build_report(manifest, records)
                    self.assertEqual(report["harmonic_mean_top1"], 1.)
                    matrix = report["seen"]["confusion_matrix"]
                    self.assertEqual((len(matrix["truth_ids"]), len(matrix["prediction_ids"])), (323 if count == 333 else 258, count))

    def test_training_only_323_headers_match_published_legacy_fingerprints(self):
        from shared_protocol import validate_protocol
        root = Path(__file__).resolve().parents[1]
        protocol = json.loads((root / "artifacts/data/development_seed42_wp1_v1.json").read_text(encoding="utf-8"))
        fitting, seen, unseen = validate_protocol(protocol, root / "DPD_Small_train.csv")
        candidates = tuple(Candidate(**r) for r in protocol["candidates"])
        spec = CandidateSpec("training-only synthetic score fixture", "1", "development", candidates,
                             protocol["candidate_sha256"], WP1_NAMESPACE, protocol["source_sha256"],
                             protocol["split_sha256"], WP1_SPLIT, "course training class-only mapping",
                             tuple(sorted({c.disease_id for c in candidates if c.disease_name == "healthy"})), synthetic=True)
        samples = tuple(Sample(str(row.image_name), fold, int(row.composition_id))
                        for fold, frame in (("fitting", fitting), ("seen", seen), ("unseen", unseen))
                        for row in frame.itertuples())
        manifest = EvaluationManifest(spec, samples, tuple(sorted(set(fitting.composition_id))),
                                      tuple(sorted(set(unseen.composition_id))), 1238, 1894,
                                      "published training-only identities; synthetic scores; no artifact approval")
        self.assertEqual(output_candidate_digest(spec, LEGACY_CANDIDATES),
                         "4cb0a24a96cf647f32311817fa7b082a46d3cf00c20ae61f4287e83cc898dfaf")
        self.assertEqual(output_split_digest(manifest, LEGACY_SPLIT),
                         "f0c7ba212449fd68f1d19c0f84cc91bf4cc1af29ae405128e5198bce89f25f85")
        self.assertEqual(output_split_digest(manifest, WP1_SPLIT), protocol["split_sha256"])
        self.assertNotEqual(spec.ids, tuple(range(323)))
        for kind in MODEL_KINDS:
            matrix = np.ones((2, 323), dtype=np.float64)
            matrix[1, -1] = np.nextafter(1., np.inf)
            adapter, packet = selected_packet(manifest, self.provenance, matrix, kind)
            packet["image_names"] = packet["image_names"][:2]
            packet["ground_truth_ids"] = packet["ground_truth_ids"][:2]
            records = adapter.adapt(packet)
            self.assertEqual(records[0].ranking(), spec.ids)
            self.assertEqual(records[1].ranking()[0], spec.ids[-1])

    def test_final_namespace_and_identity_gate_reject_development_reuse(self):
        manifest = sized_manifest(333)
        adapter, packet = selected_packet(manifest, self.provenance, np.zeros((2, 333)), WP2_CONVNEXT)
        with self.assertRaises(ValueError):
            adapter.adapt(dict(packet, candidate_hash_namespace=LEGACY_CANDIDATES))
        with self.assertRaises(ValueError):
            adapter.adapt(dict(packet, split_hash_namespace=LEGACY_SPLIT))
        with self.assertRaises(ValueError):
            output_split_digest(manifest, WP1_SPLIT)
        with self.assertRaises(ValueError):
            replace(adapter, final_evaluation_reference=[])

    def test_real_mode_guard_requires_separate_final_record_authorization(self):
        # Public class-only payloads + invented identities exercise only a deny
        # gate. No actual official labels/predictions or approved policy exist here.
        directory = Path(__file__).resolve().parents[1] / "artifacts/wp5/final_reconstruction_2026-11-07"
        payload = json.loads((directory / "verified_reconstruction_candidates.json").read_text(encoding="utf-8"))
        candidates = tuple(Candidate(**r) for r in payload["rows"])
        spec = CandidateSpec("unit-test-final-deny-gate", "1", "final", candidates,
                             candidate_digest(candidates, FINAL_NAMESPACE), FINAL_NAMESPACE,
                             fingerprint("unit-test source"), fingerprint("unit-test split"),
                             "unit-test split", "verified public class-only semantics", (), synthetic=False,
                             approval_reference="UNIT TEST fixture only; no actual policy approval",
                             final_prompt_json=(directory / "verified_reconstruction_prompts.json").read_text(encoding="utf-8"))
        seen_ids = tuple(i for i in spec.ids if i not in FINAL_UNSEEN_IDS)
        samples = tuple(Sample(f"unit-fit{i}", "fitting", seen_ids[i % 323]) for i in range(4868))
        samples += tuple(Sample(f"unit-seen{i}", "seen", seen_ids[i % 323]) for i in range(2000))
        samples += tuple(Sample(f"unit-unseen{i}", "unseen", FINAL_UNSEEN_IDS[i % 10]) for i in range(1448))
        manifest = EvaluationManifest(spec, samples, seen_ids, FINAL_UNSEEN_IDS, 2000, 1448,
                                      "unit-test invented identities; deny-gate fixture only")
        provenance = replace(self.provenance, score_semantics=SCORING[WP2_CONVNEXT])
        for reference in ("", " "):
            with self.subTest(reference=reference), self.assertRaisesRegex(ValueError, "separate owner authorization"):
                ModelOutputAdapter(manifest, provenance, WP2_CONVNEXT, reference)

    def test_adapter_outputs_are_copies_and_adaptation_performs_no_io(self):
        adapter, packet = selected_packet(self.manifest, self.provenance, self.matrix[:2], WP2_CONVNEXT)
        with patch.object(builtins, "open", side_effect=AssertionError("unexpected file read")), \
             patch.object(Path, "open", side_effect=AssertionError("unexpected path read")), \
             patch("torch.hub.download_url_to_file", side_effect=AssertionError("unexpected weights")):
            records = adapter.adapt(packet)
        packet["scores"][0, 0] = -999
        packet["selection"]["epoch"] = 100
        self.assertEqual(records[0].scores[0], 6.)
        self.assertEqual(json.loads(json.loads(records[0].provenance.preprocessing)["selection_json"])["epoch"], 1)

    def test_all_four_outputs_export_without_fabricating_real_results(self):
        with tempfile.TemporaryDirectory() as directory:
            for kind in MODEL_KINDS:
                report = build_report(self.manifest, self.records(kind))
                export_report(report, Path(directory) / kind)
                self.assertTrue(report["candidate_spec"]["synthetic"])
                self.assertTrue((Path(directory) / kind / "harmonic_means.csv").is_file())

    def test_serialized_scores_validate_original_values_before_conversion(self):
        for precision, values in (("float64", (2**53 + 1, 2**1024, float("inf"), float("nan"), True)),
                                  ("float32", (2**24 + 1, float(np.nextafter(1., np.inf)), 2**-150))):
            adapter, packet = selected_packet(self.manifest, self.provenance, self.matrix[:2], WP1_INDEPENDENT)
            adapter = replace(adapter, provenance=replace(adapter.provenance, score_precision=precision))
            packet["score_precision"] = precision
            for value in values:
                packet["scores"] = [[value, 0, 0, 0, 0, 0], [1., 0., 0., 0., 0., 0.]]
                with self.subTest(precision=precision, value=value), self.assertRaises(ValueError):
                    adapter.adapt(decoded_packet(packet))

    def test_exact_float32_json_import_keeps_native_matrix_dtype_rules(self):
        adapter, packet = selected_packet(self.manifest, self.provenance, self.matrix[:2], WP1_INDEPENDENT)
        adapter = replace(adapter, provenance=replace(adapter.provenance, score_precision="float32"))
        packet.update(score_precision="float32", scores=self.matrix[:2].tolist())
        self.assertEqual(adapter.adapt(decoded_packet(packet))[0].scores, tuple(self.matrix[0]))
        packet["scores"] = np.asarray(packet["scores"], dtype=np.float64)
        with self.assertRaisesRegex(ValueError, "dtype differs"):
            adapter.adapt(packet)
        packet["scores"] = packet["scores"].astype(np.float32)
        self.assertEqual(adapter.adapt(packet)[0].scores, tuple(self.matrix[0]))

    def test_float64_json_one_ulp_and_signed_zero_survive_import(self):
        adapter, packet = selected_packet(self.manifest, self.provenance, self.matrix[:2], WP2_CONVNEXT)
        packet["scores"] = [[1., float(np.nextafter(1., np.inf)), 0., 0., 0., -0.], [1., 1., 0., 0., 0., 0.]]
        records = adapter.adapt(decoded_packet(packet))
        self.assertEqual(records[0].ranking()[0], 75)
        self.assertEqual(records[1].ranking()[0], 10)
        self.assertTrue(np.signbit(records[0].scores[-1]))

    def test_mixed_boolean_ids_are_rejected_before_numpy_coercion(self):
        manifest = sized_manifest(333)
        adapter, packet = selected_packet(manifest, self.provenance, np.zeros((2, 333)), WP2_CONVNEXT)
        for update in ({"candidate_ids": [False, True, *range(2, 333)]},
                       {"ground_truth_ids": [0, True]}, {"independent_crop_ids": [0, True]},
                       {"independent_disease_ids": [0, False]}):
            with self.subTest(update=update.keys()), self.assertRaisesRegex(ValueError, "without Boolean"):
                adapter.adapt(dict(packet, **update))
        del packet["scores"]
        packet.update(available_k=5, ranked_ids=[[False, 1, 2, 3, 4], [0, 1, 2, 3, 4]])
        with self.assertRaisesRegex(ValueError, "without Boolean"):
            adapter.adapt(decoded_packet(packet))

    def test_adapter_and_base_clip_checkpoint_identities_must_be_distinct(self):
        adapter, packet = selected_packet(self.manifest, self.provenance, self.matrix[:2], CLIP_ADAPTER)
        packet["selection"]["base_clip_sha256"] = adapter.provenance.checkpoint_sha256
        with self.assertRaisesRegex(ValueError, "distinct checkpoint identities"):
            adapter.adapt(packet)

    def _owner_import_fixture(self):
        provenances, packets = {}, {}
        for kind in MODEL_KINDS:
            batches = []
            for fold, matrix in (("seen", self.matrix[:2]), ("unseen", self.matrix[2:])):
                adapter, packet = selected_packet(self.manifest, self.provenance, matrix, kind, fold)
                provenances[kind] = adapter.provenance
                batches.append(decoded_packet(packet))
            packets[kind] = batches
        return provenances, packets

    def test_four_model_import_validates_complete_reports_before_export(self):
        provenances, packets = self._owner_import_fixture()
        imported = import_model_packets(self.manifest, provenances, packets)
        self.assertEqual(set(imported["models"]), set(MODEL_KINDS))
        self.assertEqual(len(imported["paired"]), 6)
        with tempfile.TemporaryDirectory() as directory:
            for kind, report in imported["models"].items():
                self.assertEqual(report["harmonic_mean_top1"], .5)
                export_report(report, Path(directory) / kind)
            for pair, report in imported["paired"].items():
                self.assertEqual(report["harmonic_mean_top1_b_minus_a"], 0.)
                export_report(report, Path(directory) / "paired" / pair)

    def test_four_model_import_rejects_partial_duplicate_and_incompatible_runs(self):
        provenances, packets = self._owner_import_fixture()
        with self.assertRaisesRegex(ValueError, "exactly the four"):
            import_model_packets(self.manifest, provenances, {WP1_INDEPENDENT: packets[WP1_INDEPENDENT]})
        for bad_batches in (packets[WP2_CONVNEXT][:1], packets[WP2_CONVNEXT] + packets[WP2_CONVNEXT][:1], []):
            with self.subTest(batch_count=len(bad_batches)), self.assertRaises(ValueError):
                import_model_packets(self.manifest, provenances, dict(packets, **{WP2_CONVNEXT: bad_batches}))
        mismatched = dict(provenances, **{WP2_CONVNEXT: replace(provenances[WP2_CONVNEXT], comparison_budget="other plan")})
        with self.assertRaisesRegex(ValueError, "comparison budgets"):
            import_model_packets(self.manifest, mismatched, packets)


if __name__ == "__main__":
    unittest.main()
