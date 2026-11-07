"""Pure adapters from four selected models' canonical outputs to WP5 records.

No model construction, inference, file access, checkpoint selection or rescoring.
Input declarations require owner verification against real assets before use.
"""
from collections.abc import Mapping
from dataclasses import dataclass, replace
from itertools import combinations
from numbers import Integral

import numpy as np

from evaluation.reporting import (
    CandidateSpec, EvaluationManifest, PredictionRecord, RunProvenance, Sample,
    FINAL_NAMESPACE, FINAL_PROMPT_SHA256, WP1_NAMESPACE, candidate_digest,
    build_report, canonical_json, compare_models, fingerprint, records_from_scores,
)

WP1_INDEPENDENT = "wp1_independent_pair"
WP2_CONVNEXT = "wp2_convnext_base"
CLIP_FROZEN = "clip_vit_l14_frozen"
CLIP_ADAPTER = "clip_vit_l14_adapter"
MODEL_KINDS = (WP1_INDEPENDENT, WP2_CONVNEXT, CLIP_FROZEN, CLIP_ADAPTER)
LEGACY_CANDIDATES = "wp2-json-global-id-crop-disease-triples-v1"
WP1_SPLIT = "wp1-canonical-json-fold-filenames-v1"
LEGACY_SPLIT = "wp2-json-fit-val_seen-val_unseen-filenames-v1"
IDENTITY_MANIFEST = "wp5-identity-manifest-v1"
DEVELOPMENT_PROMPT_SHA256 = "753672c7964bcb20a1346145f21a72b40a19f55590a006b5783823f23c13f537"
CLIP_L14_SHA256 = "b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836"
SCORING = {
    WP1_INDEPENDENT: "sum_component_log_probabilities",
    WP2_CONVNEXT: "sum_log_softmax_v1",
    CLIP_FROZEN: "float64_cpu_scaled_unit_feature_cosine_v1",
    CLIP_ADAPTER: "float64_cpu_scaled_unit_feature_cosine_v1",
}
SELECTION = {
    WP2_CONVNEXT: {"architecture": "multitask_convnext_base", "epoch": 1, "input_resolution": 224},
    CLIP_FROZEN: {"architecture": "ViT-L/14"},
    CLIP_ADAPTER: {"architecture": "ViT-L/14", "epoch": 1, "training_seed": 45, "residual_cap": 0.20},
}


def _sha256(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("Expected a lowercase SHA-256 digest")


def _array(value, *, preserve_scalars=False):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=object if preserve_scalars else None)


def _id_array(value):
    # Preserve Python/JSON scalar types before NumPy can turn True into 1.
    values = _array(value, preserve_scalars=True)
    if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Integral) or v < 0 for v in values.flat):
        raise ValueError("IDs must be nonnegative integers without Boolean or numeric coercion")
    return values


def output_candidate_digest(spec: CandidateSpec, namespace):
    """Preserve source hash formats, without relabeling their different digests."""
    if namespace == LEGACY_CANDIDATES:
        return fingerprint([[c.composition_id, c.crop_id, c.disease_id] for c in spec.candidates])
    if namespace in (WP1_NAMESPACE, FINAL_NAMESPACE):
        return candidate_digest(spec.candidates, namespace)
    raise ValueError("Unsupported candidate hash namespace")


def output_split_digest(manifest: EvaluationManifest, namespace):
    if namespace == IDENTITY_MANIFEST:
        return manifest.sha256
    if manifest.spec.task_mode != "development":
        raise ValueError("Final outputs require the approved full identity-manifest fingerprint")
    if namespace == WP1_SPLIT:
        keys = ("fitting_images", "seen_validation_images", "unseen_validation_images")
    elif namespace == LEGACY_SPLIT:
        keys = ("fit", "val_seen", "val_unseen")
    else:
        raise ValueError("Unsupported split hash namespace")
    return fingerprint({key: [s.sample_id for s in manifest.samples if s.fold == fold]
                        for key, fold in zip(keys, ("fitting", "seen", "unseen"))}, ascii_only=True)


@dataclass(frozen=True)
class ModelOutputAdapter:
    manifest: EvaluationManifest
    provenance: RunProvenance
    model_kind: str
    final_evaluation_reference: str = ""

    def __post_init__(self):
        if type(self.manifest) is not EvaluationManifest or type(self.provenance) is not RunProvenance:
            raise ValueError("Require immutable approved identity and run provenance records")
        if not isinstance(self.model_kind, str) or self.model_kind not in MODEL_KINDS:
            raise ValueError("Unknown selected model kind")
        if self.provenance.score_semantics != SCORING[self.model_kind]:
            raise ValueError("Scoring declaration differs from the selected model's canonical scorer")
        if self.model_kind != WP1_INDEPENDENT and self.provenance.score_precision != "float64":
            raise ValueError("Selected WP2/CLIP outputs require canonical float64 scores")
        if not isinstance(self.final_evaluation_reference, str):
            raise ValueError("Final authorization reference must be an immutable string")
        if (self.manifest.spec.task_mode == "final" and not self.manifest.spec.synthetic
                and not self.final_evaluation_reference.strip()):
            raise ValueError("Real final record ingestion requires separate owner authorization")

    def _selection(self, packet):
        selection = packet.get("selection")
        if not isinstance(selection, Mapping):
            raise ValueError("Supply explicit selected-checkpoint/input declarations")
        for key, expected in SELECTION.get(self.model_kind, {}).items():
            value = selection.get(key)
            if type(value) is not type(expected) or value != expected:
                raise ValueError(f"Selected model declaration differs: {key}")
        if self.model_kind == WP1_INDEPENDENT:
            for task in ("crop", "disease"):
                _sha256(selection.get(task + "_checkpoint_sha256"))
                for suffix in ("_checkpoint_id", "_preprocessing"):
                    value = selection.get(task + suffix)
                    if not isinstance(value, str) or not value.strip():
                        raise ValueError("The independent pair requires both original identities and transforms")
        if self.model_kind == CLIP_ADAPTER:
            _sha256(selection.get("base_clip_sha256"))
            if selection["base_clip_sha256"] == self.provenance.checkpoint_sha256:
                raise ValueError("Base CLIP and selected adapter require distinct checkpoint identities")
            if not self.manifest.spec.synthetic and selection["base_clip_sha256"] != CLIP_L14_SHA256:
                raise ValueError("Adapter base CLIP identity differs from the frozen ViT-L/14 asset")
        if (self.model_kind == CLIP_FROZEN and not self.manifest.spec.synthetic
                and self.provenance.checkpoint_sha256 != CLIP_L14_SHA256):
            raise ValueError("Frozen CLIP checkpoint identity differs from ViT-L/14")
        # Round-trip freezes owner declarations as plain, finite JSON values.
        return canonical_json(dict(selection))

    def adapt(self, packet):
        """Copy one explicitly identified batch; build_report enforces full coverage.

        Packets are mappings, not inferred filenames attached to anonymous logits.
        Exactly one of scores[N,K] or ranked_ids[N,available_k] is required.
        """
        if not isinstance(packet, Mapping) or packet.get("schema") != "wp5-model-output-v1":
            raise ValueError("Expected an explicit wp5-model-output-v1 packet")
        for key, expected in (("model_kind", self.model_kind),
                              ("model_id", self.provenance.model_id),
                              ("run_id", self.provenance.run_id),
                              ("checkpoint_id", self.provenance.checkpoint_id),
                              ("checkpoint_sha256", self.provenance.checkpoint_sha256),
                              ("preprocessing", self.provenance.preprocessing),
                              ("code_version", self.provenance.code_version),
                              ("environment", self.provenance.environment),
                              ("score_precision", self.provenance.score_precision),
                              ("manifest_sha256", self.manifest.sha256),
                              ("source_sha256", self.manifest.spec.source_sha256),
                              ("score_semantics", self.provenance.score_semantics)):
            if packet.get(key) != expected:
                raise ValueError(f"Model output provenance mismatch: {key}")
        candidate_namespace = packet.get("candidate_hash_namespace")
        if self.manifest.spec.task_mode == "final" and candidate_namespace != FINAL_NAMESPACE:
            raise ValueError("Final outputs require the separately frozen final candidate namespace")
        if packet.get("candidate_sha256") != output_candidate_digest(self.manifest.spec, candidate_namespace):
            raise ValueError("Source candidate digest differs from the exact approved IDs/pairs/labels")
        split_namespace = packet.get("split_hash_namespace")
        if packet.get("split_sha256") != output_split_digest(self.manifest, split_namespace):
            raise ValueError("Source split digest differs from the exact approved identity order")
        if self.model_kind in (CLIP_FROZEN, CLIP_ADAPTER):
            prompt = packet.get("prompt_sha256")
            _sha256(prompt)
            if prompt != self.provenance.prompt_sha256:
                raise ValueError("CLIP output and frozen run prompt identities differ")
            if not self.manifest.spec.synthetic:
                expected_prompt = (DEVELOPMENT_PROMPT_SHA256 if self.manifest.spec.task_mode == "development"
                                   else FINAL_PROMPT_SHA256)
                if prompt != expected_prompt:
                    raise ValueError("CLIP prompt digest differs from the selected task")
        selection = self._selection(packet)
        names = packet.get("image_names")
        if isinstance(names, (str, bytes, Mapping)) or names is None:
            raise ValueError("Original image identities are required; never infer them from row order")
        names = tuple(names)
        if any(not isinstance(name, str) or not name.strip() for name in names):
            raise ValueError("Image identities must be nonempty original filename strings")
        truths = _id_array(packet.get("ground_truth_ids"))
        columns = _id_array(packet.get("candidate_ids"))
        if not names or len(set(names)) != len(names) or truths.shape != (len(names),):
            raise ValueError("Missing, duplicate or misaligned batch identities/truths")
        if columns.shape != (len(self.manifest.spec.ids),) or tuple(columns) != self.manifest.spec.ids:
            raise ValueError("Output candidate columns differ from the approved original global-ID order")
        samples = tuple(Sample(name, packet.get("fold"), truth) for name, truth in zip(names, truths))
        expected = {s.sample_id: s for s in self.manifest.samples if s.fold in ("seen", "unseen")}
        if any(expected.get(s.sample_id) != s for s in samples):
            raise ValueError("Batch identity, truth or fold differs from the approved manifest")
        provenance = replace(self.provenance, preprocessing=canonical_json({
            "declared_preprocessing": self.provenance.preprocessing,
            "model_kind": self.model_kind, "selection_json": selection,
            "candidate_hash_namespace": candidate_namespace, "candidate_sha256": packet["candidate_sha256"],
            "split_hash_namespace": split_namespace, "split_sha256": packet["split_sha256"],
            "final_evaluation_reference": self.final_evaluation_reference,
        }))
        has_scores, has_ranks = "scores" in packet, "ranked_ids" in packet
        if has_scores == has_ranks:
            raise ValueError("Supply exactly one canonical score matrix or ranked-ID matrix")
        if has_scores:
            scores = packet["scores"]
            if isinstance(scores, np.ndarray) or hasattr(scores, "detach"):
                # Keep the established dtype rules for native model matrices.
                records = records_from_scores(self.manifest.spec, provenance, samples, scores, columns)
            else:
                # JSON has no matrix dtype. Validate every original scalar at
                # the exact-precision record boundary before any numeric cast.
                values = _array(scores, preserve_scalars=True)
                if values.shape != (len(samples), len(columns)):
                    raise ValueError("Serialized scores must have aligned shape [N,K]")
                records = tuple(PredictionRecord(s, self.manifest.spec, provenance, tuple(columns), len(columns),
                                                  scores=tuple(row)) for s, row in zip(samples, values))
        else:
            available = packet.get("available_k")
            ranks = _id_array(packet["ranked_ids"])
            if type(available) is not int or ranks.shape != (len(samples), available):
                raise ValueError("Ranked-ID matrix must match the declared available_k")
            records = tuple(PredictionRecord(s, self.manifest.spec, provenance, tuple(columns), available,
                                              ranked_ids=tuple(row)) for s, row in zip(samples, ranks))
        for component in ("crop", "disease"):
            key = "independent_" + component + "_ids"
            if key in packet:
                ids = _id_array(packet[key])
                if ids.shape != (len(records),):
                    raise ValueError("Independent head decisions must align with every supplied sample")
                records = tuple(replace(r, **{"independent_" + component + "_id": value})
                                for r, value in zip(records, ids))
        return records


def import_model_packets(manifest, provenances, packets, *, final_evaluation_reference=""):
    """Validate all four identified model exports before returning any reports.

    Inputs are already decoded owner-verified packets; this function opens no
    files. Callers verify original file SHA-256 values before decoding. Full
    coverage, unique identities and common comparison budgets are mandatory.
    """
    for name, values in (("provenances", provenances), ("packets", packets)):
        if not isinstance(values, Mapping) or set(values) != set(MODEL_KINDS):
            raise ValueError(f"{name} must contain exactly the four selected model kinds")
    records, reports = {}, {}
    for kind in MODEL_KINDS:
        batches = packets[kind]
        if batches is None or isinstance(batches, (str, bytes, Mapping)):
            raise ValueError("Each model requires a sequence of identified batch packets")
        adapter = ModelOutputAdapter(manifest, provenances[kind], kind, final_evaluation_reference)
        records[kind] = tuple(record for packet in batches for record in adapter.adapt(packet))
        reports[kind] = build_report(manifest, records[kind])
    paired = {a + "__" + b: compare_models(manifest, records[a], records[b])
              for a, b in combinations(MODEL_KINDS, 2)}
    return {"schema": "wp5-model-import-v1", "manifest_sha256": manifest.sha256,
            "candidate_sha256": manifest.spec.candidate_sha256,
            "models": reports, "paired": paired}
