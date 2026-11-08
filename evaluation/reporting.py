"""WP5 record validation and reporting. No model, image, or official CSV access.

Scores must already be computed by the model's canonical scorer. This module
only ranks them; it never reconstructs logits, cosines, or probabilities.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass
from numbers import Integral, Real
from pathlib import Path

import numpy as np

from evaluation.metrics import harmonic_mean

WP1_NAMESPACE = "wp1-canonical-json-candidate-records-v1"
FINAL_NAMESPACE = "cos30082-final-candidates-v1"
FINAL_CANDIDATE_SHA256 = "ba17e94119e8b2b7a4b18603e25405d2f52bf4a93396973f80491800697eb62f"
FINAL_PROMPT_SHA256 = "c9dd871d61d4141b33583ddd71961c309fd679a8edab088cb28ba20eccef88f2"
FINAL_UNSEEN_IDS = (39, 84, 107, 124, 133, 136, 144, 166, 276, 303)
TIE_RULE = "stable_column_order_ascending_global_id"
METRIC_POLICY = "composition_hits_fraction_min5_v1"
FOLDS = ("seen", "unseen")


def canonical_json(value, *, ascii_only=False):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=ascii_only, allow_nan=False)


def fingerprint(value, *, ascii_only=False):
    return hashlib.sha256(canonical_json(value, ascii_only=ascii_only).encode("utf-8")).hexdigest()


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


def _id(value):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError("IDs must be nonnegative integers, without coercion")
    return int(value)


def _ids(values):
    return tuple(_id(v) for v in values)


def _hash(value, name):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _exact_score(value, dtype):
    """Check the original exact ratio before storing an immutable Python float."""
    message = f"Canonical scores must be finite real numbers exactly representable as {dtype.__name__}"
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(message)
    try:
        original_ratio = (int(value), 1) if isinstance(value, Integral) else value.as_integer_ratio()
        candidate = dtype(value)  # Validation probe only; never accept a rounded value.
        if not np.isfinite(candidate) or candidate.as_integer_ratio() != original_ratio:
            raise ValueError(message)
    except (AttributeError, TypeError, ValueError, OverflowError):
        raise ValueError(message) from None
    return float(candidate)  # Both supported precisions are exactly representable here.


@dataclass(frozen=True)
class Candidate:
    composition_id: int
    crop_id: int
    disease_id: int
    crop_name: str
    disease_name: str
    composition_name: str

    def __post_init__(self):
        for key in ("composition_id", "crop_id", "disease_id"):
            object.__setattr__(self, key, _id(getattr(self, key)))
        for key in ("crop_name", "disease_name", "composition_name"):
            _text(getattr(self, key), key)


def candidate_digest(candidates, namespace):
    rows = [asdict(c) for c in candidates]
    if namespace == WP1_NAMESPACE:
        return fingerprint(rows, ascii_only=True)
    if namespace == FINAL_NAMESPACE:
        return fingerprint({"schema": FINAL_NAMESPACE, "rows": rows})
    raise ValueError("Unknown candidate hash namespace")


def verify_final_payloads(candidate_payload, prompt_payload):
    """Verify supplied payloads against Issue #8, without granting approval.

    Digests cover canonical JSON payloads, not file whitespace or newline bytes.
    This function cannot obtain, reconstruct, or approve the missing payloads.
    """
    if set(candidate_payload) != {"schema", "rows"} or candidate_payload["schema"] != FINAL_NAMESPACE:
        raise ValueError("Unexpected final candidate payload schema")
    if set(prompt_payload) != {"policy", "rows"}:
        raise ValueError("Unexpected prompt payload schema")
    candidates = tuple(Candidate(**r) for r in candidate_payload["rows"])
    if len(candidates) != 333 or tuple(c.composition_id for c in candidates) != tuple(range(333)):
        raise ValueError("Final payload requires 333 original ascending global IDs")
    actual = (fingerprint(candidate_payload), fingerprint(prompt_payload))
    if actual != (FINAL_CANDIDATE_SHA256, FINAL_PROMPT_SHA256):
        raise ValueError(f"Issue #8 final hash mismatch: candidate={actual[0]}, prompt={actual[1]}")
    return {"candidate_sha256": actual[0], "prompt_sha256": actual[1],
            "verification": "payload_hashes_match_issue_8", "approval_granted": False}


@dataclass(frozen=True)
class CandidateSpec:
    protocol_id: str
    protocol_version: str
    task_mode: str
    candidates: tuple[Candidate, ...]
    candidate_sha256: str
    candidate_hash_namespace: str
    source_sha256: str
    split_sha256: str
    split_hash_namespace: str
    semantic_source: str
    healthy_disease_ids: tuple[int, ...]
    synthetic: bool = False
    approval_reference: str = ""
    final_prompt_json: str = ""
    tie_rule: str = TIE_RULE

    def __post_init__(self):
        object.__setattr__(self, "candidates", tuple(self.candidates))
        object.__setattr__(self, "healthy_disease_ids", _ids(self.healthy_disease_ids))
        for key in ("protocol_id", "protocol_version", "split_hash_namespace", "semantic_source"):
            _text(getattr(self, key), key)
        if not isinstance(self.approval_reference, str) or not isinstance(self.final_prompt_json, str):
            raise ValueError("Optional approval and prompt provenance must be immutable strings")
        if type(self.synthetic) is not bool or self.task_mode not in ("development", "final"):
            raise ValueError("Declare task_mode and a boolean synthetic flag")
        if self.tie_rule != TIE_RULE or not self.candidates or any(type(c) is not Candidate for c in self.candidates):
            raise ValueError("Require candidate records and the shared tie rule")
        ids = self.ids
        if ids != tuple(sorted(set(ids))):
            raise ValueError("Candidates require unique ascending original global IDs")
        if len({(c.crop_id, c.disease_id) for c in self.candidates}) != len(ids):
            raise ValueError("Duplicate candidate crop/disease pairs")
        for id_key, name_key in (("crop_id", "crop_name"), ("disease_id", "disease_name"),
                                 ("composition_id", "composition_name")):
            forward, reverse = {}, {}
            for c in self.candidates:
                i, n = getattr(c, id_key), getattr(c, name_key)
                if forward.setdefault(i, n) != n or reverse.setdefault(n, i) != i:
                    raise ValueError("Inconsistent component/name mapping")
        disease_ids = {c.disease_id for c in self.candidates}
        if len(set(self.healthy_disease_ids)) != len(self.healthy_disease_ids) or not set(self.healthy_disease_ids) <= disease_ids:
            raise ValueError("Healthy membership must be an explicit subset of disease IDs")
        for key in ("candidate_sha256", "source_sha256", "split_sha256"):
            _hash(getattr(self, key), key)
        if self.candidate_sha256 != candidate_digest(self.candidates, self.candidate_hash_namespace):
            raise ValueError("Candidate fingerprint mismatch")
        if not self.synthetic and self.task_mode == "development" and len(ids) != 323:
            raise ValueError("Real development requires 323 candidates")
        if self.task_mode == "final":
            _text(self.approval_reference, "explicit final policy approval_reference")
            if self.candidate_hash_namespace != FINAL_NAMESPACE:
                raise ValueError("Final candidate hash namespace required")
            if not self.synthetic:
                if not self.final_prompt_json:
                    raise ValueError("Exact final prompt payload is required")
                verify_final_payloads({"schema": FINAL_NAMESPACE, "rows": [asdict(c) for c in self.candidates]},
                                      json.loads(self.final_prompt_json))

    @property
    def ids(self):
        return tuple(c.composition_id for c in self.candidates)

    @property
    def sha256(self):
        return fingerprint(self.to_dict())

    def to_dict(self):
        return {**asdict(self), "candidate_count": len(self.candidates), "schema": "wp5-candidate-spec-v1"}

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        if data.pop("schema") != "wp5-candidate-spec-v1":
            raise ValueError("Unknown candidate specification schema")
        count = data.pop("candidate_count")
        data["candidates"] = tuple(Candidate(**r) for r in data["candidates"])
        result = cls(**data)
        if type(count) is not int or count != len(result.candidates):
            raise ValueError("Incorrect candidate count")
        return result


@dataclass(frozen=True)
class Sample:
    sample_id: str
    fold: str
    truth_id: int

    def __post_init__(self):
        _text(self.sample_id, "sample_id / original filename")
        if self.fold not in ("fitting", *FOLDS):
            raise ValueError("Unknown fold")
        object.__setattr__(self, "truth_id", _id(self.truth_id))


@dataclass(frozen=True)
class EvaluationManifest:
    spec: CandidateSpec
    samples: tuple[Sample, ...]
    seen_truth_ids: tuple[int, ...]
    unseen_truth_ids: tuple[int, ...]
    expected_seen: int
    expected_unseen: int
    identity_source: str

    def __post_init__(self):
        object.__setattr__(self, "samples", tuple(self.samples))
        if type(self.spec) is not CandidateSpec or any(type(s) is not Sample for s in self.samples):
            raise ValueError("Require immutable candidate and sample records")
        for key in ("seen_truth_ids", "unseen_truth_ids"):
            values = _ids(getattr(self, key))
            object.__setattr__(self, key, values)
            if not values or values != tuple(sorted(set(values))):
                raise ValueError("Truth axes require unique ascending IDs")
        _text(self.identity_source, "approved identity_source")
        axes = [set(self.seen_truth_ids), set(self.unseen_truth_ids)]
        if axes[0] & axes[1] or axes[0] | axes[1] != set(self.spec.ids):
            raise ValueError("Truth memberships must partition the candidate universe")
        if not self.samples or len({s.sample_id for s in self.samples}) != len(self.samples):
            raise ValueError("Duplicate or missing manifest sample identities")
        counts = Counter(s.fold for s in self.samples)
        for fold, expected in (("seen", self.expected_seen), ("unseen", self.expected_unseen)):
            if type(expected) is not int or expected <= 0 or counts[fold] != expected:
                raise ValueError(f"Empty fold or incorrect expected {fold} denominator")
        if not counts["fitting"]:
            raise ValueError("Fitting identities required for training-support diagnostics")
        for s in self.samples:
            allowed = set(self.spec.ids) if s.fold == "fitting" else set(getattr(self, s.fold + "_truth_ids"))
            if s.truth_id not in allowed:
                raise ValueError("Truth absent from approved fold/candidate membership")
        fitting_ids = {s.truth_id for s in self.samples if s.fold == "fitting"}
        if self.spec.task_mode == "development" and fitting_ids != axes[0]:
            raise ValueError("Development seen membership must equal fitting compositions")
        if not self.spec.synthetic:
            expected = (1238, 1894) if self.spec.task_mode == "development" else (2000, 1448)
            if (self.expected_seen, self.expected_unseen) != expected:
                raise ValueError("Real protocol denominators differ from frozen task")
            if counts["fitting"] != 4868:
                raise ValueError("Require the original 4,868 fitting identities")
            if self.spec.task_mode == "final" and (len(axes[0]), len(axes[1])) != (323, 10):
                raise ValueError("Final official truth axes require 323 seen / 10 unseen classes")
            if self.spec.task_mode == "final" and self.unseen_truth_ids != FINAL_UNSEEN_IDS:
                raise ValueError("Final unseen truth membership differs from the verified class-only table")

    @property
    def sha256(self):
        return fingerprint(asdict(self))

    def to_dict(self):
        return {**asdict(self), "spec": self.spec.to_dict(), "schema": "wp5-identity-manifest-v1"}

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        if data.pop("schema") != "wp5-identity-manifest-v1":
            raise ValueError("Unknown identity manifest schema")
        data["spec"] = CandidateSpec.from_dict(data["spec"])
        data["samples"] = tuple(Sample(**s) for s in data["samples"])
        return cls(**data)


def development_manifest(protocol, source_csv, *, healthy_disease_names):
    """Only this adapter reads a file: the explicitly supplied training CSV.

    WP1 validates source bytes, candidates, and exact ordered filename folds.
    No default filename, official-test path, or inferred reconciliation.
    """
    if Path(source_csv).name != "DPD_Small_train.csv":
        raise ValueError("Development ingress only accepts the course training-source filename")
    from shared_protocol import validate_protocol
    fitting, seen, unseen = validate_protocol(protocol, source_csv)
    if protocol.get("historical_reconciliation") != "ordered_filenames_match_supplied_reference":
        raise ValueError("Reconciled ordered-filename provenance required")
    if protocol.get("seed") != 42 or protocol["counts"] != [4868, 1238, 1894]:
        raise ValueError("Expected seed-42 development protocol")
    candidates = tuple(Candidate(**r) for r in protocol["candidates"])
    names = set(healthy_disease_names)
    if not names <= {c.disease_name for c in candidates}:
        raise ValueError("Unknown healthy disease source label")
    spec = CandidateSpec(protocol_id=protocol["schema"], protocol_version="1", task_mode="development",
                         candidates=candidates, candidate_sha256=protocol["candidate_sha256"],
                         candidate_hash_namespace=WP1_NAMESPACE, source_sha256=protocol["source_sha256"],
                         split_sha256=protocol["split_sha256"], split_hash_namespace=protocol["split_hash_namespace"],
                         semantic_source="WP1 validated training-source candidate table",
                         healthy_disease_ids=tuple(sorted({c.disease_id for c in candidates if c.disease_name in names})))
    samples = tuple(Sample(str(r.image_name), fold, int(r.composition_id))
                    for fold, frame in (("fitting", fitting), ("seen", seen), ("unseen", unseen))
                    for r in frame.itertuples())
    return EvaluationManifest(spec, samples, tuple(sorted(set(fitting.composition_id))),
                              tuple(sorted(set(unseen.composition_id))), 1238, 1894,
                              "WP1 source hash + validated reconciled ordered filenames")


@dataclass(frozen=True)
class RunProvenance:
    model_id: str
    run_id: str
    checkpoint_id: str
    checkpoint_sha256: str
    preprocessing: str
    code_version: str
    environment: str
    score_semantics: str
    score_precision: str
    comparison_budget: str
    prompt_sha256: str = ""
    metric_policy: str = METRIC_POLICY

    def __post_init__(self):
        if not isinstance(self.prompt_sha256, str):
            raise ValueError("prompt_sha256 must be an immutable string")
        for key, value in asdict(self).items():
            if key != "prompt_sha256":
                _text(value, key)
        _hash(self.checkpoint_sha256, "checkpoint_sha256")
        if self.prompt_sha256:
            _hash(self.prompt_sha256, "prompt_sha256")
        if self.score_precision not in ("float32", "float64") or self.metric_policy != METRIC_POLICY:
            raise ValueError("Unsupported score precision or metric semantics")


@dataclass(frozen=True)
class PredictionRecord:
    sample: Sample
    spec: CandidateSpec
    provenance: RunProvenance
    candidate_ids: tuple[int, ...]
    available_k: int
    scores: tuple[float, ...] = ()
    ranked_ids: tuple[int, ...] = ()
    independent_crop_id: int | None = None
    independent_disease_id: int | None = None

    def __post_init__(self):
        if (type(self.sample) is not Sample or type(self.spec) is not CandidateSpec
                or type(self.provenance) is not RunProvenance):
            raise ValueError("Require immutable sample, candidate and run provenance records")
        object.__setattr__(self, "candidate_ids", _ids(self.candidate_ids))
        object.__setattr__(self, "ranked_ids", _ids(self.ranked_ids))
        object.__setattr__(self, "scores", tuple(self.scores))
        if self.sample.fold not in FOLDS or self.sample.truth_id not in self.spec.ids:
            raise ValueError("Invalid prediction fold/truth")
        if self.candidate_ids != self.spec.ids:
            raise ValueError("Score-column alignment differs from candidate specification")
        if type(self.available_k) is not int or not min(5, len(self.spec.ids)) <= self.available_k <= len(self.spec.ids):
            raise ValueError("available_k must cover Top-5 (or all tiny-universe candidates)")
        if bool(self.scores) == bool(self.ranked_ids):
            raise ValueError("Supply exactly one of canonical scores or ranked global IDs")
        if self.scores:
            if len(self.scores) != len(self.spec.ids) or self.available_k != len(self.spec.ids):
                raise ValueError("Scores must cover the full declared candidate universe")
            dtype = np.dtype(self.provenance.score_precision).type
            # Overflow/underflow in the probe must reject, never warn or alter scores.
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                object.__setattr__(self, "scores", tuple(_exact_score(s, dtype) for s in self.scores))
        elif (len(self.ranked_ids) != self.available_k or len(set(self.ranked_ids)) != len(self.ranked_ids)
              or not set(self.ranked_ids) <= set(self.spec.ids)):
            raise ValueError("Ranking must contain available_k unique allowed global IDs")
        for key, allowed in (("independent_crop_id", {c.crop_id for c in self.spec.candidates}),
                             ("independent_disease_id", {c.disease_id for c in self.spec.candidates})):
            value = getattr(self, key)
            if value is not None:
                value = _id(value)
                if value not in allowed:
                    raise ValueError("Independent component prediction outside known vocabulary")
                object.__setattr__(self, key, value)

    def ranking(self):
        if self.ranked_ids:
            return self.ranked_ids
        order = np.argsort(-np.asarray(self.scores, dtype=np.float64), kind="stable")
        ids = self.spec.ids
        return tuple(ids[i] for i in order)

    def to_dict(self):
        return {"schema": "wp5-prediction-v1", **asdict(self), "spec": self.spec.to_dict()}

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        if data.pop("schema") != "wp5-prediction-v1":
            raise ValueError("Unknown prediction schema")
        data["sample"] = Sample(**data["sample"])
        data["spec"] = CandidateSpec.from_dict(data["spec"])
        data["provenance"] = RunProvenance(**data["provenance"])
        return cls(**data)


def records_from_scores(spec, provenance, samples, scores, candidate_ids):
    """Copy a canonical [N,K] numpy/torch float matrix into immutable records."""
    if hasattr(scores, "detach"):
        scores = scores.detach().cpu().numpy()
    arr = np.asarray(scores)
    samples = tuple(samples)
    if arr.dtype.name != provenance.score_precision:
        raise ValueError("Score matrix dtype differs from declared canonical precision")
    if arr.shape != (len(samples), len(spec.ids)) or not samples:
        raise ValueError("Scores require nonempty [N,K] aligned samples and columns")
    columns = _ids(candidate_ids)
    return tuple(PredictionRecord(s, spec, provenance, columns, len(columns), scores=tuple(row))
                 for s, row in zip(samples, arr))


def _validated_records(manifest, records):
    records = tuple(records)
    if not records:
        raise ValueError("Empty prediction folds are unavailable, not zero accuracy")
    expected = {s.sample_id: s for s in manifest.samples if s.fold in FOLDS}
    actual = {}
    provenance = records[0].provenance
    for record in records:
        sid = record.sample.sample_id
        if sid in actual:
            raise ValueError(f"Duplicate prediction identity: {sid}")
        if record.spec != manifest.spec or record.provenance != provenance:
            raise ValueError("Mismatched candidate or run provenance within predictions")
        if (manifest.spec.task_mode == "final" and not manifest.spec.synthetic
                and provenance.prompt_sha256 and provenance.prompt_sha256 != FINAL_PROMPT_SHA256):
            raise ValueError("Prediction prompts differ from the frozen final prompt table")
        if expected.get(sid) != record.sample:
            raise ValueError(f"Unexpected identity, truth, or fold: {sid}")
        actual[sid] = record
    missing = sorted(set(expected) - set(actual))
    if missing:
        raise ValueError(f"Missing {len(missing)} expected prediction rows: {missing[:5]}")
    return tuple(actual[sid] for sid in expected)


def _hits(records):
    ranks = [r.ranking()[:min(5, len(r.spec.ids))] for r in records]
    return [(r.sample.truth_id == rank[0], r.sample.truth_id in rank) for r, rank in zip(records, ranks)]


def _summary(records, *, expected=None):
    hits = _hits(records)
    n = len(hits)
    c1, c5 = sum(h[0] for h in hits), sum(h[1] for h in hits)
    return {"count": n, "processed": n, "expected": n if expected is None else expected,
            "correct_top1": c1, "correct_top5": c5, "top1": c1 / n if n else None,
            "top5": c5 / n if n else None}


def confusion_matrix(records, truth_ids, prediction_ids, *, candidates):
    """Rows are true classes; columns are allowed predictions, independently."""
    truth_ids, prediction_ids = _ids(truth_ids), _ids(prediction_ids)
    labels = {c.composition_id: c.composition_name for c in candidates}
    for axis in (truth_ids, prediction_ids):
        if not axis or len(set(axis)) != len(axis) or not set(axis) <= set(labels):
            raise ValueError("Confusion axes require unique known IDs")
    rows, cols = {v: i for i, v in enumerate(truth_ids)}, {v: i for i, v in enumerate(prediction_ids)}
    counts = [[0] * len(cols) for _ in rows]
    for r in records:
        truth, pred = r.sample.truth_id, r.ranking()[0]
        if truth not in rows or pred not in cols:
            raise ValueError("Confusion axes omit a truth or prediction")
        counts[rows[truth]][cols[pred]] += 1
    support = [sum(row) for row in counts]
    return {"truth_axis": "ground_truth_composition_id", "prediction_axis": "predicted_composition_id",
            "truth_ids": list(truth_ids), "prediction_ids": list(prediction_ids),
            "truth_labels": [labels[i] for i in truth_ids], "prediction_labels": [labels[i] for i in prediction_ids],
            "counts": counts, "truth_support": support,
            "row_normalized": [[v / n if n else None for v in row] for row, n in zip(counts, support)],
            "count": sum(support),
            "diagonal_hits": sum(counts[rows[i]][cols[i]] for i in truth_ids if i in cols)}


def _diagnostics(manifest, records):
    spec = manifest.spec
    by_id = {c.composition_id: c for c in spec.candidates}
    fitting = [by_id[s.truth_id] for s in manifest.samples if s.fold == "fitting"]
    supports = {key: Counter(getattr(c, key) for c in fitting)
                for key in ("crop_id", "disease_id", "composition_id")}
    healthy = set(spec.healthy_disease_ids)
    results = {}
    for key in ("crop_id", "disease_id", "composition_id"):
        groups = []
        name_key = key.replace("_id", "_name")
        for value in sorted({getattr(c, key) for c in spec.candidates}):
            subset = [r for r in records if getattr(by_id[r.sample.truth_id], key) == value]
            name = next(getattr(c, name_key) for c in spec.candidates if getattr(c, key) == value)
            groups.append({key: value, "name": name, **_summary(subset), "fitting_support": supports[key][value],
                           "low_support": len(subset) < 5, "low_support_threshold": 5})
        results["per_" + key.removesuffix("_id")] = groups
    results["health"] = {}
    for group, is_healthy in (("healthy", True), ("diseased", False)):
        subset = [r for r in records if (by_id[r.sample.truth_id].disease_id in healthy) == is_healthy]
        predicted = sum(by_id[r.ranking()[0]].disease_id in healthy for r in subset)
        results["health"][group] = {**_summary(subset), "predicted_healthy_count": predicted,
                                   "predicted_healthy_denominator": len(subset),
                                   "predicted_healthy_frequency": predicted / len(subset) if subset else None,
                                   "false_healthy_error_count": predicted if not is_healthy else 0}
    transfers = []
    for comp_id in sorted(by_id):
        subset = [r for r in records if r.sample.truth_id == comp_id]
        truth = by_id[comp_id]
        errors = Counter()
        for r in subset:
            pred = by_id[r.ranking()[0]]
            crop_right, disease_right = pred.crop_id == truth.crop_id, pred.disease_id == truth.disease_id
            errors[("both_right" if crop_right and disease_right else "crop_right_disease_wrong" if crop_right
                    else "disease_right_crop_wrong" if disease_right else "both_wrong")] += 1
            errors["predicted_fitting_composition"] += supports["composition_id"][pred.composition_id] > 0
            errors["predicted_healthy"] += pred.disease_id in healthy
        transfers.append({"composition_id": comp_id, **_summary(subset),
                          **{key: errors[key] for key in ("both_right", "crop_right_disease_wrong",
                              "disease_right_crop_wrong", "both_wrong", "predicted_fitting_composition", "predicted_healthy")},
                          "fitting_crop_support": supports["crop_id"][truth.crop_id],
                          "fitting_disease_support": supports["disease_id"][truth.disease_id],
                          "fitting_composition_support": supports["composition_id"][comp_id],
                          "predicted_fitting_composition_frequency": errors["predicted_fitting_composition"] / len(subset) if subset else None,
                          "predicted_healthy_frequency": errors["predicted_healthy"] / len(subset) if subset else None,
                          "true_healthy": truth.disease_id in healthy,
                          "low_support": len(subset) < 5, "low_support_threshold": 5})
    results["composition_component_consistency"] = transfers
    independent = {}
    for key, attr in (("crop", "independent_crop_id"), ("disease", "independent_disease_id")):
        provided = [r for r in records if getattr(r, attr) is not None]
        correct = sum(getattr(r, attr) == getattr(by_id[r.sample.truth_id], key + "_id") for r in provided)
        independent[key] = {"available": len(provided), "expected": len(records), "correct": correct,
                            "accuracy": correct / len(provided) if provided else None,
                            "meaning": "independent head prediction; provided rows only"}
    results["independent_component_accuracy"] = independent
    return results


def build_report(manifest, records):
    records = _validated_records(manifest, records)
    folds = {}
    for fold in FOLDS:
        subset = tuple(r for r in records if r.sample.fold == fold)
        folds[fold] = {**_summary(subset, expected=getattr(manifest, "expected_" + fold)),
                       **_diagnostics(manifest, subset),
                       "confusion_matrix": confusion_matrix(subset, getattr(manifest, fold + "_truth_ids"),
                                                            manifest.spec.ids, candidates=manifest.spec.candidates)}
    return {"schema": "wp5-report-v1", "candidate_spec": manifest.spec.to_dict(),
            "manifest_sha256": manifest.sha256, "identity_source": manifest.identity_source,
            "provenance": asdict(records[0].provenance), "accuracy_scale": "fraction_0_to_1",
            "missing_data_policy": "reject_any_missing_duplicate_or_unexpected_identity",
            "diagnostic_semantics": "composition hit rates grouped by truth; selected-composition component consistency",
            "low_support_policy": "flag support < 5; no significance or causal claims",
            **folds, "combined_micro": _summary(records),
            "harmonic_mean_top1": harmonic_mean(folds["seen"]["top1"], folds["unseen"]["top1"]),
            "harmonic_mean_top5": harmonic_mean(folds["seen"]["top5"], folds["unseen"]["top5"])}


def compare_models(manifest, records_a, records_b):
    """Strict full-manifest pairing; model-specific canonical scorers may differ."""
    a, b = _validated_records(manifest, records_a), _validated_records(manifest, records_b)
    if a[0].provenance.comparison_budget != b[0].provenance.comparison_budget:
        raise ValueError("Incompatible declared comparison budgets")
    reports = build_report(manifest, a), build_report(manifest, b)
    paired = []
    for fold in FOLDS:
        for comp_id in (None, *manifest.spec.ids):
            pairs = [(x, y) for x, y in zip(a, b) if x.sample.fold == fold
                     and (comp_id is None or x.sample.truth_id == comp_id)]
            for k in (1, 5):
                cells = Counter()
                for x, y in pairs:
                    ca = x.sample.truth_id in x.ranking()[:min(k, len(manifest.spec.ids))]
                    cb = y.sample.truth_id in y.ranking()[:min(k, len(manifest.spec.ids))]
                    cells["both_correct" if ca and cb else "a_only" if ca else "b_only" if cb else "neither"] += 1
                paired.append({"fold": fold, "composition_id": comp_id, "k": k, "count": len(pairs),
                               **{key: cells[key] for key in ("both_correct", "a_only", "b_only", "neither")},
                               "b_minus_a_hits": cells["b_only"] - cells["a_only"],
                               "b_minus_a_accuracy": (cells["b_only"] - cells["a_only"]) / len(pairs) if pairs else None})
    return {"schema": "wp5-paired-report-v1", "candidate_spec": manifest.spec.to_dict(),
            "manifest_sha256": manifest.sha256, "accuracy_scale": "fraction_0_to_1",
            "alignment": "exact original sample identity, truth, fold, task and candidate specification",
            "missing_data_policy": "strict_full_manifest_no_intersection",
            "a": asdict(a[0].provenance), "b": asdict(b[0].provenance), "paired": paired,
            "a_metrics": {fold: _summary([r for r in a if r.sample.fold == fold]) for fold in FOLDS},
            "b_metrics": {fold: _summary([r for r in b if r.sample.fold == fold]) for fold in FOLDS},
            "health_comparison": [{"fold": fold, "truth_group": group,
                "count": reports[0][fold]["health"][group]["count"],
                **{model + "_" + key: reports[i][fold]["health"][group][key]
                   for i, model in enumerate(("a", "b"))
                   for key in ("correct_top1", "correct_top5", "top1", "top5",
                               "predicted_healthy_count", "predicted_healthy_frequency", "false_healthy_error_count")}}
                for fold in FOLDS for group in ("healthy", "diseased")],
            **{key + "_b_minus_a": reports[1][key] - reports[0][key]
               for key in ("harmonic_mean_top1", "harmonic_mean_top5")}}


def export_report(report, output_directory):
    """Write JSON + flat CSV tables to a NEW directory; preserve all old outputs."""
    encoded = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=False)
    (output / "report.json").write_text(encoded, encoding="utf-8")

    def table(name, rows):
        rows = list(rows)
        if not rows:
            return
        keys = list(dict.fromkeys(key for row in rows for key in row))
        with (output / name).open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)

    if report["schema"] == "wp5-paired-report-v1":
        table("paired.csv", report["paired"])
        table("health_comparison.csv", report["health_comparison"])
        table("harmonic_deltas.csv", ({"metric": key, "b_minus_a": report[key + "_b_minus_a"]}
                                      for key in ("harmonic_mean_top1", "harmonic_mean_top5")))
        return output
    table("harmonic_means.csv", ({"metric": key, "accuracy": report[key]}
                                 for key in ("harmonic_mean_top1", "harmonic_mean_top5")))
    table("summary.csv", ({"fold": fold, **{k: report[fold][k] for k in _summary(())}}
                           for fold in (*FOLDS, "combined_micro")))
    for key in ("per_crop", "per_disease", "per_composition", "composition_component_consistency"):
        table(key + ".csv", ({"fold": fold, **row} for fold in FOLDS for row in report[fold][key]))
    table("health.csv", ({"fold": fold, "truth_group": group, **row}
                         for fold in FOLDS for group, row in report[fold]["health"].items()))
    table("independent_components.csv", ({"fold": fold, "component": key, **row}
          for fold in FOLDS for key, row in report[fold]["independent_component_accuracy"].items()))
    for fold in FOLDS:
        matrix = report[fold]["confusion_matrix"]
        table(f"{fold}_confusion.csv", ({"truth_id": truth, "truth_label": matrix["truth_labels"][i],
              "prediction_id": pred, "prediction_label": matrix["prediction_labels"][j],
              "count": matrix["counts"][i][j], "truth_support": matrix["truth_support"][i],
              "row_normalized": matrix["row_normalized"][i][j]}
              for i, truth in enumerate(matrix["truth_ids"]) for j, pred in enumerate(matrix["prediction_ids"])))
    return output
