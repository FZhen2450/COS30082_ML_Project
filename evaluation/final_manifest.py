"""Final-333 identity manifest and blind inference order. Authorised use only.

``final_manifest`` is the only WP5 function that reads official metadata: the two
official test CSVs, whose raw SHA-256 values must be supplied by their owner.
It never opens images. Errors report counts and fold names only, never
official filenames or labels. Fitting identities come from the already
validated development manifest, because every frozen model was trained on that
4,868-image fitting fold.

``inference_order`` derives a truth-free list for model owners. Owners need
only ordered filenames and digests to export packets, so official labels are
not distributed before inference.

Protocol status: one shared 333-class candidate universe for both official folds
is the team-proposed protocol recorded in Issue #8, pending explicit lecturer
guidance. It is not encoded here as a lecturer requirement. ``approval_reference``
must point to the recorded team decision (and any lecturer confirmation). A
different confirmed policy needs a new reviewed change, not a flag.
"""
from __future__ import annotations

import codecs
import csv
import hashlib
import io
import re
from collections.abc import Mapping
from pathlib import Path

from evaluation.reporting import (Candidate, CandidateSpec, EvaluationManifest, FINAL_NAMESPACE,
                                  FINAL_PROMPT_SHA256, Sample, candidate_digest, canonical_json, fingerprint)

OFFICIAL_CSV = {"seen": "DPD_Small_test_seen.csv", "unseen": "DPD_Small_test_unseen.csv"}
FINAL_PROTOCOL_ID = "cos30082-final-333-shared-candidates"
FINAL_SPLIT_NAMESPACE = "wp5-final-ordered-filenames-v1"
INFERENCE_ORDER_SCHEMA = "wp5-final-inference-order-v1"


def _sha(value, name):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{name} requires an owner-supplied lowercase SHA-256")
    return value


def _reference(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def official_samples(path, fold, expected_sha256, candidates_by_id):
    """Ordered (filename, truth) samples from one official CSV. Counts-only errors."""
    path = Path(path)
    if fold not in OFFICIAL_CSV or path.name != OFFICIAL_CSV[fold]:
        raise ValueError(f"Expected {OFFICIAL_CSV.get(fold, 'a known official CSV')} for the {fold} fold")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != _sha(expected_sha256, f"official {fold} CSV"):
        raise ValueError(f"Official {fold} metadata raw SHA-256 differs from the approved value")
    text = raw.decode("utf-8")
    bom = text.startswith(codecs.BOM_UTF8.decode("utf-8"))
    samples, malformed, inconsistent = [], 0, 0
    for row in csv.reader(io.StringIO(text[1:] if bom else text, newline="")):
        if not row:
            continue
        if len(row) != 7 or not row[0].strip() or any(not re.fullmatch(r"[0-9]+", v) for v in row[1:4]):
            malformed += 1
            continue
        crop_id, disease_id, composition_id = map(int, row[1:4])
        record = Candidate(composition_id, crop_id, disease_id, *row[4:7])
        if candidates_by_id.get(composition_id) != record:
            inconsistent += 1
            continue
        samples.append(Sample(row[0], fold, composition_id))
    if malformed or inconsistent:
        raise ValueError(f"Official {fold} metadata: {malformed} malformed rows (header or schema), "
                         f"{inconsistent} rows inconsistent with the frozen 333-class table")
    if len({s.sample_id for s in samples}) != len(samples):
        raise ValueError(f"Official {fold} metadata contains duplicate filenames")
    return tuple(samples), {"fold": fold, "file": path.name, "file_sha256": expected_sha256,
                            "rows": len(samples), "utf8_bom_stripped": bom}


def final_manifest(development, seen_csv, unseen_csv, *, candidate_payload, prompt_payload, expected_sha256,
                   approval_reference, evaluation_authorization, synthetic=False):
    """Build the final EvaluationManifest. Run once, only after authorisation.

    ``development`` is the validated development-323 EvaluationManifest whose
    fitting identities trained every frozen model. Seen truth axis: all of its
    323 course compositions. Unseen truth axis: the remaining final candidates
    (the 10 Issue #8 IDs for real data; EvaluationManifest enforces it).
    """
    if type(development) is not EvaluationManifest or development.spec.task_mode != "development":
        raise ValueError("Supply the validated development manifest (fitting identities)")
    if type(synthetic) is not bool or development.spec.synthetic != synthetic:
        raise ValueError("Synthetic flag must match the development manifest")
    _reference(approval_reference, "final policy approval_reference")
    _reference(evaluation_authorization, "official evaluation authorization")
    if not isinstance(expected_sha256, Mapping) or set(expected_sha256) != set(OFFICIAL_CSV):
        raise ValueError("Supply owner-approved raw SHA-256 values for exactly the seen and unseen CSVs")
    if not isinstance(candidate_payload, Mapping) or candidate_payload.get("schema") != FINAL_NAMESPACE:
        raise ValueError("Unexpected final candidate payload")
    candidates = tuple(Candidate(**row) for row in candidate_payload["rows"])
    by_id = {c.composition_id: c for c in candidates}
    if any(by_id.get(c.composition_id) != c for c in development.spec.candidates):
        raise ValueError("Development candidates differ from their rows in the final candidate table")
    if not synthetic and fingerprint(dict(prompt_payload)) != FINAL_PROMPT_SHA256:
        raise ValueError("Final prompt payload differs from Issue #8")
    seen, seen_source = official_samples(seen_csv, "seen", expected_sha256["seen"], by_id)
    unseen, unseen_source = official_samples(unseen_csv, "unseen", expected_sha256["unseen"], by_id)
    fitting = tuple(s for s in development.samples if s.fold == "fitting")
    samples = fitting + seen + unseen
    seen_axis = development.spec.ids
    unseen_axis = tuple(sorted(set(by_id) - set(seen_axis)))
    source = fingerprint({"training_source_sha256": development.spec.source_sha256,
                          "seen_csv_sha256": expected_sha256["seen"], "unseen_csv_sha256": expected_sha256["unseen"]})
    split = fingerprint({fold: [s.sample_id for s in samples if s.fold == fold]
                         for fold in ("fitting", "seen", "unseen")})
    spec = CandidateSpec(FINAL_PROTOCOL_ID, "1", "final", candidates, candidate_digest(candidates, FINAL_NAMESPACE),
                         FINAL_NAMESPACE, source, split, FINAL_SPLIT_NAMESPACE,
                         "Issue #8 verified class-only 333 table; official seen/unseen metadata",
                         development.spec.healthy_disease_ids, synthetic=synthetic,
                         approval_reference=approval_reference, final_prompt_json=canonical_json(dict(prompt_payload)))
    identity = canonical_json({"official_sources": [seen_source, unseen_source],
                               "fitting_identities_from_development_manifest": development.sha256,
                               "evaluation_authorization": evaluation_authorization})
    return EvaluationManifest(spec, samples, seen_axis, unseen_axis, len(seen), len(unseen), identity)


def inference_order(manifest):
    """Truth-free export order for owners: filenames and the digests packets must carry."""
    if type(manifest) is not EvaluationManifest or manifest.spec.task_mode != "final":
        raise ValueError("Inference orders are issued only for a final manifest")
    from evaluation.model_adapters import IDENTITY_MANIFEST
    return {"schema": INFERENCE_ORDER_SCHEMA, "manifest_sha256": manifest.sha256,
            "source_sha256": manifest.spec.source_sha256,
            "candidate_hash_namespace": FINAL_NAMESPACE, "candidate_sha256": manifest.spec.candidate_sha256,
            "candidate_ids": list(manifest.spec.ids),
            "split_hash_namespace": IDENTITY_MANIFEST, "split_sha256": manifest.sha256,
            "clip_prompt_sha256": FINAL_PROMPT_SHA256, "synthetic": manifest.spec.synthetic,
            "folds": {fold: [s.sample_id for s in manifest.samples if s.fold == fold] for fold in ("seen", "unseen")},
            "ground_truth": "withheld; WP5 attaches truths from the manifest at import"}


__all__ = ["OFFICIAL_CSV", "FINAL_PROTOCOL_ID", "FINAL_SPLIT_NAMESPACE", "INFERENCE_ORDER_SCHEMA",
           "official_samples", "final_manifest", "inference_order"]
