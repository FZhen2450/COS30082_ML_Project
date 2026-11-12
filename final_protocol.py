"""Full-training release contract, separate from the immutable development split."""
import hashlib
import json
from pathlib import Path

from data_utils import load_metadata
from evaluation.reporting import FINAL_NAMESPACE, fingerprint, verify_final_payloads
from shared_protocol import candidate_table

SCHEMA = "dpd-full8000-final333-v1"
SOURCE_SHA256 = "9f1b729ab08f120c9e8d3a5681aa1ccbdf1d9210f9aaa6143959363dcb106a7d"


def file_sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def build_final_protocol(source, candidate_payload, prompt_payload):
    verified = verify_final_payloads(candidate_payload, prompt_payload)
    frame = load_metadata(source)
    if file_sha256(source) != SOURCE_SHA256 or len(frame) != 8000 or frame.image_name.duplicated().any():
        raise ValueError("Expected the original 8,000 training-source records")
    candidates = candidate_payload["rows"]
    by_id = {row["composition_id"]: row for row in candidates}
    if any(by_id.get(row["composition_id"]) != row for row in candidate_table(frame)):
        raise ValueError("Final class semantics disagree with the original training source")
    if len(candidate_table(frame)) != 323:
        raise ValueError("Full training must retain the original 323 training compositions")
    train_images = frame.image_name.tolist()
    return {"schema": SCHEMA, "task_mode": "final", "training_scope": "full_8000",
            "authorization": "student_reported_Dr_Lee_Teams_2026-11-11",
            "source_sha256": SOURCE_SHA256, "candidate_hash_namespace": FINAL_NAMESPACE,
            "candidate_sha256": verified["candidate_sha256"],
            "prompt_sha256": verified["prompt_sha256"], "candidates": candidates,
            "candidate_payload": candidate_payload, "prompt_payload": prompt_payload,
            "training_images": train_images, "training_count": 8000,
            "training_manifest_sha256": fingerprint(train_images),
            "reference_scope": "full_8000_training_only",
            "tie_rule": "stable_column_order_ascending_global_id"}


def validate_final_protocol(protocol, source):
    if protocol.get("schema") != SCHEMA:
        raise ValueError("Unsupported full-training final protocol")
    expected = build_final_protocol(source, protocol["candidate_payload"], protocol["prompt_payload"])
    if protocol != expected:
        raise ValueError("Final protocol fields, ordered training identities or hashes differ")
    return load_metadata(source)


def load_final_protocol(path, source):
    protocol = json.loads(Path(path).read_text(encoding="utf-8"))
    return protocol, validate_final_protocol(protocol, source)


def write_new_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
