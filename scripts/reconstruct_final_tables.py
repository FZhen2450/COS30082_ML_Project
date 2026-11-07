"""Reconstruct class-only final payloads; verify exact Issue #8 hashes.

Reads course TRAINING metadata and only csv/default/train.csv in the authors'
public metadata archive. Never opens another ZIP member, images, or weights.
Outputs are prepared for review; matching hashes do not grant final approval.
"""
import argparse
import csv
import hashlib
import io
import json
import re
import sys
import unicodedata
import urllib.request
import zipfile
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.reporting import (Candidate, FINAL_NAMESPACE, canonical_json,
                                  fingerprint, verify_final_payloads)

ARCHIVE_URL = "https://zenodo.org/api/records/16879271/files/csv.zip/content"
ARCHIVE_MD5 = "611eed2050897671c5e0e2a65aa3057e"
TRAIN_MEMBER = "csv/default/train.csv"
COURSE_SOURCE_SHA256 = "9f1b729ab08f120c9e8d3a5681aa1ccbdf1d9210f9aaa6143959363dcb106a7d"
DEVELOPMENT_PROMPT_SHA256 = "753672c7964bcb20a1346145f21a72b40a19f55590a006b5783823f23c13f537"
EXTRA_IDS = (39, 84, 107, 124, 133, 136, 144, 166, 276, 303)
WP3_REF = "300f1d78bb962f67fa2dbdb7b76ed23301abd7df"
WP3_PATH = "notebooks/wp3_clip_zero_shot.ipynb"
PROMPT_POLICY = {
    "version": "label_only_single_template_v2",
    "normalization": "Unicode NFKC; underscores to spaces; collapse whitespace; lowercase; no aliases/symptoms",
    "healthy": "a photo of a healthy {crop} plant",
    "diseased": "a photo of {article} {crop} plant affected by {disease}",
    "article_rule": "an for normalized crop labels starting with a/e/i/o/u; otherwise a; diseased prompts only; verified for the current 55 crop labels",
}


def class_projection(stream):
    """Discard filenames immediately; retain only consistent six-field classes."""
    classes = {}
    row_count = 0
    for row in csv.reader(stream):
        if len(row) != 7 or any(not re.fullmatch(r"[0-9]+", v) for v in row[1:4]):
            raise ValueError("Expected the original seven-column DPD metadata schema")
        crop_id, disease_id, comp_id = map(int, row[1:4])
        candidate = Candidate(comp_id, crop_id, disease_id, *row[4:7])
        if classes.setdefault(comp_id, candidate) != candidate:
            raise ValueError("Inconsistent class mapping within source")
        row_count += 1
    candidates = tuple(classes[key] for key in sorted(classes))
    if len({(c.crop_id, c.disease_id) for c in candidates}) != len(candidates):
        raise ValueError("Duplicated class pairs")
    for id_key, name_key in (("crop_id", "crop_name"), ("disease_id", "disease_name")):
        mapping, reverse = {}, {}
        for candidate in candidates:
            i, name = getattr(candidate, id_key), getattr(candidate, name_key)
            if mapping.setdefault(i, name) != name or reverse.setdefault(name, i) != i:
                raise ValueError("Inconsistent component name mapping")
    return candidates, row_count


def normalize_label(label):
    value = " ".join(unicodedata.normalize("NFKC", label).replace("_", " ").split()).lower()
    if not value:
        raise ValueError("Empty normalized prompt label")
    return value


def prompt_payload(candidates):
    """Exact published WP3A v2 field layout, labels, article rule, and strings."""
    rows = []
    for column, c in enumerate(candidates):
        crop, disease = normalize_label(c.crop_name), normalize_label(c.disease_name)
        prompt = (PROMPT_POLICY["healthy"].format(crop=crop) if disease == "healthy" else
                  PROMPT_POLICY["diseased"].format(article="an" if crop[0] in "aeiou" else "a",
                                                   crop=crop, disease=disease))
        rows.append({"score_column": column, "global_composition_id": c.composition_id,
                     "crop_id": c.crop_id, "disease_id": c.disease_id, "crop": crop,
                     "disease": disease, "prompt": prompt})
    if len({r["prompt"] for r in rows}) != len(rows):
        raise ValueError("Duplicate normalized prompts")
    return {"policy": dict(PROMPT_POLICY), "rows": rows}


def reconstruct(course_bytes, archive_bytes):
    if hashlib.sha256(course_bytes).hexdigest() != COURSE_SOURCE_SHA256:
        raise ValueError("Course training source bytes differ from accepted CRLF source")
    if hashlib.md5(archive_bytes).hexdigest() != ARCHIVE_MD5:
        raise ValueError("Public archive checksum differs from Zenodo record")
    course, course_count = class_projection(io.StringIO(course_bytes.decode("utf-8")))
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        with archive.open(TRAIN_MEMBER) as member:
            public, public_count = class_projection(io.TextIOWrapper(member, encoding="utf-8"))
    by_id = {c.composition_id: c for c in public}
    if course_count != 8000 or len(course) != 323 or len(public) != 333:
        raise ValueError("Unexpected course/public class counts")
    if tuple(c.composition_id for c in public) != tuple(range(333)):
        raise ValueError("Public original global IDs are not the frozen ordered universe")
    if any(by_id.get(c.composition_id) != c for c in course):
        raise ValueError("Public class-only mapping differs from the 323 course classes")
    extras = tuple(sorted(set(by_id) - {c.composition_id for c in course}))
    if extras != EXTRA_IDS:
        raise ValueError("Public extra classes differ from Issue #8")
    development_prompts = prompt_payload(course)
    if fingerprint(development_prompts) != DEVELOPMENT_PROMPT_SHA256:
        raise ValueError("Published WP3A-v2 development prompt hash did not reproduce")
    candidates = {"schema": FINAL_NAMESPACE, "rows": [asdict(c) for c in public]}
    prompts = prompt_payload(public)
    old_prompts = {r["global_composition_id"]: r["prompt"] for r in development_prompts["rows"]}
    if any(r["prompt"] != old_prompts[r["global_composition_id"]] for r in prompts["rows"]
           if r["global_composition_id"] in old_prompts):
        raise ValueError("A development prompt string changed")
    try:
        verification = verify_final_payloads(candidates, prompts)
    except ValueError as exc:
        verification = {"verification": "UNVERIFIED_ALTERNATIVE", "reason": str(exc), "approval_granted": False,
                        "candidate_sha256": fingerprint(candidates), "prompt_sha256": fingerprint(prompts)}
    provenance = {**verification, "course_source_sha256": COURSE_SOURCE_SHA256,
                  "public_record": "https://zenodo.org/records/16879271", "archive_url": ARCHIVE_URL,
                  "archive_md5": ARCHIVE_MD5, "archive_sha256": hashlib.sha256(archive_bytes).hexdigest(),
                  "only_member_opened": TRAIN_MEMBER, "public_training_rows": public_count,
                  "course_training_rows": course_count, "matched_course_classes": len(course),
                  "candidate_count": len(public), "additional_ids": list(extras),
                  "wp3_notebook_ref": WP3_REF, "wp3_notebook_path": WP3_PATH,
                  "development_prompt_sha256": fingerprint(development_prompts),
                  "preserved_development_prompt_strings": len(old_prompts),
                  "hash_serialization": "UTF-8 compact sorted-key JSON; ensure_ascii=False; no trailing LF",
                  "official_test_access": False, "models_loaded": False,
                  "remaining": "explicit policy approval, frozen model inputs, approved truth manifests, and inference authorization"}
    return candidates, prompts, provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("DPD_Small_train.csv"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a new output directory; preserving existing files")
    if args.source.name != "DPD_Small_train.csv":
        raise ValueError("Only the course training-source filename is accepted")
    course_bytes = args.source.read_bytes()
    with urllib.request.urlopen(ARCHIVE_URL, timeout=60) as response:
        archive_bytes = response.read()
    candidates, prompts, provenance = reconstruct(course_bytes, archive_bytes)
    args.output.mkdir(parents=True, exist_ok=False)
    prefix = "verified_reconstruction" if provenance["verification"] == "payload_hashes_match_issue_8" else "UNVERIFIED_ALTERNATIVE"
    for name, payload in (("candidates", candidates), ("prompts", prompts)):
        with (args.output / f"{prefix}_{name}.json").open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(canonical_json(payload))
    with (args.output / "reconstruction_provenance.json").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2))
    return 0 if prefix == "verified_reconstruction" else 1


if __name__ == "__main__":
    raise SystemExit(main())
