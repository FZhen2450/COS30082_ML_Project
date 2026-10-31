"""Training-source-only development contract for WP1/WP2/WP3/WP5.

Existing WP2/WP3 artifacts remain authoritative. A reconstructed split is not
declared historically identical until its ordered filenames are reconciled.
No official-test metadata or images are loaded by this module.
"""
import argparse
import hashlib
import json
from pathlib import Path

from data_utils import load_metadata, make_validation_split

SCHEMA = "wp1-development-v1"
FOLDS = ("fitting_images", "seen_validation_images", "unseen_validation_images")


def digest(value):
    """Namespace: canonical JSON, sorted keys, compact separators, ASCII, no LF."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True).encode("ascii")).hexdigest()


def candidate_table(frame):
    cols = ["composition_id", "crop_id", "disease_id", "crop_name", "disease_name", "composition_name"]
    table = frame[cols].drop_duplicates().sort_values("composition_id")
    if table.composition_id.duplicated().any() or table.duplicated(["crop_id", "disease_id"]).any():
        raise ValueError("Inconsistent candidate/component mapping")
    return [{key: int(value) if key.endswith("_id") else value for key, value in row.items()}
            for row in table.to_dict("records")]


def validate_protocol(protocol, source_csv):
    frame = load_metadata(source_csv)
    if protocol.get("schema") != SCHEMA:
        raise ValueError("Unsupported development protocol schema")
    if protocol["source_sha256"] != hashlib.sha256(Path(source_csv).read_bytes()).hexdigest():
        raise ValueError("Source-byte hash mismatch; verify CRLF checkout before proceeding")
    if protocol["candidates"] != candidate_table(frame):
        raise ValueError("Candidate order, labels or component IDs differ from training source")
    if protocol["candidate_sha256"] != digest(protocol["candidates"]):
        raise ValueError("Candidate fingerprint mismatch")
    fold_lists = [protocol[key] for key in FOLDS]
    if any(not values or len(values) != len(set(values)) for values in fold_lists):
        raise ValueError("Empty fold or duplicate filenames")
    sets = list(map(set, fold_lists))
    if any(sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("Filename overlap between development folds")
    if set.union(*sets) != set(frame.image_name):
        raise ValueError("Folds must partition the training source exactly")
    indexed = frame.set_index("image_name", drop=False)
    folds = [indexed.loc[names].reset_index(drop=True) for names in fold_lists]
    fitting, seen, unseen = folds
    fitting_ids = set(fitting.composition_id)
    if not set(seen.composition_id) <= fitting_ids or set(unseen.composition_id) & fitting_ids:
        raise ValueError("Invalid seen/unseen development composition separation")
    for key in ("crop_id", "disease_id"):
        if set(fitting[key]) != set(frame[key]):
            raise ValueError(f"Fitting fold lost a component: {key}")
    if protocol["split_sha256"] != digest({key: protocol[key] for key in FOLDS}):
        raise ValueError("Split fingerprint mismatch")
    if protocol["counts"] != [len(fold) for fold in folds]:
        raise ValueError("Recorded fold counts do not match filenames")
    return folds


def build_protocol(source_csv, seed=42, reference=None):
    frame = load_metadata(source_csv)
    remaining, unseen = make_validation_split(frame, seed=seed, mode="heldout_composition")
    fitting, seen = make_validation_split(remaining, seed=seed, mode="stratified")
    protocol = {"schema": SCHEMA, "source_csv": Path(source_csv).name,
                "source_sha256": hashlib.sha256(Path(source_csv).read_bytes()).hexdigest(),
                "seed": seed, "algorithm": "pytorch-v1:heldout-0.2-then-stratified-0.2",
                "historical_reconciliation": "pending_ordered_filename_reference",
                "candidate_hash_namespace": "wp1-canonical-json-candidate-records-v1",
                "split_hash_namespace": "wp1-canonical-json-fold-filenames-v1",
                "counts": [len(fitting), len(seen), len(unseen)],
                "candidates": candidate_table(frame)}
    for key, fold in zip(FOLDS, (fitting, seen, unseen)):
        protocol[key] = fold.image_name.tolist()
    if reference is not None:
        # Explicit filename lists exported from the accepted WP2/WP3 artifacts.
        # Counts alone are insufficient, and historical hashes are never replaced.
        for key in FOLDS:
            if reference.get(key) != protocol[key]:
                raise ValueError(f"Historical reference differs in ordered {key}; preserve it and investigate")
        protocol["historical_reconciliation"] = "ordered_filenames_match_supplied_reference"
        protocol["reference_sha256"] = digest(reference)
    protocol["candidate_sha256"] = digest(protocol["candidates"])
    protocol["split_sha256"] = digest({key: protocol[key] for key in FOLDS})
    validate_protocol(protocol, source_csv)
    return protocol


def write_new_protocol(protocol, path):
    """Idempotent export; never overwrite an existing different artifact."""
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != protocol:
            raise FileExistsError(f"Preserving existing protocol: {path}; choose a new output path")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(protocol, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("DPD_Small_train.csv"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/data/development_seed42_wp1_v1.json"))
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        protocol = json.loads(args.output.read_text(encoding="utf-8"))
        validate_protocol(protocol, args.source)
    else:
        reference = json.loads(args.reference.read_text(encoding="utf-8")) if args.reference else None
        protocol = build_protocol(args.source, reference=reference)
        write_new_protocol(protocol, args.output)
    print(json.dumps({key: protocol[key] for key in ("schema", "counts", "candidate_sha256", "split_sha256", "historical_reconciliation")}, indent=2))


if __name__ == "__main__":
    main()
