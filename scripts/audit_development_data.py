"""Independent development data and augmentation audit for WP1 / Issue #7.

Owner: Andy Wei Jian Chu (@AA-49).
Restricted strictly to training-source development data (DPD_Small_train.csv).
Never loads, touches, or leaks official-test images or test metadata.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

# Add repo root to import path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared_protocol import FOLDS, candidate_table, digest, validate_protocol


FORBIDDEN_FILES = {
    "dpd_small_test_seen.csv",
    "dpd_small_test_unseen.csv",
}


def verify_source_safety(source_csv: Path | str) -> None:
    """Ensure official-test CSVs are strictly rejected."""
    name = Path(source_csv).name.strip().lower()
    if name in FORBIDDEN_FILES or "test" in name:
        raise PermissionError(f"Strict safety boundary: official-test file {name} must never be audited.")


def audit_metadata_partitions(
    protocol: Dict[str, Any], source_csv: Path | str
) -> Dict[str, Any]:
    """Audit partition counts, candidate distributions, singletons, and component coverage."""
    verify_source_safety(source_csv)
    fitting_df, seen_val_df, unseen_val_df = validate_protocol(protocol, source_csv)

    total_images = len(fitting_df) + len(seen_val_df) + len(unseen_val_df)
    
    # Composition counts and singletons
    fitting_comp_counts = fitting_df["composition_id"].value_counts()
    singletons = fitting_comp_counts[fitting_comp_counts == 1].index.tolist()

    # Crop and disease coverage
    crop_counts = fitting_df["crop_id"].nunique()
    disease_counts = fitting_df["disease_id"].nunique()

    # Healthy vs diseased breakdown in fitting
    # In DPD-Small, healthy composition names end with 'healthy' or disease_name == 'healthy'
    is_healthy = fitting_df["disease_name"].str.strip().str.lower() == "healthy"
    healthy_fitting_count = int(is_healthy.sum())
    diseased_fitting_count = int((~is_healthy).sum())

    summary = {
        "schema": "wp1-development-data-audit-v1",
        "auditor": "Andy Wei Jian Chu (@AA-49)",
        "source_csv": str(Path(source_csv).name),
        "source_sha256": protocol.get("source_sha256"),
        "candidate_sha256": protocol.get("candidate_sha256"),
        "split_sha256": protocol.get("split_sha256"),
        "partition_counts": {
            "total_development_images": total_images,
            "fitting_images": len(fitting_df),
            "seen_validation_images": len(seen_val_df),
            "unseen_validation_images": len(unseen_val_df),
        },
        "composition_support": {
            "fitting_compositions": int(fitting_df["composition_id"].nunique()),
            "seen_val_compositions": int(seen_val_df["composition_id"].nunique()),
            "unseen_val_heldout_compositions": int(unseen_val_df["composition_id"].nunique()),
            "singleton_compositions_in_fitting": len(singletons),
            "singleton_composition_ids": sorted(singletons),
        },
        "component_coverage": {
            "fitting_crops": crop_counts,
            "fitting_diseases": disease_counts,
            "all_components_retained": crop_counts == 55 and disease_counts == 175,
        },
        "class_balance_breakdown": {
            "fitting_healthy_images": healthy_fitting_count,
            "fitting_diseased_images": diseased_fitting_count,
            "healthy_ratio_fitting": round(healthy_fitting_count / len(fitting_df), 4),
        },
    }
    return summary


def audit_file_duplicates(
    protocol: Dict[str, Any], images_dir: Path | str
) -> Dict[str, Any]:
    """Detect exact raw-file SHA-256 duplicates across development images and folds."""
    img_dir = Path(images_dir)
    if not img_dir.exists():
        return {
            "status": "skipped",
            "reason": f"Images directory {img_dir} not accessible on this runner.",
        }

    # Restrict strictly to development allowlist
    allowlist = set(protocol["fitting_images"] + protocol["seen_validation_images"] + protocol["unseen_validation_images"])
    fold_map = {}
    for fold_name in FOLDS:
        for fname in protocol[fold_name]:
            fold_map[fname] = fold_name

    hash_to_files: Dict[str, List[str]] = {}
    unreadable_files: List[str] = []

    for fname in allowlist:
        path = img_dir / fname
        if not path.is_file():
            unreadable_files.append(fname)
            continue
        # Raw file byte SHA-256
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        hash_to_files.setdefault(sha, []).append(fname)

    # Filter to duplicates (same content, different filename)
    duplicates = {h: files for h, files in hash_to_files.items() if len(files) > 1}
    cross_fold_duplicates = []
    intra_fold_duplicates = []

    for h, files in duplicates.items():
        folds_present = {fold_map[f] for f in files}
        record = {"sha256": h, "files": files, "folds": sorted(folds_present)}
        if len(folds_present) > 1:
            cross_fold_duplicates.append(record)
        else:
            intra_fold_duplicates.append(record)

    return {
        "status": "completed",
        "scanned_allowlist_images": len(allowlist),
        "unreadable_files": unreadable_files,
        "unique_content_hashes": len(hash_to_files),
        "duplicate_groups_total": len(duplicates),
        "cross_fold_duplicate_groups": cross_fold_duplicates,
        "intra_fold_duplicate_groups": intra_fold_duplicates,
        "fold_leakage_detected": len(cross_fold_duplicates) > 0,
    }


def audit_augmentation_policy(review_json_path: Optional[Path | str] = None) -> Dict[str, Any]:
    """Audit augmentation specification and human review sign-off."""
    info = {
        "policy": "torchvision_v2_fitting_only",
        "transforms": [
            "Resize((192, 192))",
            "RandomHorizontalFlip(p=0.5)",
            "RandomVerticalFlip(p=0.5)",
            "RandomAffine(degrees=18, translate=(0.05, 0.05), scale=(0.95, 1.05))",
        ],
        "symptom_preservation_rationale": (
            "Mild affine translations (5%) and small rotations (18 deg) avoid cutting out localized leaf lesions. "
            "Horizontal and vertical flips preserve color and lesion texture without geometric distortion."
        ),
        "human_review_status": "pending_human_review",
    }
    if review_json_path and Path(review_json_path).is_file():
        try:
            data = json.loads(Path(review_json_path).read_text(encoding="utf-8"))
            info["human_review_status"] = data.get("status", "unknown")
            info["human_reviewer"] = data.get("reviewer")
            info["human_decision"] = data.get("decision")
        except Exception:
            pass
    return info


def generate_full_audit(
    protocol_path: Path | str,
    source_csv: Path | str,
    images_dir: Optional[Path | str] = None,
    review_path: Optional[Path | str] = None,
) -> Dict[str, Any]:
    """Produce the complete Issue #7 audit report."""
    verify_source_safety(source_csv)
    protocol = json.loads(Path(protocol_path).read_text(encoding="utf-8"))
    
    meta_audit = audit_metadata_partitions(protocol, source_csv)
    dup_audit = audit_file_duplicates(protocol, images_dir) if images_dir else {"status": "not_requested"}
    aug_audit = audit_augmentation_policy(review_path)

    report = {
        **meta_audit,
        "file_duplicate_audit": dup_audit,
        "augmentation_audit": aug_audit,
        "proposed_robustness_evaluation": {
            "objective": "Evaluate model resilience against real-world leaf photo corruptions (Section 3.2 research question).",
            "perturbations": [
                {"name": "brightness_variation", "severity": [-0.2, 0.2]},
                {"name": "contrast_variation", "severity": [0.8, 1.2]},
                {"name": "gaussian_blur", "kernel_size": 3, "sigma": 0.5},
            ],
            "evaluation_rule": "Clean-vs-perturbed paired ranking accuracy on internally unseen validation.",
        },
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("DPD_Small_train.csv"))
    parser.add_argument("--protocol", type=Path, default=Path("artifacts/data/development_seed42_wp1_v1.json"))
    parser.add_argument("--images", type=Path, default=None)
    parser.add_argument("--review", type=Path, default=Path("runs/wp1-augmentation-review/review.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/data/development_audit_report.json"))
    args = parser.parse_args()

    report = generate_full_audit(args.protocol, args.source, args.images, args.review)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Audit report successfully written to: {args.output}")


if __name__ == "__main__":
    main()
