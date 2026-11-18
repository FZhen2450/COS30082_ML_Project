"""Build and lock the Round 2 composition-disjoint benchmark (training-source images only).

Usage:
  python scripts/r2_build_benchmark.py --csv DPD_Small_train.csv \
      --images ../../datasets/DPD_Small_dataset --out ../../experiments/r2-compositional/benchmark
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research_r2.benchmark import build_benchmark, fold_summary, validate_benchmark  # noqa: E402
from research_r2.data import load_training_source, sha256_file, sha256_json  # noqa: E402
from research_r2.lineage import image_digest, lineage_groups  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--legacy", default="artifacts/data/heldout_composition_seed42.json")
    ap.add_argument("--threshold", type=int, default=4)
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    frame = load_training_source(a.csv)
    digest_path = out / "image_digests.json"
    if digest_path.exists():
        digests = json.loads(digest_path.read_text())
        assert [d["file"] for d in digests["images"]] == frame.file.tolist()
    else:
        t = time.time()
        paths = [Path(a.images) / f for f in frame.file]
        with ThreadPoolExecutor(8) as ex:
            res = list(ex.map(image_digest, paths))
        digests = {"schema": "r2-image-digests-v1", "dhash": "dhash64-9x8-lanczos-L",
                   "images": [{"file": f, "sha256": s, "dhash64": f"{h:016x}"} for f, (s, h) in zip(frame.file, res)],
                   "seconds": time.time() - t}
        digest_path.write_text(json.dumps(digests, indent=0))
    shas = [d["sha256"] for d in digests["images"]]
    dh = [int(d["dhash64"], 16) for d in digests["images"]]
    group, pairs = lineage_groups(shas, dh, threshold=a.threshold)
    comp = frame.comp_id.to_numpy()
    cross = [(i, j, d) for i, j, d in pairs if comp[i] != comp[j]]
    legacy = json.loads(Path(a.legacy).read_text())["validation_composition_ids"]
    bench = build_benchmark(frame, group, seed=a.seed, legacy_heldout_ids=legacy)
    checks = validate_benchmark(frame, group, bench)
    bench["provenance"] = {
        "train_csv_sha256": sha256_file(a.csv), "image_digest_list_sha256": sha256_json(shas),
        "lineage_rule": f"sha256-equal OR hamming(dhash64)<={a.threshold}",
        "lineage_groups": int(group.max() + 1), "lineage_linked_pairs": len(pairs),
        "cross_composition_linked_pairs": len(cross),
        "legacy_source": a.legacy, "official_test_files_read": False,
        "semantic_disclosure": ("Eligibility mirrors the brief's statement that unseen classes are crop-disease "
                                "compositions of known concepts and the 10 published unseen class names (all diseased, "
                                "each crop/disease known elsewhere). No official-test image, filename or per-image label "
                                "is used."),
    }
    (out / "benchmark.json").write_text(json.dumps(bench))
    (out / "lineage_groups.json").write_text(json.dumps({"group": group.tolist(), "cross_composition_pairs": [
        {"a": frame.file[i], "b": frame.file[j], "comp_a": int(comp[i]), "comp_b": int(comp[j]), "dist": d} for i, j, d in cross]}))
    summ = fold_summary(frame, bench)
    (out / "fold_summary.csv").write_text(summ.to_csv(index=False))
    (out / "checks.json").write_text(json.dumps(checks, indent=1))
    print(summ.to_string(index=False))
    print("lineage groups", group.max() + 1, "linked pairs", len(pairs), "cross-composition", len(cross))
    print("semantic sha256", bench["semantic_sha256"])
    print("\n".join(checks))


if __name__ == "__main__":
    main()
