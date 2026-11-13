"""Build benchmark v2 (outer-isolated) from the v1 lineage groups; training-source images only."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research_r2.benchmark import fold_summary  # noqa: E402
from research_r2.benchmark_v2 import build_benchmark_v2, validate_benchmark_v2  # noqa: E402
from research_r2.data import load_training_source, sha256_file  # noqa: E402


def main():
    frame = load_training_source(ROOT / "DPD_Small_train.csv")
    group = np.array(json.loads((ROOT / "artifacts/r2/lineage_groups_v1.json").read_text())["group"])
    bench = build_benchmark_v2(frame, group)
    checks = validate_benchmark_v2(frame, group, bench)
    bench["provenance"] = {"train_csv_sha256": sha256_file(ROOT / "DPD_Small_train.csv"),
                           "lineage_source": "artifacts/r2/lineage_groups_v1.json", "official_test_files_read": False,
                           "researcher_exposure": ("All 8,000 images, including every outer composition, were analysed in v1 "
                                                   "(fitting, validation and per-composition error analysis). v2 isolates the outer "
                                                   "population from v2 development by data, not from the researcher's v1 knowledge.")}
    out = ROOT / "artifacts/r2"
    (out / "benchmark_v2.json").write_text(json.dumps(bench))
    summ = fold_summary(frame, bench)
    (out / "fold_summary_v2.csv").write_text(summ.to_csv(index=False))
    (out / "checks_v2.json").write_text(json.dumps(checks, indent=1))
    print(summ.to_string(index=False))
    print("outer compositions", len(bench["outer_composition_ids"]), "outer seen", len(bench["outer_seen_rows"]),
          "outer exclusion X", len(bench["outer_exclusion_rows"]), "inner population", len(bench["inner_population_rows"]),
          "inner eligible", len(bench["inner_eligible_composition_ids"]))
    print("semantic sha256", bench["semantic_sha256"])
    print("\n".join(checks))


if __name__ == "__main__":
    main()
