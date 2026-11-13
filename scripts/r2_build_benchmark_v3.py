"""Build benchmark v3 (explicit candidate universe) from the committed v2 benchmark."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research_r2.benchmark_v3 import build_benchmark_v3, validate_benchmark_v3  # noqa: E402
from research_r2.data import load_training_source  # noqa: E402

frame = load_training_source(ROOT / "DPD_Small_train.csv")
group = np.array(json.loads((ROOT / "artifacts/r2/lineage_groups_v1.json").read_text())["group"])
v2 = json.loads((ROOT / "artifacts/r2/benchmark_v2.json").read_text())
b = build_benchmark_v3(frame, v2)
checks = validate_benchmark_v3(frame, group, b, v2)
(ROOT / "artifacts/r2/benchmark_v3.json").write_text(json.dumps(b))
(ROOT / "artifacts/r2/checks_v3.json").write_text(json.dumps(checks, indent=1))
for f in b["folds"]:
    print(f["name"], f["role"], f["candidate_counts"], "universe", len(f["candidate_composition_ids"]))
print("semantic sha256", b["semantic_sha256"])
print("\n".join(checks))
