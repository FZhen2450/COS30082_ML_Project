"""Structural validation of every saved Round 2 score packet against its fold contract (v1, v2, v3).

Read-only. Checks array set, ordered candidate ids (v3), shapes, dtypes, finite scores and truth indices
against the benchmark row lists. Computes no metric, so protected/examined folds (v1 locked, v2 outer)
are checked structurally without being re-scored. Also lists every row id present in more than one
stage directory of the same benchmark and whether its copies are byte-identical.

  python scripts/r2_validate_packets.py --out artifacts/r2/v3/hardening/packet_sweep.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research_r2.data import load_training_source  # noqa: E402
from research_r2.packets import PacketError, file_sha256, fold_contract, validate_packet  # noqa: E402

EXP = ROOT.parents[1] / "experiments" / "r2-compositional"
GROUPS = {"screen": "artifacts/r2/benchmark_v1.json", "v2": "artifacts/r2/benchmark_v2.json",
          "v3": "artifacts/r2/benchmark_v3.json"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    frame = load_training_source(ROOT / "DPD_Small_train.csv")
    rep, ok = {}, True
    for group, bpath in GROUPS.items():
        bench = json.loads((ROOT / bpath).read_text())
        F = {f["name"]: f for f in bench["folds"]}
        contracts = {}
        counts = defaultdict(int)
        failures = []
        copies = defaultdict(dict)
        for pk in sorted((EXP / group).glob("*/scores/*/*_scores.npz")):
            stage, rid, fn = pk.parts[-4], pk.parts[-2], pk.name[: -len("_scores.npz")]
            if fn not in F:
                failures.append({"packet": str(pk.relative_to(EXP)), "error": f"fold {fn} not in {bpath}"}); continue
            if fn not in contracts:
                contracts[fn] = fold_contract(frame, F[fn])
            try:
                validate_packet(np.load(pk), contracts[fn])
                counts[f"{stage}:{contracts[fn]['schema']}:{F[fn]['role']}"] += 1
            except PacketError as e:
                failures.append({"packet": str(pk.relative_to(EXP)), "error": str(e)})
            copies[(rid, fn)][stage] = file_sha256(pk)
        dup = {}
        for (rid, fn), by_stage in copies.items():
            if len(by_stage) > 1:
                d = dup.setdefault(rid, {"stages": sorted(by_stage), "folds": 0, "conflicting_folds": []})
                d["folds"] += 1
                if len(set(by_stage.values())) > 1:
                    d["conflicting_folds"].append(fn)
        rep[group] = {"benchmark": bpath, "valid_by_stage_schema_role": dict(sorted(counts.items())),
                      "n_valid": sum(counts.values()), "failures": failures, "duplicated_rows": dict(sorted(dup.items()))}
        ok &= not failures
        print(f"{group:<7} {bpath:<34} valid {sum(counts.values()):4d}  failures {len(failures)}")
        for k, v in sorted(counts.items()):
            print(f"         {k:<40} {v}")
        for rid, d in sorted(dup.items()):
            state = "CONFLICT " + ",".join(d["conflicting_folds"]) if d["conflicting_folds"] else "identical"
            print(f"         duplicate {rid:<28} in {','.join(d['stages']):<40} {d['folds']} folds: {state}")
        for f in failures[:10]:
            print("         FAIL", f["packet"], f["error"])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=1, sort_keys=True) + "\n")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
