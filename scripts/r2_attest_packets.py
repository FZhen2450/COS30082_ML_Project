"""Retrospective packet registry for the committed v3 selection receipts (v3 hardening).

Pins, by sha256, every score packet the three v3 selections consumed, together with the selected rows'
declared configuration identity, their run-time records (summaries.jsonl: producing-matrix hash and
feature identity) and each fold's contract hash. Packets are only read: nothing is regenerated, moved
or rewritten, and no provenance is retrofitted into historical files.

What it does and does not establish: it fixes the inputs so any later substitution is detected. It is
created after the packets were produced, so their origin rests on the run-time summaries and on the
independent regeneration check (scripts/r2_reproduce_packets.py), not on this file.

  python scripts/r2_attest_packets.py --out artifacts/r2/v3/packet_registry_v3.json \
      --select phase1_correction_replay=DIR_D,DIR_BC  --select phase2_screen=DIR_F  --select phase3_representation=DIR_G
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research_r2.data import load_training_source  # noqa: E402
from research_r2.packets import (PacketError, committed_matrix_index, declared_rows, file_sha256,  # noqa: E402
                                 fold_contract, git_state, resolve_packets, runtime_summary, tree_rows, validate_packet)
import numpy as np  # noqa: E402

EXP = ROOT.parents[1] / "experiments" / "r2-compositional"


def rel(p) -> str:
    return str(Path(p).resolve().relative_to(EXP.resolve()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="artifacts/r2/benchmark_v3.json")
    ap.add_argument("--protocol", default="research_r2/protocol_v3.json")
    ap.add_argument("--select", action="append", required=True, help="TREE_KEY=DIR[,DIR...] as used for the receipt")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    gs = git_state(ROOT)
    if gs["dirty"]:
        raise SystemExit("refusing to attest from a dirty tree: commit the implementation first")
    frame = load_training_source(ROOT / "DPD_Small_train.csv")
    bench = json.loads(Path(a.bench).read_text())
    proto = json.loads(Path(a.protocol).read_text())
    if bench["semantic_sha256"] != proto["benchmark_semantic_sha256"]:
        raise PacketError("benchmark semantic hash differs from the protocol")
    F = {f["name"]: f for f in bench["folds"]}
    folds = [f["name"] for f in bench["folds"] if f["role"] == "inner"]
    contracts = {fn: fold_contract(frame, F[fn]) for fn in folds}
    mindex = committed_matrix_index()
    reg = {"schema": "r2-packet-registry-v1", "retrospective": True,
           "statement": "Created after the packets were produced. Pins the exact files consumed by the committed v3 "
                        "selection receipts; their origin rests on the run-time summaries.jsonl records and on the "
                        "independent regeneration check, not on this registry.",
           "code": gs, "benchmark_semantic_sha256": bench["semantic_sha256"],
           "protocol": a.protocol, "protocol_sha256": file_sha256(a.protocol), "folds": folds,
           "contracts": {fn: contracts[fn]["sha256"] for fn in folds}, "trees": {}, "rows": {}}
    for spec in a.select:
        key, dirs = spec.split("=", 1)
        dirs = dirs.split(",")
        entry = proto[key]
        rows = tree_rows(entry["tree"], entry["start"])
        decl = declared_rows(entry, rows)
        reg["trees"][key] = {"score_dirs": [rel(d) for d in dirs], "rows": rows,
                             "matrices": dict(entry.get("matrices") or entry.get("matrix"))}
        for rid in rows:
            res = resolve_packets(rid, folds, dirs)
            stages = sorted({str(Path(c).parent.parent.parent) for f in res.values() for c in f["copies"]})
            runtime = [runtime_summary(Path(sd), rid, decl[rid], mindex) for sd in stages]
            for r in runtime:
                r["summaries"] = rel(r["summaries"])
            idents = {json.dumps(r["feature_identity"], sort_keys=True) for r in runtime}
            if len(idents) != 1:
                raise PacketError(f"{rid}: run-time records disagree on feature identity")
            fold_rec = {}
            for fn in folds:
                validate_packet(np.load(res[fn]["path"]), contracts[fn])
                fold_rec[fn] = {"sha256": res[fn]["sha256"], "contract_sha256": contracts[fn]["sha256"],
                                "copies": sorted(rel(c) for c in res[fn]["copies"])}
            new = {"config": decl[rid]["config"], "config_sha256": decl[rid]["config_sha256"],
                   "declared_in": decl[rid]["declared_in"], "runtime_records": runtime,
                   "feature_identity": runtime[0]["feature_identity"], "folds": fold_rec}
            old = reg["rows"].get(rid)
            if old is not None:
                # a row used by several trees must resolve to the same configuration and the same packets
                if old["config_sha256"] != new["config_sha256"] or \
                        {f: v["sha256"] for f, v in old["folds"].items()} != {f: v["sha256"] for f, v in fold_rec.items()}:
                    raise PacketError(f"{rid}: different configuration or packets across selection trees")
                for fn in folds:
                    old["folds"][fn]["copies"] = sorted(set(old["folds"][fn]["copies"]) | set(fold_rec[fn]["copies"]))
                seen = {r["summaries"] for r in old["runtime_records"]}
                old["runtime_records"] += [r for r in runtime if r["summaries"] not in seen]
                old["declared_in"] = sorted(set(old["declared_in"]) | set(new["declared_in"]))
            else:
                reg["rows"][rid] = new
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(reg, indent=1, sort_keys=True) + "\n")
    n_pk = sum(len(r["folds"]) for r in reg["rows"].values())
    n_cp = sum(len(f["copies"]) for r in reg["rows"].values() for f in r["folds"].values())
    print(f"registry: {len(reg['rows'])} rows, {n_pk} packets, {n_cp} copies (all byte-identical per packet), "
          f"code {gs['commit'][:7]} clean; sha256 {file_sha256(a.out)}")


if __name__ == "__main__":
    main()
