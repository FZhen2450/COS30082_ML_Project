"""Write research_r2_v4/DECLARATION.json: sha256 of the v4 protocol, matrices and plan as declared.

Run once, before any v4 scoring, and commit the result with the declared files. Later runs verify
against it (`--verify` exits 1 if any declared file changed).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research_r2.data import sha256_file  # noqa: E402

FILES = ["research_r2_v4/protocol_v4.json", "research_r2_v4/matrices/stage_h_v4.json",
         "research_r2_v4/matrices/stage_i_v4.json", "research_r2_v4/matrices/stage_j_v4.json", "docs/R2_v4_Plan.md"]
OUT = ROOT / "research_r2_v4/DECLARATION.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    for f in FILES[:4]:
        json.loads((ROOT / f).read_text())  # declared JSON must parse
    digests = {f: sha256_file(ROOT / f) for f in FILES}
    if a.verify:
        old = json.loads(OUT.read_text())["sha256"]
        bad = [f for f in FILES if old.get(f) != digests[f]]
        print("OK" if not bad else f"CHANGED: {bad}")
        sys.exit(1 if bad else 0)
    OUT.write_text(json.dumps({"schema": "r2v4-declaration-v1", "declared": "2026-11-14",
                               "note": "declared before any v4 scoring; v1-v3 evidence unchanged", "sha256": digests}, indent=1) + "\n")
    print(json.dumps(digests, indent=1))


if __name__ == "__main__":
    main()
