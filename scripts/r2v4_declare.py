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
AMENDMENTS = {1: ["research_r2_v4/amendment_1.json", "research_r2_v4/matrices/stage_i_v4_amendment1.json"],
              2: ["research_r2_v4/amendment_2.json", "research_r2_v4/DECLARATION_AMENDMENT_1.json"],
              3: ["research_r2_v4/amendment_3.json", "research_r2_v4/DECLARATION_AMENDMENT_2.json"]}  # each pins the earlier ones
DECLARED = {1: "2026-11-14", 2: "2026-11-14", 3: "2026-11-17"}


def code_pin_problems(amendment_file):
    """Amendment 3+: the amendment JSON pins code files by sha256 ('code_pins'); every pin must still match."""
    pins = json.loads((ROOT / amendment_file).read_text()).get("code_pins", {})
    return [f for f, h in pins.items() if not (ROOT / f).exists() or sha256_file(ROOT / f) != h]


def main():
    global FILES, OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--amendment", type=int, help="declare / verify amendment N (its own DECLARATION_AMENDMENT_N.json)")
    a = ap.parse_args()
    if a.amendment:
        FILES = AMENDMENTS[a.amendment] + [f"research_r2_v4/DECLARATION.json"]
        OUT = ROOT / f"research_r2_v4/DECLARATION_AMENDMENT_{a.amendment}.json"
        for f in FILES:
            json.loads((ROOT / f).read_text())
        digests = {f: sha256_file(ROOT / f) for f in FILES}
        pin_bad = code_pin_problems(AMENDMENTS[a.amendment][0])
        if a.verify:
            bad = [f for f in FILES if json.loads(OUT.read_text())["sha256"].get(f) != digests[f]] + [f"code pin {f}" for f in pin_bad]
            print("OK" if not bad else f"CHANGED: {bad}")
            sys.exit(1 if bad else 0)
        if pin_bad:
            sys.exit(f"refusing to declare: code pins do not match the files: {pin_bad}")
        if a.amendment >= 3 and OUT.exists():
            sys.exit(f"refusing to declare: {OUT.name} already exists (declarations are never rewritten)")
        OUT.write_text(json.dumps({"schema": "r2v4-declaration-v1", "amendment": a.amendment, "declared": DECLARED.get(a.amendment, "2026-11-14"),
                                   "note": "declared before any v4 scoring; the base declaration is pinned, not edited",
                                   "sha256": digests}, indent=1) + "\n")
        print(json.dumps(digests, indent=1))
        return
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
