"""Compare freshly computed v3-protocol summaries with the committed v3 results (fixed tolerance 1e-5).

Every numeric leaf of "screen_mean" and "per_fold" is compared; run-time fields (seconds, timestamps,
feature identity, matrix hash) are not scientific and are skipped. Exit code 1 on any mismatch.

  python scripts/r2v4_repro_check.py --fresh NEW/summaries.jsonl --stored artifacts/r2/v3/results_stage_f_v3.jsonl \
      --rows B1_clipL_factorised D1_synthComp_norm F10_concat_convnext --out OUT.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

TOL = 1e-5


def leaves(x, prefix=""):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from leaves(v, f"{prefix}/{k}")
    elif isinstance(x, (int, float)) and not isinstance(x, bool):
        yield prefix, float(x)
    elif isinstance(x, str):
        yield prefix, x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh", required=True)
    ap.add_argument("--stored", required=True)
    ap.add_argument("--rows", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    load = lambda p: {json.loads(l)["id"]: json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()}  # noqa: E731
    fresh, stored = load(a.fresh), load(a.stored)
    rep = {"tolerance": TOL, "rows": {}}
    ok = True
    for rid in a.rows:
        f = dict(leaves({"screen_mean": fresh[rid]["screen_mean"], "per_fold": fresh[rid]["per_fold"]}))
        s = dict(leaves({"screen_mean": stored[rid]["screen_mean"], "per_fold": stored[rid]["per_fold"]}))
        missing = sorted(set(s) ^ set(f))
        bad, worst = [], 0.0
        for k in sorted(set(s) & set(f)):
            if isinstance(s[k], str) or isinstance(f[k], str):
                if s[k] != f[k]:
                    bad.append(k)
                continue
            if math.isnan(s[k]) and math.isnan(f[k]):
                continue
            d = abs(s[k] - f[k])
            worst = max(worst, d)
            if d > TOL:
                bad.append(k)
        rep["rows"][rid] = {"leaves_compared": len(set(s) & set(f)), "max_abs_diff": worst, "mismatches": bad[:50],
                            "key_set_differences": missing[:50], "pass": not bad and not missing}
        ok &= rep["rows"][rid]["pass"]
        print(rid, rep["rows"][rid]["leaves_compared"], f"max|d|={worst:.3e}", "PASS" if rep["rows"][rid]["pass"] else "FAIL")
    rep["pass"] = ok
    Path(a.out).write_text(json.dumps(rep, indent=1))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
