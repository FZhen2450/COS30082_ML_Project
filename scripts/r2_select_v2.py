"""Mechanical recipe selection on v2 inner folds (rule in research_r2/protocol_v2.json)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research_r2.runner import gate  # noqa: E402


def select(summaries: dict, tree: dict, inner: tuple, start="B1_clipL_factorised"):
    path, trace, cur = [start], [], start
    while True:
        passing = []
        for child in tree.get(cur, []):
            g = gate(summaries[child], summaries[cur], screen=inner)
            trace.append({"parent": cur, "child": child, **g,
                          "child_primary": summaries[child]["screen_mean"]["primary_H_bal_lofo"]})
            if g["pass"]:
                passing.append(child)
        if not passing:
            return cur, path, trace
        cur = max(passing, key=lambda c: (summaries[c]["screen_mean"]["primary_H_bal_lofo"], -tree[path[-1]].index(c)))
        path.append(cur)


def main(summary_files, out):
    proto = json.loads((ROOT / "research_r2/protocol_v2.json").read_text())
    bench = json.loads((ROOT / "artifacts/r2/benchmark_v2.json").read_text())
    assert bench["semantic_sha256"] == proto["benchmark_semantic_sha256"]
    inner = tuple(f["name"] for f in bench["folds"] if f["role"] == "inner")
    summ = {}
    for f in summary_files:
        for line in Path(f).read_text().splitlines():
            s = json.loads(line)
            summ.setdefault(s["id"], s)
    sel, path, trace = select(summ, proto["mechanical_selection"]["tree"], inner)
    rep = {"selected": sel, "path": path, "trace": trace, "inner_folds": inner,
           "pooled_gamma": {k: summ[k]["screen_mean"]["pooled_gamma"] for k in
                            {sel, "B1_clipL_factorised", "B0_clipL_zeroshot_comp", "D2_synthComp_norm_text"}}}
    Path(out).write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: rep[k] for k in ("selected", "path", "pooled_gamma")}, indent=1))
    for t in trace:
        print(f"  {t['child']:<32} vs {t['parent']:<24} {t['mean_gain_pp']:+6.2f} pp  {t['folds_improved']}/{len(inner)}  "
              f"AUSUC {t['ausuc_delta']*100:+.2f}  pass={t['pass']}")


if __name__ == "__main__":
    main(sys.argv[1:-1], sys.argv[-1])
