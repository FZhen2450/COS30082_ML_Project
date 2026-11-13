"""Run a Round 2 experiment matrix on screening folds (S1..S5) and the legacy diagnostic D0.

Protected folds (role "locked" in v1, "outer" in v2) are refused unless --frozen-recipe names
an entry in research_r2/frozen_recipes.json whose path, file hash, benchmark hash and row ids
all match (research_r2/guards.py).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research_r2.data import load_training_source, sha256_file  # noqa: E402
from research_r2.guards import select_folds, verify_frozen  # noqa: E402
from research_r2.packets import git_state  # noqa: E402
from research_r2.runner import gate, run_row, summarise  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="DPD_Small_train.csv")
    ap.add_argument("--bench", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--matrix", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--frozen-recipe", help="registered name in research_r2/frozen_recipes.json; unlocks its protected folds")
    ap.add_argument("--folds", nargs="+", help="override the default unprotected fold list (e.g. v2 inner folds)")
    ap.add_argument("--save-scores", action="store_true")
    a = ap.parse_args()
    frame = load_training_source(a.csv)
    bench = json.loads(Path(a.bench).read_text())
    folds = select_folds(bench, a.folds)  # raises if a protected (locked/outer) fold is requested directly
    if a.frozen_recipe:
        entry = verify_frozen(a.frozen_recipe, a.matrix, bench)  # raises on any mismatch
        folds = folds + [f for f in entry["protected_folds"] if f not in folds]
    rows = json.loads(Path(a.matrix).read_text())
    # provenance for manifest.json written next to NEW score packets (historical packets are never touched)
    prov = {"matrix": a.matrix, "matrix_sha256": sha256_file(a.matrix), "benchmark": a.bench,
            "benchmark_semantic_sha256": bench["semantic_sha256"], "frozen_recipe": a.frozen_recipe,
            "folds": folds, "code": git_state(Path(__file__).resolve().parents[1]),
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    done = {}
    res_path = out / "summaries.jsonl"
    if res_path.exists():
        for line in res_path.read_text().splitlines():
            s = json.loads(line)
            done[s["id"]] = s
    for row in rows:
        if a.only and row["id"] not in a.only:
            continue
        if row["id"] in done:
            continue
        poison = None if a.frozen_recipe else bench.get("outer_exclusion_rows")
        results, ident, secs = run_row(frame, bench, a.features, row, folds=folds,
                                       out_dir=(out / "scores") if a.save_scores else None, poison_rows=poison,
                                       provenance=prov)
        sel = [f["name"] for f in bench["folds"] if f["role"] in ("screen", "inner")]
        s = summarise(row, results, ident, secs, screen=tuple(sel))
        s["matrix_sha256"] = sha256_file(a.matrix)
        s["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        if row.get("control") in done:
            s["gate_vs_control"] = gate(s, done[row["control"]], screen=tuple(sel))
        done[row["id"]] = s
        with open(res_path, "a") as fh:
            fh.write(json.dumps(s) + "\n")
        m = s["screen_mean"]
        g = s.get("gate_vs_control", {})
        d0 = s["per_fold"].get("D0", {}).get("pooled_gamma", {})
        print(f"{row['id']:<28} Hbal_lofo={100*m['primary_H_bal_lofo']:6.2f} Hbal_g0={100*m['H_bal_gamma0']:6.2f} "
              f"seen={100*m['seen_top1_lofo']:5.1f} unM={100*m['unseen_macro_lofo']:5.1f} "
              f"unDis={100*m['unseen_disease_lofo']:5.1f} wc0={100*m['unseen_withincrop_gamma0']:5.1f} AUSUC={100*m['ausuc']:5.2f} hs={100*m['healthy_shortcut_lofo']:4.1f} "
              f"g*={m['pooled_gamma']:.2f} | D0 Hmicro={100*d0.get('H_micro', float('nan')):5.2f} "
              f"| gate={g.get('pass', '-')} ({g.get('folds_improved', '-')}/5, {g.get('mean_gain_pp', float('nan')):+.2f}pp) "
              f"[{secs:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
