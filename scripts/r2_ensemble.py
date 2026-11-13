"""Score-level ensembles of saved Round 2 rows (equal weights; no new fitting).

  python scripts/r2_ensemble.py --id E1 --members dirA/scores/ROW1 dirB/scores/ROW2 --control-summary ... --out ...
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research_r2.data import HEALTHY, candidate_table, load_training_source  # noqa: E402
from research_r2.metrics import FoldScorer  # noqa: E402
from research_r2.runner import gate, summarise  # noqa: E402

FOLDS = ["S1", "S2", "S3", "S4", "S5", "D0"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True)
    ap.add_argument("--members", nargs="+", required=True)
    ap.add_argument("--bench", required=True)
    ap.add_argument("--control", help="summaries.jsonl:ROW_ID")
    ap.add_argument("--out", required=True)
    ap.add_argument("--hypothesis", default="")
    a = ap.parse_args()
    frame = load_training_source("DPD_Small_train.csv")
    cands = candidate_table(frame)
    crops = {c: i for i, c in enumerate(sorted(frame.crop_id.unique()))}
    dis = {d: i for i, d in enumerate(sorted(frame.disease_id.unique()))}
    cc = np.array([crops[c] for c in cands.crop_id]); cd = np.array([dis[d] for d in cands.disease_id])
    healthy = cands.disease.eq(HEALTHY).to_numpy()
    bench = json.loads(Path(a.bench).read_text())
    results = {}
    for fold in bench["folds"]:
        if fold["name"] not in FOLDS:
            continue
        zs = [np.load(Path(m) / f"{fold['name']}_scores.npz") for m in a.members]
        for z in zs[1:]:
            assert np.array_equal(z["seen_truth"], zs[0]["seen_truth"]) and np.array_equal(z["unseen_truth"], zs[0]["unseen_truth"])
        seen = np.mean([z["seen"].astype(np.float64) for z in zs], 0)
        uns = np.mean([z["unseen"].astype(np.float64) for z in zs], 0)
        held = set(fold["heldout_composition_ids"])
        fm = np.array([int(c) not in held for c in cands.comp_id])
        sc = FoldScorer(seen, zs[0]["seen_truth"], uns, zs[0]["unseen_truth"], fm, cc, cd, healthy)
        results[fold["name"]] = {"curve": sc.curve(), "ausuc": sc.ausuc(), "gamma0": sc.at(0.0, top5=True), "_scorer": sc}
    row = {"id": a.id, "backbones": [], "recipe": "score_ensemble", "cfg": {"members": a.members},
           "hypothesis": a.hypothesis, "control": a.control}
    s = summarise(row, results, {}, 0.0)
    s["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    if a.control:
        path, rid = a.control.split(":")
        ctrl = next(json.loads(l) for l in Path(path).read_text().splitlines() if json.loads(l)["id"] == rid)
        s["gate_vs_control"] = gate(s, ctrl)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "summaries.jsonl", "a") as fh:
        fh.write(json.dumps(s) + "\n")
    m = s["screen_mean"]
    print(a.id, f"Hlofo={100*m['primary_H_bal_lofo']:.2f} AUSUC={100*m['ausuc']:.2f} gate={s.get('gate_vs_control')}")


if __name__ == "__main__":
    main()
