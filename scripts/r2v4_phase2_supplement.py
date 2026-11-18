"""Stage J Phase 2 supplement (v4, post-freeze exploratory, descriptive): independent recomputation of the
three-seed gate from the score packets, the requested paired comparisons of J2 vs J1 and J2 vs J3, and the
three-seed pooling of the H_RF source diagnostic.

  python scripts/r2v4_phase2_supplement.py --eval <E>/stage_j_phase2_20261115/evaluation/evaluation.json \
      --runs <E>/stage_j_phase1_20261114 <E>/stage_j_phase2_20261115 --out <E>/stage_j_phase2_20261115/supplement.json

The independent check does not use RowTable.choose / summary / ev.gate: it rebuilds each fold's FoldScorer from the
packet, picks each fold's LOFO gamma by its own argmax (first maximum = smallest gamma, the declared tie rule),
averages over seeds per fold and applies the thresholds written out here.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from research_r2.metrics import GAMMA_GRID  # noqa: E402
from research_r2.runner import scorer_from_scores  # noqa: E402
from research_r2_v4 import confirm  # noqa: E402
from research_r2_v4.common import INNER, LABEL, env  # noqa: E402

ROWS = ["J1_clipL_last2", "J2_clipL_last4", "J3_clipL_lora8_mlp"]
SEEDS = list(confirm.CONFIRM_SEEDS)


def base_of(runs, row, seed):
    hits = [Path(b) for b in runs if (Path(b) / row / f"seed{seed}").is_dir()]
    if len(hits) != 1:
        raise SystemExit(f"{row} seed{seed}: found in {len(hits)} run dirs")
    return hits[0]


def fold_scorers(frame, F, runs, row, seed):
    v = confirm.verified_row_seed(base_of(runs, row, seed), row, seed, INNER)
    out = {}
    for fn, (p, _) in v.items():
        z = np.load(p)
        out[fn] = scorer_from_scores(frame, F[fn], z["seen"], z["seen_truth"], z["unseen"], z["unseen_truth"], z["cand_ids"])
    return out


def lofo(sc):
    H = {f: np.array([sc[f].at(float(g))["H_bal"] for g in GAMMA_GRID]) for f in INNER}
    out = {}
    for k in INNER:
        m = np.mean([H[f] for f in INNER if f != k], 0)
        gi = int(np.flatnonzero(m == m.max())[0])
        r = sc[k].at(float(GAMMA_GRID[gi]))
        out[k] = {"gamma": float(GAMMA_GRID[gi]), "H_bal": r["H_bal"], "hs": r["unseen_healthy_shortcut"], "ausuc": sc[k].ausuc()}
    return out


def independent_gate(frame, F, runs, d1_sc):
    d1 = lofo(d1_sc)
    res = {}
    for row in ROWS:
        per_seed = {s: lofo(fold_scorers(frame, F, runs, row, s)) for s in SEEDS}
        mean = {f: {k: float(np.mean([per_seed[s][f][k] for s in SEEDS])) for k in ("H_bal", "hs", "ausuc")} for f in INNER}
        d = {f: 100 * (mean[f]["H_bal"] - d1[f]["H_bal"]) for f in INNER}
        ausuc_d = np.mean([mean[f]["ausuc"] for f in INNER]) - np.mean([d1[f]["ausuc"] for f in INNER])
        hs_d = 100 * (np.mean([mean[f]["hs"] for f in INNER]) - np.mean([d1[f]["hs"] for f in INNER]))
        g = {"mean_gain_pp": float(np.mean(list(d.values()))), "folds_improved": int(sum(x > 0 for x in d.values())),
             "ausuc_delta": float(ausuc_d), "shortcut_delta_pp": float(hs_d), "per_fold_delta_pp": d,
             "per_seed_gammas": {s: {f: per_seed[s][f]["gamma"] for f in INNER} for s in SEEDS}}
        g["pass"] = bool(g["mean_gain_pp"] >= 1.0 and g["folds_improved"] >= 5 and ausuc_d >= 0 and hs_d <= 5.0)
        res[row] = g
    return res


def hrf_three_seed(dirs):
    """Pool the H_RF source / size effects over seeds: per (fold, source) pair, mean over seeds; then the cluster
    bootstrap over sources of scripts/r2v4_stage_h_summary.py. Uses gamma 0 and each seed's selected gamma."""
    sh = __import__("r2v4_stage_h_summary")
    out = {}
    for row in ROWS:
        by_pair = {}
        for s in SEEDS:
            pairs = [json.loads(l) for l in (dirs[s] / f"{row}__seed{s}" / "pairs.jsonl").read_text().splitlines() if l.strip()]
            ok = [p for p in pairs if p["status"] == "ok"]
            if len(ok) != len(pairs) or len({(p["fold"], p["source"]) for p in ok}) != len(ok):
                raise SystemExit(f"{row} seed{s}: failed or duplicate H_RF pairs")
            for p in ok:
                gs = sorted((k for k in p["base"] if k.startswith("g")), key=lambda k: float(k[1:]))
                by_pair.setdefault((p["fold"], p["source"]), {})[s] = (p, {"g0": gs[0], "gsel": gs[-1]})
        if any(len(v) != len(SEEDS) for v in by_pair.values()):
            raise SystemExit(f"{row}: H_RF pairs differ across seeds")
        row_out = {"pairs": len(by_pair), "selected_gammas": {}}
        for gname in ("g0", "gsel"):
            pooled = {}
            for m in sh.METRICS + ["ausuc"]:
                for eff in ("source_effect", "size_effect"):
                    vals = {}
                    for (fold, src), per in by_pair.items():
                        es = []
                        for s, (p, gk) in per.items():
                            e = sh.pair_effects(p, gk[gname]).get(m)
                            if e is None or (m in sh.SUBSET_N and p[sh.SUBSET_N[m]] < 10):
                                es = None
                                break
                            es.append(e[eff])
                        if es:
                            vals.setdefault(src, []).append(float(np.mean(es)))
                    pooled[f"{m}:{eff}"] = sh.boot(vals)
            row_out[gname] = pooled
        row_out["selected_gammas"] = {s: sorted({per[s][1]["gsel"] for per in by_pair.values()}) for s in SEEDS}
        out[row] = row_out
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--hrf", nargs="+", required=True, help="hrf dirs holding seed0 and seeds 1-2")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    frame, bench, F = env()
    ev_json = json.loads(Path(a.eval).read_text())
    si = __import__("r2v4_stage_i")
    import r2v4_stage_j as sj
    d1 = si.table(sj.D1, frame, F, Path("."))
    assert len(d1.configs) == 1, "D1 has one configuration"
    rep = {"label": LABEL + "; Phase 2 supplement, descriptive", "independent_three_seed_gate": independent_gate(frame, F, a.runs, {f: d1.sc[f][0] for f in INNER})}
    rep["independent_matches_evaluator"] = {}
    for row, g in rep["independent_three_seed_gate"].items():
        e = ev_json["rows"][row]["three_seed"]["gate_vs_D1"]
        rep["independent_matches_evaluator"][row] = {
            "mean_gain_pp_absdiff": abs(g["mean_gain_pp"] - e["mean_gain_pp"]), "folds_improved_equal": g["folds_improved"] == e["folds_improved"],
            "ausuc_delta_absdiff": abs(g["ausuc_delta"] - e["ausuc_delta"]), "shortcut_absdiff": abs(g["shortcut_delta_pp"] - e["shortcut_delta_pp"]),
            "pass_equal": g["pass"] == e["pass"]}
    tables = {r: {s: sj.table_from_packets(frame, F, confirm.verified_row_seed(base_of(a.runs, r, s), r, s, INNER), f"{r}/{s}") for s in SEEDS} for r in ROWS}
    rep["paired_J2"] = {f"J2_clipL_last4_vs_{o}": confirm.paired_rows(tables["J2_clipL_last4"], tables[o], INNER) for o in ("J1_clipL_last2", "J3_clipL_lora8_mlp")}
    hdirs = {}
    for s in SEEDS:
        hits = [Path(h) for h in a.hrf if any((Path(h) / f"{r}__seed{s}").is_dir() for r in ROWS)]
        hdirs[s] = hits[0]
    rep["hrf_three_seed"] = hrf_three_seed(hdirs)
    Path(a.out).write_text(json.dumps(rep, indent=1, default=float))
    for row, m in rep["independent_matches_evaluator"].items():
        print(row, m)
    for k, v in rep["paired_J2"].items():
        print(k, "mean %+.2f" % v["mean_diff_pp"], "folds J2 better", v["folds_a_better"], "seeds J2 better", v["seeds_a_better"],
              "boot3 %+.2f [%+.2f, %+.2f] p<=0 %.3f" % (v["bootstrap_three_seed"]["point_pp"], *v["bootstrap_three_seed"]["ci95_pp"], v["bootstrap_three_seed"]["p_le_0"]),
              {s: round(x["mean_diff_pp"], 2) for s, x in v["per_seed"].items()}, {f: round(x, 2) for f, x in v["per_fold_mean_diff_pp"].items()})
    for row, v in rep["hrf_three_seed"].items():
        for g in ("g0", "gsel"):
            for m in ("novel_seen_top1", "source_unseen_top1", "H_bal", "seen_top1", "unseen_top1_macro"):
                s, z = v[g][f"{m}:source_effect"], v[g][f"{m}:size_effect"]
                print(row, g, m, "source %+.2f [%+.2f, %+.2f]" % (100 * s["mean"], 100 * s["ci95"][0], 100 * s["ci95"][1]),
                      "size %+.2f [%+.2f, %+.2f]" % (100 * z["mean"], 100 * z["ci95"][0], 100 * z["ci95"][1]), s["n_pairs"], s["n_sources"])
        print(row, "selected gammas", v["selected_gammas"])


if __name__ == "__main__":
    main()
