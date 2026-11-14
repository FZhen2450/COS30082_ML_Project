"""Aggregate Stage H (v4, post-freeze exploratory): per-pair and pooled source/size effects, control spread,
cluster bootstrap over sources, learning curves. Reads <dir>/<row>/pairs.jsonl and learning_curve.json.

Pooled quantities weight every (fold, source) pair equally. A metric enters the pool only for pairs with
>= 10 images in its evaluation subset (novel-source seen, or source-s unseen); whole-fold metrics use all pairs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

METRICS = ["novel_seen_top1", "source_unseen_top1", "seen_top1", "unseen_top1_macro", "H_bal", "H_micro"]
SUBSET_N = {"novel_seen_top1": "n_novel_source_seen", "source_unseen_top1": "n_source_unseen"}


def pair_effects(p, gkey):
    out = {}
    for m in METRICS:
        b, a = p["base"][gkey].get(m), p["ablated"][gkey].get(m)
        r = [x[gkey].get(m) for x in p["random"]]
        if b is None or a is None or any(v is None for v in r):
            continue
        out[m] = {"base": b, "ablated": a, "random_mean": float(np.mean(r)), "random_sd": float(np.std(r, ddof=1)),
                  "random_min": float(min(r)), "random_max": float(max(r)),
                  "source_effect": a - float(np.mean(r)), "size_effect": float(np.mean(r)) - b,
                  "ablated_outside_control_range": bool(a < min(r) or a > max(r))}
    b, a = p["base"]["ausuc"], p["ablated"]["ausuc"]; r = [x["ausuc"] for x in p["random"]]
    out["ausuc"] = {"base": b, "ablated": a, "random_mean": float(np.mean(r)), "random_sd": float(np.std(r, ddof=1)),
                    "source_effect": a - float(np.mean(r)), "size_effect": float(np.mean(r)) - b}
    return out


def boot(vals_by_source, reps=2000, seed=0):
    rng = np.random.default_rng(seed)
    srcs = sorted(vals_by_source)
    if not srcs:
        return None
    pts = [np.mean([v for s in srcs for v in vals_by_source[s]])]
    draws = []
    for _ in range(reps):
        pick = rng.choice(len(srcs), len(srcs), replace=True)
        draws.append(np.mean([v for i in pick for v in vals_by_source[srcs[i]]]))
    return {"mean": float(pts[0]), "ci95": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
            "n_pairs": int(sum(len(v) for v in vals_by_source.values())), "n_sources": len(srcs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    a = ap.parse_args()
    d = Path(a.dir)
    rep = {"label": "post-freeze exploratory (v4); diagnostic, not a gate", "rows": {}}
    md = []
    for rdir in sorted(p for p in d.iterdir() if (p / "pairs.jsonl").exists()):
        pairs = [json.loads(l) for l in (rdir / "pairs.jsonl").read_text().splitlines() if l.strip()]
        seen_keys, dup = set(), 0
        for p in pairs:
            k = (p["fold"], p["source"]); dup += k in seen_keys; seen_keys.add(k)
        ok = [p for p in pairs if p["status"] == "ok"]
        failed = [{"fold": p["fold"], "source": p["source"], "error": p.get("error")} for p in pairs if p["status"] != "ok"]
        gk = [k for k in ok[0]["base"] if k.startswith("g")] if ok else []
        row = {"pairs_total": len(pairs), "pairs_ok": len(ok), "duplicates": dup, "failed": failed,
               "match_levels_total": {}, "per_pair": [], "pooled": {}}
        for p in ok:
            for x in p["random"]:
                for lv, n in x["match_levels"].items():
                    row["match_levels_total"][lv] = row["match_levels_total"].get(lv, 0) + n
            row["per_pair"].append({"fold": p["fold"], "source": p["source"], "n_fitting_removed": p["n_fitting_removed"],
                                    "emptied_compositions": len(p["emptied_compositions"]), "universe_common": p["universe_common"],
                                    "n_novel_source_seen": p["n_novel_source_seen"], "n_source_unseen": p["n_source_unseen"],
                                    **{g: pair_effects(p, g) for g in gk}})
        for g in gk:
            row["pooled"][g] = {}
            for m in METRICS + ["ausuc"]:
                for eff in ("source_effect", "size_effect"):
                    by_src = {}
                    for p, pp in zip(ok, row["per_pair"]):
                        e = pp[g].get(m)
                        if e is None or (m in SUBSET_N and p[SUBSET_N[m]] < 10):
                            continue
                        by_src.setdefault(p["source"], []).append(e[eff])
                    row["pooled"][g][f"{m}:{eff}"] = boot(by_src)
            sds = [pp[g][m]["random_sd"] for pp in row["per_pair"] for m in ("novel_seen_top1", "H_bal") if m in pp[g]]
            row["pooled"][g]["mean_control_sd"] = float(np.mean(sds)) if sds else None
            row["pooled"][g]["novel_seen_ablated_outside_control_range"] = int(sum(
                pp[g]["novel_seen_top1"]["ablated_outside_control_range"] for p, pp in zip(ok, row["per_pair"])
                if "novel_seen_top1" in pp[g] and p["n_novel_source_seen"] >= 10))
        lc = rdir / "learning_curve.json"
        row["learning_curve"] = json.loads(lc.read_text()) if lc.exists() else None
        rep["rows"][rdir.name] = row
        md.append(f"\n### {rdir.name}: {len(ok)}/{len(pairs)} pairs ok, {len(failed)} failed, duplicates {dup}\n")
        md.append("| gamma | metric | source effect (pp) [95% CI] | size effect (pp) [95% CI] | pairs / sources |")
        md.append("|---|---|---|---|---|")
        for g in gk:
            for m in METRICS + ["ausuc"]:
                s, z = row["pooled"][g][f"{m}:source_effect"], row["pooled"][g][f"{m}:size_effect"]
                if s is None:
                    continue
                f = lambda b: f"{100*b['mean']:+.2f} [{100*b['ci95'][0]:+.2f}, {100*b['ci95'][1]:+.2f}]"  # noqa: E731
                md.append(f"| {g} | {m} | {f(s)} | {f(z)} | {s['n_pairs']} / {s['n_sources']} |")
        if row["learning_curve"]:
            md.append("\n| fitting fraction | H_bal (LOFO gamma), per seed | pooled-oracle gamma |")
            md.append("|---|---|---|")
            for fr, v in row["learning_curve"]["fractions"].items():
                hs = ", ".join("%.2f" % (100 * x["H_bal_lofo"]) for x in v)
                gs = ", ".join(str(x["pooled_oracle_gamma"]) for x in v)
                md.append(f"| {fr} | {hs} | {gs} |")
    (d / "summary.json").write_text(json.dumps(rep, indent=1))
    (d / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
