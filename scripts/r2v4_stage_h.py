"""Stage H (v4, post-freeze exploratory): size-controlled source-shift diagnostic. Declared in
research_r2_v4/matrices/stage_h_v4.json and research_r2_v4/amendment_1.json ("stage_H").

For each row and each declared (fold, source s) pair:
  base        standard fold
  ablated     every fitting image of s removed (derive_fold keeps the v3 candidate policy)
  random k    k = 0, 1, 2: the same number of fitting images removed from other sources, matched greedily
              (same composition > same crop > same disease > any), never emptying a composition
All arms are scored on the ABLATED fold's universe and rows (research_r2_v4.common.restrict).
Also a composition-stratified learning curve (fractions 0.25-1.0, seeds 0-2, all six folds).

Results are appended to <out>/<row>/pairs.jsonl (resumable); sampled index sets go to <out>/<row>/indexsets/.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research_r2.candidates import derive_fold  # noqa: E402
from research_r2.metrics import GAMMA_GRID  # noqa: E402
from research_r2.methods import RECIPES  # noqa: E402
from research_r2_v4.common import (INNER, LABEL, append_jsonl, env, now, poisoned_features, read_jsonl,  # noqa: E402
                                   restrict, score_fold, scorer)

ROWS = {"B1_clipL_factorised": (["clip_l14"], "factorised", {}, 10.5),
        "D1_synthComp_norm": (["clip_l14"], "synth_composition", {"fuse": 1.0, "normalise": True}, 8.0),
        "F10_concat_convnext": (["clip_l14", "tv_convnext_base"], "synth_composition", {"fuse": 1.0, "normalise": True}, 8.0)}
LEVELS = ["composition", "crop", "disease", "any"]
# Stage I rows with mean H_bal >= D1 (amendment_1 "source_behaviour"); selected gamma = the row's pooled stage I gamma
I_ROWS_H = {"I6_text_direction_synthesis": ("clip_l14", "synth_v4", {"text_frac": 0.5, "text_lambda": 1.0}),
            "I7_multilayer_clipL": ("ml:16,20,24", "d1", {"fuse": 1.0, "normalise": True}),
            "I7b_multilayer_20_24": ("ml:20,24", "d1", {"fuse": 1.0, "normalise": True})}


def i_row_setup(rid, bench, stage_i_dir):
    import importlib.util
    spec = importlib.util.spec_from_file_location("r2v4_stage_i", ROOT / "scripts" / "r2v4_stage_i.py")
    si = importlib.util.module_from_spec(spec); spec.loader.exec_module(si)
    from research_r2_v4.methods_v4 import raw_fit_stats, score_synth_v4
    feat, fn, cfg = I_ROWS_H[rid]
    feats, text, ident = si.load_feat(feat, bench)
    cur = json.loads((Path(stage_i_dir) / "curves" / f"{rid}.json").read_text())
    pooled = np.mean([cur[f]["single"]["H_bal"] for f in INNER], 0)
    g_sel = float(GAMMA_GRID[int(np.argmax(pooled))])
    if fn == "synth_v4":
        prep = lambda fd, fold, x: fd.extra.__setitem__("raw_fit", raw_fit_stats(x[fold["fitting_rows"]]))  # noqa: E731
        return feats, text, ident, score_synth_v4, cfg, g_sel, prep
    return feats, text, ident, RECIPES["synth_composition"], cfg, g_sel, None


def matched_removal(frame, fit_rows, src, seed):
    """Remove len(fitting rows of src) images from other sources, greedily matched to the removed images."""
    fit = np.asarray(fit_rows)
    is_s = frame.source.to_numpy()[fit] == src
    removed, pool = fit[is_s], fit[~is_s]
    comp = frame.comp_id.to_numpy(); crop = frame.crop_id.to_numpy(); dis = frame.disease_id.to_numpy()
    rng = np.random.default_rng(seed)
    avail = np.ones(len(pool), bool)
    pc, pr, pd_ = comp[pool], crop[pool], dis[pool]
    cnt = {}
    for c in pc:
        cnt[c] = cnt.get(c, 0) + 1
    picks, levels = [], []
    for r in rng.permutation(removed):
        ok_cnt = np.array([cnt[c] >= 2 for c in pc]) & avail
        chosen = None
        for lvl, key in zip(LEVELS, (pc == comp[r], pr == crop[r], pd_ == dis[r], np.ones(len(pool), bool))):
            cand = np.flatnonzero(ok_cnt & key)
            if len(cand):
                chosen = int(rng.choice(cand)); levels.append(lvl)
                break
        if chosen is None:
            raise RuntimeError("no removable image left (would empty a composition)")
        avail[chosen] = False
        cnt[pc[chosen]] -= 1
        picks.append(int(pool[chosen]))
    kept = sorted(set(fit.tolist()) - set(picks))
    return kept, sorted(picks), {l: levels.count(l) for l in LEVELS}


def arm_metrics(sc, gammas, rows_seen_mask=None, rows_uns_mask=None):
    out = {"ausuc": sc.ausuc()}
    for g in gammas:
        r = sc.at(g)
        m = {k: r[k] for k in ("seen_top1", "unseen_top1_macro", "unseen_top1_micro", "H_bal", "H_micro",
                               "unseen_healthy_shortcut")}
        for name, part, mask in (("novel_seen_top1", "seen", rows_seen_mask), ("source_unseen_top1", "unseen", rows_uns_mask)):
            if mask is not None and mask.any():
                pred = sc.top1_pred(part, g)
                m[name] = float(np.mean(pred[mask] == sc.parts[part]["truth"][mask]))
                m[name.replace("top1", "margin_mean")] = float(np.mean(sc.parts[part]["d"][mask]))
            else:
                m[name] = None
        out[f"g{g:g}"] = m
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", nargs="+", default=list(ROWS))
    ap.add_argument("--pairs", default="artifacts/r2/v4/feasibility_20261114/source_feasibility.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-curve", action="store_true")
    ap.add_argument("--stage-i-dir", default=str(ROOT.parents[1] / "experiments/r2-compositional/v4/stage_i_20261114"))
    a = ap.parse_args()
    frame, bench, F = env()
    feas = json.loads((ROOT / a.pairs).read_text())
    pairs = [(p["fold"], p["source"]) for p in feas["useful_pairs"]]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "ineligible.json").write_text(json.dumps(
        [dict(fold=r["fold"], source=r["source"], reason=("not ablatable: held-out crop/disease coverage lost" if not r["ablatable"]
              else "fewer than 10 evaluation images")) for r in feas["rows"] if (r["fold"], r["source"]) not in set(pairs)], indent=1))
    src_all = frame.source.to_numpy()
    for rid in a.rows:
        if rid in ROWS:
            bbs, recipe_name, cfg, g_sel = ROWS[rid]
            recipe, prep = RECIPES[recipe_name], None
            feats, text, _, ident = poisoned_features(bench, bbs)
        else:
            feats, text, ident, recipe, cfg, g_sel, prep = i_row_setup(rid, bench, a.stage_i_dir)
        gammas = [0.0, g_sel]
        sf = lambda f: score_fold(frame, f, feats, text, recipe, cfg, prep=prep)  # noqa: E731
        rdir = out / rid
        done = {(d["fold"], d["source"]) for d in read_jsonl(rdir / "pairs.jsonl")}
        base_cache = {}
        for fn, s in pairs:
            if (fn, s) in done:
                continue
            t0 = time.time()
            rec = {"row": rid, "fold": fn, "source": s, "label": LABEL, "gammas": gammas, "started": now()}
            try:
                fold = F[fn]
                if fn not in base_cache:
                    base_cache[fn] = sf(fold)
                base = base_cache[fn]
                fit2 = [r for r in fold["fitting_rows"] if src_all[r] != s]
                f_abl = derive_fold(frame, fold, fit2)
                abl = sf(f_abl)
                seen_rows = np.asarray(f_abl["seen_val_rows"]); uns_rows = np.asarray(f_abl["unseen_val_rows"])
                ns_mask = src_all[seen_rows] == s if len(seen_rows) else np.zeros(0, bool)
                us_mask = src_all[uns_rows] == s
                rec.update({"n_fitting_removed": len(fold["fitting_rows"]) - len(fit2),
                            "universe_base": int(len(base["cand_ids"])), "universe_common": int(len(abl["cand_ids"])),
                            "emptied_compositions": sorted(set(int(c) for c in base["cand_ids"]) - set(int(c) for c in abl["cand_ids"])),
                            "n_seen_common": int(len(seen_rows)), "n_seen_dropped": len(f_abl.get("dropped_seen_val_rows", [])),
                            "n_unseen": int(len(uns_rows)), "n_novel_source_seen": int(ns_mask.sum()),
                            "n_novel_source_seen_compositions": int(len(set(frame.comp_id.to_numpy()[seen_rows[ns_mask]]))),
                            "n_source_unseen": int(us_mask.sum())})
                rec["base"] = arm_metrics(restrict(frame, base, f_abl), gammas, ns_mask, us_mask)
                rec["ablated"] = arm_metrics(scorer(frame, abl), gammas, ns_mask, us_mask)
                rec["random"] = []
                for k in range(3):
                    kept, picks, lv = matched_removal(frame, fold["fitting_rows"], s, seed=k)
                    f_rnd = derive_fold(frame, fold, kept)
                    assert list(f_rnd["candidate_composition_ids"]) == list(fold["candidate_composition_ids"]), "control emptied a composition"
                    idx = rdir / "indexsets" / f"{fn}_{s}_seed{k}.json"
                    idx.parent.mkdir(parents=True, exist_ok=True)
                    idx.write_text(json.dumps({"fold": fn, "source": s, "seed": k, "removed_rows": picks, "match_levels": lv}))
                    rnd = sf(f_rnd)
                    rec["random"].append({"seed": k, "match_levels": lv, "indexset": idx.name,
                                          **arm_metrics(restrict(frame, rnd, f_abl), gammas, ns_mask, us_mask)})
                rec["status"] = "ok"
            except Exception as e:  # recorded, never hidden
                rec["status"] = "failed"; rec["error"] = f"{type(e).__name__}: {e}"; rec["trace"] = traceback.format_exc()[-1500:]
            rec["seconds"] = time.time() - t0
            append_jsonl(rdir / "pairs.jsonl", rec)
            g = rec.get("ablated", {}).get(f"g{g_sel:g}", {})
            print(f"{rid} {fn} {s:5s} {rec['status']} H_bal abl={g.get('H_bal', float('nan')):.4f} [{rec['seconds']:.0f}s]", flush=True)

        if a.skip_curve or (rdir / "learning_curve.json").exists():
            continue
        t0 = time.time()
        curve = {"label": LABEL, "fractions": {}}
        for frac in (0.25, 0.5, 0.75, 1.0):
            per_seed = []
            for seed in range(3):
                rng = np.random.default_rng(seed)
                tables = {}
                for fn in INNER:
                    fr = frame.loc[F[fn]["fitting_rows"]]
                    keep = []
                    for _, idx in fr.groupby("comp_id").groups.items():
                        idx = sorted(int(i) for i in idx)
                        keep += [int(x) for x in rng.choice(idx, max(1, math.ceil(frac * len(idx))), replace=False)]
                    f3 = derive_fold(frame, F[fn], keep)
                    assert list(f3["candidate_composition_ids"]) == list(F[fn]["candidate_composition_ids"])
                    sc = scorer(frame, sf(f3))
                    tables[fn] = {"H": np.array([sc.at(g)["H_bal"] for g in GAMMA_GRID]), "n_fit": len(keep), "ausuc": sc.ausuc()}
                lofo = []
                for fn in INNER:
                    gi = int(np.argmax(np.mean([tables[o]["H"] for o in INNER if o != fn], 0)))
                    lofo.append(float(tables[fn]["H"][gi]))
                pooled = np.mean([tables[f]["H"] for f in INNER], 0)
                per_seed.append({"seed": seed, "H_bal_lofo": float(np.mean(lofo)), "pooled_oracle_gamma": float(GAMMA_GRID[pooled.argmax()]),
                                 "H_bal_pooled_oracle": float(pooled.max()), "H_bal_g_sel": float(pooled[int(np.argmin(np.abs(GAMMA_GRID - g_sel)))]),
                                 "H_bal_g0": float(pooled[0]), "ausuc": float(np.mean([tables[f]["ausuc"] for f in INNER])),
                                 "mean_fitting_images": float(np.mean([tables[f]["n_fit"] for f in INNER]))})
                if frac == 1.0:
                    break  # fraction 1.0 is deterministic (all rows kept)
            curve["fractions"][str(frac)] = per_seed
            print(rid, "curve", frac, [round(p["H_bal_lofo"], 4) for p in per_seed], flush=True)
        curve["seconds"] = time.time() - t0
        (rdir / "learning_curve.json").write_text(json.dumps(curve, indent=1))


if __name__ == "__main__":
    main()
