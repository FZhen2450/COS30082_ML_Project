"""Stage J driver (v4, post-freeze exploratory): six-fold J rows, evaluation, and the H_RF source co-endpoint.

  run       python scripts/r2v4_stage_j.py run --row J1_clipL_last2 --seed 0 --out <E>/stage_j_full [--folds I6] [--max-steps 20]
  evaluate  python scripts/r2v4_stage_j.py evaluate --out <E>/stage_j_full [--parent J1_clipL_last2]
  hrf       python scripts/r2v4_stage_j.py hrf --row J1_clipL_last2 --seed 0 --out <E>/stage_j_full
  dry-run   python scripts/r2v4_stage_j.py dryrun --out <tmp>      (synthetic backend, six folds, CPU)

Rows and rules: research_r2_v4/matrices/stage_j_v4.json, amendment_1.json, amendment_2.json.
Evaluation refuses any row/seed without six verified fold receipts.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research_r2.candidates import derive_fold  # noqa: E402
from research_r2.metrics import GAMMA_GRID  # noqa: E402
from research_r2.packets import fold_contract, validate_packet  # noqa: E402
from research_r2.runner import scorer_from_scores  # noqa: E402
from research_r2_v4 import evaluate as ev  # noqa: E402
from research_r2_v4 import jrun  # noqa: E402
from research_r2_v4.common import EXP, INNER, LABEL, V4, append_jsonl, env, now, read_jsonl, restrict, scorer  # noqa: E402

D1 = "D1_synthComp_norm"
PARENTS = ["J1_clipL_last2", "J2_clipL_last4", "J3_clipL_lora8_mlp"]
CHILDREN = ["J4a_best_geometric_aug", "J4b_best_photometric_aug", "J5_best_source_balanced", "J6_best_crosscrop_supcon", "J7_best_crop_adversary"]
INDEXSETS = V4 / "stage_h_20261114" / "D1_synthComp_norm" / "indexsets"


def _script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def frozen_clip(bench):
    rec = torch.load(EXP / "features" / "clip_l14.pt", map_location="cpu", weights_only=False)
    x = rec["features"].float().clone()
    x[bench["outer_exclusion_rows"]] = float("nan")
    return x


def backend_for(cfg):
    return {"clip_l14": jrun.ClipBackend, "convnext": jrun.ConvNextBackend, "synthetic": jrun.SyntheticBackend}[cfg.backbone]()


# ----------------------------------------------------------------------------- run
def cmd_run(a):
    frame, bench, F = env()
    rows = jrun.j_rows(a.parent)
    if a.row not in rows:
        raise SystemExit(f"unknown row {a.row} (J4a-J7 need --parent)")
    cfg = rows[a.row]
    if a.max_steps:
        cfg.max_steps = a.max_steps
    folds = [F[f] for f in (a.folds or INNER)]
    access = jrun.ImageAccess(frame, a.images)
    base = Path(a.out)
    (base / "run_log.jsonl").parent.mkdir(parents=True, exist_ok=True)
    status = jrun.run_row(cfg, a.seed, folds, frame, bench, backend_for(cfg), access, base,
                          frozen_feats=frozen_clip(bench) if cfg.concat_frozen else None,
                          log=lambda s: (print(s, flush=True), append_jsonl(base / "run_log.jsonl", {"t": now(), "msg": s})))
    append_jsonl(base / "run_log.jsonl", {"t": now(), "row": a.row, "seed": a.seed, "status": status})
    print(json.dumps(status, indent=1))
    if any(not v.startswith("ok") for v in status.values()):
        sys.exit(2)


# ----------------------------------------------------------------------------- evaluation
def table_from_packets(frame, F, packets, name):
    sc = {}
    for fn, (p, _) in packets.items():
        z = np.load(p)
        validate_packet(z, fold_contract(frame, F[fn]))
        sc[fn] = [scorer_from_scores(frame, F[fn], z["seen"], z["seen_truth"], z["unseen"], z["unseen_truth"], z["cand_ids"])]
    return ev.RowTable(name, sc, ["single"])


def mean_summary(summaries):
    folds = list(summaries[0]["per"])
    per = {f: {k: float(np.mean([s["per"][f][k] for s in summaries])) for k in ("H_bal", "ausuc", "unseen_healthy_shortcut")} for f in folds}
    return {"per": per, "primary": float(np.mean([s["primary"] for s in summaries])),
            "ausuc": float(np.mean([s["ausuc"] for s in summaries])), "hs": float(np.mean([s["hs"] for s in summaries]))}


def nested_with_parent_rule(T, control, candidates, folds, children_parent):
    """ev.nested, plus amendment_2: J4a-J7 are selectable in a held fold only if the inner-fold parent (best of
    J1-J3 on the other five folds) equals the parent they were trained from."""
    out = []
    for k in folds:
        inner = [f for f in folds if f != k]
        S = {r: T[r].summary(inner) for r in [control] + candidates}
        par = [p for p in PARENTS if p in T]
        inner_parent = max(par, key=lambda r: (S[r]["primary"], -PARENTS.index(r))) if par else None
        eligible = [r for r in candidates if r not in children_parent or children_parent[r] == inner_parent]
        passing = [r for r in eligible if ev.gate(S[r], S[control], inner)["pass"]]
        sel = max(passing, key=lambda r: (S[r]["primary"], -candidates.index(r))) if passing else control

        def held(r):
            ci, gi = T[r].choose(inner)
            return T[r].at(k, ci, gi)["H_bal"], float(GAMMA_GRID[gi])
        hs, gs = held(sel); hc, gc = held(control)
        out.append({"held_fold": k, "inner_parent": inner_parent, "excluded_children": sorted(set(candidates) - set(eligible)),
                    "selected": sel, "H_bal": hs, "gamma": gs, "control_H_bal": hc, "control_gamma": gc})
    d = [100 * (n["H_bal"] - n["control_H_bal"]) for n in out]
    return {"held": out, "mean_diff_pp": float(np.mean(d)), "folds_better": int(sum(x > 0 for x in d)),
            "meets_gate_thresholds": bool(np.mean(d) >= 1.0 and sum(x > 0 for x in d) >= len(folds) - 1)}


def cmd_evaluate(a, d1_table=None):
    frame, bench, F = env()
    base = Path(a.out)
    si = _script("r2v4_stage_i")
    T = {D1: d1_table or si.table(D1, frame, F, Path("."))}
    seeds, refused = {}, {}
    for rdir in sorted(p for p in base.iterdir() if p.is_dir() and p.name.startswith("J")):
        for sdir in sorted(rdir.glob("seed*")):
            s = int(sdir.name[4:])
            try:
                pk = jrun.row_packets(base, rdir.name, s, INNER)
            except jrun.JCheckError as e:
                refused[f"{rdir.name}/seed{s}"] = str(e)
                continue
            seeds.setdefault(rdir.name, {})[s] = table_from_packets(frame, F, pk, f"{rdir.name}/seed{s}")
    res = {"label": LABEL, "control": D1, "refused_incomplete": refused, "rows": {}, "evaluated": now()}
    for r, by_seed in seeds.items():
        T[r] = by_seed[min(by_seed)]  # seed 0 table drives screening and nested (declared)
        S0 = T[r].summary(INNER); Sd = T[D1].summary(INNER)
        e = {"seeds": sorted(by_seed), "seed0": {"primary": S0["primary"], "ausuc": S0["ausuc"], "gate_vs_D1": ev.gate(S0, Sd, INNER),
                                                  "bootstrap_vs_D1": ev.paired_bootstrap(T[r], T[D1], INNER),
                                                  "false_healthy": ev.false_healthy(T[r], INNER)}}
        if len(by_seed) >= 3:
            Sm = mean_summary([t.summary(INNER) for t in by_seed.values()])
            e["three_seed"] = {"primary": Sm["primary"], "gate_vs_D1": ev.gate(Sm, Sd, INNER),
                               "nested_per_seed": {s: nested_with_parent_rule(dict(T, **{r: t}), D1, [r], INNER, {})["mean_diff_pp"]
                                                   for s, t in by_seed.items()}}
        res["rows"][r] = e
    cands = [r for r in seeds]
    children_parent = {c: a.parent for c in CHILDREN if c in seeds} if a.parent else {}
    if cands:
        res["stage_nested"] = nested_with_parent_rule(T, D1, cands, INNER, children_parent)
        par = [p for p in PARENTS if p in seeds]
        if par:
            res["parent_by_rule"] = max(par, key=lambda r: (T[r].summary(INNER)["primary"], -PARENTS.index(r)))
    (base / "evaluation.json").write_text(json.dumps(res, indent=1, default=float))
    for r, e in res["rows"].items():
        g = e["seed0"]["gate_vs_D1"]
        print(f"{r:30s} seed0 H_bal {100*e['seed0']['primary']:6.2f} {g['mean_gain_pp']:+6.2f}pp {g['folds_improved']}/6 pass={g['pass']}")
    print("refused:", json.dumps(refused, indent=1))
    return res


# ----------------------------------------------------------------------------- H_RF source co-endpoint
def cmd_hrf(a):
    frame, bench, F = env()
    sh = _script("r2v4_stage_h")
    base = Path(a.out)
    pk = jrun.row_packets(base, a.row, a.seed, INNER)  # refuses incomplete rows
    cfgs = {json.dumps(rec["config"], sort_keys=True) for _, rec in pk.values()}
    if len(cfgs) != 1:
        raise jrun.JCheckError(f"{a.row} seed{a.seed}: folds were run with different configurations")
    cfg = jrun.JConfig(**json.loads(cfgs.pop()))  # configuration as recorded in the verified receipts
    feas = json.loads((ROOT / "artifacts/r2/v4/feasibility_20261114/source_feasibility.json").read_text())
    pairs = [(p["fold"], p["source"]) for p in feas["useful_pairs"]][:getattr(a, "max_pairs", None)]
    out = base / "hrf" / f"{a.row}__seed{a.seed}"
    done = {(d["fold"], d["source"]) for d in read_jsonl(out / "pairs.jsonl")}
    frz = frozen_clip(bench) if cfg.concat_frozen else None
    src = frame.source.to_numpy()
    feats_by_fold = {}
    for fn, (pkt, rec) in pk.items():
        adir = pkt.parent
        meta = {"row": a.row, "seed": a.seed, "fold": fn, "config_sha256": rec["config_sha256"], "checkpoint_sha256": rec["checkpoint_sha256"],
                "rows_sha256": None, "backbone": None, "schema": "r2v4-j-fold-features-v1"}
        st = torch.load(adir / "features.pt", map_location="cpu", weights_only=False)
        meta["rows_sha256"], meta["backbone"] = st["meta"]["rows_sha256"], st["meta"]["backbone"]
        x, rws = jrun.load_feature_cache(adir / "features.pt", meta)
        full = torch.full((len(frame), 2, x.shape[-1]), float("nan")); full[rws] = x.float()
        if frz is not None:
            full = torch.cat([frz / frz.norm(dim=-1, keepdim=True), full / full.norm(dim=-1, keepdim=True)], -1)
        feats_by_fold[fn] = full
    tbl = table_from_packets(frame, F, pk, a.row)
    g_sel = float(GAMMA_GRID[tbl.choose(INNER)[1]])
    gammas = [0.0, g_sel]
    from research_r2.methods import RECIPES
    from research_r2_v4.common import score_fold
    rec_fn = RECIPES["synth_composition"]
    for fn, s in pairs:
        if (fn, s) in done:
            continue
        t0 = time.time()
        r = {"row": a.row, "seed": a.seed, "fold": fn, "source": s, "label": LABEL + "; H_RF (representation fixed)", "gammas": gammas}
        try:
            fold = F[fn]; feats = feats_by_fold[fn]
            base_arm = score_fold(frame, fold, feats, None, rec_fn, jrun.D1_CFG)
            f_abl = derive_fold(frame, fold, [x for x in fold["fitting_rows"] if src[x] != s])
            abl = score_fold(frame, f_abl, feats, None, rec_fn, jrun.D1_CFG)
            sr, ur = np.asarray(f_abl["seen_val_rows"]), np.asarray(f_abl["unseen_val_rows"])
            ns, us = src[sr] == s, src[ur] == s
            r.update({"n_novel_source_seen": int(ns.sum()), "n_source_unseen": int(us.sum()),
                      "emptied_compositions": sorted(set(int(c) for c in base_arm["cand_ids"]) - set(int(c) for c in abl["cand_ids"])),
                      "universe_common": int(len(abl["cand_ids"])), "n_fitting_removed": len(fold["fitting_rows"]) - len(f_abl["fitting_rows"])})
            r["base"] = sh.arm_metrics(restrict(frame, base_arm, f_abl), gammas, ns, us)
            r["ablated"] = sh.arm_metrics(scorer(frame, abl), gammas, ns, us)
            r["random"] = []
            for k in range(3):
                saved = json.loads((INDEXSETS / f"{fn}_{s}_seed{k}.json").read_text())
                kept, picks, lv = sh.matched_removal(frame, fold["fitting_rows"], s, seed=k)
                if picks != saved["removed_rows"]:
                    raise jrun.JCheckError(f"matched control {fn}/{s}/seed{k} does not reproduce the Stage H index set")
                rnd = score_fold(frame, derive_fold(frame, fold, kept), feats, None, rec_fn, jrun.D1_CFG)
                r["random"].append({"seed": k, "match_levels": lv, **sh.arm_metrics(restrict(frame, rnd, f_abl), gammas, ns, us)})
            r["status"] = "ok"
        except Exception as e:
            r["status"] = "failed"; r["error"] = f"{type(e).__name__}: {e}"
        r["seconds"] = time.time() - t0
        append_jsonl(out / "pairs.jsonl", r)
        print(a.row, fn, s, r["status"], flush=True)


# ----------------------------------------------------------------------------- synthetic dry run
def cmd_dryrun(a):
    """Synthetic backend, real folds and real D1 downstream; exercises run -> evaluate -> hrf end to end on CPU."""
    frame, bench, F = env()
    base = Path(a.out)
    loader = jrun.synthetic_loader(frame)
    access = jrun.ImageAccess(frame, None, loader=loader)
    cfgs = [jrun.JConfig("Jsyn_A", backbone="synthetic", trainable="synthetic", epochs=1, batch=256, lr_backbone=1e-2, workers=0),
            jrun.JConfig("Jsyn_B", backbone="synthetic", trainable="synthetic", epochs=1, batch=256, lr_backbone=1e-2, workers=0,
                         extra="adversary", source_balanced=True)]
    statuses = {}
    for cfg in cfgs:
        statuses[cfg.row] = jrun.run_row(cfg, 0, [F[f] for f in INNER], frame, bench, jrun.SyntheticBackend(), access, base, device="cpu")
    return statuses, access.opened


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "evaluate", "hrf", "dryrun"):
        p = sub.add_parser(name)
        p.add_argument("--out", required=True)
        p.add_argument("--parent", choices=PARENTS)
        if name in ("run", "hrf"):
            p.add_argument("--row", required=True)
            p.add_argument("--seed", type=int, default=0)
        if name == "hrf":
            p.add_argument("--max-pairs", type=int, help="tests / dry runs only")
        if name == "run":
            p.add_argument("--images", default=str(ROOT.parents[2] / "DPD_Small_dataset"))
            p.add_argument("--folds", nargs="+")
            p.add_argument("--max-steps", type=int)
    a = ap.parse_args()
    {"run": cmd_run, "evaluate": cmd_evaluate, "hrf": cmd_hrf, "dryrun": lambda a: print(cmd_dryrun(a))}[a.cmd](a)


if __name__ == "__main__":
    main()
