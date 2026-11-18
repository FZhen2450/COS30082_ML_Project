"""Stage J driver (v4, post-freeze exploratory): six-fold J rows, evaluation, and the H_RF source co-endpoint.

  run       python scripts/r2v4_stage_j.py run --row J1_clipL_last2 --seed 0 --out <E>/stage_j_full [--folds I6] [--max-steps 20]
  evaluate  python scripts/r2v4_stage_j.py evaluate --out <E>/eval_dir [--runs <E>/runs_a <E>/runs_b] [--parent J1_clipL_last2]
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
from research_r2.packets import PacketError  # noqa: E402
from research_r2_v4 import confirm  # noqa: E402
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


mean_summary = confirm.mean_summary


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


def _seed_detail(t, ctrl, frame, F, recs):
    """Per-seed descriptive metrics at each fold's own LOFO choice (and at gamma 0)."""
    lm = confirm.lofo_metrics(t, INNER)
    g0 = {k: t.sc[k][0].at(0.0) for k in INNER}
    keys = list(ev.KEYS) + ["H5_micro", "seen_top5", "unseen_top5_micro", "ausuc", "gamma"]
    return {"summary_lofo": {key: float(np.mean([lm[f][key] for f in INNER])) for key in keys}, "per_fold_lofo": lm,
            "summary_gamma0": {key: float(np.mean([g0[f][key] for f in INNER])) for key in ev.KEYS},
            "bootstrap_vs_D1": ev.paired_bootstrap(t, ctrl, INNER), "false_healthy": ev.false_healthy(t, INNER),
            "source_strata_seen": ev.source_strata(t, frame, F, INNER),
            "receipts": {fn: {k: rec.get(k) for k in ("seconds", "peak_gpu_gib_training", "peak_gpu_gib_total", "checkpoint_sha256",
                                                     "features_sha256", "packet_sha256", "trainable_tensors_changed", "reload_max_abs_diff",
                                                     "torch_seed", "config_sha256")}
                         | {"amp_skipped": (rec.get("grad_checks") or {}).get("skipped_steps"), "steps": (rec.get("grad_checks") or {}).get("steps"),
                            "git_head": rec["code"]["git_head"], "jrun_sha256": rec["code"]["jrun_sha256"],
                            "epoch_loss": [ep.get("train_loss_mean") for ep in rec.get("epochs", [])]}
                         for fn, (_, rec) in recs.items()}}


def cmd_evaluate(a, d1_table=None):
    """Seed-0 screen, three-seed confirmation and stage-level nested selection, as separate labelled outputs.
    Runs are read from --runs (default: --out); results are written to --out. Every (row, seed) must pass
    confirm.verified_row_seed; a confirmation needs exactly the declared seeds and passes cross-seed checks."""
    frame, bench, F = env()
    base = Path(a.out)
    base.mkdir(parents=True, exist_ok=True)
    bases = [Path(p) for p in (getattr(a, "runs", None) or [a.out])]
    si = _script("r2v4_stage_i")
    T = {D1: d1_table or si.table(D1, frame, F, Path("."))}
    declared = jrun.j_rows(getattr(a, "parent", None))
    runs, refused = confirm.discover(bases)
    tables, verified = {}, {}
    for r, by in sorted(runs.items()):
        for s, b in sorted(by.items()):
            try:
                v = confirm.verified_row_seed(b, r, s, INNER, declared.get(r))
                tables.setdefault(r, {})[s] = table_from_packets(frame, F, v, f"{r}/seed{s}")
                verified.setdefault(r, {})[s] = v
            except (jrun.JCheckError, PacketError) as e:
                refused[f"{r}/seed{s}"] = f"{type(e).__name__}: {e}"
    res = {"label": LABEL, "control": D1, "run_dirs": [b.name for b in bases],
           "declared_seeds": {"screen": list(confirm.SCREEN_SEEDS), "confirmation": list(confirm.CONFIRM_SEEDS)},
           "outputs": {"rows.*.seed0": "seed-0 SCREEN (declared screen; drives the parent rule and stage_nested)",
                       "rows.*.three_seed": "three-seed CONFIRMATION (amendment_2 three_seed_gate + nested thresholds)",
                       "stage_nested": "stage-level nested selection D1 -> best of the screened rows, seed-0 tables",
                       "stage_nested_three_seed": "stage-level nested selection computed per confirmation seed and averaged"},
           "refused_incomplete": refused, "rows": {}, "evaluated": now()}
    Sd = T[D1].summary(INNER)
    conf_tables = {}
    for r, by_seed in tables.items():
        present = sorted(set(runs.get(r, {})) | {int(k.split("/seed")[1]) for k in refused if k.startswith(r + "/seed")})
        if 0 not in by_seed:
            refused[f"{r}/screen"] = "no verified seed-0 run: the declared screen is seed 0 and no other seed substitutes for it"
            continue
        T[r] = by_seed[0]  # seed 0 drives screening, the parent rule and the stage nested estimate (declared)
        S0 = T[r].summary(INNER)
        g0 = {k: {key: T[r].sc[k][0].at(0.0)[key] for key in ev.KEYS} for k in INNER}
        e = {"seeds": sorted(by_seed), "seed_dirs_present": present, "seed0_label": "seed-0 screen", "seed0": {
            "primary": S0["primary"], "ausuc": S0["ausuc"],
            "summary_lofo": {key: float(np.mean([S0["per"][f][key] for f in INNER])) for key in ev.KEYS},
            "summary_gamma0": {key: float(np.mean([g0[f][key] for f in INNER])) for key in ev.KEYS},
            "per_fold": S0["per"], "gamma0_per_fold": g0,
            "gate_vs_D1": ev.gate(S0, Sd, INNER),
            "nested_vs_D1": nested_with_parent_rule({D1: T[D1], r: T[r]}, D1, [r], INNER, {}),
            "bootstrap_vs_D1": ev.paired_bootstrap(T[r], T[D1], INNER),
            "false_healthy": ev.false_healthy(T[r], INNER),
            "source_strata_seen": ev.source_strata(T[r], frame, F, INNER),
            "receipts": {fn: {k: rec.get(k) for k in ("seconds", "peak_gpu_gib_training", "peak_gpu_gib_total", "checkpoint_sha256",
                                                     "features_sha256", "packet_sha256", "trainable_tensors_changed", "reload_max_abs_diff")}
                         | {"amp_skipped": (rec.get("grad_checks") or {}).get("skipped_steps"), "steps": (rec.get("grad_checks") or {}).get("steps")}
                         for fn, (_, rec) in verified[r][0].items()}}}
        (base / "curves").mkdir(parents=True, exist_ok=True)
        for s, t in by_seed.items():
            (base / "curves" / f"{r}__seed{s}.json").write_text(json.dumps(ev.curves(t)))
        e["three_seed"] = three_seed_entry(r, by_seed, verified[r], present, e["seed0"]["gate_vs_D1"]["pass"], T[D1], frame, F)
        if e["three_seed"]["status"] == "evaluated":
            conf_tables[r] = by_seed
        res["rows"][r] = e
    cands = [r for r in T if r != D1]
    children_parent = {c: a.parent for c in CHILDREN if c in cands} if getattr(a, "parent", None) else {}
    if cands:
        res["stage_nested"] = nested_with_parent_rule(T, D1, cands, INNER, children_parent)
        res["stage_nested"]["label"] = "seed-0 screen tables"
        if all(r in conf_tables for r in cands):
            per_seed = {s: nested_with_parent_rule({D1: T[D1], **{r: conf_tables[r][s] for r in cands}}, D1, cands, INNER, children_parent)
                        for s in confirm.CONFIRM_SEEDS}
            res["stage_nested_three_seed"] = {"candidates": cands, "label": "per confirmation seed, averaged (amendment_2)",
                                              "per_seed": per_seed, **confirm.average_nested(per_seed, INNER)}
        else:
            res["stage_nested_three_seed"] = {"status": "not evaluated",
                                              "reason": f"rows without an evaluated confirmation: {[r for r in cands if r not in conf_tables]}"}
        par3 = [p for p in PARENTS if p in conf_tables]
        if len(par3) > 1:
            lead = max(par3, key=lambda r: (np.mean([conf_tables[r][s].summary(INNER)["primary"] for s in confirm.CONFIRM_SEEDS]), -PARENTS.index(r)))
            res["paired_parent_comparisons"] = {"label": "descriptive; three-seed leader vs each other confirmed parent", "leader": lead,
                                                **{f"{lead}_vs_{o}": confirm.paired_rows(conf_tables[lead], conf_tables[o], INNER)
                                                   for o in par3 if o != lead}}
        par = [p for p in PARENTS if p in cands]
        if par:
            res["parent_by_rule"] = max(par, key=lambda r: (T[r].summary(INNER)["primary"], -PARENTS.index(r)))
            res["parent_rule_complete"] = sorted(par) == sorted(PARENTS)
            # held-fold-free parent: for each held fold k, the parent chosen on the other five folds only
            res["parent_selection_audit"] = [{"held_fold": n["held_fold"], "inner_parent": n["inner_parent"],
                                              "equals_global_parent": n["inner_parent"] == res["parent_by_rule"]}
                                             for n in res["stage_nested"]["held"]]
    (base / "evaluation.json").write_text(json.dumps(res, indent=1, default=float))
    for r, e in res["rows"].items():
        g = e["seed0"]["gate_vs_D1"]
        print(f"{r:30s} seed0 H_bal {100*e['seed0']['primary']:6.2f} {g['mean_gain_pp']:+6.2f}pp {g['folds_improved']}/6 pass={g['pass']}")
        c = e["three_seed"]
        if c["status"] == "evaluated":
            g3, n3 = c["gate_vs_D1"], c["nested_vs_D1"]
            print(f"{'':30s} 3-seed H_bal {100*c['three_seed_mean']['primary']:6.2f} {g3['mean_gain_pp']:+6.2f}pp {g3['folds_improved']}/6 "
                  f"gate={g3['pass']} nested {n3['mean_diff_pp']:+.2f}pp {n3['folds_better']}/6 -> {c['verdict']}")
        else:
            print(f"{'':30s} 3-seed: {c['status']} ({c.get('reason', '')})")
    print("refused:", json.dumps(refused, indent=1))
    return res


def three_seed_entry(r, by_seed, verified_r, present, screen_pass, ctrl, frame, F):
    """The confirmation block for one row: 'not run' (screen only), 'refused' (with the reason) or 'evaluated'."""
    want = sorted(confirm.CONFIRM_SEEDS)
    if present == sorted(confirm.SCREEN_SEEDS):
        return {"status": "not run", "reason": "seed-0 screen only"}
    if present != want:
        return {"status": "refused", "reason": f"seed directories {present}; confirmation needs exactly {want}"}
    if sorted(by_seed) != want:
        return {"status": "refused", "reason": f"verified seeds {sorted(by_seed)}; seeds {sorted(set(want) - set(by_seed))} refused (see refused_incomplete)"}
    if not screen_pass:
        return {"status": "refused", "reason": "the seed-0 screen did not pass; confirmation is declared only after a screen pass"}
    xp = confirm.cross_seed_problems(r, verified_r, INNER)
    if xp:
        return {"status": "refused", "reason": "; ".join(xp)}
    c = confirm.three_seed_confirmation(r, by_seed, ctrl, D1, INNER, nested_with_parent_rule)
    c["status"] = "evaluated"
    c["per_seed_detail"] = {s: _seed_detail(by_seed[s], ctrl, frame, F, verified_r[s]) for s in want}
    keys = list(c["per_seed_detail"][0]["summary_lofo"])
    c["three_seed_mean_metrics"] = {k: float(np.mean([c["per_seed_detail"][s]["summary_lofo"][k] for s in want])) for k in keys}
    c["bootstrap_vs_D1_three_seed"] = confirm.paired_bootstrap_seeds([by_seed[s] for s in want], [ctrl] * len(want), INNER)
    return c


# ----------------------------------------------------------------------------- H_RF source co-endpoint
def cmd_hrf(a):
    frame, bench, F = env()
    sh = _script("r2v4_stage_h")
    base = Path(a.out)
    pk = confirm.verified_row_seed(base, a.row, a.seed, INNER, jrun.j_rows(getattr(a, "parent", None)).get(a.row))  # refuses incomplete / invalid
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
        if name == "evaluate":
            p.add_argument("--runs", nargs="+", help="run directories to read (default: --out); a (row, seed) in two is refused")
        if name == "run":
            p.add_argument("--images", default=str(ROOT.parents[2] / "DPD_Small_dataset"))
            p.add_argument("--folds", nargs="+")
            p.add_argument("--max-steps", type=int)
    a = ap.parse_args()
    {"run": cmd_run, "evaluate": cmd_evaluate, "hrf": cmd_hrf, "dryrun": lambda a: print(cmd_dryrun(a))}[a.cmd](a)


if __name__ == "__main__":
    main()
