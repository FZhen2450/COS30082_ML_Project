"""Stage J run verification and three-seed confirmation (v4, post-freeze exploratory), fail closed.

Phase 2 repair of the evaluator in scripts/r2v4_stage_j.py. The rules are unchanged from the declarations:
  * seeds: matrices/stage_j_v4.json  "screen": [0], "confirmation_if_gate_passes": [0, 1, 2]
  * amendment_2.json three_seed_gate: "each seed gets its own LOFO choice per fold; the gate is applied to the
    per-fold mean over seeds of H_bal, AUSUC and healthy shortcut. The nested estimate is computed per seed and averaged."
  * protocol_v4.json nested: the nested estimate must also reach >= +1.0 pp and >= 5 of 6 held folds.

The defects this fixes: the evaluator before this commit accepted any set of >= 3 seed directories as a
confirmation (e.g. {0, 1, 3} or {0, 1, 2, 3}), took min(seeds) as "seed 0", ignored seeds whose folds were
incomplete instead of refusing the confirmation, did not check that the seeds were different runs of one declared
configuration, and only recorded each seed's nested mean (no average, no held-fold results, no verdict).

Metric definitions, the gamma grid, the candidate universe, the thresholds, the tie rules and the selection
procedure are those of research_r2_v4.evaluate and are not changed here.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path

import numpy as np

from research_r2.metrics import harmonic
from research_r2_v4 import evaluate as ev
from research_r2_v4 import jrun

ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "research_r2_v4" / "matrices" / "stage_j_v4.json"
SEED_DIR = re.compile(r"^seed(0|[1-9][0-9]*)$")
AMP_SKIP_LIMIT = 0.05  # amendment_2 execution.fail_closed


class ConfirmError(jrun.JCheckError):
    """A run or a seed set does not qualify for screening / confirmation."""


def declared_seeds():
    s = json.loads(MATRIX.read_text())["common"]["seeds"]
    return tuple(s["screen"]), tuple(s["confirmation_if_gate_passes"])


SCREEN_SEEDS, CONFIRM_SEEDS = declared_seeds()


# ----------------------------------------------------------------------------- run discovery
def seed_dirs(rdir: Path) -> dict:
    """seed -> directory. A non-canonical name (seed01, seed_1, seedX) is refused so one seed cannot appear twice."""
    out = {}
    for p in sorted(rdir.iterdir()):
        if not p.is_dir() or not p.name.startswith("seed"):
            continue
        m = SEED_DIR.match(p.name)
        if not m:
            raise ConfirmError(f"{rdir.name}: non-canonical seed directory {p.name!r}")
        out[int(m.group(1))] = p
    return out


def discover(bases, prefix="J"):
    """{row: {seed: base}} over one or more run directories, plus refusals. The same (row, seed) under two bases
    is a duplicate run: it is refused, never resolved by picking one."""
    found, refused = {}, {}
    for b in bases:
        b = Path(b)
        for rdir in sorted(p for p in b.iterdir() if p.is_dir() and p.name.startswith(prefix)):
            try:
                sd = seed_dirs(rdir)
            except ConfirmError as e:
                refused[rdir.name] = str(e)
                continue
            for s in sd:
                found.setdefault(rdir.name, {}).setdefault(s, []).append(b)
    runs = {}
    for r, by in found.items():
        for s, bs in by.items():
            if len(bs) > 1:
                refused[f"{r}/seed{s}"] = f"duplicate run: {r} seed{s} exists under {len(bs)} run directories"
            else:
                runs.setdefault(r, {})[s] = bs[0]
    return runs, refused


# ----------------------------------------------------------------------------- receipt verification
def _cfg_core(cfg: dict) -> dict:
    return {k: v for k, v in cfg.items() if k != "workers"}  # workers do not enter config_sha256 (JConfig.sha)


def receipt_problems(rec: dict, row: str, seed: int, fold: str, fold_index: int, declared: jrun.JConfig | None):
    """Every reason a verified-ok receipt still does not count as a declared run of (row, seed, fold)."""
    p = []
    for k, want in (("row", row), ("seed", seed), ("fold", fold)):
        if rec.get(k) != want:
            p.append(f"receipt {k}={rec.get(k)!r}, expected {want!r}")
    cfg = rec.get("config")
    if not isinstance(cfg, dict):
        return p + ["receipt has no configuration"]
    try:
        if jrun.JConfig(**cfg).sha() != rec.get("config_sha256"):
            p.append("config_sha256 does not match the recorded configuration")
    except TypeError as e:
        p.append(f"unreadable configuration: {e}")
    if cfg.get("max_steps") is not None:
        p.append(f"smoke run (max_steps={cfg.get('max_steps')}) cannot count")
    if declared is not None and _cfg_core(cfg) != _cfg_core(asdict(declared)):
        bad = sorted(k for k in _cfg_core(asdict(declared)) if cfg.get(k) != getattr(declared, k))
        p.append(f"configuration differs from the declared row in {bad}")
    if rec.get("torch_seed") != 1000 * seed + fold_index:
        p.append(f"torch_seed {rec.get('torch_seed')} != 1000*seed+fold_index = {1000 * seed + fold_index}")
    meta = rec.get("feature_cache_meta") or {}
    ck = rec.get("checkpoint_sha256")
    if not ck:
        p.append("missing checkpoint provenance (checkpoint_sha256)")
    for k, want in (("row", row), ("seed", seed), ("fold", fold), ("config_sha256", rec.get("config_sha256")), ("checkpoint_sha256", ck)):
        if meta.get(k) != want or want is None:
            p.append(f"feature_cache_meta {k} does not match the receipt")
    if rec.get("reload_max_abs_diff") != 0.0:
        p.append(f"checkpoint reload difference {rec.get('reload_max_abs_diff')!r} != 0.0")
    if rec.get("frozen_digest_before") is None or rec.get("frozen_digest_before") != rec.get("frozen_digest_after"):
        p.append("frozen-parameter digests missing or different")
    ch = str(rec.get("trainable_tensors_changed", ""))
    a, _, b = ch.partition("/")
    if cfg.get("epochs", 0) > 0 and not (a.isdigit() and a == b and int(a) > 0):
        p.append(f"trainable tensors changed {ch!r}")
    gc = rec.get("grad_checks")
    if cfg.get("epochs", 0) > 0:
        if not isinstance(gc, dict):
            p.append("missing gradient checks")
        else:
            if gc.get("finite_checks") != gc.get("checks") or not gc.get("checks"):
                p.append(f"gradient checks {gc.get('finite_checks')}/{gc.get('checks')} finite")
            if gc.get("steps") and gc.get("skipped_steps", 0) > AMP_SKIP_LIMIT * gc["steps"]:
                p.append(f"AMP skipped {gc.get('skipped_steps')}/{gc.get('steps')} steps > 5%")
    code = rec.get("code") or {}
    if not code.get("jrun_sha256") or not code.get("git_head"):
        p.append("missing code identity")
    elif cfg.get("backbone") != "synthetic" and code.get("worktree_clean") is not True:
        p.append("run started from a dirty worktree")
    return p


def ok_attempts(base, row, seed, fold):
    d = Path(base) / row / f"seed{seed}" / fold
    if not d.exists():
        return []
    out = []
    for a in sorted(d.glob("attempt*"), key=lambda q: int(q.name[7:])):
        rec = jrun.verify_receipt(a)  # raises if a file referenced by an ok receipt is missing or changed
        if rec is not None:
            out.append((a, rec))
    return out


def verified_row_seed(base, row, seed, fold_names, declared: jrun.JConfig | None = None):
    """{fold: (packet path, receipt)} for one (row, seed), or ConfirmError. Stricter than jrun.row_packets: exactly one
    successful attempt per fold, receipts that name this row/seed/fold, the declared configuration, complete
    checkpoint provenance, and one configuration and one training-code identity across the six folds."""
    out, missing, bad = {}, [], []
    for i, fn in enumerate(fold_names):
        oks = ok_attempts(base, row, seed, fn)
        if not oks:
            missing.append(fn)
            continue
        if len(oks) > 1:
            bad.append(f"{fn}: duplicate run, {len(oks)} successful attempts ({', '.join(a.name for a, _ in oks)})")
            continue
        adir, rec = oks[0]
        pr = receipt_problems(rec, row, seed, fn, i, declared)
        if pr:
            bad.append(f"{fn}: " + "; ".join(pr))
            continue
        out[fn] = (adir / f"{fn}_scores.npz", rec)
    if missing:
        raise ConfirmError(f"{row} seed{seed}: incomplete, missing folds {missing}")
    if bad:
        raise ConfirmError(f"{row} seed{seed}: invalid receipts: " + " | ".join(bad))
    for key, what in ((lambda r: r["config_sha256"], "configurations"), (lambda r: r["code"]["jrun_sha256"], "training-code identities")):
        vals = {key(rec) for _, rec in out.values()}
        if len(vals) != 1:
            raise ConfirmError(f"{row} seed{seed}: folds were run with different {what}")
    return out


def cross_seed_problems(row, verified: dict, fold_names):
    """verified: {seed: {fold: (packet, receipt)}}. One configuration and training code across seeds, a distinct run
    (checkpoint, features, packet, torch seed) per seed and fold, and identical candidate mapping and truth."""
    p = []
    seeds = sorted(verified)
    cfg = {s: next(iter(v.values()))[1]["config_sha256"] for s, v in verified.items()}
    if len(set(cfg.values())) != 1:
        p.append(f"incompatible configurations across seeds: {cfg}")
    code = {s: next(iter(v.values()))[1]["code"]["jrun_sha256"] for s, v in verified.items()}
    if len(set(code.values())) != 1:
        p.append(f"training code differs across seeds: { {s: c[:12] for s, c in code.items()} }")
    for fn in fold_names:
        for key in ("checkpoint_sha256", "features_sha256", "packet_sha256", "torch_seed"):
            vals = [verified[s][fn][1].get(key) for s in seeds]
            if len(set(vals)) != len(vals):
                p.append(f"{fn}: duplicate run, {key} repeats across seeds {seeds}")
        ref = None
        for s in seeds:
            z = np.load(verified[s][fn][0])
            cur = (z["cand_ids"].tolist(), z["seen_truth"].tolist(), z["unseen_truth"].tolist())
            if ref is None:
                ref = cur
            elif cur != ref:
                p.append(f"{fn}: candidate mapping or truth differs between seed {seeds[0]} and seed {s}")
    return p


# ----------------------------------------------------------------------------- three-seed statistics
def mean_summary(summaries):
    """Per-fold mean over seeds of H_bal, AUSUC and healthy shortcut; each seed keeps its own LOFO choice."""
    folds = list(summaries[0]["per"])
    per = {f: {k: float(np.mean([s["per"][f][k] for s in summaries])) for k in ("H_bal", "ausuc", "unseen_healthy_shortcut")} for f in folds}
    return {"per": per, "primary": float(np.mean([s["primary"] for s in summaries])),
            "ausuc": float(np.mean([s["ausuc"] for s in summaries])), "hs": float(np.mean([s["hs"] for s in summaries]))}


def average_nested(per_seed: dict, folds):
    """per_seed: {seed: nested result}. Average of the per-seed estimates; held-fold improvement counted on the
    per-fold mean over seeds; same thresholds as the single-seed nested estimate."""
    d = {s: {h["held_fold"]: 100 * (h["H_bal"] - h["control_H_bal"]) for h in n["held"]} for s, n in per_seed.items()}
    per_fold = {f: float(np.mean([d[s][f] for s in d])) for f in folds}
    mean = float(np.mean([n["mean_diff_pp"] for n in per_seed.values()]))
    better = int(sum(v > 0 for v in per_fold.values()))
    return {"per_seed_mean_diff_pp": {s: n["mean_diff_pp"] for s, n in per_seed.items()},
            "per_seed_folds_better": {s: n["folds_better"] for s, n in per_seed.items()},
            "per_seed_selected": {s: {h["held_fold"]: h["selected"] for h in n["held"]} for s, n in per_seed.items()},
            "per_seed_held_diff_pp": d,
            "per_fold_mean_diff_pp": per_fold, "mean_diff_pp": mean, "folds_better": better,
            "folds_worse": int(sum(v < 0 for v in per_fold.values())),
            "meets_gate_thresholds": bool(mean >= 1.0 and better >= len(folds) - 1)}


def lofo_metrics(t: ev.RowTable, folds):
    """Per-fold metrics at each fold's own LOFO choice, including H@5 (top-5) and AUSUC."""
    out = {}
    for k in folds:
        ci, gi = t.choose([f for f in folds if f != k])
        r = t.at(k, ci, gi, top5=True)
        out[k] = {**{x: r[x] for x in ev.KEYS}, "H5_micro": r["H5_micro"], "seen_top5": r["seen_top5"],
                  "unseen_top5_micro": r["unseen_top5_micro"], "ausuc": t.ausuc[k][ci], "gamma": float(ev.GAMMA_GRID[gi])}
    return out


def _prep(row: ev.RowTable, ctrl: ev.RowTable, k, folds):
    others = [f for f in folds if f != k]
    a = ev._hits(row, k, *row.choose(others)); b = ev._hits(ctrl, k, *ctrl.choose(others))
    if not (np.array_equal(a[2], b[2]) and np.array_equal(a[3], b[3])):
        raise ConfirmError("row and control truth differ")
    sc = np.unique(a[2]); uc = np.unique(a[3])
    S = lambda h, t: (np.array([h[t == c].sum() for c in sc]), np.array([(t == c).sum() for c in sc]))  # noqa: E731
    U = lambda h, t: np.array([h[t == c].mean() for c in uc])  # noqa: E731
    return S(a[0], a[2]), S(b[0], b[2]), U(a[1], a[3]), U(b[1], b[3]), len(sc), len(uc), (sc.tolist(), uc.tolist())


def paired_bootstrap_seeds(rows: list, ctrls: list, folds, reps=2000, seed=0):
    """Paired composition-cluster bootstrap of the seed-mean, fold-mean H_bal difference (rows[i] vs ctrls[i]).
    The same resampled compositions are used for every seed in a fold (compositions are identical across seeds), so
    the interval reflects composition sampling of the three-seed mean. With one seed this equals ev.paired_bootstrap."""
    rng = np.random.default_rng(seed)
    prep = {k: [_prep(r, c, k, folds) for r, c in zip(rows, ctrls)] for k in folds}
    for k, ps in prep.items():
        if len({json.dumps(p[6]) for p in ps}) != 1:
            raise ConfirmError(f"{k}: composition sets differ across seeds")

    def h(sa, ua, i, j):
        return harmonic(sa[0][i].sum() / sa[1][i].sum(), ua[j].mean())
    diffs = np.empty(reps)
    for r in range(reps):
        ds = []
        for k in folds:
            ns, nu = prep[k][0][4], prep[k][0][5]
            i = rng.integers(0, ns, ns); j = rng.integers(0, nu, nu)
            ds.append(np.mean([h(p[0], p[2], i, j) - h(p[1], p[3], i, j) for p in prep[k]]))
        diffs[r] = np.mean(ds)
    full = lambda p: (np.arange(p[4]), np.arange(p[5]))  # noqa: E731
    point = np.mean([np.mean([h(p[0], p[2], *full(p)) - h(p[1], p[3], *full(p)) for p in prep[k]]) for k in folds])
    return {"point_pp": 100 * float(point), "ci95_pp": [100 * float(np.percentile(diffs, 2.5)), 100 * float(np.percentile(diffs, 97.5))],
            "p_le_0": float(np.mean(diffs <= 0)), "reps": reps, "seed": seed, "n_seeds": len(rows)}


def three_seed_confirmation(row, tables: dict, ctrl: ev.RowTable, ctrl_name, folds, nested_fn):
    """tables: {seed: RowTable} for exactly CONFIRM_SEEDS. Returns the three-seed gate, the averaged nested estimate,
    per-seed results and the verdict (gate pass AND nested thresholds met)."""
    if tuple(sorted(tables)) != tuple(sorted(CONFIRM_SEEDS)):
        raise ConfirmError(f"{row}: confirmation needs exactly seeds {list(CONFIRM_SEEDS)}, got {sorted(tables)}")
    Sd = ctrl.summary(folds)
    S = {s: tables[s].summary(folds) for s in sorted(tables)}
    per_seed = {}
    for s, t in sorted(tables.items()):
        per_seed[s] = {"primary": S[s]["primary"], "ausuc": S[s]["ausuc"], "hs": S[s]["hs"],
                       "per_fold_H_bal": {f: S[s]["per"][f]["H_bal"] for f in folds},
                       "per_fold_gamma": {f: S[s]["per"][f]["gamma"] for f in folds},
                       "gate_vs_D1": ev.gate(S[s], Sd, folds),
                       "nested_vs_D1": nested_fn({ctrl_name: ctrl, row: t}, ctrl_name, [row], folds, {})}
    Sm = mean_summary([S[s] for s in sorted(S)])
    gate3 = ev.gate(Sm, Sd, folds)
    nest3 = average_nested({s: v["nested_vs_D1"] for s, v in per_seed.items()}, folds)
    prim = [per_seed[s]["primary"] for s in sorted(per_seed)]
    fold_spread = {f: 100 * (max(S[s]["per"][f]["H_bal"] for s in S) - min(S[s]["per"][f]["H_bal"] for s in S)) for f in folds}
    return {"seeds": sorted(tables), "rule": "amendment_2 three_seed_gate + protocol_v4 nested thresholds",
            "per_seed": per_seed,
            "three_seed_mean": {"primary": Sm["primary"], "ausuc": Sm["ausuc"], "hs": Sm["hs"], "per_fold": Sm["per"]},
            "gate_vs_D1": gate3, "nested_vs_D1": nest3,
            "seed_variation": {"primary_sd_pp": 100 * float(np.std(prim, ddof=1)), "primary_range_pp": 100 * float(max(prim) - min(prim)),
                               "per_fold_range_pp": fold_spread},
            "pass": bool(gate3["pass"] and nest3["meets_gate_thresholds"]),
            "verdict": "confirmed" if gate3["pass"] and nest3["meets_gate_thresholds"] else "not confirmed"}


def paired_rows(a_tables: dict, b_tables: dict, folds):
    """Descriptive paired comparison of two rows across the same seeds: per-seed and three-seed differences."""
    seeds = sorted(set(a_tables) & set(b_tables))
    per_seed = {}
    for s in seeds:
        A, B = a_tables[s].summary(folds), b_tables[s].summary(folds)
        per_seed[s] = {"mean_diff_pp": 100 * (A["primary"] - B["primary"]),
                       "per_fold_diff_pp": {f: 100 * (A["per"][f]["H_bal"] - B["per"][f]["H_bal"]) for f in folds},
                       "bootstrap": ev.paired_bootstrap(a_tables[s], b_tables[s], folds)}
    pf = {f: float(np.mean([per_seed[s]["per_fold_diff_pp"][f] for s in seeds])) for f in folds}
    return {"seeds": seeds, "per_seed": per_seed, "per_fold_mean_diff_pp": pf,
            "mean_diff_pp": float(np.mean([per_seed[s]["mean_diff_pp"] for s in seeds])),
            "folds_a_better": int(sum(v > 0 for v in pf.values())),
            "seeds_a_better": int(sum(per_seed[s]["mean_diff_pp"] > 0 for s in seeds)),
            "bootstrap_three_seed": paired_bootstrap_seeds([a_tables[s] for s in seeds], [b_tables[s] for s in seeds], folds)}
