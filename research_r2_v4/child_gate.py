"""PROPOSAL (amendment 3 v2, NOT declared): child-versus-parent gate for J4a-J7, fail closed.

Validated on synthetic tables only (tests/test_r2v4_child_gate.py). Nothing here is used by any registered stage, and
no real child exists. See docs/R2_v4_Amendment3_PROPOSAL_v2.md for the protocol and its rationale.

Primary gate (one parent for every fold, fixed before any child exists):
    for seed s and fold k, each row r in {child, parent} gets its own LOFO choice (config, gamma) on folds != k by mean
    H_bal (research_r2_v4.evaluate.RowTable.choose: ties -> lower config, smaller gamma). The endpoint E (default H_bal)
    is read at that choice.
        d_s(k) = E_child,s(k) - E_parent,s(k)        a_s(k) = AUSUC diff        h_s(k) = healthy-shortcut diff
    screen = seed 0 only; confirmation = exactly the declared seeds {0, 1, 2}, per-fold seed means D(k), A(k), H(k).
        pass  <=>  100*mean_k D(k) >= 1.0  and  #{k: D(k) > 0} >= n_folds - 1  and  mean_k A(k) >= 0  and  100*mean_k H(k) <= 5.0
    A child numerically identical to its parent has D = 0 everywhere and fails (0 < 1.0 pp, 0 folds improved).

Secondary "pipeline" estimate (optional; only if the child is also trained on every inner parent, on all six folds):
    pi(k) = the parent chosen on folds != k from the seed-0 screen tables (the declared parent rule, applied inside);
    d_s(k) = E(child on pi(k), s, k) - E(pi(k), s, k). Gains from switching parents cancel inside d. A missing variant is
    refused; there is no fallback to a parent, so no parent result can ever be counted as a child effect.
"""
from __future__ import annotations

import numpy as np

from research_r2.metrics import GAMMA_GRID, harmonic
from research_r2_v4 import confirm
from research_r2_v4 import evaluate as ev

THRESHOLDS = {"mean_gain_pp": 1.0, "folds_slack": 1, "ausuc_delta_min": 0.0, "shortcut_delta_max_pp": 5.0}
ENDPOINTS = ("H_bal", "unseen_disease_top1", "unseen_crop_top1", "unseen_top1_macro", "seen_top1")


class GateError(ValueError):
    """A child / parent comparison cannot be made (missing, incomplete or incompatible inputs)."""


# ----------------------------------------------------------------------------- input checks
def check_pair(child: ev.RowTable, parent: ev.RowTable, folds):
    """Same fold identities in the same order, one configuration, identical candidates (order, crop, disease, healthy,
    fitting mask) and identical truth in every fold, so the comparison is paired image by image."""
    if list(child.folds) != list(folds) or list(parent.folds) != list(folds):
        raise GateError(f"fold identities differ: child {list(child.folds)}, parent {list(parent.folds)}, expected {list(folds)}")
    for k in folds:
        if len(child.sc[k]) != 1 or len(parent.sc[k]) != 1:
            raise GateError(f"{k}: child and parent must each have exactly one configuration")
        c, p = child.sc[k][0], parent.sc[k][0]
        for attr in ("cand_crop", "cand_dis", "cand_healthy", "fit_mask"):
            if not np.array_equal(getattr(c, attr), getattr(p, attr)):
                raise GateError(f"{k}: candidate order / identity differs ({attr})")
        for part in ("seen", "unseen"):
            if not np.array_equal(c.parts[part]["truth"], p.parts[part]["truth"]):
                raise GateError(f"{k}: {part} truth differs (different rows or fold)")


def check_seeds(child_by_seed: dict, parent_by_seed: dict, mode: str):
    if mode == "screen":
        want = tuple(confirm.SCREEN_SEEDS)
    elif mode == "confirm":
        want = tuple(confirm.CONFIRM_SEEDS)
    else:
        raise GateError(f"unknown mode {mode!r}")
    for who, d in (("child", child_by_seed), ("parent", parent_by_seed)):
        have = tuple(sorted(d))
        if mode == "confirm" and have != tuple(sorted(want)):
            raise GateError(f"{who}: confirmation needs exactly seeds {list(want)}, got {list(have)}")
        if mode == "screen" and not set(want) <= set(have):
            raise GateError(f"{who}: screen needs seed {list(want)}, got {list(have)}")
    return list(want)


# ----------------------------------------------------------------------------- per-fold values
def fold_values(t: ev.RowTable, k, folds, endpoint="H_bal", gi=None):
    """Endpoint, AUSUC and healthy shortcut of fold k at the row's own LOFO choice (gi=None) or at a given gamma index."""
    if endpoint not in ENDPOINTS:
        raise GateError(f"unknown endpoint {endpoint!r}")
    ci, g = t.choose([f for f in folds if f != k])
    if gi is not None:
        g = gi
    r = t.at(k, ci, g)
    return {"E": r[endpoint], "ausuc": t.ausuc[k][ci], "hs": r["unseen_healthy_shortcut"], "gi": int(g), "gamma": float(GAMMA_GRID[g])}


def gate_from(per: dict, folds):
    """per: {seed: {fold: {"d", "a", "h"}}} -> the threshold rule on per-fold seed means."""
    seeds = sorted(per)
    D = {k: float(np.mean([per[s][k]["d"] for s in seeds])) for k in folds}
    A = {k: float(np.mean([per[s][k]["a"] for s in seeds])) for k in folds}
    H = {k: float(np.mean([per[s][k]["h"] for s in seeds])) for k in folds}
    g = {"seeds": seeds, "per_fold_delta_pp": {k: 100 * v for k, v in D.items()},
         "mean_gain_pp": 100 * float(np.mean(list(D.values()))),
         "folds_improved": int(sum(v > 0 for v in D.values())), "folds_regressed": int(sum(v < 0 for v in D.values())),
         "folds_required": len(folds) - THRESHOLDS["folds_slack"],
         "ausuc_delta": float(np.mean(list(A.values()))), "shortcut_delta_pp": 100 * float(np.mean(list(H.values())))}
    g["pass"] = bool(g["mean_gain_pp"] >= THRESHOLDS["mean_gain_pp"] and g["folds_improved"] >= g["folds_required"]
                     and g["ausuc_delta"] >= THRESHOLDS["ausuc_delta_min"] and g["shortcut_delta_pp"] <= THRESHOLDS["shortcut_delta_max_pp"])
    return g


def _deltas(child_of, parent_of, folds, seeds, endpoint, matched=False):
    per, detail = {}, {}
    for s in seeds:
        per[s], detail[s] = {}, {}
        for k in folds:
            pt = parent_of(k, s); ct = child_of(k, s)
            check_pair(ct, pt, folds)
            p = fold_values(pt, k, folds, endpoint)
            c = fold_values(ct, k, folds, endpoint, gi=p["gi"] if matched else None)
            per[s][k] = {"d": c["E"] - p["E"], "a": c["ausuc"] - p["ausuc"], "h": c["hs"] - p["hs"]}
            detail[s][k] = {"child": c, "parent": p, "parent_row": pt.name, "child_row": ct.name}
    return per, detail


# ----------------------------------------------------------------------------- uncertainty
def _hits(t, k, gi):
    s = t.sc[k][0]; g = float(GAMMA_GRID[gi])
    return ((s.top1_pred("seen", g) == s.parts["seen"]["truth"]).astype(float), s.parts["seen"]["truth"],
            (s.top1_pred("unseen", g) == s.parts["unseen"]["truth"]).astype(float), s.parts["unseen"]["truth"])


def paired_bootstrap(detail, child_of, parent_of, folds, seeds, reps=2000, seed=0):
    """Composition-cluster bootstrap of the seed-mean, fold-mean H_bal difference child - parent, each at its own
    chosen gamma; the same resampled compositions are used for child and parent and for every seed in a fold."""
    rng = np.random.default_rng(seed)
    prep = {}
    for k in folds:
        rows = []
        for s in seeds:
            c = _hits(child_of(k, s), k, detail[s][k]["child"]["gi"]); p = _hits(parent_of(k, s), k, detail[s][k]["parent"]["gi"])
            sc, uc = np.unique(c[1]), np.unique(c[3])
            S = lambda h, t: (np.array([h[t == x].sum() for x in sc]), np.array([(t == x).sum() for x in sc]))  # noqa: E731
            U = lambda h, t: np.array([h[t == x].mean() for x in uc])  # noqa: E731
            rows.append((S(c[0], c[1]), U(c[2], c[3]), S(p[0], p[1]), U(p[2], p[3])))
        prep[k] = (rows, len(np.unique(_hits(child_of(k, seeds[0]), k, 0)[1])), len(np.unique(_hits(child_of(k, seeds[0]), k, 0)[3])))
    H = lambda Sx, Ux, i, j: harmonic(Sx[0][i].sum() / Sx[1][i].sum(), Ux[j].mean())  # noqa: E731
    draws = np.empty(reps)
    for r in range(reps):
        vals = []
        for k in folds:
            rows, ns, nu = prep[k]
            i = rng.integers(0, ns, ns); j = rng.integers(0, nu, nu)
            vals.append(np.mean([H(a, b, i, j) - H(c, d, i, j) for a, b, c, d in rows]))
        draws[r] = np.mean(vals)
    return {"ci95_pp": [100 * float(np.percentile(draws, 2.5)), 100 * float(np.percentile(draws, 97.5))],
            "p_le_0": float(np.mean(draws <= 0)), "reps": reps, "seed": seed}


def holm(pvals: dict):
    """Holm step-down adjusted p-values for disclosure across the child rows evaluated (not part of the gate)."""
    items = sorted(pvals.items(), key=lambda kv: kv[1]); m = len(items); out, run = {}, 0.0
    for i, (k, p) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p)); out[k] = run
    return out


# ----------------------------------------------------------------------------- primary gate
def primary_gate(child_by_seed: dict, parent_by_seed: dict, folds, mode="screen", endpoint="H_bal", reps=2000):
    """Child vs ONE parent (the declared global parent) on every fold."""
    seeds = check_seeds(child_by_seed, parent_by_seed, mode)
    child_of = lambda k, s: child_by_seed[s]  # noqa: E731
    parent_of = lambda k, s: parent_by_seed[s]  # noqa: E731
    per, detail = _deltas(child_of, parent_of, folds, seeds, endpoint)
    per_m, _ = _deltas(child_of, parent_of, folds, seeds, endpoint, matched=True)
    out = {"mode": mode, "endpoint": endpoint, "parent_per_fold": {k: parent_by_seed[seeds[0]].name for k in folds},
           "gate": gate_from(per, folds), "calibration_matched_sensitivity": gate_from(per_m, folds), "detail": detail}
    if endpoint == "H_bal":
        out["bootstrap"] = paired_bootstrap(detail, child_of, parent_of, folds, seeds, reps)
    return out


# ----------------------------------------------------------------------------- secondary pipeline estimate
def inner_parents(parents_seed0: dict, folds, order):
    """pi(k): best parent by seed-0 screen H_bal on folds != k (each inner fold at its own inner LOFO gamma); ties ->
    earlier in ``order``. The held fold k never enters its own choice."""
    out = {}
    for k in folds:
        inner = [f for f in folds if f != k]
        out[k] = max(order, key=lambda r: (parents_seed0[r].summary(inner)["primary"], -order.index(r)))
    return out


def pipeline_estimate(variants: dict, parents_by_seed: dict, folds, order, mode="screen", endpoint="H_bal", d1=None):
    """variants: {parent: {seed: child-on-parent RowTable}}; parents_by_seed: {parent: {seed: RowTable}}."""
    pi = inner_parents({p: parents_by_seed[p][0] for p in order}, folds, order)
    need = sorted(set(pi.values()))
    for p in need:
        if p not in variants:
            raise GateError(f"missing child variant trained on parent {p} (needed for held folds "
                            f"{[k for k in folds if pi[k] == p]}); no fallback to the parent")
        check_seeds(variants[p], parents_by_seed[p], mode)
    seeds = check_seeds(variants[need[0]], parents_by_seed[need[0]], mode)
    child_of = lambda k, s: variants[pi[k]][s]  # noqa: E731
    parent_of = lambda k, s: parents_by_seed[pi[k]][s]  # noqa: E731
    per, detail = _deltas(child_of, parent_of, folds, seeds, endpoint)
    out = {"mode": mode, "endpoint": endpoint, "inner_parent": pi, "gate": gate_from(per, folds), "detail": detail}
    if d1 is not None:  # decomposition: child - D1 = (child - parent) + (parent - D1); only the first is the child's
        dec = {}
        for k in folds:
            dd = fold_values(d1, k, folds, endpoint)["E"]
            par = float(np.mean([detail[s][k]["parent"]["E"] for s in seeds])); ch = float(np.mean([detail[s][k]["child"]["E"] for s in seeds]))
            dec[k] = {"child_attributable_pp": 100 * (ch - par), "parent_attributable_pp": 100 * (par - dd), "total_vs_D1_pp": 100 * (ch - dd),
                      "parent": pi[k]}
        out["decomposition_vs_D1"] = dec
    return out
