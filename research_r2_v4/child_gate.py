"""PROPOSAL (amendment 3 v3, NOT declared): child-versus-parent gate for J4a-J7, fail closed.

Validated on synthetic tables only (tests/test_r2v4_child_gate.py, tests/test_r2v4_child_inputs.py). Nothing here is
used by any registered stage, and no real child exists. Protocol: docs/R2_v4_Amendment3_PROPOSAL_v3.md.

Inputs must come from research_r2_v4.child_inputs (verified receipts, registered fold, ordered image identity, packet
re-derivation, one registered modification); a table without that identity record is refused here.

Primary gate (PRIMARY ENDPOINT LOCKED TO H_bal; one parent J2 for every fold, fixed before any child exists):
    for seed s and fold k, child and parent each get their own LOFO choice (config, gamma) on folds != k by mean H_bal
    (research_r2_v4.evaluate.RowTable.choose: ties -> lower config, smaller gamma); H_bal is read at that choice.
        d_s(k) = H_bal child - H_bal parent      a_s(k) = AUSUC diff      h_s(k) = healthy-shortcut diff
    screen = seed 0 only; confirmation = exactly seeds {0, 1, 2}; D(k), A(k), H(k) = means over seeds.
        pass  <=>  100*mean_k D(k) >= 1.0  and  #{k: D(k) > 0} >= 5  and  mean_k A(k) >= 0  and  100*mean_k H(k) <= 5.0
    A child numerically identical to its parent has D = 0 everywhere and fails.
Diagnostics (other endpoints, calibration-matched sensitivity, bootstrap) carry no pass/fail and cannot promote.
Secondary pipeline estimate (optional): child on the held fold's inner parent vs that parent, all six folds, H_bal only.
"""
from __future__ import annotations

import numpy as np

from research_r2.metrics import GAMMA_GRID, harmonic
from research_r2_v4 import confirm
from research_r2_v4 import evaluate as ev

PRIMARY_ENDPOINT = "H_bal"
THRESHOLDS = {"mean_gain_pp": 1.0, "folds_slack": 1, "ausuc_delta_min": 0.0, "shortcut_delta_max_pp": 5.0}
DIAGNOSTIC_ENDPOINTS = ("unseen_disease_top1", "unseen_crop_top1", "unseen_top1_macro", "unseen_withincrop_macro", "seen_top1", "H_micro")


class GateError(ValueError):
    """A child / parent comparison cannot be made, or a decision is requested from a non-primary result."""


# ----------------------------------------------------------------------------- input checks
def _identity(t):
    idt = getattr(t, "identity", None)
    if not isinstance(idt, dict) or idt.get("verified") is not True or "folds" not in idt:
        raise GateError(f"{getattr(t, 'name', t)}: unverified input (no identity record from child_inputs.load_verified)")
    return idt


def check_pair(child: ev.RowTable, parent: ev.RowTable, folds):
    """Verified identity on both sides, the same ordered evaluation images per fold, fold identities in the same order,
    one configuration, identical candidates (order, crop, disease, healthy, fitting mask) and identical truth."""
    ci, pi = _identity(child), _identity(parent)
    if list(child.folds) != list(folds) or list(parent.folds) != list(folds):
        raise GateError(f"fold identities differ: child {list(child.folds)}, parent {list(parent.folds)}, expected {list(folds)}")
    for k in folds:
        a, b = ci["folds"].get(k), pi["folds"].get(k)
        if a is None or b is None or a.get("fold") != k or b.get("fold") != k:
            raise GateError(f"{k}: identity record missing or names another fold")
        if a["identity_sha256"] != b["identity_sha256"]:
            raise GateError(f"{k}: evaluation image identity / order differs between child and parent")
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
        for s in want:
            if _identity(d[s]).get("seed") != s:
                raise GateError(f"{who}: table under seed {s} is identified as seed {_identity(d[s]).get('seed')}")
    return list(want)


# ----------------------------------------------------------------------------- per-fold values
def fold_values(t: ev.RowTable, k, folds, endpoint=PRIMARY_ENDPOINT, gi=None):
    """Endpoint, AUSUC and healthy shortcut of fold k at the row's own LOFO choice (gi=None) or at a given gamma index."""
    if endpoint != PRIMARY_ENDPOINT and endpoint not in DIAGNOSTIC_ENDPOINTS:
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


# ----------------------------------------------------------------------------- uncertainty (descriptive)
def _hits(t, k, gi):
    s = t.sc[k][0]; g = float(GAMMA_GRID[gi])
    return ((s.top1_pred("seen", g) == s.parts["seen"]["truth"]).astype(float), s.parts["seen"]["truth"],
            (s.top1_pred("unseen", g) == s.parts["unseen"]["truth"]).astype(float), s.parts["unseen"]["truth"])


def paired_bootstrap(detail, child_of, parent_of, folds, seeds, reps=2000, seed=0):
    """Composition-cluster bootstrap CI of the seed-mean, fold-mean H_bal difference child - parent, each at its own
    chosen gamma; the same resampled compositions for child and parent and for every seed in a fold.
    ``bootstrap_sign_proportion_le_0`` is the share of bootstrap draws <= 0: DESCRIPTIVE, not a p-value, and it is
    never multiplicity-adjusted or used as a significance test."""
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
        h0 = _hits(child_of(k, seeds[0]), k, 0)
        prep[k] = (rows, len(np.unique(h0[1])), len(np.unique(h0[3])))
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
            "bootstrap_sign_proportion_le_0": float(np.mean(draws <= 0)),
            "note": "descriptive share of bootstrap draws <= 0; not a p-value; not multiplicity-adjusted",
            "reps": reps, "seed": seed}


# ----------------------------------------------------------------------------- primary gate and decision
def primary_gate(child_by_seed: dict, parent_by_seed: dict, folds, mode="screen", endpoint=PRIMARY_ENDPOINT, reps=2000):
    """Child vs ONE fixed parent (J2) on every fold; the endpoint is locked to H_bal."""
    if endpoint != PRIMARY_ENDPOINT:
        raise GateError(f"the primary endpoint is {PRIMARY_ENDPOINT}; {endpoint!r} is diagnostic only and cannot gate or promote")
    seeds = check_seeds(child_by_seed, parent_by_seed, mode)
    child_of = lambda k, s: child_by_seed[s]  # noqa: E731
    parent_of = lambda k, s: parent_by_seed[s]  # noqa: E731
    per, detail = _deltas(child_of, parent_of, folds, seeds, PRIMARY_ENDPOINT)
    return {"role": "primary", "mode": mode, "endpoint": PRIMARY_ENDPOINT,
            "parent_per_fold": {k: parent_by_seed[seeds[0]].identity["row"] for k in folds},
            "gate": gate_from(per, folds), "detail": detail,
            "bootstrap": paired_bootstrap(detail, child_of, parent_of, folds, seeds, reps)}


def diagnostics(child_by_seed: dict, parent_by_seed: dict, folds, mode="screen", endpoints=DIAGNOSTIC_ENDPOINTS):
    """Diagnostic differences (no pass/fail): other endpoints at each row's own gamma, and the calibration-matched H_bal
    difference (child scored at the parent's gamma)."""
    seeds = check_seeds(child_by_seed, parent_by_seed, mode)
    child_of = lambda k, s: child_by_seed[s]  # noqa: E731
    parent_of = lambda k, s: parent_by_seed[s]  # noqa: E731
    out = {"role": "diagnostic", "mode": mode, "endpoints": {}}
    for e in endpoints:
        if e == PRIMARY_ENDPOINT:
            raise GateError("H_bal is the primary endpoint, not a diagnostic")
        per, _ = _deltas(child_of, parent_of, folds, seeds, e)
        out["endpoints"][e] = {"per_fold_delta_pp": {k: 100 * float(np.mean([per[s][k]["d"] for s in seeds])) for k in folds},
                               "mean_delta_pp": 100 * float(np.mean([per[s][k]["d"] for s in seeds for k in folds]))}
    per_m, _ = _deltas(child_of, parent_of, folds, seeds, PRIMARY_ENDPOINT, matched=True)
    out["calibration_matched_H_bal"] = {"per_fold_delta_pp": {k: 100 * float(np.mean([per_m[s][k]["d"] for s in seeds])) for k in folds},
                                        "mean_delta_pp": 100 * float(np.mean([per_m[s][k]["d"] for s in seeds for k in folds]))}
    return out


def promotion_decision(result: dict):
    """The only decision function: reads the primary H_bal gate; any other result is refused."""
    if result.get("role") != "primary" or result.get("endpoint") != PRIMARY_ENDPOINT or "gate" not in result:
        raise GateError("a child can only pass on the primary H_bal gate; diagnostic or alternative-endpoint results cannot promote")
    return {"mode": result["mode"], "pass": bool(result["gate"]["pass"]),
            "note": "screen pass -> run seeds 1-2; confirmation pass -> 'confirmed vs J2' (exploratory; never above D1 without review)"}


# ----------------------------------------------------------------------------- secondary pipeline estimate
def inner_parents(parents_seed0: dict, folds, order):
    """pi(k): best parent by seed-0 screen H_bal on folds != k (each inner fold at its own inner LOFO gamma); ties ->
    earlier in ``order``. The held fold k never enters its own choice."""
    out = {}
    for k in folds:
        inner = [f for f in folds if f != k]
        out[k] = max(order, key=lambda r: (parents_seed0[r].summary(inner)["primary"], -order.index(r)))
    return out


def pipeline_estimate(variants: dict, parents_by_seed: dict, folds, order, mode="screen", d1=None):
    """Secondary, H_bal only. variants: {parent: {seed: child-on-parent table}}; parents_by_seed: {parent: {seed: table}}."""
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
    per, detail = _deltas(child_of, parent_of, folds, seeds, PRIMARY_ENDPOINT)
    out = {"role": "secondary", "mode": mode, "endpoint": PRIMARY_ENDPOINT, "inner_parent": pi, "gate": gate_from(per, folds), "detail": detail}
    if d1 is not None:  # decomposition: child - D1 = (child - parent) + (parent - D1); only the first is the child's
        dec = {}
        for k in folds:
            dd = fold_values(d1, k, folds)["E"]
            par = float(np.mean([detail[s][k]["parent"]["E"] for s in seeds])); ch = float(np.mean([detail[s][k]["child"]["E"] for s in seeds]))
            dec[k] = {"child_attributable_pp": 100 * (ch - par), "parent_attributable_pp": 100 * (par - dd), "total_vs_D1_pp": 100 * (ch - dd),
                      "parent": pi[k]}
        out["decomposition_vs_D1"] = dec
    return out
