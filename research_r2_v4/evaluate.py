"""Row evaluation for v4: LOFO over (config, gamma), the v3 gate, nested LOFO, paired bootstrap, extra diagnostics.

A row's fold entry holds one FoldScorer per *config* (e.g. I1's gamma1 values; ordinary rows have one config).
LOFO for fold k = argmax over (config, gamma) of mean H_bal over the other folds; ties -> lower config index,
then smaller gamma (np.argmax on a config-major flattening), which for one config is exactly the v3 rule.
The gate and nested estimate mirror scripts/r2_select.py (gate: >= +1 pp, >= n-1 of n folds, AUSUC not lower,
healthy shortcut <= +5 pp; nested: rerun on n-1 folds with n-2 required, score held fold at pooled choice).
"""
from __future__ import annotations

import numpy as np

from research_r2.metrics import GAMMA_GRID, harmonic

KEYS = ("H_bal", "H_micro", "seen_top1", "unseen_top1_macro", "unseen_top1_micro", "unseen_disease_top1",
        "unseen_crop_top1", "unseen_healthy_shortcut", "unseen_withincrop_macro", "H_withincrop")


class RowTable:
    def __init__(self, name, scorers: dict[str, list], config_labels: list):
        self.name, self.sc, self.configs = name, scorers, config_labels
        self.folds = list(scorers)
        self.H = {f: np.array([[s.at(g)["H_bal"] for g in GAMMA_GRID] for s in scorers[f]]) for f in self.folds}
        self.ausuc = {f: [s.ausuc() for s in scorers[f]] for f in self.folds}

    def choose(self, folds):
        m = np.mean([self.H[f] for f in folds], 0)
        ci, gi = np.unravel_index(int(np.argmax(m)), m.shape)
        return int(ci), int(gi)

    def at(self, fold, ci, gi, top5=False):
        return self.sc[fold][ci].at(float(GAMMA_GRID[gi]), top5=top5)

    def summary(self, folds):
        per = {}
        for k in folds:
            ci, gi = self.choose([f for f in folds if f != k])
            r = self.at(k, ci, gi)
            per[k] = {"config": self.configs[ci], "ci": ci, "gamma": float(GAMMA_GRID[gi]), "gi": gi,
                      "ausuc": self.ausuc[k][ci], **{x: r[x] for x in KEYS}}
        return {"per": per, "primary": float(np.mean([v["H_bal"] for v in per.values()])),
                "ausuc": float(np.mean([v["ausuc"] for v in per.values()])),
                "hs": float(np.mean([v["unseen_healthy_shortcut"] for v in per.values()]))}


def gate(c, k, folds):
    d = [c["per"][f]["H_bal"] - k["per"][f]["H_bal"] for f in folds]
    g = {"mean_gain_pp": 100 * (c["primary"] - k["primary"]), "folds_improved": int(sum(x > 0 for x in d)),
         "folds_regressed": int(sum(x < 0 for x in d)), "folds_required": len(folds) - 1,
         "ausuc_delta": c["ausuc"] - k["ausuc"], "shortcut_delta_pp": 100 * (c["hs"] - k["hs"]),
         "per_fold_delta_pp": {f: 100 * x for f, x in zip(folds, d)}}
    g["pass"] = bool(g["mean_gain_pp"] >= 1.0 and g["folds_improved"] >= len(folds) - 1 and g["ausuc_delta"] >= 0
                     and g["shortcut_delta_pp"] <= 5.0)
    return g


def nested(tables: dict, control: str, candidates: list, folds):
    """Nested LOFO for the selection 'control -> best gate-passing candidate' (one level, as r2_select)."""
    out = []
    for k in folds:
        inner = [f for f in folds if f != k]
        S = {r: tables[r].summary(inner) for r in [control] + candidates}
        passing = [r for r in candidates if gate(S[r], S[control], inner)["pass"]]
        sel = max(passing, key=lambda r: (S[r]["primary"], -candidates.index(r))) if passing else control
        def held(r):
            ci, gi = tables[r].choose(inner)
            x = tables[r].at(k, ci, gi)
            return x["H_bal"], x["H_micro"], float(GAMMA_GRID[gi]), tables[r].configs[ci]
        hs, ms, gs, cs = held(sel); hc, mc, gc, _ = held(control)
        out.append({"held_fold": k, "selected": sel, "H_bal": hs, "H_micro": ms, "gamma": gs, "config": cs,
                    "control_H_bal": hc, "control_H_micro": mc, "control_gamma": gc})
    d = [100 * (n["H_bal"] - n["control_H_bal"]) for n in out]
    return {"held": out, "mean_diff_pp": float(np.mean(d)), "folds_better": int(sum(x > 0 for x in d)),
            "folds_worse": int(sum(x < 0 for x in d)),
            "mean_H_bal_selected": float(np.mean([n["H_bal"] for n in out])),
            "selection_counts": {r: sum(n["selected"] == r for n in out) for r in sorted({n["selected"] for n in out})},
            "meets_gate_thresholds": bool(np.mean(d) >= 1.0 and sum(x > 0 for x in d) >= len(folds) - 1)}


def _hits(t: RowTable, fold, ci, gi):
    s = t.sc[fold][ci]; g = float(GAMMA_GRID[gi])
    hs = s.top1_pred("seen", g) == s.parts["seen"]["truth"]
    hu = s.top1_pred("unseen", g) == s.parts["unseen"]["truth"]
    return hs.astype(float), hu.astype(float), s.parts["seen"]["truth"], s.parts["unseen"]["truth"]


def paired_bootstrap(row: RowTable, ctrl: RowTable, folds, reps=2000, seed=0):
    """Paired composition-cluster bootstrap of mean_f [H_bal(row) - H_bal(ctrl)] at each fold's LOFO choice.
    Seen: resample seen compositions (image clusters); unseen: resample held-out compositions (macro)."""
    rng = np.random.default_rng(seed)
    prep = []
    for k in folds:
        others = [f for f in folds if f != k]
        a = _hits(row, k, *row.choose(others)); b = _hits(ctrl, k, *ctrl.choose(others))
        assert np.array_equal(a[2], b[2]) and np.array_equal(a[3], b[3]), "row and control truth differ"
        sc = np.unique(a[2]); uc = np.unique(a[3])
        S = lambda h, t: (np.array([h[t == c].sum() for c in sc]), np.array([(t == c).sum() for c in sc]))  # noqa: E731
        U = lambda h, t: np.array([h[t == c].mean() for c in uc])  # noqa: E731
        prep.append((S(a[0], a[2]), S(b[0], b[2]), U(a[1], a[3]), U(b[1], b[3]), len(sc), len(uc)))
    diffs = np.empty(reps)
    for r in range(reps):
        ds = []
        for (sa, sb, ua, ub, ns, nu) in prep:
            i = rng.integers(0, ns, ns); j = rng.integers(0, nu, nu)
            ha = harmonic(sa[0][i].sum() / sa[1][i].sum(), ua[j].mean())
            hb = harmonic(sb[0][i].sum() / sb[1][i].sum(), ub[j].mean())
            ds.append(ha - hb)
        diffs[r] = np.mean(ds)
    point = np.mean([harmonic(p[0][0].sum() / p[0][1].sum(), p[2].mean()) - harmonic(p[1][0].sum() / p[1][1].sum(), p[3].mean())
                     for p in prep])
    return {"point_pp": 100 * float(point), "ci95_pp": [100 * float(np.percentile(diffs, 2.5)), 100 * float(np.percentile(diffs, 97.5))],
            "p_le_0": float(np.mean(diffs <= 0)), "reps": reps, "seed": seed}


def false_healthy(t: RowTable, folds):
    out = {}
    for k in folds:
        ci, gi = t.choose([f for f in folds if f != k])
        s = t.sc[k][ci]; g = float(GAMMA_GRID[gi])
        ps, pu = s.top1_pred("seen", g), s.top1_pred("unseen", g)
        ts, tu = s.parts["seen"]["truth"], s.parts["unseen"]["truth"]
        dis_s = ~s.cand_healthy[ts]
        out[k] = {"seen_diseased_pred_healthy": float(np.mean(s.cand_healthy[ps][dis_s])),
                  "unseen_pred_healthy": float(np.mean(s.cand_healthy[pu])), "unseen_truth_healthy": int(s.cand_healthy[tu].sum())}
    return {"per_fold": out, "mean_seen_diseased_pred_healthy": float(np.mean([v["seen_diseased_pred_healthy"] for v in out.values()])),
            "mean_unseen_pred_healthy": float(np.mean([v["unseen_pred_healthy"] for v in out.values()]))}


def source_strata(t: RowTable, frame, folds_meta, folds):
    """Seen-val Top-1 at the LOFO choice by image source relative to its composition's fitting sources:
    dominant fitting source / other fitting source / source absent from that composition's fitting images."""
    src = frame.source.to_numpy(); comp = frame.comp_id.to_numpy()
    acc = {"dominant": [0, 0], "other_fitting_source": [0, 0], "novel_source": [0, 0]}
    for k in folds:
        f = folds_meta[k]
        fit = np.asarray(f["fitting_rows"])
        dom, have = {}, {}
        for c in np.unique(comp[fit]):
            ss, cnt = np.unique(src[fit][comp[fit] == c], return_counts=True)
            dom[c] = ss[np.argmax(cnt)]; have[c] = set(ss)
        h, _, _, _ = _hits(t, k, *t.choose([x for x in folds if x != k]))
        for i, r in enumerate(f["seen_val_rows"]):
            c = comp[r]
            key = "dominant" if src[r] == dom[c] else ("other_fitting_source" if src[r] in have[c] else "novel_source")
            acc[key][0] += h[i]; acc[key][1] += 1
    return {k: {"top1": (v[0] / v[1]) if v[1] else None, "n_image_folds": v[1]} for k, v in acc.items()}


def curves(t: RowTable):
    return {f: {t.configs[ci]: {"gamma": GAMMA_GRID.tolist(),
                                "H_bal": [s.at(g)["H_bal"] for g in GAMMA_GRID],
                                "seen_top1": [s.at(g)["seen_top1"] for g in GAMMA_GRID],
                                "unseen_top1_macro": [s.at(g)["unseen_top1_macro"] for g in GAMMA_GRID]}
                for ci, s in enumerate(t.sc[f])} for f in t.folds}
