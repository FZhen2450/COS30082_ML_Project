"""Round 2 compositional metrics on top of the WP5 Top-k contract.

Columns are candidates in ascending global composition ID order, so the WP5 stable
tie rule (lowest column wins) equals "lowest global ID wins". Calibrated stacking
(Chao et al., ECCV 2016) subtracts gamma from fitting-composition candidates only.
Top-1 under any gamma is derived exactly from each row's best fitting candidate and
best held-out candidate, which also gives an exact AUSUC.
"""
from __future__ import annotations

import numpy as np

GAMMA_GRID = np.round(np.arange(0.0, 16.0 + 1e-9, 0.25), 2)


def harmonic(a: float, b: float) -> float:
    return 0.0 if a + b == 0 else 2 * a * b / (a + b)


class FoldScorer:
    def __init__(self, seen_scores, seen_truth_col, unseen_scores, unseen_truth_col, fit_mask,
                 cand_crop, cand_disease, cand_healthy):
        self.fit_mask = np.asarray(fit_mask, bool)
        self.cand_crop, self.cand_dis = np.asarray(cand_crop), np.asarray(cand_disease)
        self.cand_healthy = np.asarray(cand_healthy, bool)
        self.parts = {}
        for name, s, t in (("seen", seen_scores, seen_truth_col), ("unseen", unseen_scores, unseen_truth_col)):
            s = np.asarray(s, np.float64)
            if not np.isfinite(s).all():
                raise ValueError("non-finite scores")
            t = np.asarray(t, int)
            sf = np.where(self.fit_mask[None], s, -np.inf)
            su = np.where(~self.fit_mask[None], s, -np.inf)
            bf, bu = sf.argmax(1), su.argmax(1)  # argmax returns first max -> lowest column
            self.parts[name] = dict(scores=s, truth=t, bf=bf, bu=bu,
                                    d=s[np.arange(len(t)), bf] - s[np.arange(len(t)), bu])
        t = self.parts["unseen"]["truth"]
        comps, inv = np.unique(t, return_inverse=True)
        self.u_weight = 1.0 / (len(comps) * np.bincount(inv)[inv])

    def top1_pred(self, part, gamma):
        p = self.parts[part]
        choose_fit = (p["d"] > gamma) | ((p["d"] == gamma) & (p["bf"] < p["bu"]))
        return np.where(choose_fit, p["bf"], p["bu"])

    def at(self, gamma: float, top5: bool = False) -> dict:
        ps, pu = self.top1_pred("seen", gamma), self.top1_pred("unseen", gamma)
        ts, tu = self.parts["seen"]["truth"], self.parts["unseen"]["truth"]
        seen = float(np.mean(ps == ts))
        hit_u = (pu == tu)
        un_macro, un_micro = float(np.sum(hit_u * self.u_weight)), float(np.mean(hit_u))
        r = {"gamma": float(gamma), "seen_top1": seen, "unseen_top1_macro": un_macro, "unseen_top1_micro": un_micro,
             "H_bal": harmonic(seen, un_macro), "H_micro": harmonic(seen, un_micro),
             "unseen_crop_top1": float(np.mean(self.cand_crop[pu] == self.cand_crop[tu])),
             "unseen_disease_top1": float(np.mean(self.cand_dis[pu] == self.cand_dis[tu])),
             "unseen_healthy_shortcut": float(np.mean(self.cand_healthy[pu])),
             "unseen_pred_fitting_comp": float(np.mean(self.fit_mask[pu])),
             "seen_pred_heldout_comp": float(np.mean(~self.fit_mask[ps]))}
        # Disease-transfer diagnostic: rank only the TRUE crop's candidates (oracle crop), so the
        # held-out pair must beat that crop's fitting diseases on disease evidence alone.
        for part, key in (("unseen", "unseen_withincrop_macro"), ("seen", "seen_withincrop")):
            if not np.isfinite(gamma):
                r[key] = float("nan")
                continue
            p = self.parts[part]
            same = self.cand_crop[None, :] == self.cand_crop[p["truth"]][:, None]
            s = np.where(same, p["scores"] - gamma * self.fit_mask[None], -np.inf)
            hit = s.argmax(1) == p["truth"]
            r[key] = float(np.sum(hit * self.u_weight)) if part == "unseen" else float(np.mean(hit))
        r["H_withincrop"] = harmonic(r["seen_withincrop"], r["unseen_withincrop_macro"])
        if top5:
            for part, key in (("seen", "seen_top5"), ("unseen", "unseen_top5_micro")):
                p = self.parts[part]
                s = p["scores"] - gamma * self.fit_mask[None]
                order = np.argsort(-s, axis=1, kind="stable")[:, :5]
                r[key] = float(np.mean((order == p["truth"][:, None]).any(1)))
            r["H5_micro"] = harmonic(r["seen_top5"], r["unseen_top5_micro"])
        return r

    def curve(self, grid=GAMMA_GRID):
        return [self.at(g) for g in grid]

    def ausuc(self) -> float:
        """Exact area under (unseen-macro, seen) Top-1 as gamma sweeps -inf..+inf."""
        ds = np.concatenate([self.parts["seen"]["d"], self.parts["unseen"]["d"]])
        g = np.unique(ds)
        pts = [self.at(-np.inf)] + [self.at(x) for x in g] + [self.at(np.inf)]
        xs = np.array([p["unseen_top1_macro"] for p in pts])
        ys = np.array([p["seen_top1"] for p in pts])
        o = np.argsort(xs, kind="stable")
        xs, ys = xs[o], ys[o]
        # at equal x keep the max seen accuracy (upper envelope of the step curve)
        ux = np.unique(xs)
        uy = np.array([ys[xs == x].max() for x in ux])
        return float(np.trapezoid(uy, ux)) if len(ux) > 1 else 0.0


def lofo_calibrated(curves: dict[str, list[dict]], grid=GAMMA_GRID, key="H_bal") -> dict[str, dict]:
    """Leave-one-fold-out gamma: for fold f choose argmax mean `key` over the other folds."""
    names = list(curves)
    table = np.array([[c[key] for c in curves[n]] for n in names])  # folds x grid
    out = {}
    for i, n in enumerate(names):
        others = np.delete(table, i, axis=0).mean(0)
        j = int(np.argmax(others))  # first max -> smallest gamma
        out[n] = dict(curves[n][j], gamma_source="lofo")
    return out


def pooled_gamma(curves: dict[str, list[dict]], grid=GAMMA_GRID, key="H_bal") -> float:
    table = np.array([[c[key] for c in curves[n]] for n in curves])
    return float(grid[int(np.argmax(table.mean(0)))])
