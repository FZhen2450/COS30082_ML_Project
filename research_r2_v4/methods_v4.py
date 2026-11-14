"""Stage I methods (v4, post-freeze exploratory). D1's machinery is reused; each row changes one thing.

``synth_set`` rebuilds D1's composition-head training set. With mode "mean" it is the exact D1 set
(asserted against research_r2.methods.score_synth_composition in tests), so a row that only swaps the
head (I3) or the synthesis (I6) differs from D1 in that component alone.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from research_r2.methods import (DEV, FoldData, _views, comp_balanced_weight, crop_means, fit_softmax, log_probs,
                                 score_factorised, score_synth_composition, standardise)


def _l2(t):
    return t / t.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def raw_fit_stats(feats_rows_views: torch.Tensor):
    """Exactly research_r2.methods.standardise's statistics: L2 per view, then mean/sd over all fitting views."""
    xf = _l2(feats_rows_views.float())
    flat = xf.reshape(-1, xf.shape[-1])
    return flat, flat.mean(0), flat.std(0).clamp_min(1e-6)


def synth_set(fd: FoldData, per=64, seed=0, text_frac=0.0, text_lambda=1.0, raw=None):
    """Composition-head training set: real fitting views + ``per`` synthetic features per held-out candidate.

    text_frac = 0: D1 mean-shift synthesis x - mu_crop(x) + mu_c' (identical random stream to D1).
    text_frac > 0: that fraction uses x_raw + lambda (t(c', d) - t(c, d)) in L2-normalised raw CLIP space,
    mapped into the standardised space with the fitting statistics (``raw`` = (raw_flat, mu, sd)).
    """
    x, crop, dis, comp_col, _ = _views(fd)
    mus = crop_means(x, crop, comp_col, fd.n_crop)
    xs, ys = [x], [comp_col]
    g = torch.Generator().manual_seed(int(seed))
    n_text = int(round(per * text_frac))
    n_mean = per - n_text
    if n_text:
        raw_flat, mu, sd = raw
        T = fd.text["comp"].float()
        assert T.shape[0] == len(fd.cand_crop), "text bank must align with the candidate columns"
    for col in np.nonzero(~fd.fit_mask)[0]:
        c, d = int(fd.cand_crop[col]), int(fd.cand_dis[col])
        srcs = torch.nonzero((dis == d) & (crop != c)).flatten()
        if len(srcs) == 0:
            continue
        parts = []
        if n_mean:
            pick = srcs[torch.randint(0, len(srcs), (n_mean,), generator=g)]
            parts.append(x[pick] - mus[crop[pick]] + mus[c])
        if n_text:
            pick = srcs[torch.randint(0, len(srcs), (n_text,), generator=g)]
            out_raw = raw_flat[pick] + text_lambda * (T[int(col)] - T[comp_col[pick]])
            parts.append(((out_raw - mu) / sd).to(x.dtype))
        hx = torch.cat(parts)
        xs.append(hx); ys.append(torch.full((len(hx),), int(col)))
    X, Y = torch.cat(xs), torch.cat(ys)
    return X, Y, comp_balanced_weight(Y)


def comp_head_logprobs(fd, X, Y, w, head="logistic", tau=0.05, wd=5e-4):
    K = len(fd.cand_crop)
    if head == "logistic":
        h = fit_softmax(X, Y, K, w, wd=wd)
        return log_probs(h, fd.x_seen), log_probs(h, fd.x_uns)
    if head == "cosine":  # prototype = mean of L2-normalised training features of each candidate
        Xn = _l2(X.double())
        P = torch.zeros(K, X.shape[1], dtype=torch.float64).index_add_(0, Y, Xn)
        P = _l2(P / torch.bincount(Y, minlength=K).clamp_min(1)[:, None].double())
        f = lambda q: F.log_softmax(_l2(q.double()) @ P.T / tau, 1)  # noqa: E731
        return f(fd.x_seen), f(fd.x_uns)
    raise KeyError(head)


def score_synth_v4(fd: FoldData, cfg: dict):
    """D1 with a replaceable synthesis (cfg text_frac) and composition head (cfg head, tau); fuse 1, normalise."""
    X, Y, w = synth_set(fd, per=cfg.get("per_comp", 64), seed=cfg.get("seed", 0), text_frac=cfg.get("text_frac", 0.0),
                        text_lambda=cfg.get("text_lambda", 1.0), raw=fd.extra.get("raw_fit"))
    ls, lu = comp_head_logprobs(fd, X, Y, w, head=cfg.get("head", "logistic"), tau=cfg.get("tau", 0.05))
    fs, fu = score_factorised(fd, {})
    return (fs + ls) / 2.0, (fu + lu) / 2.0


# ----------------------------------------------------------------------------- learned projections (I4, I5)
def _cka(a, b):
    a = a - a.mean(0); b = b - b.mean(0)
    hab = (a.T @ b).pow(2).sum()
    return hab / ((a.T @ a).pow(2).sum().sqrt() * (b.T @ b).pow(2).sum().sqrt()).clamp_min(1e-12)


def fit_factor_projection(x, crop, dis, n_crop, n_dis, lam=1.0, dim=256, steps=200, seed=0):
    """Linear crop and disease codes with CE heads and a normalised linear-kernel HSIC (linear CKA) penalty."""
    torch.manual_seed(seed)
    x, crop, dis = x.float().to(DEV), crop.to(DEV), dis.to(DEV)
    d = x.shape[1]
    wc, wd_ = torch.nn.Linear(d, dim).to(DEV), torch.nn.Linear(d, dim).to(DEV)
    hc, hd = torch.nn.Linear(dim, n_crop).to(DEV), torch.nn.Linear(dim, n_dis).to(DEV)
    params = [p for m in (wc, wd_, hc, hd) for p in m.parameters()]
    opt = torch.optim.Adam(params, lr=1e-3)
    log = []
    for s in range(steps):
        opt.zero_grad()
        zc, zd = wc(x), wd_(x)
        ce = F.cross_entropy(hc(zc), crop) + F.cross_entropy(hd(zd), dis)
        dep = _cka(zc, zd)
        loss = ce + lam * dep
        loss.backward(); opt.step()
        if s in (0, steps - 1):
            log.append({"step": s, "ce": float(ce), "cka": float(dep)})
    proj = lambda t: torch.cat([wc(t.float().to(DEV)), wd_(t.float().to(DEV))], -1).detach().cpu()  # noqa: E731
    return proj, log


def fit_supcon_projection(x, crop, dis, cross_crop=True, epochs=30, batch=512, tau=0.1, seed=0):
    """MLP 768-1024-512; supervised contrastive on disease. cross_crop: positives must have a different crop."""
    torch.manual_seed(seed)
    g = torch.Generator(device="cpu").manual_seed(seed)
    x, crop, dis = x.float().to(DEV), crop.to(DEV), dis.to(DEV)
    net = torch.nn.Sequential(torch.nn.Linear(x.shape[1], 1024), torch.nn.GELU(), torch.nn.Linear(1024, 512)).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    n, log = len(x), []
    for ep in range(epochs):
        tot, cnt = 0.0, 0
        for b in torch.randperm(n, generator=g).to(DEV).split(batch):
            z = F.normalize(net(x[b]), dim=-1)
            sim = z @ z.T / tau
            eye = torch.eye(len(b), dtype=torch.bool, device=DEV)
            pos = (dis[b][:, None] == dis[b][None]) & ~eye
            if cross_crop:
                pos &= crop[b][:, None] != crop[b][None]
            has = pos.any(1)
            if not has.any():
                continue
            logp = sim.masked_fill(eye, -1e9).log_softmax(1)
            loss = -(logp * pos).sum(1)[has] / pos.sum(1)[has]
            loss = loss.mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * int(has.sum()); cnt += int(has.sum())
        if ep in (0, epochs - 1):
            log.append({"epoch": ep, "supcon": tot / max(cnt, 1), "anchors_with_positive": cnt})
    net.eval()
    proj = lambda t: net(t.float().to(DEV)).detach().cpu()  # noqa: E731
    return proj, log


def projected_fold(fd: FoldData, proj) -> FoldData:
    """Apply a projection fitted on fitting views, then re-standardise with fitting statistics (standardise)."""
    n, v, d = fd.x_fit.shape
    with torch.no_grad():
        zf = proj(fd.x_fit.reshape(n * v, d)).reshape(n, v, -1)
        zs, zu = proj(fd.x_seen), proj(fd.x_uns)
    fd2 = FoldData(x_fit=zf, x_seen=zs, x_uns=zu, fit_crop=fd.fit_crop, fit_dis=fd.fit_dis, fit_comp_col=fd.fit_comp_col,
                   fit_source=fd.fit_source, cand_crop=fd.cand_crop, cand_dis=fd.cand_dis, fit_mask=fd.fit_mask,
                   n_crop=fd.n_crop, n_dis=fd.n_dis, n_source=fd.n_source, text=None,
                   extra={"cand_ids": fd.extra["cand_ids"]})
    return standardise(fd2)


def score_projected_d1(fd: FoldData, cfg: dict):
    x, crop, dis, _, _ = _views(fd)
    if cfg["projection"] == "factor":
        proj, log = fit_factor_projection(x, crop, dis, fd.n_crop, fd.n_dis, lam=cfg.get("hsic_lambda", 1.0), seed=cfg.get("seed", 0))
    elif cfg["projection"] == "supcon":
        proj, log = fit_supcon_projection(x, crop, dis, cross_crop=cfg.get("cross_crop", True), seed=cfg.get("seed", 0))
    else:
        raise KeyError(cfg["projection"])
    fd2 = projected_fold(fd, proj)
    fd.extra["projection_log"] = log
    return score_synth_composition(fd2, {"fuse": 1.0, "normalise": True})
