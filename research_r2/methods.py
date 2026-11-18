"""Frozen-feature compositional methods for Round 2 screening.

All fitted quantities (standardisation, erasers, crop means, prototypes, heads) use
fitting rows of the fold only. Validation rows are touched only to produce scores.
Every method returns composition log-scores [N, K] over the fold's candidate table.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------------------------------------------------------- erasure
class LeaceEraser:
    """LEACE (Belrose et al., NeurIPS 2023): least-squares linear concept erasure.

    r(x) = x - W^+ P W (x - mu), W = Sigma_xx^{-1/2}, P = projector onto col(W Sigma_xz).
    After erasure no linear predictor can beat a constant at predicting the concept z.
    """

    def __init__(self, x: torch.Tensor, z_onehot: torch.Tensor, ridge: float = 1e-4, strength: float = 1.0):
        x = x.double()
        z = z_onehot.double()
        self.mu = x.mean(0)
        xc, zc = x - self.mu, z - z.mean(0)
        n, d = x.shape
        sxx = xc.T @ xc / n
        sxx = sxx + ridge * torch.trace(sxx) / d * torch.eye(d, dtype=sxx.dtype, device=sxx.device)
        sxz = xc.T @ zc / n
        evals, evecs = torch.linalg.eigh(sxx)
        w = evecs @ torch.diag(evals.rsqrt()) @ evecs.T
        w_inv = evecs @ torch.diag(evals.sqrt()) @ evecs.T
        u, s, _ = torch.linalg.svd(w @ sxz, full_matrices=False)
        u = u[:, s > s.max() * 1e-6]
        self.proj = (w_inv @ u @ u.T @ w).T  # applied as row-vector map
        self.strength = strength

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        xd = x.double()
        return (xd - self.strength * (xd - self.mu) @ self.proj).to(x.dtype)


# ----------------------------------------------------------------------------- heads
def fit_softmax(x, y, n_classes, weight=None, wd=5e-4, hidden=0, steps=200, seed=0, epochs=30):
    """Weighted multinomial logistic regression (L-BFGS) or 1-hidden-layer MLP (Adam)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    x, y = x.float().to(DEV), y.long().to(DEV)
    w_s = torch.ones(len(y), device=DEV) if weight is None else weight.float().to(DEV)
    w_s = w_s / w_s.mean()
    d = x.shape[1]
    if hidden == 0:
        lin = torch.nn.Linear(d, n_classes).to(DEV)
        torch.nn.init.zeros_(lin.weight), torch.nn.init.zeros_(lin.bias)
        opt = torch.optim.LBFGS(lin.parameters(), lr=1, max_iter=steps, history_size=20,
                                line_search_fn="strong_wolfe", tolerance_grad=1e-6, tolerance_change=1e-9)

        def closure():
            opt.zero_grad()
            loss = (F.cross_entropy(lin(x), y, reduction="none") * w_s).mean() + wd * lin.weight.pow(2).sum()
            loss.backward()
            return loss
        opt.step(closure)
        return lin.eval()
    torch.manual_seed(int(torch.randint(0, 2**31 - 1, (1,), generator=g)))
    net = torch.nn.Sequential(torch.nn.Linear(d, hidden), torch.nn.GELU(), torch.nn.Dropout(0.3),
                              torch.nn.Linear(hidden, n_classes)).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=0.05)
    n = len(y)
    for _ in range(epochs):
        idx = torch.randperm(n, device=DEV)
        for b in idx.split(512):
            opt.zero_grad()
            (F.cross_entropy(net(x[b]), y[b], reduction="none") * w_s[b]).mean().backward()
            opt.step()
    return net.eval()


@torch.no_grad()
def log_probs(head, x):
    return F.log_softmax(head(x.float().to(DEV)), 1).double().cpu()


# ----------------------------------------------------------------------------- fold data
@dataclass
class FoldData:
    x_fit: torch.Tensor       # [n, V, d] all views
    x_seen: torch.Tensor      # [ns, d] centre view
    x_uns: torch.Tensor       # [nu, d]
    fit_crop: torch.Tensor
    fit_dis: torch.Tensor
    fit_comp_col: torch.Tensor  # candidate column of each fitting image
    fit_source: torch.Tensor
    cand_crop: np.ndarray     # crop index per candidate column
    cand_dis: np.ndarray
    fit_mask: np.ndarray      # candidate is a fitting composition
    n_crop: int
    n_dis: int
    n_source: int
    text: dict | None = None
    extra: dict = field(default_factory=dict)


def standardise(fd: FoldData) -> FoldData:
    """L2-normalise then z-score with fitting statistics (all views)."""
    def l2(t):
        return t / t.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    xf = l2(fd.x_fit.float())
    flat = xf.reshape(-1, xf.shape[-1])
    mu, sd = flat.mean(0), flat.std(0).clamp_min(1e-6)
    fd.extra["raw_seen"], fd.extra["raw_uns"] = l2(fd.x_seen.float()), l2(fd.x_uns.float())
    fd.x_fit = (xf - mu) / sd
    fd.x_seen = (l2(fd.x_seen.float()) - mu) / sd
    fd.x_uns = (l2(fd.x_uns.float()) - mu) / sd
    return fd


def _views(fd):
    n, v, d = fd.x_fit.shape
    rep = lambda t: t.repeat_interleave(v)  # noqa: E731
    return fd.x_fit.reshape(n * v, d), rep(fd.fit_crop), rep(fd.fit_dis), rep(fd.fit_comp_col), rep(fd.fit_source)


def comp_balanced_weight(comp_col):
    _, inv, cnt = torch.unique(comp_col, return_inverse=True, return_counts=True)
    return 1.0 / cnt[inv].double()


def crop_means(x, crop, comp_col, n_crop):
    """Composition-balanced crop means (mean of composition means) to limit disease confounding."""
    mus = torch.zeros(n_crop, x.shape[1], dtype=x.dtype)
    for c in torch.unique(crop):
        m = crop == c
        comp_means = [x[m & (comp_col == k)].mean(0) for k in torch.unique(comp_col[m])]
        mus[c] = torch.stack(comp_means).mean(0)
    return mus


def synth_crop_shift(x, crop, dis, comp_col, n_crop, per_image=2, seed=0):
    """Hallucinate disease-d features on other crops: x - mu_c + mu_c' (c' != c, c' observed)."""
    g = torch.Generator().manual_seed(seed)
    mus = crop_means(x, crop, comp_col, n_crop)
    present = torch.unique(crop)
    xs, ds, cs = [], [], []
    for _ in range(per_image):
        tgt = present[torch.randint(0, len(present), (len(crop),), generator=g)]
        same = tgt == crop
        tgt[same] = present[(torch.searchsorted(present, tgt[same]) + 1) % len(present)]
        xs.append(x - mus[crop] + mus[tgt])
        ds.append(dis)
        cs.append(tgt)
    return torch.cat(xs), torch.cat(ds), torch.cat(cs)


def zero_shot_logprobs(fd: FoldData, part: str, kind: str):
    """CLIP zero-shot log-softmax over crops, diseases or candidate compositions (raw normalised features)."""
    t = fd.text
    img = fd.extra["raw_seen" if part == "seen" else "raw_uns"].double()
    bank = t[kind].double()
    logits = t["logit_scale"] * img @ bank.T
    return F.log_softmax(logits, 1)


# ----------------------------------------------------------------------------- Phase 2 helpers (v3)
def within_concept_source_weight(concept, src):
    """Reweight so every source contributes equally *within* each concept (crop or disease), keeping each
    concept's total weight equal to its image count: w = n_concept / (n_sources(concept) * n(concept, source))."""
    key = concept.long() * 100003 + src.long()
    _, inv_k, cnt_k = torch.unique(key, return_inverse=True, return_counts=True)
    _, inv_c, cnt_c = torch.unique(concept, return_inverse=True, return_counts=True)
    w = torch.zeros(len(concept), dtype=torch.float64)
    for c in torch.unique(inv_c):
        m = inv_c == c
        n_src = len(torch.unique(key[m]))
        w[m] = cnt_c[c].double() / (n_src * cnt_k[inv_k[m]].double())
    return w


def style_bank(x, comp_col, src, min_images=2):
    """Composition-controlled source offsets: delta = mean(x | comp, source) - mean over sources of that
    composition, for compositions observed in >= 2 sources (each with >= min_images fitting views)."""
    deltas = []
    for k in torch.unique(comp_col):
        m = comp_col == k
        srcs = [s for s in torch.unique(src[m]) if int((m & (src == s)).sum()) >= min_images]
        if len(srcs) < 2:
            continue
        means = torch.stack([x[m & (src == s)].mean(0) for s in srcs])
        deltas.append(means - means.mean(0))
    return torch.cat(deltas) if deltas else torch.zeros(0, x.shape[1], dtype=x.dtype)


def style_augment(x, ys, bank, k=1, seed=0):
    """Add ``k`` copies of x shifted by random style offsets; labels in ``ys`` are repeated unchanged."""
    if len(bank) == 0:
        return x, ys, 0
    g = torch.Generator().manual_seed(seed)
    xs = [x + bank[torch.randint(0, len(bank), (len(x),), generator=g)].to(x.dtype) for _ in range(k)]
    return torch.cat([x] + xs), [torch.cat([y] + [y] * k) for y in ys], len(x) * k


# ----------------------------------------------------------------------------- recipes
def score_factorised(fd: FoldData, cfg: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Independent crop and disease heads; composition score = log p(c|x) + log p(d|x) (+ extras)."""
    x, crop, dis, comp_col, src = _views(fd)
    seen_c, uns_c = fd.x_seen, fd.x_uns
    seen_d, uns_d = fd.x_seen, fd.x_uns
    x_c, x_d = x, x
    if cfg.get("erase_source"):
        er = LeaceEraser(x.to(DEV), F.one_hot(src, fd.n_source).to(DEV))
        x_c = x_d = er(x.to(DEV)).cpu()
        seen_c = seen_d = er(fd.x_seen.to(DEV)).cpu()
        uns_c = uns_d = er(fd.x_uns.to(DEV)).cpu()
    if cfg.get("erase_crop_from_disease"):
        er = LeaceEraser(x_d.to(DEV), F.one_hot(crop, fd.n_crop).to(DEV), strength=cfg.get("erase_strength", 1.0))
        x_d, seen_d, uns_d = (er(t.to(DEV)).cpu() for t in (x_d, seen_d, uns_d))
    w_d = comp_balanced_weight(comp_col) if cfg.get("balance_disease") else None
    w_c = comp_balanced_weight(comp_col) if cfg.get("balance_crop") else None
    if cfg.get("source_balance"):
        w_c = within_concept_source_weight(crop, src)
        w_d = within_concept_source_weight(dis, src)
    if cfg.get("style_aug"):
        if cfg.get("erase_source") or cfg.get("erase_crop_from_disease"):
            raise ValueError("style_aug is not defined together with erasure")
        sa = cfg["style_aug"]
        n0, k = len(x), int(sa.get("k", 1))
        x_aug, (crop, dis, comp_col, src), n_aug = style_augment(x, (crop, dis, comp_col, src), style_bank(x, comp_col, src), k, sa.get("seed", 0))
        x_c = x_d = x_aug
        if n_aug:
            sw = float(sa.get("w", 0.5))
            def _ext(w):
                base = torch.ones(n0, dtype=torch.float64) if w is None else w / w.mean()
                return torch.cat([base] + [base * sw] * k)
            w_c, w_d = _ext(w_c), _ext(w_d)
    ccd = cfg.get("crop_centred_disease")
    if ccd:
        mus_cc = crop_means(x_d.double(), crop, comp_col, fd.n_crop).to(x_d.dtype)
        x_d = x_d - mus_cc[crop]
    xd_train, yd_train, wd_train = x_d, dis, w_d
    xc_train, yc_train, wc_train = x_c, crop, w_c
    if cfg.get("synth"):
        xs, ds, cs = synth_crop_shift(x_d, crop, dis, comp_col, fd.n_crop, per_image=cfg.get("synth_k", 2))
        base_w = torch.ones(len(dis), dtype=torch.float64) if w_d is None else w_d / w_d.mean()
        sw = torch.full((len(ds),), cfg.get("synth_w", 0.5), dtype=torch.float64)
        if w_d is not None:
            sw = sw * base_w.repeat(cfg.get("synth_k", 2))
        xd_train, yd_train = torch.cat([x_d, xs]), torch.cat([dis, ds])
        wd_train = torch.cat([base_w, sw])
    hid = cfg.get("hidden", 0)
    wd = cfg.get("wd", 5e-4)
    hc = fit_softmax(xc_train, yc_train, fd.n_crop, wc_train, wd=wd, hidden=hid)
    hd = fit_softmax(xd_train, yd_train, fd.n_dis, wd_train, wd=wd, hidden=hid)
    out = []
    for part, (xc_, xd_) in (("seen", (seen_c, seen_d)), ("unseen", (uns_c, uns_d))):
        lc = log_probs(hc, xc_)
        if ccd:  # test-time crop unknown: centre on the crop-probability-weighted crop mean
            xd_ = xd_ - (lc.exp().to(xd_.dtype) @ mus_cc)
        ld = log_probs(hd, xd_)
        if cfg.get("text_crop_beta"):
            lc = lc + cfg["text_crop_beta"] * zero_shot_logprobs(fd, part, "crop")
        if cfg.get("text_dis_beta"):
            ld = ld + cfg["text_dis_beta"] * zero_shot_logprobs(fd, part, "disease")
        s = lc[:, fd.cand_crop] + ld[:, fd.cand_dis]
        if cfg.get("text_comp_beta"):
            s = s + cfg["text_comp_beta"] * zero_shot_logprobs(fd, part, "comp")
        out.append(s)
    return out[0], out[1]


def score_zero_shot(fd: FoldData, cfg: dict):
    return zero_shot_logprobs(fd, "seen", "comp"), zero_shot_logprobs(fd, "unseen", "comp")


def score_flat(fd: FoldData, cfg: dict):
    """Seen-composition memoriser: one softmax over all candidates trained on fitting labels only."""
    x, _, _, comp_col, _ = _views(fd)
    h = fit_softmax(x, comp_col, len(fd.cand_crop), wd=cfg.get("wd", 5e-4))
    return log_probs(h, fd.x_seen), log_probs(h, fd.x_uns)


def score_synth_composition(fd: FoldData, cfg: dict):
    """Generative-style composition head: real fitting features for fitting compositions plus
    crop-shifted hallucinations for held-out candidate compositions (names only), fused with
    the factorised score."""
    x, crop, dis, comp_col, source = _views(fd)
    mus = crop_means(x, crop, comp_col, fd.n_crop)
    mode = cfg.get("synth_mode", "mean")
    if mode == "subspace":  # shift only along the crop-discriminative (LEACE) subspace
        proj = LeaceEraser(x.to(DEV), F.one_hot(crop, fd.n_crop).to(DEV)).proj.cpu().to(x.dtype)
    noise_sd = None
    if cfg.get("synth_noise"):  # diagonal within-composition spread, estimated on fitting rows
        n_cols = int(comp_col.max()) + 1
        sums = torch.zeros(n_cols, x.shape[1], dtype=torch.float64).index_add_(0, comp_col, x.double())
        cnt = torch.bincount(comp_col, minlength=n_cols).clamp_min(1).double()[:, None]
        noise_sd = (cfg["synth_noise"] * (x.double() - (sums / cnt)[comp_col]).std(0)).to(x.dtype)
    xs, ys, ws = [x], [comp_col], [None]
    g = torch.Generator().manual_seed(int(cfg.get("seed", 0)))
    per = cfg.get("per_comp", 64)

    def hallucinate(c, d, n):
        srcs = torch.nonzero((dis == d) & (crop != c)).flatten()
        if len(srcs) == 0:
            return None
        pick = srcs[torch.randint(0, len(srcs), (n,), generator=g)]
        if mode == "subspace":
            out = x[pick] + (mus[c] - mus[crop[pick]]) @ proj
        else:  # v1/v2 arithmetic order kept exactly, so historical D1/D2 scores reproduce bit-for-bit
            out = x[pick] - mus[crop[pick]] + mus[c]
        if noise_sd is not None:
            out = out + torch.randn(out.shape, generator=g).to(out.dtype) * noise_sd
        return out

    for col in np.nonzero(~fd.fit_mask)[0]:
        hx = hallucinate(int(fd.cand_crop[col]), int(fd.cand_dis[col]), per)
        if hx is not None:
            xs.append(hx); ys.append(torch.full((per,), int(col))); ws.append(None)
    seen_w = cfg.get("synth_seen", 0.0)
    if seen_w:  # same hallucination for fitting pairs, so real/synthetic is not a seen/unseen cue
        per_s = cfg.get("per_seen", 16)
        for col in np.nonzero(fd.fit_mask)[0]:
            hx = hallucinate(int(fd.cand_crop[col]), int(fd.cand_dis[col]), per_s)
            if hx is not None:
                xs.append(hx); ys.append(torch.full((per_s,), int(col))); ws.append(seen_w)
    X, Y = torch.cat(xs), torch.cat(ys)
    if seen_w:
        # each group (real/unseen-synthetic per column, seen-synthetic per column) totals 1; seen-synthetic x seen_w
        is_seen_syn = torch.cat([torch.full((len(t),), ww is not None) for t, ww in zip(xs, ws)])
        w = comp_balanced_weight(torch.where(is_seen_syn, Y + len(fd.cand_crop), Y))
        w = torch.where(is_seen_syn, w * float(seen_w), w)
    else:
        w = comp_balanced_weight(Y)
    if cfg.get("source_balance_comp"):  # within each fitting composition, every source gets an equal share
        w = w.clone()
        w[:len(x)] = w[:len(x)] * within_concept_source_weight(comp_col, source)
    h = fit_softmax(X, Y, len(fd.cand_crop), w, wd=cfg.get("wd", 5e-4))
    fs, fu = score_factorised(fd, cfg.get("factorised", {}))
    a = cfg.get("fuse", 1.0)
    z = (1.0 + a) if cfg.get("normalise", False) else 1.0  # keep fused scores on one log-prob scale
    return (fs + a * log_probs(h, fd.x_seen)) / z, (fu + a * log_probs(h, fd.x_uns)) / z


RECIPES = {"factorised": score_factorised, "zero_shot": score_zero_shot, "flat": score_flat,
           "synth_composition": score_synth_composition}
