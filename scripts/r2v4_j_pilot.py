"""Stage J, limited (v4, post-freeze exploratory): J0 pipeline reproduction and a single-fold J1-J3 pilot.

  --j0      extract all 8,000 training rows with the UNCHANGED encoder through the shared J extraction path,
            write it as a v3-schema feature file, then (separately) run the v3 D1 row on it and compare with
            the committed v3 results at 1e-5 (scripts/r2_screen.py + scripts/r2v4_repro_check.py).
  --pilot   J1 (last 2 blocks), J2 (last 4), J3 (LoRA r8 MLP): 2 epochs each on the pilot fold's pilot-train split,
            with pilot-internal validation; mechanics checks; nothing from evaluation rows is read or scored.
Pilot fold and split rules: research_r2_v4/amendment_1.json ("J_pilot").
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research_r2.data import load_training_source, sha256_file, sha256_json  # noqa: E402
from research_r2.features import WEIGHTS  # noqa: E402
from research_r2_v4.common import EXP, LABEL, env, now  # noqa: E402
from research_r2_v4.finetune import build, extract_with_model, param_digest  # noqa: E402

ROWS = {"J1_clipL_last2": "last2", "J2_clipL_last4": "last4", "J3_clipL_lora8_mlp": "lora8_mlp"}
EPOCHS_DECLARED, EPOCHS_PILOT = 5, 2


def pilot_fold():
    """Inner fold whose committed v3 D1 LOFO H_bal is closest to D1's six-fold mean (ties -> lower index)."""
    rec = next(json.loads(l) for l in (ROOT / "artifacts/r2/v3/results_stage_f_v3.jsonl").read_text().splitlines()
               if json.loads(l)["id"] == "D1_synthComp_norm")
    h = {f: rec["per_fold"][f]["lofo"]["H_bal"] for f in ("I1", "I2", "I3", "I4", "I5", "I6")}
    m = float(np.mean(list(h.values())))
    return min(h, key=lambda f: (abs(h[f] - m), f))


def pilot_split(frame, fold, seed=0, frac=0.10):
    """90/10 split of the fold's FITTING rows only: lineage-group aware, composition-stratified (compositions
    with >= 5 fitting images give about 10% of their images, in whole lineage groups); never empties a composition."""
    group = np.array(json.loads((ROOT / "artifacts/r2/lineage_groups_v1.json").read_text())["group"])
    fit = np.asarray(sorted(fold["fitting_rows"]))
    comp = frame.comp_id.to_numpy()
    rng = np.random.default_rng(seed)
    rows_of_group = {}
    for r in fit:
        rows_of_group.setdefault(int(group[r]), []).append(int(r))
    val = set()
    for c in sorted(np.unique(comp[fit])):
        rows_c = fit[comp[fit] == c]
        if len(rows_c) < 5:
            continue
        target = max(1, int(round(frac * len(rows_c))))
        got = len([r for r in rows_c if r in val])
        for gid in rng.permutation(sorted(set(int(group[r]) for r in rows_c))):
            if got >= target:
                break
            new = [r for r in rows_of_group[int(gid)] if r not in val]
            trial = val | set(new)
            if any(not (set(fit[comp[fit] == comp[r]]) - trial) for r in new):
                continue  # would empty a composition in pilot-train
            val = trial
            got = len([r for r in rows_c if r in val])
    train = sorted(set(fit.tolist()) - val)
    return train, sorted(val)


class Train(Dataset):
    def __init__(self, paths, crop, dis, tf, flip):
        self.paths, self.crop, self.dis, self.tf, self.flip = list(paths), crop, dis, tf, flip

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            x = self.tf(im.convert("RGB"))
        if self.flip and torch.rand(1).item() < 0.5:
            x = torch.flip(x, dims=[2])
        return x, self.crop[i], self.dis[i]


def evaluate_loss(model, heads, loader):
    model.eval(); heads.eval()
    tot = {"loss": 0.0, "crop_acc": 0.0, "dis_acc": 0.0, "n": 0}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for x, yc, yd in loader:
            x, yc, yd = x.cuda(), yc.cuda(), yd.cuda()
            zc, zd = heads(model.encode_image(x).float())
            tot["loss"] += float(F.cross_entropy(zc.float(), yc, reduction="sum") + F.cross_entropy(zd.float(), yd, reduction="sum"))
            tot["crop_acc"] += float((zc.argmax(1) == yc).sum()); tot["dis_acc"] += float((zd.argmax(1) == yd).sum())
            tot["n"] += len(x)
    n = tot.pop("n")
    return {k: v / n for k, v in tot.items()} | {"n": n}


class Heads(nn.Module):
    def __init__(self, d, nc, nd):
        super().__init__()
        self.c, self.d = nn.Linear(d, nc), nn.Linear(d, nd)

    def forward(self, z):
        return self.c(z), self.d(z)


def pilot_row(rid, trainable, frame, F_, fn, images, tmp, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    fold = F_[fn]
    tr, va = pilot_split(frame, fold, seed=0)
    forbidden = set(fold["seen_val_rows"]) | set(fold["unseen_val_rows"])
    assert not (set(tr) | set(va)) & forbidden and set(tr) | set(va) <= set(fold["fitting_rows"])
    crops = sorted(frame.crop_id.unique()); dis = sorted(frame.disease_id.unique())
    ci = {c: i for i, c in enumerate(crops)}; di = {d: i for i, d in enumerate(dis)}
    lab = lambda rows: (torch.tensor([ci[c] for c in frame.crop_id[rows]]), torch.tensor([di[d] for d in frame.disease_id[rows]]))  # noqa: E731
    model, tf, wfile = build(trainable)
    heads = Heads(768, len(crops), len(dis)).cuda()
    backbone_params = [p for p in model.parameters() if p.requires_grad]
    frozen = [p for p in model.parameters() if not p.requires_grad]
    frozen_before = param_digest(frozen)
    train_before = [p.detach().clone() for p in backbone_params]
    rep = {"row": rid, "trainable": trainable, "fold": fn, "label": LABEL, "n_pilot_train": len(tr), "n_pilot_val": len(va),
           "pilot_split_sha256": sha256_json({"train": tr, "val": va}),
           "trainable_backbone_params": int(sum(p.numel() for p in backbone_params)),
           "trainable_head_params": int(sum(p.numel() for p in heads.parameters())),
           "frozen_params": int(sum(p.numel() for p in frozen)), "weights_file": wfile}
    yc, yd = lab(tr)
    comp = frame.comp_id.to_numpy()[tr]
    _, inv, cnt = np.unique(comp, return_inverse=True, return_counts=True)
    g = torch.Generator().manual_seed(seed)
    sampler = WeightedRandomSampler(torch.tensor(1.0 / cnt[inv]), num_samples=len(tr), replacement=True, generator=g)
    paths = lambda rows: [Path(images) / f for f in frame.file[rows]]  # noqa: E731
    dl = DataLoader(Train(paths(tr), yc, yd, tf, flip=True), batch_size=32, sampler=sampler, num_workers=8, drop_last=True,
                    worker_init_fn=lambda w: torch.manual_seed(seed * 100 + w))
    vyc, vyd = lab(va)
    vl = DataLoader(Train(paths(va), vyc, vyd, tf, flip=False), batch_size=64, num_workers=8)
    opt = torch.optim.AdamW([{"params": backbone_params, "lr": 1e-5}, {"params": heads.parameters(), "lr": 1e-3}], weight_decay=0.05)
    steps_per_epoch = len(dl); total = EPOCHS_DECLARED * steps_per_epoch; warm = steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    scaler = torch.amp.GradScaler("cuda")
    rep["schedule"] = f"declared {EPOCHS_DECLARED}-epoch schedule (1 warm-up epoch, cosine), truncated after {EPOCHS_PILOT} pilot epochs"
    rep["epochs"] = [{"epoch": 0, "pilot_val": evaluate_loss(model, heads, vl)}]
    torch.cuda.reset_peak_memory_stats()
    first_grad, step = None, 0
    for ep in range(1, EPOCHS_PILOT + 1):
        model.train(); heads.train()
        t0, losses, n_img = time.time(), [], 0
        for x, c, d in dl:
            x, c, d = x.cuda(non_blocking=True), c.cuda(), d.cuda()
            with torch.autocast("cuda", dtype=torch.float16):
                zc, zd = heads(model.encode_image(x).float())
                loss = F.cross_entropy(zc.float(), c) + F.cross_entropy(zd.float(), d)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            step += 1
            if first_grad is None and step == 5:  # after the AMP loss scale has settled
                scaler.unscale_(opt)
                gn = [p.grad for p in backbone_params]
                first_grad = {"step": step, "scaler_scale": float(scaler.get_scale()), "trainable_with_grad": int(sum(x is not None for x in gn)), "trainable_total": len(gn),
                              "all_finite": bool(all(x is not None and torch.isfinite(x).all() for x in gn)),
                              "grad_norm_total": float(torch.norm(torch.stack([x.float().norm() for x in gn if x is not None]))),
                              "zero_norm_tensors": int(sum(x is not None and float(x.norm()) == 0 for x in gn)),
                              "frozen_with_grad": int(sum(p.grad is not None for p in frozen))}
            scaler.step(opt); scaler.update(); sched.step()
            losses.append(float(loss)); n_img += len(x)
        torch.cuda.synchronize()
        sec = time.time() - t0
        rep["epochs"].append({"epoch": ep, "train_loss_mean": float(np.mean(losses)), "train_loss_last50": float(np.mean(losses[-50:])),
                              "images": n_img, "seconds": sec, "images_per_s": n_img / sec, "lr_end": sched.get_last_lr(),
                              "pilot_val": evaluate_loss(model, heads, vl)})
        print(rid, json.dumps(rep["epochs"][-1]), flush=True)
    rep["first_backward"] = first_grad
    rep["peak_gpu_gib"] = torch.cuda.max_memory_allocated() / 2**30
    rep["frozen_unchanged"] = param_digest(frozen) == frozen_before
    deltas = [float((p.detach() - b).norm()) for p, b in zip(backbone_params, train_before)]
    rep["trainable_update"] = {"tensors": len(deltas), "changed": int(sum(x > 0 for x in deltas)), "delta_norm_total": float(np.linalg.norm(deltas))}
    # checkpoint save -> reload into a freshly built model -> identical outputs on a fixed pilot-val batch
    ck = Path(tmp) / f"{rid}_pilot.pt"
    ck.parent.mkdir(parents=True, exist_ok=True)
    names = {id(p): n for n, p in model.named_parameters()}
    torch.save({"backbone": {names[id(p)]: p.detach().cpu() for p in backbone_params}, "heads": heads.state_dict()}, ck)
    xb = next(iter(vl))[0][:16].cuda()
    model.eval(); heads.eval()
    with torch.no_grad():
        ref = torch.cat(heads(model.encode_image(xb).float()), 1)
    m2, _, _ = build(trainable)
    h2 = Heads(768, len(crops), len(dis)).cuda()
    st = torch.load(ck, map_location="cuda")
    missing = m2.load_state_dict(st["backbone"], strict=False)
    h2.load_state_dict(st["heads"])
    m2.eval(); h2.eval()
    with torch.no_grad():
        out2 = torch.cat(h2(m2.encode_image(xb).float()), 1)
    rep["checkpoint_reload"] = {"max_abs_diff": float((ref - out2).abs().max()), "unexpected_keys": list(missing.unexpected_keys),
                                "checkpoint_bytes": ck.stat().st_size, "trainable_keys_saved": len(st["backbone"])}
    ck.unlink()
    rep["checkpoint_deleted"] = not ck.exists()
    del model, m2
    torch.cuda.empty_cache()
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--j0", action="store_true")
    ap.add_argument("--pilot", nargs="*")
    a = ap.parse_args()
    frame, bench, F_ = env()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if a.j0:
        t0 = time.time()
        model, tf, wfile = build("none")
        feats = extract_with_model(model, tf, [Path(a.images) / f for f in frame.file])
        v3 = torch.load(EXP / "features" / "clip_l14.pt", map_location="cpu", weights_only=False)
        digests = json.loads((EXP / "benchmark" / "image_digests.json").read_text())["images"]
        rec = {"schema": "r2-frozen-features-v1", "backbone": "clip_l14", "views": ["centre", "hflip"], "features": feats,
               "files": frame.file.tolist(), "image_sha256_list_sha256": sha256_json([d["sha256"] for d in digests]),
               "weights_file": wfile, "weights_sha256": sha256_file(WEIGHTS / wfile), "pipeline": "research_r2_v4.finetune.extract_with_model, trainable=none, 0 epochs",
               "seconds": time.time() - t0}
        (out / "features").mkdir(exist_ok=True)
        torch.save(rec, out / "features" / "clip_l14.pt")
        cmp = {"bit_identical_to_v3": bool(torch.equal(feats, v3["features"])),
               "max_abs_diff_vs_v3": float((feats.float() - v3["features"].float()).abs().max()), "seconds": rec["seconds"]}
        (out / "j0_feature_check.json").write_text(json.dumps(cmp, indent=1))
        print(json.dumps(cmp))
    if a.pilot is not None:
        fn = pilot_fold()
        rows = a.pilot or list(ROWS)
        res = {"label": LABEL, "pilot_fold": fn, "rule": "closest committed v3 D1 LOFO H_bal to the six-fold mean", "started": now(), "rows": {}}
        for rid in rows:
            try:
                res["rows"][rid] = pilot_row(rid, ROWS[rid], frame, F_, fn, a.images, out / "tmp_checkpoints")
                res["rows"][rid]["status"] = "ok"
            except Exception as e:
                import traceback
                res["rows"][rid] = {"status": "failed", "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-2000:]}
            (out / "pilot.json").write_text(json.dumps(res, indent=1, default=float))
        res["finished"] = now()
        (out / "pilot.json").write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
