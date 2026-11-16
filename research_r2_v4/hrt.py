"""H_RT: retrained source exclusion (v4, post-freeze exploratory; amendment_2 source_co_endpoint.H_RT), fail closed.

For each three-seed-passing J row and each registered (fold, source) pair, the row is RETRAINED from the original
pretrained weights at seed 0 on (a) the fold's fitting rows without any image of the source ("ablated") and (b) the
fitting rows minus the saved Stage H matched-random removal of seed 0 ("random"). The base is the row's Phase 1
seed-0 fold model. All three are scored with the unchanged D1 downstream and compared on the ablated fold's
candidate universe and evaluation rows. Diagnostic only: never a gate, never used to select.

Training is jrun.run_fold, unchanged. See docs/R2_v4_HRT_Plan.md.
"""
from __future__ import annotations

import copy
import importlib.util
import json
import time
import traceback
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from research_r2.candidates import derive_fold
from research_r2.metrics import GAMMA_GRID, harmonic
from research_r2.packets import fold_contract, validate_packet
from research_r2_v4 import confirm, jrun
from research_r2_v4.common import INNER, V4, restrict

ROOT = Path(__file__).resolve().parents[1]
ROWS = ["J1_clipL_last2", "J2_clipL_last4", "J3_clipL_lora8_mlp"]
ARMS = ("ablated", "random")
SEED = 0
CONTROL_SEED = 0
SCHEMA = "r2v4-hrt-v1"
INDEXSETS = V4 / "stage_h_20261114" / "D1_synthComp_norm" / "indexsets"
INDEX_SUMS = ROOT / "artifacts/r2/v4/stage_h_20261114/D1_synthComp_norm/indexsets_SHA256SUMS"
FEASIBILITY = ROOT / "artifacts/r2/v4/feasibility_20261114/source_feasibility.json"
AMENDMENT_2 = ROOT / "research_r2_v4/amendment_2.json"
PHASE1 = V4 / "stage_j_phase1_20261114"


class HRTError(jrun.JCheckError):
    """An H_RT arm, its data or its provenance fails a check."""


def _stage_h():
    spec = importlib.util.spec_from_file_location("r2v4_stage_h", ROOT / "scripts" / "r2v4_stage_h.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


# ----------------------------------------------------------------------------- registered scope
def declared_pairs():
    """The registered pairs; the pair rule is re-derived from source_feasibility.json and must agree."""
    h = json.loads(AMENDMENT_2.read_text())["source_co_endpoint"]["H_RT"]
    pairs = [tuple(p) for p in h["pairs"]]
    feas = json.loads(FEASIBILITY.read_text())["useful_pairs"]
    order = sorted(feas, key=lambda p: (-p["novel_seen_images"], INNER.index(p["fold"])))
    picked, seen = [], set()
    for p in order:
        if p["source"] not in seen:
            picked.append((p["fold"], p["source"])); seen.add(p["source"])
        if len(picked) == len(pairs):
            break
    if picked != pairs:
        raise HRTError(f"registered pairs {pairs} differ from the re-derived pair rule {picked}")
    return pairs


def arm_id(row, fold, source, arm):
    if arm not in ARMS:
        raise HRTError(f"unknown arm {arm!r}")
    return f"{row}@HRT_{fold}_{source}_{arm}"


def arm_config(row, fold, source, arm, declared: jrun.JConfig | None = None) -> jrun.JConfig:
    """The declared row configuration; only the row id changes (it names the pair and arm)."""
    base = declared if declared is not None else jrun.j_rows()[row]
    cfg = copy.deepcopy(base)
    cfg.row = arm_id(row, fold, source, arm)
    return cfg


def _index_set(fold, source, indexsets=INDEXSETS, sums=INDEX_SUMS):
    p = Path(indexsets) / f"{fold}_{source}_seed{CONTROL_SEED}.json"
    if not p.exists():
        raise HRTError(f"saved matched-removal index set missing: {p.name}")
    table = {}
    for line in Path(sums).read_text().splitlines():
        if line.strip():
            h, n = line.split()
            table[n.lstrip("./")] = h
    sha = jrun.sha256_file(p)
    if table.get(p.name) != sha:
        raise HRTError(f"index set {p.name}: sha256 {sha[:12]} does not match the Stage H manifest")
    saved = json.loads(p.read_text())
    if (saved.get("fold"), saved.get("source"), saved.get("seed")) != (fold, source, CONTROL_SEED):
        raise HRTError(f"index set {p.name} names {saved.get('fold')}/{saved.get('source')}/seed{saved.get('seed')}")
    return saved, sha


def verify_arm_fold(frame, fold, f_arm, source, arm, removed):
    """Independent checks of an arm's fold; returns the composition-support record."""
    src, comp = frame.source.to_numpy(), frame.comp_id.to_numpy()
    fit = [int(r) for r in fold["fitting_rows"]]
    rem = sorted(int(r) for r in removed)
    kept = sorted(int(r) for r in f_arm["fitting_rows"])
    p = []
    if len(set(rem)) != len(rem) or not set(rem) <= set(fit):
        p.append("removed rows are not distinct fitting rows")
    if kept != sorted(set(fit) - set(rem)):
        p.append("arm fitting rows != fold fitting rows minus removed rows")
    n_s = int((src[fit] == source).sum())
    if arm == "ablated":
        if int((src[kept] == source).sum()) != 0 or len(rem) != n_s or any(src[r] != source for r in rem):
            p.append(f"incomplete source removal: {int((src[kept] == source).sum())} {source} fitting rows remain")
    elif arm == "random":
        if any(src[r] == source for r in rem):
            p.append(f"matched-random removal contains {source} rows")
        if len(rem) != n_s:
            p.append(f"matched-random removal size {len(rem)} != {n_s} {source} fitting rows")
        if set(comp[kept].tolist()) != set(comp[fit].tolist()):
            p.append("matched-random removal emptied a composition")
        if list(f_arm["seen_val_rows"]) != list(fold["seen_val_rows"]):
            p.append("matched-random arm changed the seen-val rows")
    else:
        p.append(f"unknown arm {arm!r}")
    if list(f_arm["unseen_val_rows"]) != list(fold["unseen_val_rows"]):
        p.append("unseen-val rows changed")
    if not set(f_arm["seen_val_rows"]) <= set(fold["seen_val_rows"]):
        p.append("seen-val rows outside the base fold")
    if not set(f_arm.get("candidate_composition_ids", [])) <= set(fold.get("candidate_composition_ids", [])):
        p.append("arm candidate universe is not inside the base universe")
    if f_arm.get("name") != fold.get("name"):
        p.append("arm fold name differs")
    if p:
        raise HRTError(f"{fold['name']}/{source}/{arm}: " + "; ".join(p))
    emptied = sorted(set(comp[fit].tolist()) - set(comp[kept].tolist()))
    return {"n_fitting_base": len(fit), "n_fitting": len(kept), "n_removed": len(rem),
            "fitting_compositions_base": len(set(comp[fit].tolist())), "fitting_compositions": len(set(comp[kept].tolist())),
            "emptied_compositions": [int(c) for c in emptied],
            "seen_val_rows": len(f_arm["seen_val_rows"]), "dropped_seen_val_rows": len(f_arm.get("dropped_seen_val_rows", [])),
            "unseen_val_rows": len(f_arm["unseen_val_rows"]), "universe": len(f_arm.get("candidate_composition_ids", [])),
            "removed_rows_sha256": jrun.sha256_json(rem), "fitting_rows_sha256": jrun.sha256_json(kept)}


def arm_fold(frame, fold, source, arm, indexsets=INDEXSETS, sums=INDEX_SUMS):
    """(arm fold, removed rows, provenance). The random arm uses the saved Stage H index set, which must also be
    reproduced by matched_removal(seed=0)."""
    src = frame.source.to_numpy()
    fit = [int(r) for r in fold["fitting_rows"]]
    prov = {"fold": fold["name"], "source": source, "arm": arm}
    if arm == "ablated":
        removed = sorted(r for r in fit if src[r] == source)
    elif arm == "random":
        saved, sha = _index_set(fold["name"], source, indexsets, sums)
        _, picks, lv = _stage_h().matched_removal(frame, fit, source, seed=CONTROL_SEED)
        if picks != saved["removed_rows"] or lv != saved.get("match_levels"):
            raise HRTError(f"{fold['name']}/{source}: matched_removal(seed=0) does not reproduce the saved index set")
        removed = sorted(saved["removed_rows"])
        prov.update({"index_set": f"{fold['name']}_{source}_seed{CONTROL_SEED}.json", "index_set_sha256": sha, "match_levels": lv})
    else:
        raise HRTError(f"unknown arm {arm!r}")
    f_arm = derive_fold(frame, fold, sorted(set(fit) - set(removed)))
    prov["support"] = verify_arm_fold(frame, fold, f_arm, source, arm, removed)
    return f_arm, removed, prov


# ----------------------------------------------------------------------------- guarded access
class ExclusionAccess(jrun.ImageAccess):
    """ImageAccess that additionally refuses every removed fitting row in every phase, before opening it."""

    def __init__(self, frame, image_root, excluded, loader=None):
        super().__init__(frame, image_root, loader=loader)
        self.excluded = frozenset(int(r) for r in excluded)
        self.opened_rows: set[int] = set()  # main process only (DataLoader workers enforce the check but do not report)

    def allow(self, phase, rows):
        rows = [int(r) for r in rows]
        bad = self.excluded & set(rows)
        if bad:
            raise HRTError(f"phase {phase!r} would allow {len(bad)} excluded rows")
        super().allow(phase, rows)

    def check(self, row):
        if int(row) in self.excluded:
            raise HRTError(f"image access refused: row {int(row)} is excluded in this H_RT arm")
        return super().check(row)

    def get(self, row):
        x = super().get(row)
        self.opened_rows.add(int(row))
        return x


# ----------------------------------------------------------------------------- base model
def base_fold(row, fold, phase1=PHASE1, declared=None):
    """The row's Phase 1 seed-0 model of this fold (six-fold verified as in the confirmation)."""
    v = confirm.verified_row_seed(Path(phase1), row, SEED, INNER, declared if declared is not None else jrun.j_rows().get(row))
    return v[fold]


def arm_dir(out, row, fold, source, arm):
    return Path(out) / row / f"{fold}_{source}" / arm


def _ok_attempts(d):
    out = []
    if not Path(d).exists():
        return out
    for a in sorted(Path(d).glob("attempt*"), key=lambda q: int(q.name[7:])):
        h = a / "hrt_receipt.json"
        if h.exists() and json.loads(h.read_text()).get("status") == "ok":
            out.append(a)
    return out


# ----------------------------------------------------------------------------- verification
def verify_arm(out, row, fold_meta, source, arm, frame, phase1=PHASE1, declared=None):
    """(attempt dir, receipt, hrt receipt, arm fold) for a completed arm, or HRTError."""
    fn = fold_meta["name"]
    d = arm_dir(out, row, fn, source, arm)
    oks = _ok_attempts(d)
    if not oks:
        raise HRTError(f"{row} {fn}/{source}/{arm}: no successful attempt")
    if len(oks) > 1:
        raise HRTError(f"{row} {fn}/{source}/{arm}: duplicate run, {len(oks)} successful attempts")
    adir = oks[0]
    rec = jrun.verify_receipt(adir)  # receipt status ok and files unchanged since their receipt
    if rec is None:
        raise HRTError(f"{row} {fn}/{source}/{arm}: hrt receipt present but run receipt is not ok")
    h = json.loads((adir / "hrt_receipt.json").read_text())
    cfg = arm_config(row, fn, source, arm, declared)
    p = confirm.receipt_problems(rec, cfg.row, SEED, fn, INNER.index(fn), cfg)
    if jrun.sha256_file(adir / "receipt.json") != h.get("receipt_sha256"):
        p.append("receipt.json changed since the H_RT success marker")
    if (h.get("schema"), h.get("row"), h.get("arm_row"), h.get("fold"), h.get("source"), h.get("arm")) != (SCHEMA, row, cfg.row, fn, source, arm):
        p.append("H_RT success marker names another row, pair or arm")
    if not (rec.get("grad_checks") or {}).get("steps"):
        p.append("no optimizer steps recorded")
    f_arm, removed, prov = arm_fold(frame, fold_meta, source, arm)
    if h.get("support", {}).get("fitting_rows_sha256") != prov["support"]["fitting_rows_sha256"] or \
            h.get("support", {}).get("removed_rows_sha256") != prov["support"]["removed_rows_sha256"]:
        p.append("fitting / removed rows differ from the registered derivation")
    if rec.get("n_fitting") != prov["support"]["n_fitting"]:
        p.append("trained on a different number of fitting rows")
    if set(h.get("opened_rows_excluded", [None])) != set():
        p.append("excluded rows were opened")
    _, brec = base_fold(row, fn, phase1, declared)
    for k in ("frozen_digest_before", "weights_file", "torch_seed", "frozen_params", "trainable_backbone_params"):
        if rec.get(k) != brec.get(k):
            p.append(f"{k} differs from the base model (initialisation / architecture not identical)")
    bc = {k: v for k, v in brec["config"].items() if k != "row"}
    if {k: v for k, v in rec["config"].items() if k != "row"} != bc:
        p.append("configuration differs from the base model beyond the row id")
    if rec.get("checkpoint_sha256") == brec.get("checkpoint_sha256"):
        p.append("checkpoint identical to the base model")
    if h.get("base_checkpoint_sha256") != brec.get("checkpoint_sha256"):
        p.append("success marker names another base model")
    try:
        meta = torch.load(adir / "checkpoint.pt", map_location="cpu", weights_only=False)["meta"]
    except Exception as e:  # noqa: BLE001
        meta = {}; p.append(f"checkpoint unreadable: {type(e).__name__}")
    for k, v in (("row", cfg.row), ("seed", SEED), ("fold", fn), ("config_sha256", cfg.sha())):
        if meta.get(k) != v:
            p.append(f"checkpoint metadata {k} does not name this arm")
    rows = sorted(set(f_arm["fitting_rows"]) | set(f_arm["seen_val_rows"]) | set(f_arm["unseen_val_rows"]))
    fm = rec.get("feature_cache_meta") or {}
    if fm.get("rows_sha256") != jrun.sha256_json([int(r) for r in rows]):
        p.append("feature cache rows are not this arm's rows")
    try:
        jrun.load_feature_cache(adir / "features.pt", fm)
    except jrun.JCheckError as e:
        p.append(f"feature cache: {e}")
    try:
        validate_packet(np.load(adir / f"{fn}_scores.npz"), fold_contract(frame, f_arm))
    except Exception as e:  # noqa: BLE001
        p.append(f"score packet: {type(e).__name__}: {e}")
    if p:
        raise HRTError(f"{row} {fn}/{source}/{arm}: " + "; ".join(p))
    return adir, rec, h, f_arm


# ----------------------------------------------------------------------------- execution
def run_arm(row, fold_meta, source, arm, frame, bench, backend, out, image_root=None, loader=None, device="cuda",
            phase1=PHASE1, declared=None, log=print):
    fn = fold_meta["name"]
    d = arm_dir(out, row, fn, source, arm)
    if _ok_attempts(d):
        return "ok (existing)"
    d.mkdir(parents=True, exist_ok=True)
    adir = d / f"attempt{1 + len([q for q in d.iterdir() if q.name.startswith('attempt')])}"
    t0 = time.time()
    try:
        f_arm, removed, prov = arm_fold(frame, fold_meta, source, arm)
        bpk, brec = base_fold(row, fn, phase1, declared)
        cfg = arm_config(row, fn, source, arm, declared)
        access = ExclusionAccess(frame, image_root, removed, loader=loader)
        rec = jrun.run_fold(cfg, SEED, f_arm, INNER.index(fn), frame, bench, backend, access, adir, device)
        opened_bad = sorted(access.opened_rows & set(removed))
        h = {"schema": SCHEMA, "status": "ok", "row": row, "arm_row": cfg.row, "fold": fn, "source": source, "arm": arm,
             "seed": SEED, "label": "post-freeze exploratory (v4); H_RT diagnostic only", **prov,
             "opened_rows_main_process": len(access.opened_rows), "opened_rows_excluded": opened_bad,
             "base_checkpoint_sha256": brec["checkpoint_sha256"], "base_packet_sha256": brec["packet_sha256"],
             "receipt_sha256": jrun.sha256_file(adir / "receipt.json"), "seconds_total": time.time() - t0,
             "finished": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        if opened_bad:
            raise HRTError(f"excluded rows were opened: {opened_bad[:5]}")
        jrun.save_atomic_json(adir / "hrt_receipt.json", h)
        verify_arm(out, row, fold_meta, source, arm, frame, phase1, declared)  # the marker must also pass evaluation
        log(json.dumps({"arm_row": cfg.row, "seconds": rec["seconds"], "n_fitting": rec["n_fitting"], "reload": rec["reload_max_abs_diff"]}))
        return "ok"
    except Exception as e:  # noqa: BLE001
        adir.mkdir(parents=True, exist_ok=True)
        if (adir / "hrt_receipt.json").exists():
            (adir / "hrt_receipt.json").rename(adir / "hrt_receipt.REJECTED.json")
        jrun.save_atomic_json(adir / "failure.json", {"row": row, "fold": fn, "source": source, "arm": arm, "status": "failed",
                                                     "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-3000:]})
        log(f"FAILED {row} {fn}/{source}/{arm}: {type(e).__name__}: {e}")
        return f"failed: {type(e).__name__}: {e}"


# ----------------------------------------------------------------------------- evaluation
def _arm_scores(frame, packet, fold):
    z = np.load(packet)
    return {"seen": z["seen"], "unseen": z["unseen"], "seen_truth": z["seen_truth"], "unseen_truth": z["unseen_truth"],
            "cand_ids": z["cand_ids"], "fold": fold}


def arm_outcomes(sc, gammas, ns, us):
    m = _stage_h().arm_metrics(sc, gammas, ns, us)
    for g in gammas:
        r = sc.at(g, top5=True)
        ps, pu = sc.top1_pred("seen", g), sc.top1_pred("unseen", g)
        ts = sc.parts["seen"]["truth"]
        m[f"g{g:g}"].update({k: r[k] for k in ("seen_top5", "unseen_top5_micro", "H5_micro", "unseen_disease_top1", "unseen_crop_top1",
                                                 "unseen_withincrop_macro")})
        m[f"g{g:g}"]["false_healthy_seen_diseased"] = float(np.mean(sc.cand_healthy[ps][~sc.cand_healthy[ts]]))
        m[f"g{g:g}"]["false_healthy_unseen"] = float(np.mean(sc.cand_healthy[pu]))
    return m


def _hits(sc, part, g):
    return (sc.top1_pred(part, g) == sc.parts[part]["truth"]).astype(float), sc.parts[part]["truth"]


def paired_bootstrap_pair(sa, sb, g, ns, reps=2000, seed=0):
    """a - b on the same images: novel-source seen Top-1 (clusters = compositions of the novel-source images) and
    H_bal (seen compositions and unseen compositions resampled, as research_r2_v4.evaluate.paired_bootstrap)."""
    rng = np.random.default_rng(seed)
    ha, ts = _hits(sa, "seen", g); hb, tsb = _hits(sb, "seen", g)
    ua, tu = _hits(sa, "unseen", g); ub, tub = _hits(sb, "unseen", g)
    if not (np.array_equal(ts, tsb) and np.array_equal(tu, tub)):
        raise HRTError("arms differ in evaluation truth")
    sc_, uc_ = np.unique(ts), np.unique(tu)
    S = lambda h: (np.array([h[ts == c].sum() for c in sc_]), np.array([(ts == c).sum() for c in sc_]))  # noqa: E731
    U = lambda h: np.array([h[tu == c].mean() for c in uc_])  # noqa: E731
    Sa, Sb, Ua, Ub = S(ha), S(hb), U(ua), U(ub)
    nc = np.unique(ts[ns])
    Na = np.array([ha[ns][ts[ns] == c].sum() for c in nc]); Nb = np.array([hb[ns][ts[ns] == c].sum() for c in nc])
    Nn = np.array([(ts[ns] == c).sum() for c in nc])
    hd, nd = np.empty(reps), np.empty(reps)
    for r in range(reps):
        i = rng.integers(0, len(sc_), len(sc_)); j = rng.integers(0, len(uc_), len(uc_)); k = rng.integers(0, len(nc), len(nc))
        hd[r] = harmonic(Sa[0][i].sum() / Sa[1][i].sum(), Ua[j].mean()) - harmonic(Sb[0][i].sum() / Sb[1][i].sum(), Ub[j].mean())
        nd[r] = Na[k].sum() / Nn[k].sum() - Nb[k].sum() / Nn[k].sum()
    point_h = harmonic(Sa[0].sum() / Sa[1].sum(), Ua.mean()) - harmonic(Sb[0].sum() / Sb[1].sum(), Ub.mean())
    point_n = Na.sum() / Nn.sum() - Nb.sum() / Nn.sum()
    ci = lambda x: [100 * float(np.percentile(x, 2.5)), 100 * float(np.percentile(x, 97.5))]  # noqa: E731
    return {"H_bal": {"point_pp": 100 * float(point_h), "ci95_pp": ci(hd), "draws": hd},
            "novel_seen_top1": {"point_pp": 100 * float(point_n), "ci95_pp": ci(nd), "draws": nd, "n_images": int(ns.sum()),
                                "n_compositions": int(len(nc))},
            "reps": reps, "seed": seed}


def evaluate_row(row, pairs, out, frame, F, g_sel, phase1=PHASE1, declared=None, reps=2000):
    """Per pair: base / ablated / random outcomes on the ablated fold's universe and rows, effects and bootstrap."""
    res = {}
    for fn, s in pairs:
        fold = F[fn]
        bpk, brec = base_fold(row, fn, phase1, declared)
        arms = {a: verify_arm(out, row, fold, s, a, frame, phase1, declared) for a in ARMS}
        f_abl, f_rnd = arms["ablated"][3], arms["random"][3]
        u = lambda f: set(f["candidate_composition_ids"])  # noqa: E731
        if not (u(f_abl) <= u(f_rnd) <= u(fold)):
            raise HRTError(f"{row} {fn}/{s}: candidate universes do not nest (ablated <= random <= base)")
        if not set(f_abl["seen_val_rows"]) <= set(f_rnd["seen_val_rows"]):
            raise HRTError(f"{row} {fn}/{s}: ablated evaluation rows are not inside the random arm's")
        if [x[1]["checkpoint_sha256"] for x in arms.values()][0] == [x[1]["checkpoint_sha256"] for x in arms.values()][1]:
            raise HRTError(f"{row} {fn}/{s}: the two arms share a checkpoint")
        sc = {"base": restrict(frame, _arm_scores(frame, bpk, fold), f_abl),
              "ablated": restrict(frame, _arm_scores(frame, arms["ablated"][0] / f"{fn}_scores.npz", f_abl), f_abl),
              "random": restrict(frame, _arm_scores(frame, arms["random"][0] / f"{fn}_scores.npz", f_rnd), f_abl)}
        src = frame.source.to_numpy()
        ns = src[np.asarray(f_abl["seen_val_rows"])] == s
        us = src[np.asarray(f_abl["unseen_val_rows"])] == s
        gammas = [0.0, float(g_sel)]
        outc = {a: arm_outcomes(x, gammas, ns, us) for a, x in sc.items()}
        eff = {}
        for g in [f"g{x:g}" for x in gammas]:
            eff[g] = {}
            for k, v in outc["ablated"][g].items():
                if not isinstance(v, (int, float)) or outc["random"][g].get(k) is None or outc["base"][g].get(k) is None:
                    continue
                eff[g][k] = {"source_effect": v - outc["random"][g][k], "size_effect": outc["random"][g][k] - outc["base"][g][k],
                             "total_effect": v - outc["base"][g][k]}
        eff["ausuc"] = {"source_effect": outc["ablated"]["ausuc"] - outc["random"]["ausuc"],
                        "size_effect": outc["random"]["ausuc"] - outc["base"]["ausuc"],
                        "total_effect": outc["ablated"]["ausuc"] - outc["base"]["ausuc"]}
        bs = {k: paired_bootstrap_pair(sc["ablated"], sc[k], float(g_sel), ns, reps) for k in ("random", "base")}
        res[f"{fn}_{s}"] = {
            "fold": fn, "source": s, "gammas": gammas, "n_novel_source_seen": int(ns.sum()), "n_source_unseen": int(us.sum()),
            "evaluation": {"universe": len(f_abl["candidate_composition_ids"]), "seen_rows": len(f_abl["seen_val_rows"]),
                           "unseen_rows": len(f_abl["unseen_val_rows"])},
            "support": {"base": {"n_fitting": len(fold["fitting_rows"]), "universe": len(fold["candidate_composition_ids"])},
                        "ablated": arms["ablated"][2]["support"], "random": arms["random"][2]["support"]},
            "match_levels": arms["random"][2].get("match_levels"),
            "outcomes": outc, "effects": eff, "bootstrap_ablated_minus": bs,
            "receipts": {a: {"attempt": arms[a][0].name, "checkpoint_sha256": arms[a][1]["checkpoint_sha256"],
                             "features_sha256": arms[a][1]["features_sha256"], "packet_sha256": arms[a][1]["packet_sha256"],
                             "seconds": arms[a][1]["seconds"], "peak_gpu_gib_training": arms[a][1]["peak_gpu_gib_training"],
                             "peak_gpu_gib_total": arms[a][1]["peak_gpu_gib_total"], "n_fitting": arms[a][1]["n_fitting"],
                             "steps": arms[a][1]["grad_checks"]["steps"], "amp_skipped": arms[a][1]["grad_checks"]["skipped_steps"],
                             "grad_checks": arms[a][1]["grad_checks"], "trainable_tensors_changed": arms[a][1]["trainable_tensors_changed"],
                             "reload_max_abs_diff": arms[a][1]["reload_max_abs_diff"], "torch_seed": arms[a][1]["torch_seed"],
                             "epoch_loss": [e["train_loss_mean"] for e in arms[a][1]["epochs"]], "git_head": arms[a][1]["code"]["git_head"]}
                         for a in ARMS} | {"base": {"checkpoint_sha256": brec["checkpoint_sha256"], "packet_sha256": brec["packet_sha256"],
                                                    "n_fitting": brec["n_fitting"], "epoch_loss": [e["train_loss_mean"] for e in brec["epochs"]]}}}
    return res


def aggregate(per_pair: dict, g_key: str):
    """Equal-weight mean over pairs of the source effect, stratified bootstrap (draws are paired within pairs)."""
    out = {}
    for k in ("H_bal", "novel_seen_top1"):
        pts = [v["bootstrap_ablated_minus"]["random"][k]["point_pp"] for v in per_pair.values()]
        draws = np.mean([v["bootstrap_ablated_minus"]["random"][k]["draws"] for v in per_pair.values()], 0)
        out[k] = {"mean_source_effect_pp": float(np.mean(pts)), "ci95_pp": [float(np.percentile(draws, 2.5)) * 100, float(np.percentile(draws, 97.5)) * 100],
                  "per_pair_pp": dict(zip(per_pair, pts)), "n_pairs": len(pts), "gamma": g_key}
    return out


def strip_draws(obj):
    if isinstance(obj, dict):
        return {k: strip_draws(v) for k, v in obj.items() if k != "draws"}
    if isinstance(obj, list):
        return [strip_draws(v) for v in obj]
    return obj


__all__ = ["declared_pairs", "arm_fold", "verify_arm_fold", "ExclusionAccess", "run_arm", "verify_arm", "evaluate_row",
           "aggregate", "arm_config", "arm_id", "HRTError", "ROWS", "ARMS", "GAMMA_GRID", "asdict"]
