"""Stage I (v4, post-freeze exploratory): screen the declared frozen-feature rows against D1 on I1-I6.

Rows: research_r2_v4/matrices/stage_i_v4.json + stage_i_v4_amendment1.json; definitions as in amendment_1.json.
Controls B1, D1, F10 come from the REGISTERED v3 packets (read-only, sha256-checked against the v3 registry).
New rows write packets to <out>/scores/<row>/[<config>/]<fold>_scores.npz (strictly validated) plus a manifest.

  python scripts/r2v4_stage_i.py --out ../../experiments/r2-compositional/v4/stage_i_20261114 [--only ROW ...]
  python scripts/r2v4_stage_i.py --out ... --evaluate      # gate / nested / bootstrap / diagnostics over all rows
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research_r2.data import sha256_json  # noqa: E402
from research_r2.methods import RECIPES  # noqa: E402
from research_r2.packets import file_sha256, fold_contract, resolve_packets, validate_packet  # noqa: E402
from research_r2.runner import make_fold, scorer_from_scores  # noqa: E402
from research_r2_v4 import evaluate as ev  # noqa: E402
from research_r2_v4.common import EXP, INNER, LABEL, V4, env, now, poisoned_features  # noqa: E402
from research_r2_v4.methods_v4 import raw_fit_stats, score_projected_d1, score_synth_v4  # noqa: E402

D1 = "D1_synthComp_norm"
V3_DIRS = [EXP / "v3" / d / "scores" for d in ("inner_stage_bc_v1", "inner_stage_d_v1", "inner_stage_f_v3", "repro_hardening_20261113")]
SYNTH = {"fuse": 1.0, "normalise": True}
ROWS = {
    "B1_clipL_factorised": {"kind": "v3"}, D1: {"kind": "v3"}, "F10_concat_convnext": {"kind": "v3"},
    "I1_count_aware_calibration": {"kind": "posthoc", "rule": "count", "g1": [0.0, 0.25, 0.5, 1.0, 2.0]},
    "I2_crop_conditional_calibration": {"kind": "posthoc", "rule": "crop", "g1": [0.0, 1.0, 2.0, 4.0]},
    "I3_cosine_prototype_head": {"kind": "fresh", "feat": "clip_l14", "fn": "synth_v4", "cfg": {"head": "cosine", "tau": 0.05}},
    "I3b_cosine_prototype_tau010": {"kind": "fresh", "feat": "clip_l14", "fn": "synth_v4", "cfg": {"head": "cosine", "tau": 0.10}},
    "I4_orthogonal_factor_projection": {"kind": "fresh", "feat": "clip_l14", "fn": "proj", "cfg": {"projection": "factor", "hsic_lambda": 1.0, "seed": 0}, "stochastic": True},
    "I4b_projection_no_hsic": {"kind": "fresh", "feat": "clip_l14", "fn": "proj", "cfg": {"projection": "factor", "hsic_lambda": 0.0, "seed": 0}, "stochastic": True},
    "I5_crosscrop_supcon_projection": {"kind": "fresh", "feat": "clip_l14", "fn": "proj", "cfg": {"projection": "supcon", "cross_crop": True, "seed": 0}, "stochastic": True},
    "I5b_supcon_any_crop": {"kind": "fresh", "feat": "clip_l14", "fn": "proj", "cfg": {"projection": "supcon", "cross_crop": False, "seed": 0}, "stochastic": True},
    "I6_text_direction_synthesis": {"kind": "fresh", "feat": "clip_l14", "fn": "synth_v4", "cfg": {"text_frac": 0.5, "text_lambda": 1.0}},
    "I6b_text_direction_only": {"kind": "fresh", "feat": "clip_l14", "fn": "synth_v4", "cfg": {"text_frac": 1.0, "text_lambda": 1.0}},
    "I7_multilayer_clipL": {"kind": "fresh", "feat": "ml:16,20,24", "fn": "d1", "cfg": SYNTH},
    "I7b_multilayer_20_24": {"kind": "fresh", "feat": "ml:20,24", "fn": "d1", "cfg": SYNTH},
    "I8_multicrop_views": {"kind": "fresh", "feat": "mc", "fn": "d1", "cfg": SYNTH},
}
I_ROWS = [r for r in ROWS if r.startswith("I")]


def load_feat(spec, bench):
    if spec == "clip_l14":
        feats, text, _, ident = poisoned_features(bench, ["clip_l14"])
        return feats, text, ident
    rec = torch.load(V4 / "features" / (("clip_l14_ml" if spec.startswith("ml") else "clip_l14_mc") + ".pt"),
                     map_location="cpu", weights_only=False)
    x = rec["features"].float()
    ident = {"file": rec["schema"], "features_sha256": rec["features_sha256"], "weights_sha256": rec["weights_sha256"],
             "image_sha256_list_sha256": rec["image_sha256_list_sha256"]}
    if spec.startswith("ml"):
        layers = [int(v) for v in spec.split(":")[1].split(",")]
        blocks = [x[..., rec["layers"].index(L) * 768:(rec["layers"].index(L) + 1) * 768] for L in layers]
        x = torch.cat([b / b.norm(dim=-1, keepdim=True) for b in blocks], -1)  # per-layer L2, as v3 multi-backbone concat
        ident["layers"] = layers
    else:
        v3 = torch.load(EXP / "features" / "clip_l14.pt", map_location="cpu", weights_only=False)["features"]
        assert torch.equal(rec["features"][:, :2], v3), "I8 views 0-1 must equal the v3 clip_l14 features"
    x[bench["outer_exclusion_rows"]] = float("nan")
    return x, None, ident


def save_packet(path, fold, frame, arm):
    pk = {"cand_ids": arm["cand_ids"].astype(np.int64), "seen": arm["seen"].astype(np.float32), "unseen": arm["unseen"].astype(np.float32),
          "seen_truth": arm["seen_truth"].astype(np.int64), "unseen_truth": arm["unseen_truth"].astype(np.int64)}
    validate_packet(pk, fold_contract(frame, fold))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **pk)
    return file_sha256(path)


def v3_packets(rid, frame, F):
    res = resolve_packets(rid, INNER, [d for d in V3_DIRS if d.is_dir()])
    reg = json.loads((ROOT / "artifacts/r2/v3/packet_registry_v3.json").read_text())["rows"].get(rid)
    out = {}
    for fn in INNER:
        z = np.load(res[fn]["path"])
        validate_packet(z, fold_contract(frame, F[fn]))
        if reg is not None:
            assert reg["folds"][fn]["sha256"] == res[fn]["sha256"], f"{rid}/{fn} not the registered packet"
        out[fn] = {k: z[k] for k in z.files}
        out[fn]["sha256"] = res[fn]["sha256"]
    return out, reg is not None


def run_row(rid, frame, bench, F, out):
    spec = ROWS[rid]
    t0 = time.time()
    rdir = out / "scores" / rid
    man = {"row": rid, "label": LABEL, "spec": {k: v for k, v in spec.items()}, "started": now(), "folds": {}}
    if spec["kind"] == "posthoc":
        base, _ = v3_packets(D1, frame, F)
        for fn in INNER:
            fold = F[fn]; z = base[fn]
            ids = z["cand_ids"]; fit = fold_contract(frame, fold)["fit_mask"]
            if spec["rule"] == "count":
                n = frame.loc[fold["fitting_rows"]].comp_id.value_counts()
                off = np.log1p(np.array([n.get(int(c), 0) for c in ids], float)) * fit  # fitting rows only
                offs = {"seen": np.broadcast_to(off, z["seen"].shape), "unseen": np.broadcast_to(off, z["unseen"].shape)}
            else:
                cand = frame.drop_duplicates("comp_id").set_index("comp_id").loc[ids]
                crops = cand.crop_id.to_numpy()
                phi = {k: float(fit[crops == k].mean()) for k in np.unique(crops)}  # candidate names only
                offs = {}
                for part in ("seen", "unseen"):
                    s = z[part].astype(np.float64)
                    ks = np.unique(crops)
                    lse = np.stack([np.logaddexp.reduce(s[:, crops == k], axis=1) for k in ks], 1)
                    khat = ks[lse.argmax(1)]
                    offs[part] = np.array([phi[k] for k in khat])[:, None] * fit[None, :]
            man["folds"][fn] = {"base_packet_sha256": z["sha256"], "configs": {}}
            for g1 in spec["g1"]:
                arm = {"cand_ids": ids, "seen": z["seen"].astype(np.float64) - g1 * offs["seen"],
                       "unseen": z["unseen"].astype(np.float64) - g1 * offs["unseen"],
                       "seen_truth": z["seen_truth"], "unseen_truth": z["unseen_truth"]}
                man["folds"][fn]["configs"][f"g1={g1:g}"] = save_packet(rdir / f"g1={g1:g}" / f"{fn}_scores.npz", fold, frame, arm)
    else:
        feats, text, ident = load_feat(spec["feat"], bench)
        man["feature_identity"] = ident
        for fn in INNER:
            fold = F[fn]
            torch.manual_seed(0)
            fd, truth, _ = make_fold(frame, fold, feats, text)
            if spec["fn"] == "synth_v4":
                if spec["cfg"].get("text_frac"):
                    fd.extra["raw_fit"] = raw_fit_stats(feats[fold["fitting_rows"]])
                s, u = score_synth_v4(fd, spec["cfg"])
            elif spec["fn"] == "proj":
                s, u = score_projected_d1(fd, spec["cfg"])
                man.setdefault("projection_logs", {})[fn] = fd.extra.get("projection_log")
            else:
                s, u = RECIPES["synth_composition"](fd, spec["cfg"])
            arm = {"cand_ids": np.asarray(fd.extra["cand_ids"]), "seen": s.numpy(), "unseen": u.numpy(),
                   "seen_truth": truth["seen"], "unseen_truth": truth["unseen"]}
            man["folds"][fn] = {"packet_sha256": save_packet(rdir / f"{fn}_scores.npz", fold, frame, arm),
                                "contract_sha256": fold_contract(frame, fold)["sha256"]}
    man["config_sha256"] = sha256_json(man["spec"])
    man["seconds"] = time.time() - t0
    man["finished"] = now()
    (rdir / "manifest.json").write_text(json.dumps(man, indent=1, default=float))
    return man


def table(rid, frame, F, out):
    spec = ROWS[rid]
    if spec["kind"] == "v3":
        pk, registered = v3_packets(rid, frame, F)
        sc = {fn: [scorer_from_scores(frame, F[fn], pk[fn]["seen"], pk[fn]["seen_truth"], pk[fn]["unseen"], pk[fn]["unseen_truth"], pk[fn]["cand_ids"])]
              for fn in INNER}
        return ev.RowTable(rid, sc, ["v3 packet"])
    configs = [f"g1={g:g}" for g in spec["g1"]] if spec["kind"] == "posthoc" else ["single"]
    sc = {}
    for fn in INNER:
        sc[fn] = []
        for c in configs:
            p = out / "scores" / rid / (f"{c}/" if spec["kind"] == "posthoc" else "") / f"{fn}_scores.npz"
            z = np.load(p)
            validate_packet(z, fold_contract(frame, F[fn]))
            sc[fn].append(scorer_from_scores(frame, F[fn], z["seen"], z["seen_truth"], z["unseen"], z["unseen_truth"], z["cand_ids"]))
    return ev.RowTable(rid, sc, configs)


def evaluate(frame, F, out, rows):
    T = {r: table(r, frame, F, out) for r in rows}
    S = {r: T[r].summary(INNER) for r in rows}
    res = {"label": LABEL, "folds": INNER, "control": D1, "rows": {}, "evaluated": now()}
    for r in rows:
        man = json.loads((out / "scores" / r / "manifest.json").read_text()) if ROWS[r]["kind"] != "v3" else {}
        g0 = {}
        for k in INNER:
            ci = S[r]["per"][k]["ci"]
            x = T[r].sc[k][ci].at(0.0)
            g0[k] = {key: x[key] for key in ev.KEYS}
        entry = {"summary": {"primary_H_bal_lofo": S[r]["primary"], "ausuc": S[r]["ausuc"], "healthy_shortcut_lofo": S[r]["hs"],
                             **{f"{k}_lofo": float(np.mean([S[r]["per"][f][k] for f in INNER])) for k in ev.KEYS},
                             **{f"{k}_gamma0": float(np.mean([g0[f][k] for f in INNER])) for k in ev.KEYS}},
                 "per_fold": S[r]["per"], "gamma0_per_fold": g0,
                 "false_healthy": ev.false_healthy(T[r], INNER),
                 "source_strata_seen": ev.source_strata(T[r], frame, F, INNER),
                 "seconds": man.get("seconds"), "config_sha256": man.get("config_sha256"),
                 "packets": man.get("folds")}
        if r != D1:
            entry["gate_vs_D1"] = ev.gate(S[r], S[D1], INNER)
            entry["nested_vs_D1"] = ev.nested(T, D1, [r], INNER)
            entry["bootstrap_vs_D1"] = ev.paired_bootstrap(T[r], T[D1], INNER)
        res["rows"][r] = entry
        (out / "curves").mkdir(parents=True, exist_ok=True)
        (out / "curves" / f"{r}.json").write_text(json.dumps(ev.curves(T[r])))
    i_rows = [r for r in rows if r in I_ROWS]
    res["stage_nested_D1_to_all_I_rows"] = ev.nested(T, D1, i_rows, INNER)
    (out / "results.json").write_text(json.dumps(res, indent=1, default=float))
    print(f"{'row':36s} {'H_bal':>7s} {'dpp':>6s} {'f+':>3s} {'f-':>3s} {'AUSUC':>6s} {'nest':>6s} {'n+':>3s} {'CI95':>15s} gate")
    for r in rows:
        e = res["rows"][r]; g = e.get("gate_vs_D1", {}); n = e.get("nested_vs_D1", {}); b = e.get("bootstrap_vs_D1", {})
        ci = b.get("ci95_pp", [float("nan")] * 2)
        print(f"{r:36s} {100*e['summary']['primary_H_bal_lofo']:7.2f} {g.get('mean_gain_pp', 0):+6.2f} {g.get('folds_improved', '-'):>3} "
              f"{g.get('folds_regressed', '-'):>3} {100*e['summary']['ausuc']:6.2f} {n.get('mean_diff_pp', 0):+6.2f} {n.get('folds_better', '-'):>3} "
              f"[{ci[0]:+5.2f},{ci[1]:+5.2f}] {g.get('pass', '-')}")
    print("stage nested:", json.dumps({k: v for k, v in res["stage_nested_D1_to_all_I_rows"].items() if k != "held"}))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--evaluate", action="store_true")
    a = ap.parse_args()
    frame, bench, F = env()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.evaluate:
        rows = [r for r in ROWS if ROWS[r]["kind"] == "v3" or (out / "scores" / r / "manifest.json").exists()]
        evaluate(frame, F, out, rows)
        return
    for rid in (a.only or I_ROWS):
        if (out / "scores" / rid / "manifest.json").exists():
            print("done", rid); continue
        try:
            man = run_row(rid, frame, bench, F, out)
            status = {"row": rid, "status": "ok", "seconds": man["seconds"]}
        except Exception as e:  # recorded, never hidden
            status = {"row": rid, "status": "failed", "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-2000:]}
        status["time"] = now()
        with open(out / "run_status.jsonl", "a") as fh:
            fh.write(json.dumps(status, default=float) + "\n")
        print(json.dumps({k: v for k, v in status.items() if k != "trace"}), flush=True)


if __name__ == "__main__":
    main()
