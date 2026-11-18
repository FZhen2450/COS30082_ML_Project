"""Compare regenerated score packets with the registered (historical) ones. Read-only on both.

For every row x fold: exact candidate ids, exact truth indices, max |score difference| (seen, unseen)
against the declared tolerance, Top-1 agreement, and H_bal / H_micro / seen / unseen over the complete
gamma grid. Exit 1 if any packet exceeds the tolerance or any metric differs; mismatches are reported,
never absorbed (the tolerance is fixed by the approved plan and is not a parameter).

  python scripts/r2_compare_repro.py --regen DIR/scores --rows B1 D1 F10 --out OUT.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research_r2.data import load_training_source  # noqa: E402
from research_r2.metrics import GAMMA_GRID  # noqa: E402
from research_r2.packets import file_sha256, fold_contract, validate_packet  # noqa: E402
from research_r2.runner import scorer_from_npz  # noqa: E402

TOL = 1e-5  # approved plan; investigate, do not relax
EXP = ROOT.parents[1] / "experiments" / "r2-compositional"
KEYS = ("H_bal", "H_micro", "seen_top1", "unseen_top1_macro", "unseen_top1_micro")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="artifacts/r2/benchmark_v3.json")
    ap.add_argument("--registry", default="artifacts/r2/v3/packet_registry_v3.json")
    ap.add_argument("--regen", required=True)
    ap.add_argument("--rows", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    frame = load_training_source(ROOT / "DPD_Small_train.csv")
    F = {f["name"]: f for f in json.loads(Path(a.bench).read_text())["folds"]}
    reg = json.loads(Path(a.registry).read_text())
    rep, ok = {"tolerance": TOL, "registry_sha256": file_sha256(a.registry), "rows": {}}, True
    for rid in a.rows:
        rr = reg["rows"][rid]
        man = json.loads((Path(a.regen) / rid / "manifest.json").read_text())
        same_cfg = man["config_sha256"] == rr["config_sha256"]
        same_feat = man["feature_identity"] == rr["feature_identity"]
        ok &= same_cfg and same_feat
        rows = {"config_identical": same_cfg, "feature_identity_identical": same_feat, "folds": {}}
        for fn, rec in rr["folds"].items():
            con = fold_contract(frame, F[fn])
            old_p, new_p = EXP / rec["copies"][0], Path(a.regen) / rid / f"{fn}_scores.npz"
            if file_sha256(old_p) != rec["sha256"]:
                raise SystemExit(f"{old_p}: registered packet changed")
            zo, zn = np.load(old_p), np.load(new_p)
            validate_packet(zo, con); validate_packet(zn, con)
            d_seen = float(np.abs(zo["seen"].astype(np.float64) - zn["seen"]).max())
            d_uns = float(np.abs(zo["unseen"].astype(np.float64) - zn["unseen"]).max())
            top1 = bool((zo["seen"].argmax(1) == zn["seen"].argmax(1)).all() and (zo["unseen"].argmax(1) == zn["unseen"].argmax(1)).all())
            so, sn = scorer_from_npz(frame, F[fn], zo), scorer_from_npz(frame, F[fn], zn)
            co, cn = [so.at(g) for g in GAMMA_GRID], [sn.at(g) for g in GAMMA_GRID]
            dm = {k: float(max(abs(x[k] - y[k]) for x, y in zip(co, cn))) for k in KEYS}
            n_diff = sum(any(x[k] != y[k] for k in KEYS) for x, y in zip(co, cn))
            r = {"cand_ids_identical": bool(np.array_equal(zo["cand_ids"], zn["cand_ids"])),
                 "truth_identical": bool(np.array_equal(zo["seen_truth"], zn["seen_truth"]) and np.array_equal(zo["unseen_truth"], zn["unseen_truth"])),
                 "max_abs_diff_seen": d_seen, "max_abs_diff_unseen": d_uns, "top1_identical": top1,
                 "gamma_points": len(GAMMA_GRID), "gamma_points_with_any_metric_difference": n_diff,
                 "max_abs_metric_diff": dm, "ausuc_diff": float(abs(so.ausuc() - sn.ausuc())),
                 "regen_sha256": file_sha256(new_p), "registered_sha256": rec["sha256"],
                 "manifest_sha256_matches": man["packets"][fn]["sha256"] == file_sha256(new_p)}
            r["pass"] = bool(r["cand_ids_identical"] and r["truth_identical"] and max(d_seen, d_uns) <= TOL and top1
                             and n_diff == 0 and r["manifest_sha256_matches"])
            ok &= r["pass"]
            rows["folds"][fn] = r
            print(f"{'PASS' if r['pass'] else 'FAIL'} {rid:<22} {fn} ids={r['cand_ids_identical']} truth={r['truth_identical']} "
                  f"max|d| seen {d_seen:.2e} unseen {d_uns:.2e} top1={top1} metric-diff gamma pts {n_diff}/{len(GAMMA_GRID)} "
                  f"max|dH_bal| {dm['H_bal']:.1e} |dAUSUC| {r['ausuc_diff']:.1e}")
        print(f"     {rid}: config identical {same_cfg}, feature identity identical {same_feat}")
        rep["rows"][rid] = rows
    rep["pass"] = bool(ok)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=1, sort_keys=True) + "\n")
    print("ALL PASS" if ok else "MISMATCH: investigate (tolerance is not relaxed)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
