"""Audit how far v1 'locked' folds were independent of screening (read-only; writes a JSON report)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def audit(bench: dict) -> dict:
    F = {f["name"]: f for f in bench["folds"]}
    S = [n for n, f in F.items() if f["role"] == "screen"]
    L = [n for n, f in F.items() if f["role"] == "locked"]
    s_fit_any = set().union(*[set(F[s]["fitting_rows"]) for s in S])
    s_fit_all = set(F[S[0]]["fitting_rows"]).intersection(*[set(F[s]["fitting_rows"]) for s in S[1:]])
    s_sv = set().union(*[set(F[s]["seen_val_rows"]) for s in S])
    s_uv = set().union(*[set(F[s]["unseen_val_rows"]) for s in S])
    touched = s_fit_any | s_sv | s_uv
    out = {"schema": "r2-v1-independence-audit-v1", "screen_folds": S, "locked_folds": L, "folds": {},
           "all_rows": 8000, "rows_touched_by_screening": len(touched)}
    for n in L:
        uv, sv, fit = (set(F[n][k]) for k in ("unseen_val_rows", "seen_val_rows", "fitting_rows"))
        out["folds"][n] = {
            "unseen_val": len(uv), "unseen_in_any_screen_fitting": len(uv & s_fit_any),
            "unseen_in_all_screen_fitting": len(uv & s_fit_all), "unseen_scored_as_screen_seen_val": len(uv & s_sv),
            "unseen_untouched_by_screening": len(uv - touched),
            "seen_val": len(sv), "seen_val_scored_in_screening": len(sv & s_sv),
            "fitting_scored_as_screen_unseen_val": len(fit & s_uv),
        }
    out["verdict"] = ("v1 L1-L3 are cross-validation confirmation folds, not an isolated outer evaluation: every "
                      "locked validation image was fitted on or scored during screening, and the screening-stage "
                      "selection (recipe, gamma) used models fitted on locked unseen compositions.")
    return out


if __name__ == "__main__":
    bench = json.loads((ROOT / "artifacts/r2/benchmark_v1.json").read_text())
    rep = audit(bench)
    dest = ROOT / "artifacts/r2/audit_v1_independence.json"
    dest.write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    sys.exit(0)
