"""Amendment-3 child runner (J4b), fail closed. Uses jrun.run_fold unchanged.

For each registered inner fold: a NEW attempt directory (never reused, never inside the J2 / H_RT evidence), the
declared child configuration from jrun.j_rows("J2_clipL_last4") (J2 plus exactly the registered modification),
seed-derived initialisation from the original pretrained weights, then post-run verification (receipt checks,
packet re-derivation, registered fold, lineage) and only then ``child_identity.json``. A failed run keeps its
``failure.json``; a run that fails post-run verification keeps ``child_rejected.json`` and gets no sidecar.

Production entry: ``run_child(child_row, seed, out, image_root)``; it requires the amendment-3 declaration to verify
and the (row, seed) to be authorised in research_r2_v4/amendment_3.json. ``_run`` takes explicit configurations
and switches for synthetic tests only.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path

from research_r2_v4 import child_inputs as ci
from research_r2_v4 import confirm, jrun
from research_r2_v4.common import INNER, V4

ROOT = Path(__file__).resolve().parents[1]
AMENDMENT = ROOT / "research_r2_v4" / "amendment_3.json"
DECLARATION = ROOT / "research_r2_v4" / "DECLARATION_AMENDMENT_3.json"
PROTECTED = [V4 / d for d in ("stage_j_phase1_20261114", "stage_j_phase2_20261115", "stage_hrt_20261116", "stage_h_20261114",
                              "stage_i_20261114", "stage_j_20261114", "stage_j_audit_20261114", "stage_j_audit_20261114_v2", "features")]


class ChildRunError(jrun.JCheckError):
    """The child run may not start or an attempt failed verification."""


def worktree_clean():
    out = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True)
    return out.returncode == 0 and out.stdout.strip() == ""


def check_out_dir(out):
    o = Path(out).resolve()
    for p in PROTECTED:
        p = p.resolve()
        if o == p or p in o.parents or o in p.parents:
            raise ChildRunError(f"output directory {o} overlaps protected evidence {p}")


def check_declaration(child_row, seed):
    if not AMENDMENT.exists() or not DECLARATION.exists():
        raise ChildRunError("amendment 3 is not declared (amendment_3.json / DECLARATION_AMENDMENT_3.json missing)")
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "r2v4_declare.py"), "--verify", "--amendment", "3"], cwd=ROOT,
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise ChildRunError(f"amendment 3 declaration does not verify: {r.stdout.strip()} {r.stderr.strip()[-300:]}")
    a = json.loads(AMENDMENT.read_text())
    auth = a.get("authorised_now", {})
    if child_row not in auth.get("rows", []) or seed not in auth.get("seeds", []):
        raise ChildRunError(f"{child_row} seed {seed} is not authorised by amendment 3 (authorised: {auth})")


def _verify_attempt(adir, rec, cfg, seed, fold, frame, lineage):
    p = confirm.receipt_problems(rec, cfg.row, seed, fold["name"], INNER.index(fold["name"]), cfg)
    if p:
        raise ChildRunError("; ".join(p))
    ci.check_lineage(fold, lineage)
    return ci.rederive_packet(adir, rec, frame, fold)


def _run(cfg, seed, out, frame, bench, F, backend, access, parent_row=ci.PARENT_ROW, device="cuda", worktree_check=worktree_clean,
         log=print):
    """Shared implementation (tests pass synthetic configurations and switches; production calls run_child)."""
    if not worktree_check():
        raise ChildRunError("refusing to start: the worktree is not clean")
    check_out_dir(out)
    lin = ci.lineage_groups()
    out = Path(out)
    status = {}
    for i, fn in enumerate(INNER):
        fold = F[fn]
        ci.check_registered_fold(fold, F[fn]); ci.check_lineage(fold, lin)
        d = out / cfg.row / f"seed{seed}" / fn
        accepted = [a for a in sorted(d.glob("attempt*")) if (a / ci.SIDECAR).exists()] if d.exists() else []
        if accepted:
            status[fn] = "ok (existing)"; continue
        _, adir = jrun.fold_dir(out, cfg.row, seed, fn)  # always a new attempt directory
        t0 = time.time()
        try:
            rec = jrun.run_fold(cfg, seed, fold, i, frame, bench, backend, access, adir, device)
        except Exception as e:  # noqa: BLE001
            adir.mkdir(parents=True, exist_ok=True)
            jrun.save_atomic_json(adir / "failure.json", {"row": cfg.row, "seed": seed, "fold": fn, "status": "failed",
                                                         "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-3000:]})
            status[fn] = f"failed: {type(e).__name__}: {e}"; log(f"FAILED {cfg.row} seed{seed} {fn}: {e}"); continue
        try:
            diff = _verify_attempt(adir, rec, cfg, seed, fold, frame, lin)
        except Exception as e:  # noqa: BLE001
            jrun.save_atomic_json(adir / "child_rejected.json", {"row": cfg.row, "seed": seed, "fold": fn, "status": "rejected",
                                                                "error": f"{type(e).__name__}: {e}"})
            status[fn] = f"rejected: {type(e).__name__}: {e}"; log(f"REJECTED {cfg.row} seed{seed} {fn}: {e}"); continue
        ci.write_child_identity(adir, rec, frame, fold, parent_row=parent_row, lineage=lin)
        status[fn] = "ok"
        log(json.dumps({"row": cfg.row, "seed": seed, "fold": fn, "seconds": rec["seconds"], "rederive_max_abs_diff": diff,
                        "wall": time.time() - t0}))
    if not worktree_check():
        status["_worktree"] = "CHANGED DURING THE RUN"
    return status


def run_child(child_row, seed, out, image_root, device="cuda", log=print):
    """PRODUCTION: the authorised child row, declared configuration, CLIP backend, real images (training CSV only)."""
    from research_r2_v4.common import env
    if child_row not in ci.AUTHORISED_CHILDREN:
        raise ChildRunError(f"{child_row!r} is not an authorised child row")
    check_declaration(child_row, seed)
    cfg = ci.declared_child_config(child_row, ci.PARENT_ROW)
    if ci.modification(asdict(cfg), asdict(jrun.j_rows()[ci.PARENT_ROW])) != ci.AUTHORISED_MODIFICATIONS[child_row]:
        raise ChildRunError(f"{child_row}: configuration is not J2 plus the authorised modification")
    frame, bench, F = env()
    access = jrun.ImageAccess(frame, image_root)
    return _run(cfg, seed, out, frame, bench, F, jrun.ClipBackend(), access, ci.PARENT_ROW, device, worktree_clean, log)
