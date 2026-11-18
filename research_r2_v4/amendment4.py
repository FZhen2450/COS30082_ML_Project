"""Amendment 4 (append-only): J4b_best_photometric_aug seeds 1-2 and the three-seed confirmation against J2.

Amendment 3 (research_r2_v4/amendment_3.json, sha256 ba3f2442...3456) authorises J4b seed 0 only and pins 24 files with
code_pins. Nothing it pins is edited; this module ADDS:

  * an authorisation record research_r2_v4/amendment_4.json and its declaration DECLARATION_AMENDMENT_4.json
    (sha256 of amendment_4.json, DECLARATION_AMENDMENT_3.json, amendment_3.json and DECLARATION.json; its own code_pins);
  * ``check_authorised(row, seed, purpose)``: amendment 3 still verifies with the PINNED tool and still has its recorded
    hashes, amendment 4 verifies, and (row, seed, purpose) is listed in amendment 4 ``authorised_now``;
  * ``run_child(child_row, seed, ...)``: PRODUCTION runner entry. Authorisation first, then the declared J4b configuration
    (child_inputs.declared_child_config, J2 + augment=photometric only), the pinned fold identities, and only then the
    unchanged amendment-3 shared implementation child_run._run (jrun.run_fold; new attempt directories; failures and
    rejections kept; post-run verification and re-derivation; child_identity.json). No configuration, backend or switch
    can be passed; the output directory is fixed by seed;
  * ``verified_confirmation_pair(frame, F)``: PRODUCTION evaluator input. Fixed per-seed child bases (seed 0 from the
    amendment-3 stage, seeds 1-2 from the amendment-4 stage), registered J2 bases, child_inputs.load_verified (mandatory
    re-derivation, sidecar, ordered image identity, contract, lineage, sealed tables) and the child_inputs._pair checks;
  * ``confirmation_decision(C, P)``: child_gate.promotion_decision(C, P, "confirm") on all six folds only.

``_confirmation_pair`` takes explicit bases and configurations for SYNTHETIC TESTS only.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

from research_r2.data import sha256_file
from research_r2.packets import fold_contract
from research_r2_v4 import child_gate as cg
from research_r2_v4 import child_inputs as ci
from research_r2_v4 import child_run as cr
from research_r2_v4 import confirm, jrun
from research_r2_v4.common import INNER, V4

ROOT = Path(__file__).resolve().parents[1]
AMENDMENT_REL = "research_r2_v4/amendment_4.json"
DECLARATION_REL = "research_r2_v4/DECLARATION_AMENDMENT_4.json"
DECLARED_FILES = [AMENDMENT_REL, "research_r2_v4/DECLARATION_AMENDMENT_3.json", "research_r2_v4/amendment_3.json",
                  "research_r2_v4/DECLARATION.json"]
A3_SHA256 = {"research_r2_v4/amendment_3.json": "ba3f24425f0c6d5dff0411a40eeb775a868fc44f0afc2a70e014e048e7da3456",
             "research_r2_v4/DECLARATION_AMENDMENT_3.json": "95358ee8cc6350918907a991dfe22d15288828282a924e1c24870d9d79419f54"}
DECLARED_DATE = "2026-11-17"

CHILD_ROW = "J4b_best_photometric_aug"
RUN_SEEDS = (1, 2)
SEED0_STAGE = V4 / "stage_j4b_20261117"
CONFIRM_STAGE = V4 / "stage_j4b_confirm_20261117"
CHILD_BASES = {0: SEED0_STAGE / "runs", 1: CONFIRM_STAGE / "runs", 2: CONFIRM_STAGE / "runs"}
PROTECTED = list(cr.PROTECTED) + [SEED0_STAGE, ROOT / "artifacts"]


class A4Error(cr.ChildRunError):
    """Amendment 4 does not authorise this, or its declaration / inputs do not verify."""


# ----------------------------------------------------------------------------- declaration and authorisation
def a3_problems(root=ROOT):
    """Amendment 3 must still verify with its own PINNED tool (files and 24 code pins) and keep its recorded hashes."""
    p = [f"{f} changed (sha256 != recorded {h[:12]})" for f, h in A3_SHA256.items()
         if not (Path(root) / f).exists() or sha256_file(Path(root) / f) != h]
    r = subprocess.run([sys.executable, str(Path(root) / "scripts" / "r2v4_declare.py"), "--verify", "--amendment", "3"], cwd=root,
                       capture_output=True, text=True)
    if r.returncode != 0:
        p.append(f"amendment 3 does not verify: {r.stdout.strip()} {r.stderr.strip()[-300:]}")
    return p


def declaration_problems(root=ROOT):
    """Amendment 4: DECLARATION_AMENDMENT_4.json matches the declared files; every amendment-4 code pin matches."""
    root = Path(root)
    if not (root / AMENDMENT_REL).exists() or not (root / DECLARATION_REL).exists():
        return ["amendment 4 is not declared (amendment_4.json / DECLARATION_AMENDMENT_4.json missing)"]
    d = json.loads((root / DECLARATION_REL).read_text())
    p = []
    if d.get("amendment") != 4 or sorted(d.get("sha256", {})) != sorted(DECLARED_FILES):
        p.append("DECLARATION_AMENDMENT_4.json does not declare exactly the amendment-4 file set")
    for f in DECLARED_FILES:
        if not (root / f).exists() or d.get("sha256", {}).get(f) != sha256_file(root / f):
            p.append(f"declared file changed: {f}")
    a = json.loads((root / AMENDMENT_REL).read_text())
    if a.get("amendment") != 4 or a.get("extends", {}).get("amendment_3_sha256") != A3_SHA256["research_r2_v4/amendment_3.json"]:
        p.append("amendment_4.json does not extend the recorded amendment 3")
    pins = a.get("code_pins") or {}
    if not pins:
        p.append("amendment_4.json has no code pins")
    p += [f"code pin {f}" for f, h in pins.items() if not (root / f).exists() or sha256_file(root / f) != h]
    return p


def declare(root=ROOT):
    """Write DECLARATION_AMENDMENT_4.json once (refused if it exists or a code pin does not match)."""
    root = Path(root)
    if (root / DECLARATION_REL).exists():
        raise A4Error(f"{DECLARATION_REL} already exists (declarations are never rewritten)")
    a = json.loads((root / AMENDMENT_REL).read_text())
    bad = [f for f, h in (a.get("code_pins") or {}).items() if not (root / f).exists() or sha256_file(root / f) != h]
    if bad or not a.get("code_pins"):
        raise A4Error(f"refusing to declare: code pins do not match the files: {bad or 'no pins'}")
    for f in DECLARED_FILES:
        json.loads((root / f).read_text())
    dec = {"schema": "r2v4-declaration-v1", "amendment": 4, "declared": DECLARED_DATE,
           "declared_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "note": "declared before any J4b seed-1/2 training or three-seed scoring; amendments 1-3 are pinned, not edited",
           "sha256": {f: sha256_file(root / f) for f in DECLARED_FILES}}
    (root / DECLARATION_REL).write_text(json.dumps(dec, indent=1) + "\n")
    return dec


def check_authorised(child_row, seed, purpose, root=ROOT, a3_check=None):
    """purpose 'run' (one training seed) or 'evaluate' (the three-seed confirmation). Raises A4Error otherwise."""
    p = (a3_problems(root) if a3_check is None else a3_check()) + declaration_problems(root)
    if p:
        raise A4Error("; ".join(p))
    auth = json.loads((Path(root) / AMENDMENT_REL).read_text()).get("authorised_now", {})
    if child_row not in auth.get("rows", []):
        raise A4Error(f"{child_row!r} is not authorised by amendment 4 (authorised: {auth.get('rows')})")
    if purpose == "run":
        if not isinstance(seed, int) or seed not in auth.get("seeds", []):
            raise A4Error(f"{child_row} seed {seed} is not authorised by amendment 4 (authorised training seeds: {auth.get('seeds')})")
    elif purpose == "evaluate":
        ev_ = auth.get("evaluation", {})
        if ev_.get("mode") != "confirm" or ev_.get("seeds") != list(confirm.CONFIRM_SEEDS) or ev_.get("folds") != list(INNER):
            raise A4Error(f"amendment 4 does not authorise the three-seed confirmation as declared: {ev_}")
    else:
        raise A4Error(f"unknown purpose {purpose!r}")
    return auth


def check_out_dir(out):
    o = Path(out).resolve()
    for p in PROTECTED:
        p = Path(p).resolve()
        if o == p or p in o.parents or o in p.parents:
            raise A4Error(f"output directory {o} overlaps protected evidence {p}")


# ----------------------------------------------------------------------------- pinned inputs
def fold_pins(frame, F):
    """Ordered evaluation identity, contract and fitting-row hash of each registered inner fold."""
    lin = ci.lineage_groups()
    return {fn: {"identity_sha256": ci.eval_identity(frame, F[fn], lin)["identity_sha256"], "contract_sha256": fold_contract(frame, F[fn])["sha256"],
                 "fitting_rows_sha256": jrun.sha256_json([int(r) for r in F[fn]["fitting_rows"]])} for fn in INNER}


def evidence_pins():
    """sha256 of the J2 receipts (seeds 0-2) and the J4b seed-0 receipts and sidecars used by the confirmation."""
    out = {"J2": {}, "J4b_seed0": {}}
    for s, base in ci.PARENT_BASES.items():
        v = confirm.verified_row_seed(base, ci.PARENT_ROW, s, INNER, jrun.j_rows()[ci.PARENT_ROW])
        out["J2"][str(s)] = {fn: jrun.sha256_file(p.parent / "receipt.json") for fn, (p, _) in v.items()}
    v = confirm.verified_row_seed(CHILD_BASES[0], CHILD_ROW, 0, INNER, ci.declared_child_config(CHILD_ROW))
    out["J4b_seed0"] = {fn: {"receipt_sha256": jrun.sha256_file(p.parent / "receipt.json"), "sidecar_sha256": jrun.sha256_file(p.parent / ci.SIDECAR)}
                        for fn, (p, _) in v.items()}
    return out


def check_pinned(frame, F, evidence=True, root=ROOT):
    a = json.loads((Path(root) / AMENDMENT_REL).read_text())
    if fold_pins(frame, F) != a["fold_pins"]:
        raise A4Error("registered fold identities differ from the amendment-4 pins")
    cfg, par = ci.declared_child_config(CHILD_ROW), jrun.j_rows()[ci.PARENT_ROW]
    if cfg.sha() != a["child"]["config_sha256"] or par.sha() != a["parent"]["config_sha256"]:
        raise A4Error("child or parent configuration differs from the amendment-4 pins")
    if ci.modification(asdict(cfg), asdict(par)) != ci.AUTHORISED_MODIFICATIONS[CHILD_ROW]:
        raise A4Error("child configuration is not J2 plus the registered photometric augmentation")
    if evidence and evidence_pins() != a["evidence_pins"]:
        raise A4Error("J2 or J4b seed-0 evidence differs from the amendment-4 pins")
    return cfg, par


# ----------------------------------------------------------------------------- production runner
def run_child(child_row, seed, image_root, device="cuda", log=print):
    """PRODUCTION: one authorised J4b seed (1 or 2) on I1-I6 into the fixed amendment-4 run directory."""
    from research_r2_v4.common import env
    if child_row != CHILD_ROW:
        raise A4Error(f"{child_row!r} is not the amendment-4 child row ({CHILD_ROW})")
    check_authorised(child_row, seed, "run")
    out = CHILD_BASES[seed]
    check_out_dir(out)
    frame, bench, F = env()
    cfg, _ = check_pinned(frame, F)
    access = jrun.ImageAccess(frame, image_root)
    return cr._run(cfg, seed, out, frame, bench, F, jrun.ClipBackend(), access, ci.PARENT_ROW, device, cr.worktree_clean, log)


# ----------------------------------------------------------------------------- confirmation pair
def _confirmation_pair(child_bases, parent_bases, child_row, parent_row, child_cfg, parent_cfg, frame, F):
    """TEST path (explicit bases / configurations). Exactly the confirmation seeds; each base holds exactly the seeds
    mapped to it; mandatory re-derivation; the child_inputs._pair checks."""
    seeds = sorted(confirm.CONFIRM_SEEDS)
    if sorted(child_bases) != seeds:
        raise A4Error(f"child bases cover seeds {sorted(child_bases)}; confirmation needs exactly {seeds}")
    mod = ci.modification(asdict(child_cfg), asdict(parent_cfg))
    for base in {Path(b) for b in child_bases.values()}:
        want = sorted(s for s, b in child_bases.items() if Path(b) == base)
        have = sorted(confirm.seed_dirs(base / child_row)) if (base / child_row).exists() else []
        if have != want:
            raise A4Error(f"{base}: holds {child_row} seeds {have}, declared for seeds {want}")
    lin = ci.lineage_groups()
    C, P, VC, VP = {}, {}, {}, {}
    for s in seeds:  # child first
        C[s], VC[s] = ci.load_verified(child_bases[s], child_row, s, frame, F, child_cfg, child=True, parent_row=parent_row, lineage=lin)
    for s in seeds:
        if s not in parent_bases:
            raise A4Error(f"no registered parent run for seed {s}")
        P[s], VP[s] = ci.load_verified(parent_bases[s], parent_row, s, frame, F, parent_cfg, child=False, parent_row=parent_row, lineage=lin)
    for r, V in ((child_row, VC), (parent_row, VP)):
        xp = confirm.cross_seed_problems(r, V, INNER)
        if xp:
            raise A4Error(f"{r}: " + "; ".join(xp))
    for s in seeds:
        if C[s].identity["jrun_sha256"] != P[s].identity["jrun_sha256"]:
            raise A4Error(f"seed{s}: child and parent were trained by different training code")
        for fn in INNER:
            cf, pf = C[s].identity["folds"][fn], P[s].identity["folds"][fn]
            if cf["identity_sha256"] != pf["identity_sha256"]:
                raise A4Error(f"seed{s} {fn}: child and parent evaluation identity differ")
            if cf["checkpoint_sha256"] == pf["checkpoint_sha256"]:
                raise A4Error(f"seed{s} {fn}: child checkpoint is the parent's checkpoint")
    return C, P, {"modification": mod, "seeds": seeds, "mode": "confirm", "child_row": child_row, "parent_row": parent_row,
                  "child_bases": {s: str(b) for s, b in child_bases.items()}, "parent_bases": {s: str(parent_bases[s]) for s in seeds}}


def verified_confirmation_pair(frame, F):
    """PRODUCTION: J4b seeds 0-2 (fixed bases) vs the registered J2 seeds 0-2; authorisation and pins checked first."""
    check_authorised(CHILD_ROW, None, "evaluate")
    cfg, par = check_pinned(frame, F)
    return _confirmation_pair(CHILD_BASES, ci.PARENT_BASES, CHILD_ROW, ci.PARENT_ROW, cfg, par, frame, F)


def confirmation_decision(C, P, folds=None):
    """The amendment-3 decision function in confirm mode, on all six registered folds (a subset is refused)."""
    folds = list(INNER) if folds is None else list(folds)
    if folds != list(INNER):
        raise A4Error(f"the confirmation gate needs all six folds {INNER}, got {folds}")
    return cg.promotion_decision(C, P, "confirm", folds)


# ----------------------------------------------------------------------------- report (diagnostics separate from the gate)
UNITS = ROOT / "artifacts/r2/v4/stage_hrt_20261116_post_review_v3/strata_units.json"


def _mean(xs):
    import numpy as np
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def _delta(c, p, m):
    if c is None or p is None:
        return None
    return c - p if m == "gamma" else 100 * (c - p)


def confirmation_report(C, P, meta, frame, F, child_cfg, reps=2000):
    """Primary gate + decision (recomputed from the verified tables) and, separately, the diagnostics: per seed and
    fold, seed means, LOFO metrics incl. Top-5, source strata, false-healthy, receipts. No diagnostic has pass/fail."""
    import numpy as np
    from research_r2_v4 import evaluate as ev
    from research_r2_v4.common import LABEL, now
    seeds = meta["seeds"]
    prim = cg.primary_gate(C, P, INNER, mode="confirm", reps=reps)
    decision = confirmation_decision(C, P)
    diag = cg.diagnostics(C, P, INNER, "confirm")
    lm = {s: {"child": confirm.lofo_metrics(C[s], INNER), "parent": confirm.lofo_metrics(P[s], INNER)} for s in seeds}
    keys = list(ev.KEYS) + ["H5_micro", "seen_top5", "unseen_top5_micro", "ausuc", "gamma"]
    seed_mean = {k: {} for k in INNER}
    for k in INNER:
        for m in keys:
            c = _mean([lm[s]["child"][k][m] for s in seeds]); p = _mean([lm[s]["parent"][k][m] for s in seeds])
            seed_mean[k][m] = {"child": c, "parent": p, "delta_pp": _delta(c, p, m)}
    overall = {}
    for m in keys:
        c = _mean([seed_mean[k][m]["child"] for k in INNER]); p = _mean([seed_mean[k][m]["parent"] for k in INNER])
        overall[m] = {"child": c, "parent": p, "delta_pp": _delta(c, p, m)}
    D = prim["detail"]
    per_seed = {}
    for s in seeds:
        d = {k: D[s][k]["child"]["E"] - D[s][k]["parent"]["E"] for k in INNER}
        per_seed[s] = {"per_fold_delta_pp": {k: 100 * v for k, v in d.items()}, "mean_delta_pp": 100 * float(np.mean(list(d.values()))),
                       "folds_improved": int(sum(v > 0 for v in d.values())),
                       "ausuc_delta_pp": 100 * float(np.mean([D[s][k]["child"]["ausuc"] - D[s][k]["parent"]["ausuc"] for k in INNER])),
                       "shortcut_delta_pp": 100 * float(np.mean([D[s][k]["child"]["hs"] - D[s][k]["parent"]["hs"] for k in INNER])),
                       "per_fold_shortcut_delta_pp": {k: 100 * (D[s][k]["child"]["hs"] - D[s][k]["parent"]["hs"]) for k in INNER}}
    strata = {s: {"child": ev.source_strata(C[s], frame, F, INNER), "parent": ev.source_strata(P[s], frame, F, INNER)} for s in seeds}
    strata_mean = {}
    for key in strata[seeds[0]]["child"]:
        c = _mean([strata[s]["child"][key]["top1"] for s in seeds]); p = _mean([strata[s]["parent"][key]["top1"] for s in seeds])
        strata_mean[key] = {"child_top1": c, "parent_top1": p, "delta_pp": _delta(c, p, "top1"),
                            "per_seed_delta_pp": {s: _delta(strata[s]["child"][key]["top1"], strata[s]["parent"][key]["top1"], "top1") for s in seeds}}
    fh = {s: {"child": ev.false_healthy(C[s], INNER), "parent": ev.false_healthy(P[s], INNER)} for s in seeds}
    rec = {}
    for s in seeds:
        v = confirm.verified_row_seed(Path(meta["child_bases"][s]), meta["child_row"], s, INNER, child_cfg)
        rec[s] = {fn: {k: r.get(k) for k in ("seconds", "peak_gpu_gib_training", "peak_gpu_gib_total", "checkpoint_sha256", "features_sha256",
                                             "packet_sha256", "trainable_tensors_changed", "reload_max_abs_diff", "torch_seed", "config_sha256")}
                   | {"attempt": p.parent.name, "grad_checks": r.get("grad_checks"), "git_head": r["code"]["git_head"], "jrun_sha256": r["code"]["jrun_sha256"],
                      "receipt_sha256": jrun.sha256_file(p.parent / "receipt.json"), "sidecar_sha256": jrun.sha256_file(p.parent / ci.SIDECAR),
                      "rederive_max_abs_diff": C[s].identity["folds"][fn]["rederive_max_abs_diff"]}
                   for fn, (p, r) in v.items()}
    return {"label": LABEL + "; amendment 4 three-seed child vs J2 confirmation (exploratory)", "child_row": meta["child_row"],
            "parent_row": meta["parent_row"], "mode": "confirm", "seeds": seeds, "modification": meta["modification"],
            "child_bases": meta["child_bases"], "parent_bases": meta["parent_bases"], "evaluated": now(),
            "primary": {"endpoint": prim["endpoint"], "gate": prim["gate"], "bootstrap_exploratory": prim["bootstrap"], "per_seed": per_seed,
                        "per_fold_gamma": {s: {k: {"child": D[s][k]["child"]["gamma"], "parent": D[s][k]["parent"]["gamma"]} for k in INNER} for s in seeds},
                        "per_seed_fold": {s: {k: {"child_H_bal": D[s][k]["child"]["E"], "parent_H_bal": D[s][k]["parent"]["E"],
                                                  "child_ausuc": D[s][k]["child"]["ausuc"], "parent_ausuc": D[s][k]["parent"]["ausuc"],
                                                  "child_shortcut": D[s][k]["child"]["hs"], "parent_shortcut": D[s][k]["parent"]["hs"]}
                                              for k in INNER} for s in seeds}},
            "decision": decision, "diagnostics": diag,
            "diagnostic_lofo_metrics_per_seed": lm, "diagnostic_lofo_metrics_seed_mean": seed_mean, "diagnostic_lofo_metrics_overall": overall,
            "diagnostic_source_strata_seen": {"per_seed": strata, "seed_mean": strata_mean, "units": json.loads(UNITS.read_text())},
            "diagnostic_false_healthy": fh, "child_receipts": rec,
            "parent_identity": {s: {fn: {k: P[s].identity["folds"][fn][k] for k in ("identity_sha256", "checkpoint_sha256", "packet_sha256",
                                                                                     "rederive_max_abs_diff")} for fn in INNER} for s in seeds}}
