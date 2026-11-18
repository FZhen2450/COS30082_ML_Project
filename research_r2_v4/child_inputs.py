"""Verified-input boundary for the amendment-3 child gate (PROPOSAL v3; NOT declared), fail closed.

research_r2_v4.child_gate refuses any table that does not carry an identity record from ``load_verified`` here.
Checks (reusing the existing receipts, fold contracts and feature caches; no new identity format for old runs):

  1. ordered evaluation image identity: the ordered seen-val / unseen-val row ids and training-CSV file names of the
     REGISTERED fold (benchmark v3), hashed into ``identity_sha256``; a child run must carry a ``child_identity.json``
     sidecar naming the same identity hash and its own receipt hash. Because neither the fold contract (labels and
     candidates only) nor the feature-cache row hash (a sorted set) encodes ORDER, every packet is also RE-DERIVED from
     its own verified feature cache with the registered fold order and the unchanged D1 recipe: an exchange of two
     images with the same label changes the scores and is refused.
  2. fold identity and lineage: fold lists equal the registered benchmark fold (ordered); no lineage group is shared
     between fitting and evaluation rows; receipt contract sha256 = fold_contract(registered fold).
  3. candidates: ids, order, crop / disease / healthy mapping and fitting mask (contract + child_gate.check_pair).
  4. training recipe and receipts: confirm.verified_row_seed (single successful attempt, file hashes, declared
     configuration, torch seed, gradient / AMP / frozen / reload checks, code identity, clean worktree).
  5. seeds: exact declared seed sets; cross-seed checks; child checkpoints distinct from the parent's and each other.
  6. exactly one registered modification: the child configuration equals the declared J4a-J7 row built on J2
     (jrun.j_rows(parent)) and differs from J2's configuration in exactly one field (row id excluded).
  7. feature caches: metadata (row, seed, fold, config, checkpoint, row set) verified on load; stale or foreign caches refused.
  8. code: the same training-code hash (jrun.py) for child and parent runs.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import secrets
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from research_r2.packets import fold_contract, validate_packet
from research_r2.runner import scorer_from_scores
from research_r2_v4 import confirm, jrun
from research_r2_v4 import evaluate as ev
from research_r2_v4.common import INNER, V4

ROOT = Path(__file__).resolve().parents[1]
LINEAGE = ROOT / "artifacts/r2/lineage_groups_v1.json"
PARENT_ROW = "J2_clipL_last4"
SIDECAR = "child_identity.json"
SCHEMA = "r2v4-child-identity-v1"
REDERIVE_TOL = 1e-5  # the R2 reproduction tolerance (CPU vs GPU); swaps of distinct images differ by orders more


class ChildInputError(jrun.JCheckError):
    """A child or parent input fails identity or provenance verification."""


def lineage_groups():
    return np.asarray(json.loads(LINEAGE.read_text())["group"])


# ----------------------------------------------------------------------------- identity
def eval_identity(frame, fold, lineage=None):
    """Ordered identity of a fold's evaluation population (registered fold expected)."""
    seen = [int(r) for r in fold["seen_val_rows"]]; uns = [int(r) for r in fold["unseen_val_rows"]]
    fit = sorted(int(r) for r in fold["fitting_rows"])
    con = fold_contract(frame, fold)
    rec = {"fold": fold["name"], "seen_rows_sha256": jrun.sha256_json(seen), "unseen_rows_sha256": jrun.sha256_json(uns),
           "seen_files_sha256": jrun.sha256_json([frame.file.iloc[r] for r in seen]),
           "unseen_files_sha256": jrun.sha256_json([frame.file.iloc[r] for r in uns]),
           "fitting_rows_sha256": jrun.sha256_json(fit), "contract_sha256": con["sha256"],
           "candidate_ids_sha256": jrun.sha256_json(con["cand_ids"].tolist()), "n_seen": len(seen), "n_unseen": len(uns)}
    if lineage is not None:
        rec["lineage_eval_sha256"] = jrun.sha256_json(lineage[seen + uns].tolist())
    rec["identity_sha256"] = jrun.sha256_json(rec)
    return rec


def check_registered_fold(fold, registered):
    """For a child runner: the fold it trains on must be the registered benchmark fold, lists in the same order."""
    norm = lambda v: [int(x) for x in v] if isinstance(v, (list, tuple, np.ndarray)) else v  # noqa: E731
    for k in ("name", "fitting_rows", "seen_val_rows", "unseen_val_rows", "candidate_composition_ids"):
        if norm(fold.get(k)) != norm(registered.get(k)):
            raise ChildInputError(f"{registered['name']}: fold {k} differs from the registered benchmark fold")


def check_lineage(fold, lineage):
    fit = set(lineage[np.asarray(fold["fitting_rows"])].tolist())
    ev_rows = np.asarray(list(fold["seen_val_rows"]) + list(fold["unseen_val_rows"]))
    if fit & set(lineage[ev_rows].tolist()):
        raise ChildInputError(f"{fold['name']}: a lineage group is shared between fitting and evaluation rows")


# ----------------------------------------------------------------------------- configuration
def declared_child_config(child_row, parent_row=PARENT_ROW) -> jrun.JConfig:
    rows = jrun.j_rows(parent_row)
    if child_row not in rows or child_row in jrun.j_rows():
        raise ChildInputError(f"{child_row!r} is not a registered J4a-J7 child row")
    return rows[child_row]


def modification(child_cfg: dict, parent_cfg: dict):
    """The single configuration field (row id and workers excluded) in which child and parent differ."""
    keys = sorted(k for k in set(child_cfg) | set(parent_cfg) if k not in ("row", "workers") and child_cfg.get(k) != parent_cfg.get(k))
    if len(keys) != 1:
        raise ChildInputError(f"child must differ from its parent in exactly one registered field, differs in {keys}")
    return keys[0], child_cfg[keys[0]]


# ----------------------------------------------------------------------------- packet re-derivation
def rederive_packet(adir, rec, frame, fold, tol=REDERIVE_TOL):
    """Re-score the D1 downstream from the verified feature cache in the registered fold order (as jrun.run_fold does)
    and compare with the stored packet; binds the packet's row order to the images' own features."""
    if rec["config"].get("concat_frozen"):
        raise ChildInputError("re-derivation for concatenated frozen features is not implemented (not needed for J2 children)")
    feats, rows = jrun.load_feature_cache(Path(adir) / "features.pt", rec["feature_cache_meta"])
    full = torch.full((len(frame), 2, feats.shape[-1]), float("nan"))
    full[rows] = feats.float()
    torch.manual_seed(0)
    fd, truth, _ = jrun.make_fold(frame, fold, full, None)
    s, u = jrun.RECIPES["synth_composition"](fd, jrun.D1_CFG)
    z = np.load(Path(adir) / f"{fold['name']}_scores.npz")
    if not (np.array_equal(z["seen_truth"], truth["seen"]) and np.array_equal(z["unseen_truth"], truth["unseen"])
            and np.array_equal(z["cand_ids"], np.asarray(fd.extra["cand_ids"]))):
        raise ChildInputError(f"{fold['name']}: packet truth / candidates differ from the re-derivation")
    diff = max(float(np.abs(s.numpy() - z["seen"]).max()), float(np.abs(u.numpy() - z["unseen"]).max()))
    if not diff <= tol:
        raise ChildInputError(f"{fold['name']}: packet does not re-derive from its feature cache (max abs diff {diff:.3g} > {tol})")
    return diff


# ----------------------------------------------------------------------------- child sidecar
def write_child_identity(adir, rec, frame, fold, parent_row=PARENT_ROW, lineage=None):
    """Written by a child runner after jrun.run_fold succeeded, from the fold object actually trained and scored."""
    ident = eval_identity(frame, fold, lineage)
    sc = {"schema": SCHEMA, "row": rec["row"], "seed": rec["seed"], "fold": rec["fold"], "parent_row": parent_row,
          "identity_sha256": ident["identity_sha256"], "receipt_sha256": jrun.sha256_file(Path(adir) / "receipt.json"),
          "checkpoint_sha256": rec["checkpoint_sha256"], "packet_sha256": rec["packet_sha256"]}
    jrun.save_atomic_json(Path(adir) / SIDECAR, sc)
    return sc



# ----------------------------------------------------------------------------- sealing (verified-table authenticity)
_SEAL_KEY = secrets.token_bytes(32)  # per process; a seal cannot be produced without going through _load


def _table_digest(t):
    h = hashlib.sha256()
    for fn in t.folds:
        h.update(fn.encode())
        for s in t.sc[fn]:
            for arr in (s.parts["seen"]["scores"], s.parts["seen"]["truth"], s.parts["unseen"]["scores"], s.parts["unseen"]["truth"],
                        s.fit_mask, s.cand_crop, s.cand_dis, s.cand_healthy):
                a = np.ascontiguousarray(arr); h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def _mac(identity, t):
    body = {k: v for k, v in identity.items() if k != "seal"}
    return hmac.new(_SEAL_KEY, (jrun.sha256_json(body) + _table_digest(t)).encode(), hashlib.sha256).hexdigest()


def _seal(t, identity):
    """Attach a sealed identity record. Production code reaches this only through _load (re-derivation mandatory);
    tests may call it directly on synthetic tables."""
    identity = dict(identity)
    identity["seal"] = _mac(identity, t)
    t.identity = identity
    return t


def is_sealed(t):
    idt = getattr(t, "identity", None)
    return isinstance(idt, dict) and isinstance(idt.get("seal"), str) and hmac.compare_digest(idt["seal"], _mac(idt, t))


# ----------------------------------------------------------------------------- loading
def _load(base, row, seed, frame, F, declared, child, parent_row, lineage, rederive):
    v = confirm.verified_row_seed(base, row, seed, INNER, declared)
    sc, folds = {}, {}
    for fn, (pkt, rec) in v.items():
        reg = F[fn]
        check_lineage(reg, lineage)
        con = fold_contract(frame, reg)
        if rec.get("contract_sha256") != con["sha256"]:
            raise ChildInputError(f"{row} seed{seed} {fn}: receipt contract is not the registered fold contract")
        want_rows = jrun.sha256_json(sorted(set(int(r) for r in reg["fitting_rows"]) | set(int(r) for r in reg["seen_val_rows"])
                                            | set(int(r) for r in reg["unseen_val_rows"])))
        if rec["feature_cache_meta"].get("rows_sha256") != want_rows:
            raise ChildInputError(f"{row} seed{seed} {fn}: feature cache rows are not the registered fold's rows")
        ident = eval_identity(frame, reg, lineage)
        adir = pkt.parent
        if child:
            p = adir / SIDECAR
            if not p.exists():
                raise ChildInputError(f"{row} seed{seed} {fn}: missing child provenance ({SIDECAR})")
            s = json.loads(p.read_text())
            want = {"schema": SCHEMA, "row": row, "seed": seed, "fold": fn, "parent_row": parent_row, "identity_sha256": ident["identity_sha256"],
                    "receipt_sha256": jrun.sha256_file(adir / "receipt.json"), "checkpoint_sha256": rec["checkpoint_sha256"],
                    "packet_sha256": rec["packet_sha256"]}
            bad = sorted(k for k in want if s.get(k) != want[k])
            if bad:
                raise ChildInputError(f"{row} seed{seed} {fn}: child provenance does not match ({bad})")
        diff = rederive_packet(adir, rec, frame, reg) if rederive else None
        z = np.load(pkt)
        validate_packet(z, con)
        sc[fn] = [scorer_from_scores(frame, reg, z["seen"], z["seen_truth"], z["unseen"], z["unseen_truth"], z["cand_ids"])]
        folds[fn] = {**ident, "checkpoint_sha256": rec["checkpoint_sha256"], "packet_sha256": rec["packet_sha256"],
                     "torch_seed": rec["torch_seed"], "rederive_max_abs_diff": diff}
    t = ev.RowTable(f"{row}/seed{seed}", sc, ["single"])
    one = next(iter(v.values()))[1]
    _seal(t, {"verified": True, "rederived": bool(rederive), "row": row, "seed": seed, "config": one["config"],
              "config_sha256": one["config_sha256"], "jrun_sha256": one["code"]["jrun_sha256"], "folds": folds})
    return t, v


def load_verified(base, row, seed, frame, F, declared, child=False, parent_row=PARENT_ROW, lineage=None):
    """A sealed RowTable that passed every check, INCLUDING packet re-derivation (not optional)."""
    return _load(base, row, seed, frame, F, declared, child, parent_row, lineage_groups() if lineage is None else lineage, True)


def _pair(child_base, parent_bases, child_row, parent_row, child_cfg, parent_cfg, mode, frame, F, rederive=True):
    """Shared implementation. TEST USE with synthetic configurations; production uses ``verified_pair``.
    rederive=False exists only to demonstrate the weakness in tests: such tables are marked rederived=False and the
    gate refuses them."""
    mod = modification(asdict(child_cfg), asdict(parent_cfg))
    if mode not in ("screen", "confirm"):
        raise ChildInputError(f"unknown mode {mode!r}")
    seeds = list(confirm.SCREEN_SEEDS) if mode == "screen" else list(confirm.CONFIRM_SEEDS)
    if mode == "confirm":
        present = sorted(confirm.seed_dirs(Path(child_base) / child_row))
        if present != sorted(confirm.CONFIRM_SEEDS):
            raise ChildInputError(f"{child_row}: seed directories {present}; confirmation needs exactly {sorted(confirm.CONFIRM_SEEDS)}")
    lin = lineage_groups()
    C, P, VC, VP = {}, {}, {}, {}
    for s in seeds:  # child first: a wrong child configuration is refused before any parent work
        C[s], VC[s] = _load(child_base, child_row, s, frame, F, child_cfg, True, parent_row, lin, rederive)
    for s in seeds:
        if s not in parent_bases:
            raise ChildInputError(f"no registered parent run for seed {s}")
        P[s], VP[s] = _load(parent_bases[s], parent_row, s, frame, F, parent_cfg, False, parent_row, lin, rederive)
    if mode == "confirm":
        for r, V in ((child_row, VC), (parent_row, VP)):
            xp = confirm.cross_seed_problems(r, V, INNER)
            if xp:
                raise ChildInputError(f"{r}: " + "; ".join(xp))
    for s in seeds:
        if C[s].identity["jrun_sha256"] != P[s].identity["jrun_sha256"]:
            raise ChildInputError(f"seed{s}: child and parent were trained by different training code")
        for fn in INNER:
            cf, pf = C[s].identity["folds"][fn], P[s].identity["folds"][fn]
            if cf["identity_sha256"] != pf["identity_sha256"]:
                raise ChildInputError(f"seed{s} {fn}: child and parent evaluation identity differ")
            if cf["checkpoint_sha256"] == pf["checkpoint_sha256"]:
                raise ChildInputError(f"seed{s} {fn}: child checkpoint is the parent's checkpoint (no distinct child run)")
    return C, P, {"modification": mod, "seeds": seeds, "mode": mode, "child_row": child_row, "parent_row": parent_row}


# ----------------------------------------------------------------------------- production
PARENT_BASES = {0: V4 / "stage_j_phase1_20261114", 1: V4 / "stage_j_phase2_20261115", 2: V4 / "stage_j_phase2_20261115"}
AUTHORISED_CHILDREN = ("J4b_best_photometric_aug",)
AUTHORISED_MODIFICATIONS = {"J4b_best_photometric_aug": ("augment", "photometric")}


def verified_pair(child_base, mode, frame, F, child_row="J4b_best_photometric_aug"):
    """PRODUCTION: the authorised child row on its declared configuration vs the registered J2 runs; no overrides,
    re-derivation always on."""
    if child_row not in AUTHORISED_CHILDREN:
        raise ChildInputError(f"{child_row!r} is not an authorised child row (authorised: {list(AUTHORISED_CHILDREN)})")
    parent_cfg = jrun.j_rows()[PARENT_ROW]
    child_cfg = declared_child_config(child_row, PARENT_ROW)
    if modification(asdict(child_cfg), asdict(parent_cfg)) != AUTHORISED_MODIFICATIONS[child_row]:
        raise ChildInputError(f"{child_row}: registered modification differs from the authorised one")
    return _pair(child_base, PARENT_BASES, child_row, PARENT_ROW, child_cfg, parent_cfg, mode, frame, F, rederive=True)
