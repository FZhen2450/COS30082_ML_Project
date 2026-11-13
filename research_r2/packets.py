"""Score-packet contract, validation and provenance (v3 hardening; review of 2fdbe19).

A *score packet* is one ``<fold>_scores.npz`` written by ``runner.run_row``. Its meaning depends on
three things that are NOT self-evident from the file: which candidate composition each score column
is, which column is the truth for each validation image, and which model configuration produced it.

``fold_contract`` derives the first two from the benchmark fold and the training CSV alone, for
both packet schemas:

  v3+ folds (``candidate_policy`` present): columns = the policy universe F u H in ascending global
      composition ID (the order ``make_fold`` uses); the packet must carry ``cand_ids``.
  legacy v1/v2 folds: columns = all compositions in ascending ID; historical packets carry no
      ``cand_ids`` and are accepted without it (they are never rewritten).

``validate_packet`` rejects any packet that differs from the contract: extra or missing arrays, a
candidate list that is a subset, superset, reordering or contains duplicates, a wrong shape or dtype,
non-finite scores, or truth indices that differ from the benchmark's row lists.

``row_config`` / ``config_sha256`` define a row's configuration identity (the fields that change its
scores: backbones, recipe, cfg, tta_hflip); descriptive fields (id, hypothesis, control) are excluded.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from .candidates import candidate_universe
from .data import candidate_table, sha256_json

LEGACY_POLICY = "legacy-all-compositions (benchmark v1/v2)"
V3_KEYS = frozenset({"cand_ids", "seen", "unseen", "seen_truth", "unseen_truth"})
LEGACY_KEYS = frozenset({"seen", "unseen", "seen_truth", "unseen_truth"})
CONFIG_FIELDS = ("backbones", "recipe", "cfg", "tta_hflip")


class PacketError(ValueError):
    """A score packet does not satisfy its fold contract or its registered provenance."""


def fold_contract(frame: pd.DataFrame, fold: dict) -> dict:
    cands = candidate_table(frame)
    if "candidate_policy" in fold:
        cu = candidate_universe(frame, fold)
        if cu["universe"] != list(fold["candidate_composition_ids"]) or cu["policy"] != fold["candidate_policy"]:
            raise PacketError(f"{fold['name']}: stored candidate universe does not match the policy")
        cands = cands[cands.comp_id.isin(cu["universe"])].reset_index(drop=True)
        fit_mask = cands.comp_id.isin(cu["fitting"]).to_numpy()
        policy = cu["policy"]
    else:
        held = set(int(c) for c in fold["heldout_composition_ids"])
        fit_mask = np.array([int(c) not in held for c in cands.comp_id])
        policy = LEGACY_POLICY
    ids = cands.comp_id.to_numpy().astype(np.int64)
    col = {int(c): i for i, c in enumerate(ids)}
    seen_truth = np.array([col[int(c)] for c in frame.comp_id.loc[fold["seen_val_rows"]]], dtype=np.int64)
    unseen_truth = np.array([col[int(c)] for c in frame.comp_id.loc[fold["unseen_val_rows"]]], dtype=np.int64)
    c = {"fold": fold["name"], "policy": policy, "schema": "v3" if "candidate_policy" in fold else "legacy",
         "cand_ids": ids, "seen_truth": seen_truth, "unseen_truth": unseen_truth, "fit_mask": fit_mask}
    c["sha256"] = sha256_json({k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in c.items()})
    return c


def _int1d(a, name):
    a = np.asarray(a)
    if a.ndim != 1 or not np.issubdtype(a.dtype, np.integer):
        raise PacketError(f"{name}: expected a 1-D integer array, got dtype {a.dtype} shape {a.shape}")
    return a


def validate_packet(z, contract: dict) -> None:
    """Raise PacketError unless the packet (npz or mapping of arrays) matches the fold contract exactly."""
    keys = frozenset(z.files if hasattr(z, "files") else z.keys())
    if contract["schema"] == "v3":
        if keys != V3_KEYS:
            raise PacketError(f"{contract['fold']}: packet arrays {sorted(keys)} != {sorted(V3_KEYS)}")
    elif keys not in (LEGACY_KEYS, V3_KEYS):
        raise PacketError(f"{contract['fold']}: legacy packet arrays {sorted(keys)} not recognised")
    want = contract["cand_ids"]
    if "cand_ids" in keys:
        ids = _int1d(z["cand_ids"], "cand_ids")
        if len(np.unique(ids)) != len(ids):
            raise PacketError(f"{contract['fold']}: duplicate candidate ids")
        if len(ids) != len(want):
            raise PacketError(f"{contract['fold']}: {len(ids)} candidate ids, contract has {len(want)} "
                              f"(missing {sorted(set(want.tolist()) - set(ids.tolist()))[:5]}, "
                              f"extra {sorted(set(ids.tolist()) - set(want.tolist()))[:5]})")
        if set(ids.tolist()) != set(want.tolist()):
            raise PacketError(f"{contract['fold']}: candidate ids differ from the fold universe "
                              f"(extra {sorted(set(ids.tolist()) - set(want.tolist()))[:5]})")
        if not np.array_equal(ids, want):
            raise PacketError(f"{contract['fold']}: candidate ids are reordered relative to the contract")
    for split in ("seen", "unseen"):
        s = np.asarray(z[split])
        n = len(contract[f"{split}_truth"])
        if s.ndim != 2 or s.shape != (n, len(want)):
            raise PacketError(f"{contract['fold']}: {split} scores shape {s.shape} != {(n, len(want))}")
        if not np.issubdtype(s.dtype, np.floating):
            raise PacketError(f"{contract['fold']}: {split} scores dtype {s.dtype} is not floating")
        if not np.isfinite(s).all():
            raise PacketError(f"{contract['fold']}: {split} scores contain non-finite values")
        t = _int1d(z[f"{split}_truth"], f"{split}_truth")
        if not np.array_equal(t, contract[f"{split}_truth"]):
            bad = int((t != contract[f"{split}_truth"]).sum()) if t.shape == contract[f"{split}_truth"].shape else "shape"
            raise PacketError(f"{contract['fold']}: {split}_truth differs from the benchmark row lists ({bad} mismatches)")


def row_config(row: dict) -> dict:
    """Configuration identity of an experiment row (fields that determine its scores)."""
    return {"backbones": list(row["backbones"]), "recipe": row["recipe"], "cfg": row.get("cfg", {}) or {},
            "tta_hflip": bool(row.get("tta_hflip", False))}


def config_sha256(row: dict) -> str:
    return sha256_json(row_config(row))


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_state(root) -> dict:
    def run(*args):
        return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
    try:
        return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain", "--untracked-files=no"))}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def write_manifest(row_dir, row: dict, provenance: dict, ident: dict, packets: dict) -> Path:
    """Provenance record written next to NEW packets (run_row with ``provenance``). Never retrofitted."""
    m = {"schema": "r2-packet-manifest-v1", "row_id": row["id"], "config": row_config(row),
         "config_sha256": config_sha256(row), "feature_identity": ident, **provenance, "packets": packets}
    p = Path(row_dir) / "manifest.json"
    p.write_text(json.dumps(m, indent=1, sort_keys=True))
    return p


# ---------------------------------------------------------------------------------------------
# Selection-time provenance: declared matrices, run-time summaries, packet copies, registry
# ---------------------------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
MATRIX_DIR = Path(__file__).resolve().parent / "matrices"


def tree_rows(tree: dict, start: str) -> list[str]:
    rows = [start]
    for parent, children in tree.items():
        for r in [parent, *children]:
            if r not in rows:
                rows.append(r)
    return rows


def declared_matrices(entry: dict) -> dict:
    """{path: sha256} declared by a protocol entry ("matrix" or "matrices"); every file hash is verified."""
    decl = entry.get("matrices") or entry.get("matrix")
    if not isinstance(decl, dict) or not decl:
        raise PacketError("protocol entry declares no matrix hashes")
    for rel, want in decl.items():
        got = file_sha256(ROOT / rel)
        if got != want:
            raise PacketError(f"matrix {rel}: sha256 {got} != declared {want}")
    return dict(decl)


def declared_rows(entry: dict, rows: list[str]) -> dict:
    """Configuration identity for each selected row, from the protocol's declared (hash-verified) matrices.

    A row must appear in at least one declared matrix; if it appears in several, its configuration must
    be identical in all of them. Matrix rows outside the selection tree are allowed (screening rows)."""
    decl = declared_matrices(entry)
    found: dict[str, list] = {}
    for rel in decl:
        for r in json.loads((ROOT / rel).read_text()):
            found.setdefault(r["id"], []).append((rel, r))
    out = {}
    for rid in rows:
        if rid not in found:
            raise PacketError(f"selected row {rid!r} is not declared in any protocol matrix {sorted(decl)}")
        hashes = {config_sha256(r) for _, r in found[rid]}
        if len(hashes) != 1:
            raise PacketError(f"row {rid!r} has conflicting configurations across declared matrices")
        out[rid] = {"config": row_config(found[rid][0][1]), "config_sha256": hashes.pop(),
                    "declared_in": [rel for rel, _ in found[rid]]}
    return out


def committed_matrix_index() -> dict:
    """sha256 -> {path, rows by id} for every committed matrix file."""
    idx = {}
    for p in sorted(MATRIX_DIR.glob("*.json")):
        idx[file_sha256(p)] = {"path": str(p.relative_to(ROOT)), "rows": {r["id"]: r for r in json.loads(p.read_text())}}
    return idx


def runtime_summary(stage_dir: Path, rid: str, want: dict, matrix_index: dict) -> dict:
    """Check the run-time record (summaries.jsonl, written when the packets were produced) for ``rid``:
    it must name a committed matrix by hash whose row ``rid`` has the declared configuration, and its own
    recorded backbones / recipe / cfg must equal the declared ones."""
    path = Path(stage_dir) / "summaries.jsonl"
    if not path.exists():
        raise PacketError(f"{stage_dir}: no summaries.jsonl run-time record")
    recs = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    recs = [s for s in recs if s["id"] == rid]
    if len(recs) != 1:
        raise PacketError(f"{path}: expected exactly one record for {rid!r}, found {len(recs)}")
    s = recs[0]
    m = matrix_index.get(s.get("matrix_sha256"))
    if m is None:
        raise PacketError(f"{path}: {rid!r} was produced from matrix {s.get('matrix_sha256')} which is not committed")
    if rid not in m["rows"] or config_sha256(m["rows"][rid]) != want["config_sha256"]:
        raise PacketError(f"{path}: {rid!r} was produced from {m['path']} with a different configuration")
    rec = {"backbones": list(s["backbones"]), "recipe": s["recipe"], "cfg": s.get("cfg", {}) or {}}
    if rec != {k: want["config"][k] for k in ("backbones", "recipe", "cfg")}:
        raise PacketError(f"{path}: recorded configuration of {rid!r} differs from the declared one")
    return {"summaries": str(path), "matrix": m["path"], "matrix_sha256": s["matrix_sha256"],
            "feature_identity": s.get("feature_identity"), "finished_utc": s.get("finished_utc")}


def resolve_packets(rid: str, folds: list[str], score_dirs: list) -> dict:
    """All copies of ``rid``'s packets across ``score_dirs``. Every fold must exist in at least one dir, and
    all copies of a fold must be byte-identical (conflicting copies are rejected, never silently shadowed)."""
    dirs = [Path(d) / rid for d in score_dirs if (Path(d) / rid).is_dir()]
    if not dirs:
        raise PacketError(f"no score directory contains row {rid!r}")
    out = {}
    for fn in folds:
        copies = [d / f"{fn}_scores.npz" for d in dirs if (d / f"{fn}_scores.npz").exists()]
        if not copies:
            raise PacketError(f"{rid}/{fn}: packet missing")
        hashes = {str(c): file_sha256(c) for c in copies}
        if len(set(hashes.values())) != 1:
            raise PacketError(f"{rid}/{fn}: conflicting packet copies {hashes}")
        out[fn] = {"path": copies[0], "sha256": next(iter(hashes.values())), "copies": [str(c) for c in copies]}
    return out
