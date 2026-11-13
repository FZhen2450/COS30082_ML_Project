"""Frozen-recipe guard for scoring protected folds (v1 locked L*, v2 outer O*).

A protected stage may run only a recipe registered in ``frozen_recipes.json`` (a committed,
reviewable file). The caller supplies the registered NAME; the guard checks the matrix path,
the matrix file hash, the benchmark semantic hash and the row identities against the
registry. Supplying an arbitrary matrix together with its own freshly computed hash is
therefore rejected.
"""
from __future__ import annotations

import json
from pathlib import Path

from .data import sha256_file

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = Path(__file__).resolve().parent / "frozen_recipes.json"


class FrozenRecipeError(RuntimeError):
    pass


def load_registry(path: Path = REGISTRY) -> dict:
    return json.loads(Path(path).read_text())["recipes"]


def verify_frozen(name: str, matrix_path, bench: dict, registry: dict | None = None) -> dict:
    reg = load_registry() if registry is None else registry
    if name not in reg:
        raise FrozenRecipeError(f"recipe {name!r} is not registered in frozen_recipes.json")
    entry = reg[name]
    given = Path(matrix_path).resolve()
    declared = (ROOT / entry["matrix"]).resolve()
    if given != declared:
        raise FrozenRecipeError(f"matrix path {given} differs from registered {declared}")
    actual = sha256_file(given)
    if actual != entry["matrix_sha256"]:
        raise FrozenRecipeError(f"matrix hash {actual} differs from registered {entry['matrix_sha256']}")
    if bench.get("semantic_sha256") != entry["benchmark_semantic_sha256"]:
        raise FrozenRecipeError("benchmark semantic hash differs from the one this recipe was frozen for")
    ids = [r["id"] for r in json.loads(given.read_text())]
    if ids != entry["row_ids"]:
        raise FrozenRecipeError(f"row ids {ids} differ from registered {entry['row_ids']}")
    return entry


PROTECTED_ROLES = ("locked", "outer")


def select_folds(bench: dict, requested: list[str] | None = None) -> list[str]:
    """Unprotected folds to score. Requesting a protected fold directly is an error."""
    role = {f["name"]: f["role"] for f in bench["folds"]}
    if requested is None:
        return [n for n, r in role.items() if r not in PROTECTED_ROLES]
    unknown = [n for n in requested if n not in role]
    if unknown:
        raise FrozenRecipeError(f"unknown folds {unknown}")
    blocked = [n for n in requested if role[n] in PROTECTED_ROLES]
    if blocked:
        raise FrozenRecipeError(f"protected folds {blocked} may only be scored through a registered frozen recipe")
    return list(requested)
