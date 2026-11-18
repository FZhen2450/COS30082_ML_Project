"""Find approved local model files by content without loading any checkpoint."""
import hashlib
import json
import os
from pathlib import Path
import threading

ROOT = Path(__file__).resolve().parents[1]
ASSETS = {
    "crop": ("f0c82ca78153f492078a3a39a11eac1a847cc85eb496cf3b3d4d74db752c7d14",
             "runs/wp1-release-20261106-final/crop/best.pt"),
    "disease": ("f497e19da62b68dfbcfbae58b9a5318a26b1989ba59b7a486223cad0d56e31a5",
                "runs/wp1-release-20261106-final/disease/best.pt"),
    "clip_weights": ("b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836",
                     "runs/wp4-issue13-20261107/private_assets/ViT-L-14.pt"),
    "adapter_checkpoint": ("8835a18b9eb933ae224bd17866f18bc56406e24d8612743f4e7c88848a8135cc",
                           "runs/wp4-issue13-20261107/private_assets/adapter_epoch_001_e84ba1dd.pt"),
}
FIELDS = {"wp1": ("crop", "disease"), "frozen_clip": ("clip_weights",),
          "m6": ("clip_weights", "adapter_checkpoint")}
SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", "dpd_small_dataset",
             ".agents", ".codex", "site-packages"}
_HASH_CACHE = {}
_CACHE_LOCK = threading.Lock()


def default_search_roots():
    """Limit automatic searches to project asset folders and the CLIP cache."""
    return [ROOT / name for name in ("runs", "models", "weights", "checkpoints", "assets", "artifacts")] + [
        Path.home() / ".cache/clip"]


def _file_hash(path, stop):
    before = path.stat()
    identity = (str(path.resolve()), before.st_size, before.st_mtime_ns, before.st_ctime_ns)
    with _CACHE_LOCK:
        cached = _HASH_CACHE.get(identity)
    if cached:
        return cached
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while not stop.is_set():
            block = stream.read(4 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
        else:
            return None
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_size, after.st_mtime_ns, after.st_ctime_ns):
        return None
    value = digest.hexdigest()
    with _CACHE_LOCK:
        _HASH_CACHE[identity] = value
    return value


def discover_weight_files(kind, roots=None, stop=None, final_training_sha=None):
    """Return exact approved hashes; names alone never establish model identity.

    A custom root search has no implicit access to the project's private files.
    Loading still performs the existing runtime, schema and provenance checks.
    """
    required = FIELDS[kind]
    assets = ASSETS
    if kind == 'wp1' and final_training_sha is not None:
        registry = ROOT/'artifacts/models/wp1_full8000_release.json'
        try:
            release = json.loads(registry.read_text(encoding='utf-8'))
            from final_protocol import SOURCE_SHA256
            if (release.get('schema') != 'wp1-full8000-release-v1' or release.get('training_count') != 8000
                    or release.get('source_sha256') != SOURCE_SHA256
                    or release.get('training_manifest_sha256') != final_training_sha):
                raise ValueError('Final WP1 registry does not match the loaded training contract')
            assets = {key: (release['checkpoints'][key]['sha256'], release['checkpoints'][key]['path'])
                      for key in required}
        except (OSError, ValueError, KeyError, TypeError):
            return {'paths': {}, 'missing': list(required), 'warnings': ['Full-8,000 WP1 release is not registered'],
                    'cancelled': False}
    stop = stop if stop is not None else threading.Event()
    found, visited, warnings = {}, set(), []

    def inspect(path):
        if stop.is_set() or len(found) == len(required):
            return
        try:
            resolved = path.resolve(strict=True)
            if resolved in visited or not resolved.is_file():
                return
            visited.add(resolved)
            value = _file_hash(resolved, stop)
            for key in required:
                if key not in found and value == assets[key][0]:
                    found[key] = str(resolved)
        except OSError:
            warnings.append(f"Could not read {path.name}")

    if roots is None:
        for key in required:
            inspect(ROOT / assets[key][1])
        roots = default_search_roots()
    for root in roots:
        root = Path(root)
        if stop.is_set() or len(found) == len(required):
            break
        if not root.is_dir():
            continue
        for directory, folders, files in os.walk(root, followlinks=False,
                onerror=lambda exc: warnings.append(f"Could not search {exc.filename}")):
            folders[:] = sorted(name for name in folders if name.lower() not in SKIP_DIRS
                                and not (Path(directory) / name).is_symlink())
            for name in sorted(files):
                if Path(name).suffix.lower() in (".pt", ".pth"):
                    inspect(Path(directory) / name)
                if stop.is_set() or len(found) == len(required):
                    break
            if stop.is_set() or len(found) == len(required):
                break
    return {"paths": found, "missing": [key for key in required if key not in found],
            "warnings": warnings, "cancelled": stop.is_set()}
