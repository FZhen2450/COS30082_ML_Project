"""Bounded, data-only packet IO. Never executes archive contents."""
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tempfile
import zipfile
import uuid

from evaluation.reporting import canonical_json

MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_PACKET_BYTES = 192 * 1024 * 1024


def sha256_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def require_sha(value):
    if not isinstance(value, str) or re.fullmatch('[0-9a-f]{64}', value) is None:
        raise ValueError('Expected a lowercase SHA-256 pin')
    return value


def unique_pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError('Duplicate JSON key')
        result[key] = value
    return result


def json_bytes(value):
    return (canonical_json(value)+'\n').encode('utf-8')


def read_json(path, expected_sha256=None):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError('JSON input missing, redirected or oversized')
    raw = path.read_bytes()
    if expected_sha256 is not None and sha256_bytes(raw) != require_sha(expected_sha256):
        raise ValueError('Raw JSON file hash mismatch')
    return decode_json(raw)


def decode_json(raw):
    def finite_float(value):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError('Nonfinite JSON value')
        return result
    return json.loads(raw.decode('utf-8'), object_pairs_hook=unique_pairs,
                      parse_float=finite_float,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON value')))


def relative_path(name):
    if (not isinstance(name, str) or not name or '\\' in name or ':' in name or '\x00' in name
            or PurePosixPath(name).is_absolute()):
        raise ValueError('Unsafe packet path')
    parts = name.split('/')
    reserved = {'CON', 'PRN', 'AUX', 'NUL', *['COM'+str(i) for i in range(1, 10)],
                *['LPT'+str(i) for i in range(1, 10)]}
    if any(part in ('', '.', '..') or part.endswith(('.', ' ')) or part.split('.')[0].upper() in reserved
           or any(ord(c) < 32 or c in '<>"|?*' for c in part) for part in parts):
        raise ValueError('Unsafe packet path component')
    return PurePosixPath(name)


def child(root, name):
    relative_path(name)
    root = Path(root).resolve(strict=True)
    path = root/name
    if any(p.is_symlink() for p in [path, *list(path.parents)[:len(PurePosixPath(name).parts)-1]]):
        raise ValueError('Symlink packet path forbidden')
    if not path.resolve().is_relative_to(root):
        raise ValueError('Packet path escapes its root')
    return path


def inspect_zip(archive):
    infos = archive.infolist()
    if len(infos) > 64:
        raise ValueError('Too many archive members')
    seen, total = set(), 0
    for info in infos:
        relative_path(info.filename)
        normalized = info.filename.casefold()
        mode = info.external_attr >> 16
        if normalized in seen or info.is_dir() or stat.S_IFMT(mode) not in (0, stat.S_IFREG):
            raise ValueError('Duplicate, case-colliding or nonregular archive member')
        seen.add(normalized)
        total += info.file_size
        if (info.flag_bits & 1 or info.file_size > MAX_FILE_BYTES or total > MAX_PACKET_BYTES or
                info.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)):
            raise ValueError('Encrypted or oversized archive')
    return infos


@contextmanager
def packet_directory(root, entry):
    if not isinstance(entry, dict) or entry.get('format') not in ('directory', 'zip'):
        raise ValueError('Explicit directory/zip packet format required')
    required = {'format', 'path', 'receipt_sha256'} | ({'archive_sha256'} if entry['format'] == 'zip' else set())
    if set(entry) != required:
        raise ValueError('Unexpected owner packet locator fields')
    require_sha(entry['receipt_sha256'])
    path = child(root, entry['path'])
    if entry['format'] == 'directory':
        if not path.is_dir():
            raise ValueError('Owner packet directory unavailable')
        yield path
        return
    if not path.is_file() or path.stat().st_size > MAX_PACKET_BYTES:
        raise ValueError('Owner archive unavailable or oversized')
    raw = path.read_bytes()
    if sha256_bytes(raw) != require_sha(entry['archive_sha256']):
        raise ValueError('Archive raw hash mismatch')
    # Work only from the authenticated byte snapshot; no check/reopen race.
    import io
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            infos = inspect_zip(archive)  # inspect every path BEFORE any extraction
            with tempfile.TemporaryDirectory(prefix='wp5_data_only_', dir=root) as temp:
                target = Path(temp)
                for info in infos:
                    output = target/info.filename
                    output.parent.mkdir(parents=True, exist_ok=True)
                    with output.open('xb') as stream:
                        stream.write(archive.read(info))
                yield target
    except (zipfile.BadZipFile, EOFError, NotImplementedError) as error:
        raise ValueError('Invalid owner ZIP archive') from error


def write_new_directory(directory, files, *, completion_name):
    """Write payloads first, then atomically publish a flushed completion marker.

    Publishing uses a same-filesystem hard link, which cannot replace an existing
    marker. Unsupported filesystems fail closed. Interrupted writes retain their
    private partial file without exposing a recognized completion filename.
    """
    directory = Path(directory)
    for name in files:
        relative_path(name)
    if completion_name not in files:
        raise ValueError('Completion file required')
    directory.mkdir(parents=True, exist_ok=False)
    for name in [n for n in sorted(files) if n != completion_name]:
        path = directory/name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            stream.write(files[name]); stream.flush(); os.fsync(stream.fileno())
    completion = directory/completion_name
    completion.parent.mkdir(parents=True, exist_ok=True)
    partial = completion.parent/('.wp5-completion-'+uuid.uuid4().hex+'.partial')
    with partial.open('xb') as stream:
        stream.write(files[completion_name]); stream.flush(); os.fsync(stream.fileno())
    os.link(partial, completion)  # atomic publication with no overwrite
    partial.unlink()
