"""Image-lineage grouping: exact byte duplicates plus bounded perceptual near-duplicates.

Groups are connected components of the relation
    sha256 equal  OR  Hamming(dHash64) <= threshold
computed over training-source images only. A group is the unit that must never be
split across fitting and any validation fold.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image


def dhash64(path) -> int:
    """64-bit difference hash of a 9x8 greyscale thumbnail (decode: PIL RGB, no EXIF transpose)."""
    with Image.open(path) as im:
        g = im.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    a = np.asarray(g, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    return int(np.packbits(bits).view(">u8")[0])


def image_digest(path) -> tuple[str, int]:
    data = Path(path).read_bytes()
    return hashlib.sha256(data).hexdigest(), dhash64(path)


_POP8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def hamming_matrix_rows(hashes: np.ndarray, rows: slice) -> np.ndarray:
    x = hashes[rows, None] ^ hashes[None, :]
    return _POP8[x.view(np.uint8)].reshape(x.shape + (8,)).sum(-1, dtype=np.uint16)


class _UnionFind:
    def __init__(self, n):
        self.p = np.arange(n)

    def find(self, i):
        root = i
        while self.p[root] != root:
            root = self.p[root]
        while self.p[i] != root:
            self.p[i], i = root, self.p[i]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def lineage_groups(sha256s, dhashes, threshold: int = 4, chunk: int = 512):
    """Return (group_id per image, list of linked pairs with distance)."""
    n = len(sha256s)
    hashes = np.asarray([np.uint64(h) for h in dhashes], dtype=np.uint64)
    uf = _UnionFind(n)
    pairs = []
    first = {}
    for i, s in enumerate(sha256s):
        if s in first:
            uf.union(first[s], i)
            pairs.append((first[s], i, -1))
        else:
            first[s] = i
    for start in range(0, n, chunk):
        stop = min(n, start + chunk)
        dist = hamming_matrix_rows(hashes, slice(start, stop))
        ii, jj = np.nonzero(dist <= threshold)
        for a, b in zip(ii + start, jj):
            if a < b:
                uf.union(int(a), int(b))
                pairs.append((int(a), int(b), int(dist[a - start, b])))
    roots = np.array([uf.find(i) for i in range(n)])
    _, group = np.unique(roots, return_inverse=True)
    return group.astype(int), pairs
