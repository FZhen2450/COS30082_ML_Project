"""Controlled, development-only image-perturbation robustness for WP5.

The suite measures how a FROZEN model's decisions change when the image it
actually receives is degraded in a controlled, reproducible way. It does not
train, tune, select or modify models and it never opens official-test CSVs.

Two versioned protocols (see docs/WP5_Robustness_2026-11-07.md):

* v1 ``wp5-robustness-suite-v1`` (``run_suite``): corruptions act on the uint8
  array AFTER each model's own resize/crop. Retained unchanged so the archived
  ablation-C v1 run stays reproducible. Because resize/crop differ by model,
  v1 does NOT give different architectures identical corrupted pixels; only
  the declared parameters and noise seeds are shared.
* v2 ``wp5-robustness-suite-v2`` (``run_native_suite``), required for
  cross-model comparison: each image is decoded once by the declared policy,
  corrupted once at native resolution, SHA-256-recorded, and the identical
  array is given to every model's unchanged frozen preprocessing.

Blur sigma is a fraction of the shorter side in both versions. Gaussian noise is
seeded from SHA-256(suite, seed, sample key, condition). Synthetic corruptions
are a sensitivity probe: not field images and not deployment accuracy.
"""
from __future__ import annotations

import hashlib
import io
import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from typing import Callable, Mapping, Protocol

import numpy as np

from evaluation.reporting import fingerprint

SUITE_VERSION = "wp5-robustness-suite-v1"
SUITE_SEED = 20261107
FAMILIES = ("clean", "darken", "brighten", "gaussian_blur", "jpeg", "gaussian_noise")
OFFICIAL_TEST_CSVS = frozenset({"DPD_Small_test_seen.csv", "DPD_Small_test_unseen.csv"})


@dataclass(frozen=True)
class Perturbation:
    """One controlled condition. ``parameter`` meaning depends on ``family``.

    darken/brighten: multiplicative intensity factor (clipped to [0, 255]).
    gaussian_blur:   sigma as a fraction of min(height, width).
    jpeg:            libjpeg quality (4:2:0 chroma subsampling, no optimize).
    gaussian_noise:  standard deviation on the [0, 1] intensity scale.
    """
    family: str
    severity: int
    parameter: float

    def __post_init__(self):
        if self.family not in FAMILIES:
            raise ValueError(f"unknown perturbation family: {self.family}")
        if isinstance(self.severity, bool) or not isinstance(self.severity, int):
            raise TypeError("severity must be an integer")
        if self.family == "clean":
            if self.severity != 0:
                raise ValueError("clean must have severity 0")
        elif self.severity < 1:
            raise ValueError("perturbed conditions require severity >= 1")
        value = float(self.parameter)
        if not math.isfinite(value):
            raise ValueError("parameter must be finite")
        if self.family in ("darken", "brighten") and value <= 0:
            raise ValueError("intensity factor must be positive")
        if self.family == "darken" and value > 1:
            raise ValueError("darken factor must be <= 1")
        if self.family == "brighten" and value < 1:
            raise ValueError("brighten factor must be >= 1")
        if self.family in ("gaussian_blur", "gaussian_noise") and value < 0:
            raise ValueError("sigma must be nonnegative")
        if self.family == "jpeg" and not (1 <= value <= 100 and value == int(value)):
            raise ValueError("jpeg quality must be an integer in [1, 100]")
        object.__setattr__(self, "parameter", value)

    @property
    def name(self):
        return "clean" if self.family == "clean" else f"{self.family}_s{self.severity}"


def _suite(spec):
    rows = [Perturbation("clean", 0, 0.0)]
    for family, values in spec:
        rows.extend(Perturbation(family, i, v) for i, v in enumerate(values, start=1))
    return tuple(rows)


# Prespecified before any model was run; do not tune severities on results.
DEFAULT_SUITE = _suite((
    ("darken", (0.7, 0.5, 0.3)),
    ("brighten", (1.3, 1.6, 2.0)),
    ("gaussian_blur", (0.004, 0.008, 0.016)),
    ("jpeg", (40, 20, 10)),
    ("gaussian_noise", (0.02, 0.05, 0.10)),
))


def validate_suite(suite):
    suite = tuple(suite)
    if not suite or suite[0] != Perturbation("clean", 0, 0.0):
        raise ValueError("suite must start with the clean reference condition")
    names = [p.name for p in suite]
    if len(set(names)) != len(names):
        raise ValueError("duplicate condition names")
    return suite


def suite_fingerprint(suite=DEFAULT_SUITE, *, seed=SUITE_SEED):
    suite = validate_suite(suite)
    return fingerprint({"version": SUITE_VERSION, "seed": int(seed),
                        "conditions": [asdict(p) for p in suite]})


# --------------------------------------------------------------------------- #
# Perturbations on uint8 HxWx3 arrays
# --------------------------------------------------------------------------- #

def _check_image(image):
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
        raise TypeError("image must be a uint8 numpy array")
    if image.ndim != 3 or image.shape[2] != 3 or 0 in image.shape:
        raise ValueError("image must have shape [H, W, 3]")
    return image


def _gaussian_kernel(sigma):
    radius = max(1, int(math.ceil(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    return kernel / kernel.sum(), radius


def _blur(image, sigma):
    kernel, radius = _gaussian_kernel(sigma)
    out = image.astype(np.float64)
    for axis in (0, 1):
        pad = [(0, 0)] * 3
        pad[axis] = (radius, radius)
        padded = np.pad(out, pad, mode="reflect" if out.shape[axis] > radius else "edge")
        acc = np.zeros_like(out)
        for offset, weight in enumerate(kernel):
            index = [slice(None)] * 3
            index[axis] = slice(offset, offset + out.shape[axis])
            acc += weight * padded[tuple(index)]
        out = acc
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def _jpeg(image, quality):
    from PIL import Image
    buffer = io.BytesIO()
    Image.fromarray(image, mode="RGB").save(buffer, format="JPEG", quality=int(quality),
                                            subsampling=2, optimize=False)
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        return np.asarray(decoded.convert("RGB"), dtype=np.uint8).copy()


def noise_seed(sample_key, condition, *, seed=SUITE_SEED):
    """Deterministic 64-bit seed, identical across models for one image/condition."""
    if not isinstance(sample_key, str) or not sample_key:
        raise ValueError("sample_key must be a nonempty string")
    digest = hashlib.sha256(f"{SUITE_VERSION}|{int(seed)}|{sample_key}|{condition}".encode("utf-8"))
    return int.from_bytes(digest.digest()[:8], "big")


def apply_perturbation(image, perturbation, sample_key, *, seed=SUITE_SEED):
    """Return a NEW uint8 array; the input is never modified."""
    image = _check_image(image)
    p = perturbation
    if p.family == "clean":
        return image.copy()
    if p.family in ("darken", "brighten"):
        if p.parameter == 1.0:
            return image.copy()
        return np.clip(np.rint(image.astype(np.float64) * p.parameter), 0, 255).astype(np.uint8)
    if p.family == "gaussian_blur":
        sigma = p.parameter * min(image.shape[:2])
        return image.copy() if sigma == 0 else _blur(image, sigma)
    if p.family == "jpeg":
        return _jpeg(image, p.parameter)
    if p.family == "gaussian_noise":
        if p.parameter == 0:
            return image.copy()
        rng = np.random.default_rng(noise_seed(sample_key, p.name, seed=seed))
        noisy = image.astype(np.float64) + rng.normal(0.0, p.parameter * 255.0, size=image.shape)
        return np.clip(np.rint(noisy), 0, 255).astype(np.uint8)
    raise AssertionError("unreachable")


# --------------------------------------------------------------------------- #
# Development-only guards
# --------------------------------------------------------------------------- #

def require_development_samples(sample_keys, allowed_keys, *, sources=()):
    """Reject official-test sources and any sample outside the approved fold.

    ``allowed_keys`` must come from a training-source development manifest
    (e.g. WP1 ``development_seed42_wp1_v1.json``); official CSVs are never read.
    """
    for source in sources:
        if str(source).replace("\\", "/").rsplit("/", 1)[-1] in OFFICIAL_TEST_CSVS:
            raise PermissionError(f"official-test source is not permitted: {source}")
    keys = list(sample_keys)
    if not keys:
        raise ValueError("no samples supplied")
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate sample keys")
    allowed = set(allowed_keys)
    outside = [k for k in keys if k not in allowed]
    if outside:
        raise PermissionError(f"{len(outside)} samples are outside the approved development fold, "
                              f"e.g. {outside[0]!r}")
    return keys


# --------------------------------------------------------------------------- #
# Model-agnostic runner
# --------------------------------------------------------------------------- #

class RobustnessAdapter(Protocol):
    """A frozen model split at the point where perturbations are applied.

    ``geometric(sample_key)`` loads one image and applies ONLY the model's own
    resize/crop, returning uint8 [H, W, 3]. ``forward(batch)`` receives uint8
    [B, H, W, 3] and performs tensor conversion, normalization, the forward
    pass and the model's canonical scorer, returning at least ``scores`` [B, K]
    over the declared candidate columns. Optional extra arrays (for example
    ``crop_logits``/``disease_logits``) are carried through unchanged.
    """

    def geometric(self, sample_key: str) -> np.ndarray: ...

    def forward(self, batch: np.ndarray) -> Mapping[str, np.ndarray]: ...


def run_suite(adapter, sample_keys, *, suite=DEFAULT_SUITE, batch_size=4, workers=4,
              seed=SUITE_SEED, progress: Callable[[int, int], None] | None = None):
    """Evaluate every condition on every sample. Returns {condition: {key: array}}.

    Batches contain the same consecutive samples for every condition, so the
    clean condition reproduces the model's ordinary fixed-batch inference.
    """
    suite = validate_suite(suite)
    keys = list(sample_keys)
    if not keys or len(set(keys)) != len(keys):
        raise ValueError("sample_keys must be nonempty and unique")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    collected = {p.name: {} for p in suite}
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        for start in range(0, len(keys), batch_size):
            chunk = keys[start:start + batch_size]
            bases = list(pool.map(adapter.geometric, chunk))
            shapes = {_check_image(b).shape for b in bases}
            if len(shapes) != 1:
                raise ValueError("geometric preprocessing must return a fixed input shape")
            for p in suite:
                images = list(pool.map(lambda kb, p=p: apply_perturbation(kb[1], p, kb[0], seed=seed),
                                       zip(chunk, bases)))
                outputs = adapter.forward(np.stack(images))
                if "scores" not in outputs:
                    raise ValueError("adapter.forward must return 'scores'")
                for name, value in outputs.items():
                    value = np.asarray(value)
                    if value.shape[0] != len(chunk):
                        raise ValueError(f"output {name!r} batch dimension mismatch")
                    collected[p.name].setdefault(name, []).append(value)
            if progress is not None:
                progress(min(start + batch_size, len(keys)), len(keys))
    return {cond: {name: np.concatenate(parts) for name, parts in arrays.items()}
            for cond, arrays in collected.items()}


# --------------------------------------------------------------------------- #
# Outcomes and paired statistics
# --------------------------------------------------------------------------- #

def rank_outcomes(scores, candidate_ids, truth_ids):
    """Stable descending ranking over explicit global-ID columns.

    Ties keep column order, matching ``evaluation.metrics.topk_accuracy``.
    """
    scores = np.asarray(scores, dtype=np.float64)
    ids = np.asarray([int(v) for v in candidate_ids], dtype=np.int64)
    truth = np.asarray([int(v) for v in truth_ids], dtype=np.int64)
    if scores.ndim != 2 or scores.shape[1] != len(ids) or scores.shape[0] != len(truth):
        raise ValueError("scores must be [N, K] matching truth and candidate ids")
    if len(set(ids.tolist())) != len(ids):
        raise ValueError("candidate ids must be unique")
    if not np.isfinite(scores).all():
        raise ValueError("scores contain NaN or infinity")
    if not set(truth.tolist()) <= set(ids.tolist()):
        raise ValueError("ground truth absent from candidate universe")
    ranked = ids[np.argsort(-scores, axis=1, kind="stable")]
    truth_rank = np.argmax(ranked == truth[:, None], axis=1) + 1
    k = min(5, len(ids))
    return {"top1_id": ranked[:, 0], "top5_ids": ranked[:, :k], "truth_rank": truth_rank,
            "top1": truth_rank == 1, "top5": truth_rank <= k}


def mcnemar_exact(b, c):
    """Two-sided exact McNemar p-value from discordant counts (exact integers)."""
    b, c = int(b), int(c)
    if b < 0 or c < 0:
        raise ValueError("discordant counts must be nonnegative")
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1))
    return min(1.0, 2 * tail / 2 ** n)


def cluster_bootstrap_delta(reference, perturbed, clusters, *, replicates=10000, seed=SUITE_SEED,
                            level=0.95):
    """Percentile CI for mean(perturbed) - mean(reference), resampling whole clusters.

    Images of the same composition are not independent; resampling compositions
    (not images) keeps that dependence. Returns (estimate, low, high).
    """
    ref = np.asarray(reference, dtype=np.float64)
    per = np.asarray(perturbed, dtype=np.float64)
    groups = np.asarray(clusters)
    if not (ref.shape == per.shape == groups.shape) or ref.ndim != 1 or ref.size == 0:
        raise ValueError("reference, perturbed and clusters must be equal-length 1-D arrays")
    labels, inverse = np.unique(groups, return_inverse=True)
    diff_sum = np.bincount(inverse, weights=per - ref, minlength=len(labels))
    counts = np.bincount(inverse, minlength=len(labels)).astype(np.float64)
    estimate = float(diff_sum.sum() / counts.sum())
    if len(labels) < 2:
        return estimate, None, None
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(labels), size=(int(replicates), len(labels)))
    stats = diff_sum[draws].sum(axis=1) / counts[draws].sum(axis=1)
    alpha = (1 - level) / 2
    low, high = np.quantile(stats, [alpha, 1 - alpha])
    return estimate, float(low), float(high)


def _rate(mask, denominator_mask=None):
    mask = np.asarray(mask, dtype=bool)
    if denominator_mask is not None:
        mask = mask[np.asarray(denominator_mask, dtype=bool)]
    return {"hits": int(mask.sum()), "n": int(mask.size),
            "rate": (float(mask.mean()) if mask.size else None)}


def summarize_conditions(results, *, candidate_ids, truth_ids, clusters, healthy_ids=(),
                         head_truth=None, replicates=10000, seed=SUITE_SEED):
    """Per-condition accuracy, stability, paired flips and cluster-bootstrap CIs.

    ``results`` is the output of :func:`run_suite`. ``healthy_ids`` are global
    composition IDs whose disease is healthy (explicit, never inferred from
    names). ``head_truth`` optionally maps ``crop_logits``/``disease_logits``
    output names to integer truth arrays for separate head analysis.
    """
    if "clean" not in results:
        raise ValueError("results must include the clean reference")
    truth = np.asarray([int(v) for v in truth_ids], dtype=np.int64)
    clusters = np.asarray(clusters)
    healthy_list = sorted({int(v) for v in healthy_ids})
    healthy = np.isin(truth, np.asarray(healthy_list, dtype=np.int64))
    outcomes = {cond: rank_outcomes(arr["scores"], candidate_ids, truth) for cond, arr in results.items()}
    clean = outcomes["clean"]
    clean_pred_healthy = np.isin(clean["top1_id"], healthy_list)
    head_truth = dict(head_truth or {})
    clean_heads = {name: np.asarray(results["clean"][name]).argmax(axis=1) for name in head_truth}
    rows = []
    for cond, out in outcomes.items():
        pred_healthy = np.isin(out["top1_id"], healthy_list)
        row = {"condition": cond, "n": int(truth.size),
               "top1": _rate(out["top1"]), "top5": _rate(out["top5"]),
               "top1_consistency_with_clean": _rate(out["top1_id"] == clean["top1_id"]),
               "mean_truth_rank": float(out["truth_rank"].mean()),
               "false_healthy_among_diseased": _rate(pred_healthy, ~healthy),
               "predicted_healthy_among_healthy": _rate(pred_healthy, healthy)}
        for metric in ("top1", "top5"):
            b = int((clean[metric] & ~out[metric]).sum())
            c = int((~clean[metric] & out[metric]).sum())
            est, low, high = cluster_bootstrap_delta(clean[metric], out[metric], clusters,
                                                     replicates=replicates, seed=seed)
            row[f"{metric}_paired"] = {
                "lost": b, "gained": c, "delta": est, "delta_ci95_cluster": [low, high],
                "mcnemar_exact_p": mcnemar_exact(b, c),
                "delta_without_largest_cluster":
                    largest_cluster_sensitivity(clean[metric], out[metric], clusters)["delta_without_largest"]}
        diseased = ~healthy
        if healthy_list and diseased.any():
            est, low, high = cluster_bootstrap_delta(clean_pred_healthy[diseased], pred_healthy[diseased],
                                                     clusters[diseased], replicates=replicates, seed=seed)
            row["false_healthy_delta"] = {
                "delta": est, "delta_ci95_cluster": [low, high],
                "delta_without_largest_cluster": largest_cluster_sensitivity(
                    clean_pred_healthy[diseased], pred_healthy[diseased],
                    clusters[diseased])["delta_without_largest"]}
        for name, head_t in head_truth.items():
            pred = np.asarray(results[cond][name]).argmax(axis=1)
            row[f"{name}_top1"] = _rate(pred == np.asarray(head_t))
            row[f"{name}_consistency_with_clean"] = _rate(pred == clean_heads[name])
        rows.append(row)
    return rows


def family_means(rows, suite=DEFAULT_SUITE, metric="top1"):
    """Mean rate over the severities of each family, beside the clean value."""
    by_name = {r["condition"]: r for r in rows}
    clean = by_name["clean"][metric]["rate"]
    out = []
    for family in dict.fromkeys(p.family for p in suite if p.family != "clean"):
        values = [by_name[p.name][metric]["rate"] for p in suite if p.family == family]
        mean = float(np.mean(values))
        out.append({"family": family, f"clean_{metric}": clean, f"mean_{metric}": mean,
                    "mean_delta": mean - clean,
                    "relative_retention": (mean / clean if clean else None)})
    return out


def log_margin(logits):
    """Top-1 minus top-2 log-softmax value per row (equal to the logit gap)."""
    x = np.sort(np.asarray(logits, dtype=np.float64), axis=1)
    return x[:, -1] - x[:, -2]


def margin_matched_head_flips(clean_a, perturbed_a, clean_b, perturbed_b, clusters, *, bins=5,
                              replicates=2000, seed=SUITE_SEED):
    """Compare decision-flip rates of two heads after matching clean confidence.

    A head with smaller clean margins flips more easily regardless of what
    features it uses. Margins from both heads are pooled to define common
    quantile bins; within each bin the flip-rate difference (b - a) is taken,
    then averaged with weights equal to the pooled bin count. The CI resamples
    whole clusters (compositions). Returns a JSON-ready dictionary.
    """
    ca, pa, cb, pb = (np.asarray(v, dtype=np.float64) for v in (clean_a, perturbed_a, clean_b, perturbed_b))
    groups = np.asarray(clusters)
    n = ca.shape[0]
    if not (pa.shape == ca.shape and pb.shape == cb.shape and cb.shape[0] == n and groups.shape == (n,)):
        raise ValueError("head arrays and clusters must share the sample dimension")
    flip_a = ca.argmax(1) != pa.argmax(1)
    flip_b = cb.argmax(1) != pb.argmax(1)
    ma, mb = log_margin(ca), log_margin(cb)
    edges = np.quantile(np.concatenate([ma, mb]), np.linspace(0, 1, bins + 1)[1:-1])
    bin_a = np.searchsorted(edges, ma, side="right")
    bin_b = np.searchsorted(edges, mb, side="right")

    def statistic(index):
        total = weight_sum = 0.0
        per_bin = []
        fa_i, fb_i, ba_i, bb_i = flip_a[index], flip_b[index], bin_a[index], bin_b[index]
        for k in range(bins):
            sa, sb = fa_i[ba_i == k], fb_i[bb_i == k]
            if sa.size and sb.size:
                diff = sb.mean() - sa.mean()
                total += (sa.size + sb.size) * diff
                weight_sum += sa.size + sb.size
                per_bin.append((k, int(sa.size), float(sa.mean()), int(sb.size), float(sb.mean())))
            else:
                per_bin.append((k, int(sa.size), None, int(sb.size), None))
        return (total / weight_sum if weight_sum else None), per_bin

    estimate, per_bin = statistic(np.arange(n))
    labels, inverse = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inverse == g) for g in range(len(labels))]
    rng = np.random.default_rng(seed)
    stats = []
    if len(labels) > 1:
        for _ in range(int(replicates)):
            pick = rng.integers(0, len(labels), size=len(labels))
            value, _ = statistic(np.concatenate([members[g] for g in pick]))
            if value is not None:
                stats.append(value)
    low, high = (np.quantile(stats, [0.025, 0.975]).tolist() if len(stats) > 1 else (None, None))
    return {"flip_rate_a": float(flip_a.mean()), "flip_rate_b": float(flip_b.mean()),
            "raw_difference_b_minus_a": float(flip_b.mean() - flip_a.mean()),
            "margin_matched_difference_b_minus_a": estimate,
            "ci95_cluster": [low, high], "bins": bins, "bin_edges": edges.tolist(),
            "per_bin": [{"bin": k, "n_a": na, "flip_a": fa, "n_b": nb, "flip_b": fb}
                        for k, na, fa, nb, fb in per_bin],
            "valid_bootstrap_replicates": len(stats)}


def cluster_design(clusters):
    """How much independent information the clusters carry.

    The Kish effective number of clusters, (sum n)^2 / sum n^2, falls far below
    the raw count when one composition dominates. Percentile cluster-bootstrap
    intervals are approximate and tend to under-cover when it is small.
    """
    _, counts = np.unique(np.asarray(clusters), return_counts=True)
    counts = np.sort(counts)[::-1].astype(np.int64)
    total = int(counts.sum())
    return {"clusters": int(counts.size), "images": total,
            "kish_effective_clusters": float(total ** 2 / (counts.astype(np.float64) ** 2).sum()),
            "largest_cluster_images": int(counts[0]), "largest_cluster_share": float(counts[0] / total),
            "clusters_below_5_images": int((counts < 5).sum())}


def largest_cluster_sensitivity(reference, perturbed, clusters):
    """Paired mean difference recomputed without the single largest cluster."""
    ref = np.asarray(reference, dtype=np.float64)
    per = np.asarray(perturbed, dtype=np.float64)
    groups = np.asarray(clusters)
    labels, counts = np.unique(groups, return_counts=True)
    largest = labels[np.argmax(counts)]
    keep = groups != largest
    return {"excluded_cluster": largest.item() if hasattr(largest, "item") else largest,
            "excluded_images": int((~keep).sum()),
            "delta_without_largest": (float((per[keep] - ref[keep]).mean()) if keep.any() else None)}


def environment_versions():
    import PIL
    from PIL import features
    return {"numpy": np.__version__, "pillow": PIL.__version__,
            "libjpeg": features.version("jpg")}


# --------------------------------------------------------------------------- #
# v2: decode once, corrupt native RGB once, identical pixels for every model
# --------------------------------------------------------------------------- #

SUITE_V2_VERSION = "wp5-robustness-suite-v2"
DECODE_POLICY = "PIL.Image.open(path).convert('RGB'); no EXIF transpose; alpha discarded by convert"
APPLICATION_POINT_V2 = "native decoded RGB, before each model's unchanged frozen preprocessing"
BLUR_OPERATOR_V2 = "Pillow ImageFilter.GaussianBlur(radius=sigma_px), sigma_px = parameter * min(H, W)"
NATIVE_SUITE = DEFAULT_SUITE  # identical declared families, severities and parameters


def suite_v2_fingerprint(suite=NATIVE_SUITE, *, seed=SUITE_SEED):
    suite = validate_suite(suite)
    return fingerprint({"version": SUITE_V2_VERSION, "seed": int(seed),
                        "application_point": APPLICATION_POINT_V2, "decode_policy": DECODE_POLICY,
                        "blur_operator": BLUR_OPERATOR_V2,
                        "conditions": [asdict(p) for p in suite]})


def decode_rgb(path):
    """Decode exactly as the WP1, WP2, WP3 and ablation-C evaluation pipelines do.

    No EXIF transpose: those pipelines call ``Image.open(path).convert("RGB")``.
    (The WP4 GUI transposes EXIF orientation, so GUI and evaluation inputs differ
    for EXIF-rotated photos; that is a GUI property, not changed here.)
    """
    from PIL import Image
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()


def apply_native_perturbation(image, perturbation, sample_key, *, seed=SUITE_SEED):
    """v2 corruption of a native-resolution uint8 RGB array; returns a new array.

    Identical to v1 except blur, which uses Pillow's Gaussian blur so that
    multi-megapixel images remain tractable. Identity parameters are exact.
    """
    image = _check_image(image)
    p = perturbation
    if p.family == "gaussian_blur":
        sigma = p.parameter * min(image.shape[:2])
        if sigma == 0:
            return image.copy()
        from PIL import Image, ImageFilter
        blurred = Image.fromarray(image, mode="RGB").filter(ImageFilter.GaussianBlur(radius=sigma))
        return np.asarray(blurred, dtype=np.uint8).copy()
    if p.family in ("darken", "brighten", "gaussian_noise") and not (
            p.parameter == (1.0 if p.family != "gaussian_noise" else 0.0)):
        return _blockwise(image, p, sample_key, seed)
    return apply_perturbation(image, p, sample_key, seed=seed)


def _blockwise(image, p, sample_key, seed, block_rows=256):
    """Bit-identical to ``apply_perturbation`` but bounded memory on multi-megapixel images.

    Row blocks are processed in C order; the noise Generator fills sequentially,
    so blockwise draws equal one full-size draw.
    """
    out = np.empty_like(image)
    rng = (np.random.default_rng(noise_seed(sample_key, p.name, seed=seed))
           if p.family == "gaussian_noise" else None)
    for start in range(0, image.shape[0], block_rows):
        block = image[start:start + block_rows].astype(np.float64)
        if rng is None:
            block *= p.parameter
        else:
            block += rng.normal(0.0, p.parameter * 255.0, size=block.shape)
        out[start:start + block_rows] = np.clip(np.rint(block), 0, 255).astype(np.uint8)
    return out


def pixel_digest(image):
    """SHA-256 over shape and bytes: proves two models received identical pixels."""
    image = _check_image(image)
    digest = hashlib.sha256(f"{image.shape[0]}x{image.shape[1]}x3|".encode("ascii"))
    digest.update(np.ascontiguousarray(image).tobytes())
    return digest.hexdigest()


def _lcm(values):
    out = 1
    for v in values:
        out = out * int(v) // math.gcd(out, int(v))
    return out


def run_native_suite(predictors, sample_keys, *, load=decode_rgb, suite=NATIVE_SUITE, batch_sizes=None,
                     outputs=None, workers=4, seed=SUITE_SEED,
                     progress: Callable[[int, int], None] | None = None):
    """Run every predictor on identical corrupted native pixels (protocol v2).

    ``predictors`` maps a name to an object whose ``predict(images)`` takes a list
    of PIL RGB images and applies that model's own frozen preprocessing and
    canonical scorer, returning a mapping that includes ``scores`` [B, K].
    ``batch_sizes`` maps names to each model's historical inference batch size;
    chunks span their least common multiple so batch boundaries keep the
    sequential order. Each predictor receives freshly built PIL images, so a
    predictor cannot alter what another predictor sees. ``outputs`` maps names
    to the per-image arrays to keep (default ``("scores",)``); each must have
    the batch as its first dimension.

    Returns {"pixel_sha256": {condition: [hex, ...]}, name: {condition: {key: array}}}.
    """
    from PIL import Image
    suite = validate_suite(suite)
    keys = list(sample_keys)
    if not keys or len(set(keys)) != len(keys):
        raise ValueError("sample_keys must be nonempty and unique")
    if not predictors:
        raise ValueError("at least one predictor is required")
    wanted = {name: tuple((outputs or {}).get(name, ("scores",))) for name in predictors}
    if any("scores" not in w for w in wanted.values()):
        raise ValueError("every predictor must collect 'scores'")
    sizes = {name: int((batch_sizes or {}).get(name, 1)) for name in predictors}
    if any(size < 1 for size in sizes.values()):
        raise ValueError("batch sizes must be positive")
    chunk_size = _lcm(sizes.values())
    digests = {p.name: [] for p in suite}
    collected = {name: {p.name: {} for p in suite} for name in predictors}
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        for start in range(0, len(keys), chunk_size):
            chunk = keys[start:start + chunk_size]
            natives = list(pool.map(load, chunk))
            frozen = [n.copy() for n in natives]
            for p in suite:
                corrupted = list(pool.map(lambda kn, p=p: apply_native_perturbation(kn[1], p, kn[0], seed=seed),
                                          zip(chunk, natives)))
                digests[p.name].extend(pixel_digest(c) for c in corrupted)
                for name, predictor in predictors.items():
                    for b in range(0, len(chunk), sizes[name]):
                        images = [Image.fromarray(c, mode="RGB") for c in corrupted[b:b + sizes[name]]]
                        returned = predictor.predict(images)
                        for out_name in wanted[name]:
                            if out_name not in returned:
                                raise ValueError(f"predictor {name!r} did not return {out_name!r}")
                            value = returned[out_name]
                            if hasattr(value, "detach"):
                                value = value.detach().cpu().numpy()
                            value = np.array(value, copy=True)
                            if value.ndim == 0 or value.shape[0] != len(images):
                                raise ValueError(f"{name!r} output {out_name!r} batch dimension mismatch")
                            collected[name][p.name].setdefault(out_name, []).append(value)
            if any(not np.array_equal(a, b) for a, b in zip(natives, frozen)):
                raise AssertionError("decoded source arrays were modified during corruption")
            if progress is not None:
                progress(min(start + chunk_size, len(keys)), len(keys))
    result = {"pixel_sha256": digests}
    for name, by_condition in collected.items():
        result[name] = {cond: {k: np.concatenate(v) for k, v in arrays.items()}
                        for cond, arrays in by_condition.items()}
    return result


__all__ = [
    "SUITE_VERSION", "SUITE_SEED", "DEFAULT_SUITE", "Perturbation", "validate_suite",
    "suite_fingerprint", "apply_perturbation", "noise_seed", "require_development_samples",
    "RobustnessAdapter", "run_suite", "rank_outcomes", "mcnemar_exact",
    "cluster_bootstrap_delta", "summarize_conditions", "family_means", "log_margin",
    "margin_matched_head_flips", "cluster_design", "largest_cluster_sensitivity",
    "environment_versions", "SUITE_V2_VERSION", "DECODE_POLICY", "NATIVE_SUITE",
    "suite_v2_fingerprint", "decode_rgb", "apply_native_perturbation", "pixel_digest",
    "run_native_suite",
]
