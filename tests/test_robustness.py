"""Synthetic regressions for the WP5 robustness suite. No images, weights or tests."""
import unittest
from dataclasses import replace

import numpy as np

from evaluation.metrics import topk_accuracy
from evaluation.robustness import (
    DEFAULT_SUITE, NATIVE_SUITE, Perturbation, apply_native_perturbation, apply_perturbation,
    cluster_bootstrap_delta, cluster_design, decode_rgb, family_means, largest_cluster_sensitivity,
    margin_matched_head_flips, mcnemar_exact, noise_seed, pixel_digest, rank_outcomes,
    require_development_samples, run_native_suite, run_suite, suite_fingerprint, suite_v2_fingerprint,
    summarize_conditions, validate_suite,
)

# Pinned so the prespecified suite cannot drift silently after results are seen.
PINNED_SUITE_SHA256 = "7a36aebcc1ec695f41b4656a01b0f5d4eddd90bd71bc150ea61cf566b17f63a3"


def smooth_image(h=64, w=48, seed=0):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w]
    base = 128 + 60 * np.sin(x / 5.0)[..., None] * np.cos(y / 7.0)[..., None]
    img = base + rng.normal(0, 8, size=(h, w, 3))
    return np.clip(np.rint(img), 0, 255).astype(np.uint8)


def stripes(size=128, period=16):
    y, x = np.mgrid[0:size, 0:size]
    img = 128 + 100 * np.sin(2 * np.pi * x / period) * np.sin(2 * np.pi * y / period)
    return np.repeat(np.rint(img).astype(np.uint8)[..., None], 3, axis=2)


def high_frequency_energy(img):
    g = img.astype(np.float64).mean(axis=2)
    return float(np.abs(np.diff(g, axis=0)).mean() + np.abs(np.diff(g, axis=1)).mean())


class PerturbationTests(unittest.TestCase):
    def test_identity_parameters_are_exact_and_return_new_arrays(self):
        img = smooth_image()
        before = img.copy()
        for p in (Perturbation("clean", 0, 0), Perturbation("darken", 1, 1.0),
                  Perturbation("brighten", 1, 1.0), Perturbation("gaussian_blur", 1, 0.0),
                  Perturbation("gaussian_noise", 1, 0.0)):
            out = apply_perturbation(img, p, "a.jpg")
            self.assertTrue(np.array_equal(out, img), p)
            self.assertIsNot(out, img)
            self.assertEqual(out.dtype, np.uint8)
        self.assertTrue(np.array_equal(img, before), "input must never be modified")

    def test_every_default_condition_preserves_shape_dtype_and_input(self):
        img = smooth_image()
        before = img.copy()
        for p in DEFAULT_SUITE:
            out = apply_perturbation(img, p, "a.jpg")
            self.assertEqual(out.shape, img.shape)
            self.assertEqual(out.dtype, np.uint8)
        self.assertTrue(np.array_equal(img, before))

    def test_perturbations_are_deterministic_and_noise_is_keyed_by_sample(self):
        img = smooth_image()
        for p in DEFAULT_SUITE:
            self.assertTrue(np.array_equal(apply_perturbation(img, p, "a.jpg"),
                                           apply_perturbation(img, p, "a.jpg")), p.name)
        noise = Perturbation("gaussian_noise", 2, 0.05)
        self.assertFalse(np.array_equal(apply_perturbation(img, noise, "a.jpg"),
                                        apply_perturbation(img, noise, "b.jpg")))
        self.assertNotEqual(noise_seed("a.jpg", "gaussian_noise_s1"),
                            noise_seed("a.jpg", "gaussian_noise_s2"))

    def test_severities_are_monotonic_within_each_family(self):
        img = smooth_image(96, 96)
        pattern = stripes()
        mid = np.full((32, 32, 3), 128, np.uint8)

        def ladder(family):
            return [p for p in DEFAULT_SUITE if p.family == family]

        dark = [apply_perturbation(img, p, "k").mean() for p in ladder("darken")]
        self.assertTrue(img.mean() > dark[0] > dark[1] > dark[2])
        bright = [apply_perturbation(img, p, "k").mean() for p in ladder("brighten")]
        self.assertTrue(img.mean() < bright[0] < bright[1] < bright[2])
        blur = [high_frequency_energy(apply_perturbation(pattern, p, "k")) for p in ladder("gaussian_blur")]
        self.assertTrue(high_frequency_energy(pattern) > blur[0] > blur[1] > blur[2], blur)
        jpeg = [np.mean((apply_perturbation(img, p, "k").astype(float) - img) ** 2) for p in ladder("jpeg")]
        self.assertTrue(0 < jpeg[0] < jpeg[1] < jpeg[2], jpeg)
        noise = [np.std(apply_perturbation(mid, p, "k").astype(float) - 128) for p in ladder("gaussian_noise")]
        for measured, p in zip(noise, ladder("gaussian_noise")):
            self.assertAlmostEqual(measured, p.parameter * 255, delta=0.12 * p.parameter * 255)

    def test_blur_preserves_constants_and_scales_with_resolution(self):
        flat = np.full((40, 30, 3), 77, np.uint8)
        blur = Perturbation("gaussian_blur", 3, 0.016)
        self.assertTrue(np.array_equal(apply_perturbation(flat, blur, "k"), flat))
        # A 2x larger rendering of the same scene blurred with the same relative
        # sigma should, after 2x2 averaging, closely match the small blurred image.
        small = smooth_image(48, 48, seed=3)
        large = np.repeat(np.repeat(small, 2, axis=0), 2, axis=1)
        p = Perturbation("gaussian_blur", 2, 0.04)
        a = apply_perturbation(small, p, "k").astype(float)
        b = apply_perturbation(large, p, "k").astype(float).reshape(48, 2, 48, 2, 3).mean(axis=(1, 3))
        self.assertLess(np.abs(a - b).mean(), 1.5)

    def test_brighten_clips_and_darken_rounds(self):
        img = np.array([[[200, 100, 0]]], np.uint8)
        self.assertEqual(apply_perturbation(img, Perturbation("brighten", 3, 2.0), "k").tolist(),
                         [[[255, 200, 0]]])
        self.assertEqual(apply_perturbation(img, Perturbation("darken", 2, 0.5), "k").tolist(),
                         [[[100, 50, 0]]])

    def test_invalid_conditions_and_images_are_rejected(self):
        bad = [("blur", 1, 1), ("clean", 1, 0), ("darken", 0, 0.5), ("darken", 1, 1.2),
               ("brighten", 1, 0.8), ("jpeg", 1, 0), ("jpeg", 1, 101), ("jpeg", 1, 50.5),
               ("gaussian_noise", 1, -0.1), ("darken", 1, float("nan"))]
        for family, severity, parameter in bad:
            with self.assertRaises(ValueError, msg=(family, severity, parameter)):
                Perturbation(family, severity, parameter)
        with self.assertRaises(TypeError):
            Perturbation("darken", True, 0.5)
        with self.assertRaises(TypeError):
            apply_perturbation(smooth_image().astype(np.float32), DEFAULT_SUITE[1], "k")
        with self.assertRaises(ValueError):
            apply_perturbation(np.zeros((4, 4), np.uint8), DEFAULT_SUITE[1], "k")


class SuiteTests(unittest.TestCase):
    def test_default_suite_is_prespecified_and_pinned(self):
        self.assertEqual(len(DEFAULT_SUITE), 16)
        self.assertEqual(DEFAULT_SUITE[0].name, "clean")
        self.assertEqual(len({p.name for p in DEFAULT_SUITE}), 16)
        self.assertEqual(suite_fingerprint(), PINNED_SUITE_SHA256)
        changed = list(DEFAULT_SUITE)
        changed[3] = replace(changed[3], parameter=0.31)
        self.assertNotEqual(suite_fingerprint(changed), suite_fingerprint())

    def test_suite_requires_clean_first_and_unique_names(self):
        with self.assertRaises(ValueError):
            validate_suite(DEFAULT_SUITE[1:])
        with self.assertRaises(ValueError):
            validate_suite(DEFAULT_SUITE + (DEFAULT_SUITE[1],))

    def test_development_guard_rejects_official_sources_outsiders_and_duplicates(self):
        allowed = ["a.jpg", "b.jpg", "c.jpg"]
        self.assertEqual(require_development_samples(["b.jpg", "a.jpg"], allowed), ["b.jpg", "a.jpg"])
        for source in ("DPD_Small_test_seen.csv", r"C:\x\DPD_Small_test_unseen.csv"):
            with self.assertRaises(PermissionError):
                require_development_samples(["a.jpg"], allowed, sources=[source])
        with self.assertRaises(PermissionError):
            require_development_samples(["a.jpg", "z.jpg"], allowed)
        with self.assertRaises(ValueError):
            require_development_samples(["a.jpg", "a.jpg"], allowed)
        with self.assertRaises(ValueError):
            require_development_samples([], allowed)


class StatisticsTests(unittest.TestCase):
    def test_rank_outcomes_matches_existing_metric_including_ties(self):
        rng = np.random.default_rng(7)
        ids = [3, 10, 11, 40, 75, 200, 332]
        scores = rng.integers(0, 4, size=(300, len(ids))).astype(float)  # many exact ties
        truth = rng.choice(ids, size=300)
        out = rank_outcomes(scores, ids, truth)
        self.assertAlmostEqual(out["top1"].mean(), topk_accuracy(scores, truth, ids, k=1))
        self.assertAlmostEqual(out["top5"].mean(), topk_accuracy(scores, truth, ids, k=5))
        tie = rank_outcomes([[1.0, 1.0, 0.0]], [9, 4, 2], [4])
        self.assertEqual(int(tie["top1_id"][0]), 9, "ties keep column order")
        self.assertEqual(int(tie["truth_rank"][0]), 2)
        with self.assertRaises(ValueError):
            rank_outcomes([[1.0, 2.0]], [1, 2], [5])
        with self.assertRaises(ValueError):
            rank_outcomes([[1.0, np.nan]], [1, 2], [1])

    def test_mcnemar_exact_known_values(self):
        self.assertEqual(mcnemar_exact(0, 0), 1.0)
        self.assertEqual(mcnemar_exact(0, 5), 0.0625)
        self.assertEqual(mcnemar_exact(1, 9), 2 * (1 + 10) / 1024)
        self.assertEqual(mcnemar_exact(9, 1), mcnemar_exact(1, 9))
        self.assertEqual(mcnemar_exact(5, 5), 1.0)
        self.assertLess(mcnemar_exact(0, 2000), 1e-300)

    def test_cluster_bootstrap_is_exact_for_no_change_and_seeded(self):
        ref = np.array([1, 0, 1, 1, 0, 0], bool)
        clusters = np.array([1, 1, 2, 2, 3, 3])
        self.assertEqual(cluster_bootstrap_delta(ref, ref, clusters), (0.0, 0.0, 0.0))
        per = np.array([0, 0, 1, 1, 0, 1], bool)
        a = cluster_bootstrap_delta(ref, per, clusters, replicates=500)
        self.assertEqual(a, cluster_bootstrap_delta(ref, per, clusters, replicates=500))
        self.assertAlmostEqual(a[0], per.mean() - ref.mean())
        self.assertLessEqual(a[1], a[0])
        self.assertGreaterEqual(a[2], a[0])
        self.assertEqual(cluster_bootstrap_delta(ref, per, np.zeros(6))[1:], (None, None))

    def test_margin_matching_removes_pure_confidence_confound(self):
        # Both heads flip with identical probability at a given clean margin;
        # head B merely has more low-margin samples. Raw flip rates differ,
        # the margin-matched difference must be zero.
        def head(n_low, n_high):
            clean, pert = [], []
            for i in range(n_low):
                c = np.array([0.1, 0.0, -9.0])
                clean.append(c)
                pert.append(np.array([0.0, 1.0, -9.0]) if i % 5 else c)  # 80% flip
            for _ in range(n_high):
                c = np.array([5.0, 0.0, -9.0])
                clean.append(c)
                pert.append(c)  # never flips
            return np.array(clean), np.array(pert)

        ca, pa = head(20, 80)
        cb, pb = head(80, 20)
        clusters = np.arange(100) % 10
        result = margin_matched_head_flips(ca, pa, cb, pb, clusters, bins=2, replicates=200)
        self.assertAlmostEqual(result["flip_rate_a"], 0.16)
        self.assertAlmostEqual(result["flip_rate_b"], 0.64)
        self.assertAlmostEqual(result["margin_matched_difference_b_minus_a"], 0.0)
        # A genuine extra sensitivity in B survives matching.
        pb2 = pb.copy()
        pb2[80:] = np.array([0.0, 6.0, -9.0])
        result2 = margin_matched_head_flips(ca, pa, cb, pb2, clusters, bins=2, replicates=200)
        self.assertAlmostEqual(result2["margin_matched_difference_b_minus_a"], 0.5)
        low, high = result2["ci95_cluster"]
        self.assertLessEqual(low, 0.5)
        self.assertGreaterEqual(high, 0.5)


class FakeAdapter:
    """Scores depend on mean brightness, so darkening changes decisions."""

    def __init__(self, images):
        self.images = images
        self.calls = []

    def geometric(self, key):
        return self.images[key].copy()

    def forward(self, batch):
        self.calls.append(batch.shape)
        if batch.dtype != np.uint8:
            raise AssertionError("adapter must receive uint8 arrays")
        brightness = batch.astype(float).mean(axis=(1, 2, 3)) / 255.0
        scores = np.stack([brightness, 0.5 * np.ones_like(brightness), 1 - brightness], axis=1)
        crop = np.stack([brightness, 1 - brightness], axis=1) * 10
        disease = np.stack([brightness, 0.45 * np.ones_like(brightness)], axis=1) * 10
        return {"scores": scores, "crop_logits": crop, "disease_logits": disease}


LEVELS = [230, 220, 200, 180, 30, 160, 150, 20, 240]


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.keys = [f"img_{i}.jpg" for i in range(len(LEVELS))]
        self.images = {k: np.full((8, 8, 3), v, np.uint8) for k, v in zip(self.keys, LEVELS)}
        self.adapter = FakeAdapter(self.images)
        self.suite = (DEFAULT_SUITE[0], Perturbation("darken", 3, 0.3), Perturbation("jpeg", 1, 40))

    def test_clean_condition_reproduces_direct_inference_with_same_batches(self):
        results = run_suite(self.adapter, self.keys, suite=self.suite, batch_size=4, workers=2)
        direct = FakeAdapter(self.images).forward(np.stack([self.images[k] for k in self.keys]))
        for name in ("scores", "crop_logits", "disease_logits"):
            np.testing.assert_array_equal(results["clean"][name], direct[name])
        self.assertEqual(self.adapter.calls, [(4, 8, 8, 3)] * 6 + [(1, 8, 8, 3)] * 3)
        for key, level in zip(self.keys, LEVELS):
            self.assertTrue((self.images[key] == level).all(), "source images untouched")

    def test_runner_rejects_bad_adapters_and_inputs(self):
        class NoScores(FakeAdapter):
            def forward(self, batch):
                return {"other": np.zeros((len(batch), 2))}

        class WrongBatch(FakeAdapter):
            def forward(self, batch):
                return {"scores": np.zeros((len(batch) + 1, 3))}

        class Ragged(FakeAdapter):
            def geometric(self, key):
                return np.zeros((8 + int(key[4]) % 2, 8, 3), np.uint8)

        for adapter in (NoScores(self.images), WrongBatch(self.images), Ragged(self.images)):
            with self.assertRaises(ValueError):
                run_suite(adapter, self.keys, suite=self.suite)
        with self.assertRaises(ValueError):
            run_suite(self.adapter, self.keys + self.keys[:1], suite=self.suite)

    def test_summary_counts_paired_flips_health_and_heads(self):
        results = run_suite(self.adapter, self.keys, suite=self.suite, batch_size=4)
        candidate_ids = [5, 7, 9]  # column 0 favours bright, column 2 favours dark
        truth = [5, 5, 5, 5, 9, 5, 7, 9, 5]
        clusters = [1, 1, 2, 2, 3, 2, 4, 3, 1]
        rows = summarize_conditions(results, candidate_ids=candidate_ids, truth_ids=truth,
                                    clusters=clusters, healthy_ids=[9],
                                    head_truth={"crop_logits": [0] * 9, "disease_logits": [0] * 9},
                                    replicates=300)
        by = {r["condition"]: r for r in rows}
        clean, dark = by["clean"], by["darken_s3"]
        # Clean: levels > 127.5 predict 5, the two dark images predict 9.
        self.assertEqual(clean["top1"]["hits"], 8)
        # x0.3 makes every image dark, so all predict 9; only two truths are 9.
        self.assertEqual(dark["top1"]["hits"], 2)
        self.assertEqual((dark["top1_paired"]["lost"], dark["top1_paired"]["gained"]), (6, 0))
        self.assertAlmostEqual(dark["top1_paired"]["delta"], -6 / 9)
        self.assertEqual(dark["top1_paired"]["mcnemar_exact_p"], 2 / 64)
        self.assertEqual(dark["top1_consistency_with_clean"]["hits"], 2)
        self.assertEqual(dark["false_healthy_among_diseased"], {"hits": 7, "n": 7, "rate": 1.0})
        self.assertEqual(clean["false_healthy_among_diseased"]["hits"], 0)
        self.assertAlmostEqual(dark["false_healthy_delta"]["delta"], 1.0)
        self.assertEqual(clean["top1_paired"]["mcnemar_exact_p"], 1.0)
        self.assertEqual(dark["crop_logits_consistency_with_clean"]["n"], 9)
        means = family_means(rows, suite=self.suite)
        self.assertEqual([m["family"] for m in means], ["darken", "jpeg"])
        self.assertAlmostEqual(means[0]["mean_delta"], 2 / 9 - 8 / 9)


class RecordingPredictor:
    """Records exactly what it receives, then resizes internally like a real model would."""

    def __init__(self, size, mutate=False):
        self.size, self.mutate, self.received, self.calls = size, mutate, [], []

    def predict(self, images):
        self.calls.append(len(images))
        self.received.extend(np.asarray(im).copy() for im in images)
        if self.mutate:
            for im in images:
                im.paste((255, 0, 0), (0, 0, im.size[0], im.size[1]))
        small = [np.asarray(im.resize((self.size, self.size))) for im in images]
        means = np.array([s.mean() for s in small])
        return {"scores": np.stack([means, 255 - means], axis=1), "candidate_ids": np.array([1, 2])}


class NativeSuiteV2Tests(unittest.TestCase):
    PINNED_V2 = "8f68a0e00cf7af3739696748c76513095fe8abe20bd6b26e8aae5c7116dd006b"

    def setUp(self):
        self.images = {f"k{i}.jpg": smooth_image(40 + 7 * i, 64 - 3 * i, seed=i) for i in range(6)}
        self.suite = (NATIVE_SUITE[0], Perturbation("gaussian_blur", 2, 0.04),
                      Perturbation("gaussian_noise", 1, 0.05), Perturbation("jpeg", 1, 40))

    def load(self, key):
        return self.images[key].copy()

    def test_v2_fingerprint_is_pinned_and_distinct_from_v1(self):
        self.assertEqual(suite_v2_fingerprint(), self.PINNED_V2)
        self.assertNotEqual(suite_v2_fingerprint(), suite_fingerprint())
        self.assertEqual(suite_fingerprint(), PINNED_SUITE_SHA256, "v1 must stay reproducible")

    def test_v1_does_not_give_identical_pixels_across_architectures(self):
        # Documents the v1 limitation: corruption after each model's own resize.
        class Geometric:
            def __init__(self, size):
                self.size, self.seen = size, []

            def geometric(self, key):
                from PIL import Image
                return np.asarray(Image.fromarray(self_images[key]).resize((self.size, self.size)))

            def forward(self, batch):
                self.seen.append(batch.copy())
                return {"scores": batch.reshape(len(batch), -1)[:, :2].astype(float)}

        self_images = self.images
        a, b = Geometric(224), Geometric(288)
        run_suite(a, list(self.images), suite=self.suite[:2], batch_size=2)
        run_suite(b, list(self.images), suite=self.suite[:2], batch_size=2)
        self.assertNotEqual(a.seen[1].shape, b.seen[1].shape)

    def test_v2_gives_identical_verified_pixels_to_every_model(self):
        small, large = RecordingPredictor(224), RecordingPredictor(288)
        result = run_native_suite({"small": small, "large": large}, list(self.images), load=self.load,
                                  suite=self.suite, batch_sizes={"small": 2, "large": 2})
        self.assertEqual(len(small.received), len(large.received))
        for x, y in zip(small.received, large.received):
            self.assertTrue(np.array_equal(x, y))
        digests = [d for p in self.suite for d in result["pixel_sha256"][p.name]]
        # Received order is condition-major within each chunk; compare as multisets.
        self.assertEqual(sorted(digests), sorted(pixel_digest(x) for x in small.received))
        clean = result["pixel_sha256"]["clean"]
        self.assertEqual(clean, [pixel_digest(self.images[k]) for k in self.images])
        self.assertNotEqual(result["pixel_sha256"]["gaussian_noise_s1"], clean)
        # Running a model alone later reproduces the same pixels (digests are a cross-run check).
        alone = run_native_suite({"small": RecordingPredictor(224)}, list(self.images), load=self.load,
                                 suite=self.suite)
        self.assertEqual(alone["pixel_sha256"], result["pixel_sha256"])

    def test_v2_batches_follow_each_models_declared_size_and_models_are_isolated(self):
        a, b = RecordingPredictor(16, mutate=True), RecordingPredictor(16)
        run_native_suite({"a": a, "b": b}, list(self.images), load=self.load, suite=self.suite[:1],
                         batch_sizes={"a": 2, "b": 4})
        self.assertEqual(a.calls, [2, 2, 2])
        self.assertEqual(b.calls, [4, 2])
        for k, received in zip(self.images, b.received):
            self.assertTrue(np.array_equal(received, self.images[k]), "mutation must not leak")

    def test_v2_rejects_missing_or_misshaped_outputs(self):
        class Missing(RecordingPredictor):
            def predict(self, images):
                return {"other": np.zeros((len(images), 2))}

        class Misshaped(RecordingPredictor):
            def predict(self, images):
                return {"scores": np.zeros((len(images) + 1, 2))}

        for predictor in (Missing(8), Misshaped(8)):
            with self.assertRaises(ValueError):
                run_native_suite({"m": predictor}, list(self.images), load=self.load, suite=self.suite[:1])
        with self.assertRaises(ValueError):
            run_native_suite({"m": RecordingPredictor(8)}, list(self.images), load=self.load,
                             suite=self.suite[:1], outputs={"m": ("crop_logits",)})

    def test_decode_matches_frozen_pipelines_without_exif_transpose(self):
        import tempfile
        from pathlib import Path
        from PIL import Image, ImageOps
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            Image.fromarray(smooth_image(30, 50)).save(root / "plain.jpg", quality=95)
            rgba = np.dstack([smooth_image(20, 20), np.full((20, 20), 90, np.uint8)])
            Image.fromarray(rgba, mode="RGBA").save(root / "alpha.png")
            exif = Image.Exif()
            exif[0x0112] = 6
            Image.fromarray(smooth_image(30, 50)).save(root / "rotated.jpg", quality=95, exif=exif)
            for name in ("plain.jpg", "alpha.png", "rotated.jpg"):
                with Image.open(root / name) as im:
                    expected = np.asarray(im.convert("RGB"))
                self.assertTrue(np.array_equal(decode_rgb(root / name), expected), name)
            with Image.open(root / "rotated.jpg") as im:
                transposed = np.asarray(ImageOps.exif_transpose(im).convert("RGB"))
            self.assertNotEqual(decode_rgb(root / "rotated.jpg").shape, transposed.shape)

    def test_native_blur_is_exact_identity_monotonic_and_relative(self):
        flat = np.full((50, 70, 3), 77, np.uint8)
        self.assertTrue(np.array_equal(apply_native_perturbation(flat, Perturbation("gaussian_blur", 3, 0.016), "k"),
                                       flat))
        img = smooth_image()
        self.assertTrue(np.array_equal(apply_native_perturbation(img, Perturbation("gaussian_blur", 1, 0.0), "k"),
                                       img))
        pattern = stripes(256)
        energy = [high_frequency_energy(apply_native_perturbation(pattern, p, "k"))
                  for p in NATIVE_SUITE if p.family == "gaussian_blur"]
        self.assertTrue(high_frequency_energy(pattern) > energy[0] > energy[1] > energy[2], energy)
        for p in NATIVE_SUITE:
            if p.family != "gaussian_blur":
                self.assertTrue(np.array_equal(apply_native_perturbation(img, p, "k"),
                                               apply_perturbation(img, p, "k")), p.name)

    def test_blockwise_native_operators_are_bit_identical_across_block_boundaries(self):
        from evaluation.robustness import _blockwise
        img = smooth_image(37, 23, seed=5)  # rows not divisible by the block size
        for p in NATIVE_SUITE:
            if p.family in ("darken", "brighten", "gaussian_noise"):
                reference = apply_perturbation(img, p, "k")
                for rows in (1, 4, 10, 37, 100):
                    self.assertTrue(np.array_equal(_blockwise(img, p, "k", 20261107, block_rows=rows),
                                                   reference), (p.name, rows))


class ClusterStatisticsTests(unittest.TestCase):
    def test_vectorized_cluster_bootstrap_equals_brute_force_paired_resampling(self):
        rng = np.random.default_rng(3)
        clusters = rng.choice(["a", "b", "c", "d", "e"], size=60, p=[.5, .2, .1, .1, .1])
        ref = rng.random(60) < .4
        per = np.where(rng.random(60) < .3, ~ref, ref)
        est, low, high = cluster_bootstrap_delta(ref, per, clusters, replicates=400, seed=11)
        labels = np.unique(clusters)
        draws = np.random.default_rng(11).integers(0, len(labels), size=(400, len(labels)))
        brute = []
        for row in draws:
            idx = np.concatenate([np.flatnonzero(clusters == labels[g]) for g in row])
            brute.append(per[idx].mean() - ref[idx].mean())  # same images in both arms: paired
        self.assertAlmostEqual(est, per.mean() - ref.mean())
        self.assertAlmostEqual(low, float(np.quantile(brute, .025)))
        self.assertAlmostEqual(high, float(np.quantile(brute, .975)))

    def test_cluster_design_reports_kish_effective_clusters(self):
        design = cluster_design(["a"] * 6 + ["b"] * 2 + ["c"] * 2)
        self.assertEqual((design["clusters"], design["images"], design["largest_cluster_images"]), (3, 10, 6))
        self.assertAlmostEqual(design["kish_effective_clusters"], 100 / 44)
        self.assertEqual(design["clusters_below_5_images"], 2)

    def test_largest_cluster_sensitivity(self):
        out = largest_cluster_sensitivity([1, 1, 1, 0, 0], [0, 0, 0, 0, 1], ["x", "x", "x", "y", "z"])
        self.assertEqual(out["excluded_cluster"], "x")
        self.assertEqual(out["excluded_images"], 3)
        self.assertAlmostEqual(out["delta_without_largest"], 0.5)


if __name__ == "__main__":
    unittest.main()
