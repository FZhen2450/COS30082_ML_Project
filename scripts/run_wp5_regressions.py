"""Run all local regressions applicable under the WP5 no-training constraint."""
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = {
    "test_independent_baseline.IndependentTests.test_separate_parameters_and_gradient_updates",
    "test_independent_baseline.IndependentTests.test_synthetic_training_records_and_inference_adapter",
}


def flatten(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from flatten(item)
        else:
            yield item


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = "2"
    sys.path.insert(0, str(ROOT))
    tests = list(flatten(unittest.defaultTestLoader.discover(str(ROOT / "tests"))))
    if not EXCLUDED <= {test.id() for test in tests}:
        raise RuntimeError("Expected training-test exclusions changed; review before running")
    print("EXCLUDED (no-training constraint):", *sorted(EXCLUDED), sep="\n", flush=True)
    suite = unittest.TestSuite(test for test in tests if test.id() not in EXCLUDED)
    print("APPLICABLE TESTS:", suite.countTestCases(), flush=True)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
