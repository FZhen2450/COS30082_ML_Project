"""Verify an extracted WP1 development export using the pinned WP5 adapters."""
import argparse
import copy
import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import sys


def receipt_path(root, name):
    """Resolve either receipt separator style within the explicit source root."""
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Receipt path must be a nonempty relative string")
    relative = PurePosixPath(name.replace("\\", "/"))
    if relative.is_absolute() or PureWindowsPath(name).drive or ".." in relative.parts:
        raise ValueError("Receipt path must stay within its declared root")
    root = Path(root).resolve(strict=True)
    path = root.joinpath(*relative.parts).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("Receipt path must identify a file within its declared root")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--wp5-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.bundle.resolve(strict=True)
    wp5 = args.wp5_root.resolve(strict=True)
    sys.path.insert(0, str(root / "source"))
    sys.path.insert(0, str(wp5))
    from evaluation.model_adapters import ModelOutputAdapter, WP1_INDEPENDENT
    from evaluation.reporting import EvaluationManifest, RunProvenance, build_report

    def read(name):
        return json.loads((root / name).read_text(encoding="utf-8"))

    receipt = read("receipt.json")
    if receipt["complete"] is not True:
        raise ValueError("Incomplete export")
    for name, expected in receipt["files_sha256"].items():
        path = receipt_path(root, name)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("File checksum mismatch: " + name)
    for name, expected in receipt["wp5_source_sha256"].items():
        if hashlib.sha256(receipt_path(wp5, name).read_bytes()).hexdigest() != expected:
            raise ValueError("Use the pinned WP5 source bytes: " + name)
    manifest = EvaluationManifest.from_dict(read("identity_manifest.json"))
    provenance = RunProvenance(**read("provenance.json"))
    adapter = ModelOutputAdapter(manifest, provenance, WP1_INDEPENDENT)
    packets = [read(fold + "_packet.json") for fold in ("seen", "unseen")]
    records = tuple(r for packet in packets for r in adapter.adapt(packet))
    report = build_report(manifest, records)
    saved = read("wp5_report/report.json")
    if json.loads(json.dumps(report, allow_nan=False)) != saved:
        raise ValueError("Rebuilt WP5 report differs from exported report")
    expected = {"seen": (1238, 661, 960), "unseen": (1894, 13, 164)}
    for fold, totals in expected.items():
        if tuple(report[fold][k] for k in ("count", "correct_top1", "correct_top5")) != totals:
            raise ValueError("Historical hit totals differ")
    passed = []

    def reject(label, operation):
        try:
            operation()
        except ValueError:
            passed.append(label)
        else:
            raise AssertionError("Malformed export accepted: " + label)

    for label, mutate in (
        ("duplicate filename", lambda p: p["image_names"].__setitem__(1, p["image_names"][0])),
        ("wrong truth", lambda p: p["ground_truth_ids"].__setitem__(0, 999999)),
        ("wrong candidate order", lambda p: p["candidate_ids"].reverse()),
        ("nonfinite score", lambda p: p["scores"][0].__setitem__(0, float("nan"))),
        ("wrong float32 precision", lambda p: p["scores"][0].__setitem__(0, 0.1)),
        ("missing score row", lambda p: p["scores"].pop()),
        ("wrong fold", lambda p: p.__setitem__("fold", "unseen")),
    ):
        packet = copy.deepcopy(packets[0])
        mutate(packet)
        reject(label, lambda: adapter.adapt(packet))
    reject("missing identity", lambda: build_report(manifest, records[:-1]))
    reject("duplicate identity across packets", lambda: build_report(manifest, records + records[:1]))
    print(json.dumps({"verified": True, "rows": len(records), "manifest_sha256": manifest.sha256,
                      "H1": report["harmonic_mean_top1"], "H5": report["harmonic_mean_top5"],
                      "negative_checks_passed": passed}, indent=2))


if __name__ == "__main__":
    main()
