"""Verify the three immutable CSVs and historical manifests without ML dependencies.

Run from any working directory: python scripts/verify_csv_checkout.py
This command never rewrites files or updates provenance.
"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

EXPECTED = {
    "DPD_Small_train.csv": (8000, "b3054d6f20b1e18453133519cfb856f19590af1c", "31d19a852f74b6f1095ea13121243a95074d50576d3d5fc5b48baa0ff6eb0994", "9f1b729ab08f120c9e8d3a5681aa1ccbdf1d9210f9aaa6143959363dcb106a7d"),
    "DPD_Small_test_seen.csv": (2000, "037b8bf5d2481aecd6692883688cc0cc79b6046d", "0c60fbe2a9c5f04ba7fd74064c23a4f8bde450b62e91099c1a0c0cf82fe3cee0", "45d31d48d2260f56cc28871b44434b5243100dc90dc95652c28c377b35b45e83"),
    "DPD_Small_test_unseen.csv": (1448, "ef187426c48d997e3ae68c3cf39d18b91a175451", "2d19e6cc934945fcfde2a772c251cd13ce39bd924de3304eaa6b5bd6ce7a3587", "757e3c1fdcc894004a69c21dbc00d6b818f563aa671db9ab8d315c4790c2490b"),
}


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], stderr=subprocess.PIPE)


def verify(root):
    root = Path(root)
    report = {"status": "passed", "files": {}}
    for name, (rows, blob_id, lf_hash, crlf_hash) in EXPECTED.items():
        raw = (root / name).read_bytes()
        stored = git(root, "show", f"HEAD:{name}")
        actual_blob = hashlib.sha1(b"blob " + str(len(stored)).encode() + b"\0" + stored).hexdigest()
        attrs = git(root, "check-attr", "text", "eol", "--", name).decode()
        problems = []
        if sha256(stored) != lf_hash or actual_blob != blob_id:
            problems.append("Committed CSV content differs from upstream main")
        if sha256(raw) != crlf_hash or raw.count(b"\r\n") != rows or raw.count(b"\n") != rows:
            problems.append("Working CSV must contain the exact historical CRLF bytes")
        if raw.replace(b"\r\n", b"\n") != stored:
            problems.append("Working and committed CSV rows differ")
        if f"{name}: text: set" not in attrs or f"{name}: eol: crlf" not in attrs:
            problems.append("Scoped text/eol attributes are missing")
        report["files"][name] = {"rows": rows, "git_blob": actual_blob,
                                "working_sha256": sha256(raw), "problems": problems}
        if problems:
            report["status"] = "failed"
    for mode in ("stratified", "heldout_composition"):
        path = root / "artifacts" / "data" / f"{mode}_seed42.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest["source_sha256"] != report["files"]["DPD_Small_train.csv"]["working_sha256"]:
            report["status"] = "failed"
            report.setdefault("manifest_errors", []).append(str(path))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    try:
        report = verify(args.repo)
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as exc:
        report = {"status": "failed", "error": str(exc)}
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
