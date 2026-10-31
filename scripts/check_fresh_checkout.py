"""Verify scoped CRLF checkout under false/true/input autocrlf in fresh clones.

The source must contain the proposed changes in a local commit. By default,
fresh repositories copy objects and use read-tree/checkout-index; --method clone
uses fresh local clones. Both exercise Git conversion. No push is performed.
"""
import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--method", choices=("objects", "clone"), default="objects",
                        help="objects uses init/object-copy/read-tree/checkout-index; clone uses local clone")
    args = parser.parse_args()
    results = []
    source_sha = subprocess.check_output(["git", "-C", str(args.repo), "rev-parse", "HEAD"], text=True).strip()
    with tempfile.TemporaryDirectory(prefix="wp1-checkout-") as folder:
        for setting in ("false", "true", "input"):
            target = Path(folder) / setting
            if args.method == "clone":
                subprocess.run(["git", "clone", "--quiet", "--local", "--config", f"core.autocrlf={setting}",
                                str(args.repo.resolve()), str(target)], check=True)
            else:
                subprocess.run(["git", "init", "--quiet", str(target)], check=True)
                git_dir = Path(subprocess.check_output(
                    ["git", "-C", str(args.repo), "rev-parse", "--absolute-git-dir"], text=True).strip())
                shutil.copytree(git_dir / "objects", target / ".git/objects", dirs_exist_ok=True)
                for command in (("config", "core.autocrlf", setting),
                                ("update-ref", "refs/heads/wp1-check", source_sha),
                                ("symbolic-ref", "HEAD", "refs/heads/wp1-check"),
                                ("read-tree", "HEAD"), ("checkout-index", "--all", "--force")):
                    subprocess.run(["git", "-C", str(target), *command], check=True)
            result = subprocess.run([sys.executable, str(target / "scripts/verify_csv_checkout.py")],
                                    capture_output=True, text=True, check=False)
            status = subprocess.check_output(["git", "-C", str(target), "status", "--porcelain"], text=True)
            results.append({"core.autocrlf": setting, "exit_code": result.returncode,
                            "clean_worktree": not status.strip(), "verification": json.loads(result.stdout)})
    report = {"source_commit": source_sha, "checkout_method": args.method, "fresh_checkouts": results,
              "passed": all(r["exit_code"] == 0 and r["clean_worktree"] for r in results)}
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
