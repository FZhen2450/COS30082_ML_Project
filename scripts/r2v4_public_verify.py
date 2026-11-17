"""PUBLIC-checkout verification of the R2 v4 research evidence (publication-only; not part of any declaration).

  python scripts/r2v4_public_verify.py            evidence hashes + declarations
  python scripts/r2v4_public_verify.py --tests    ... + every R2 test whose prerequisites are present here

The public history replaces local machine paths in 46 evidence files (artifacts/r2/v4/PUBLICATION_SANITIZATION_MAP.json).
The stage SHA256SUMS manifests are published UNCHANGED and record the PRIVATE ORIGINAL hashes. This script therefore
reports every manifest entry as one of:

  ORIGINAL                published bytes == manifest hash (byte-identical to the private original)
  SANITIZED_DERIVATIVE    published bytes == public_sanitized_sha256 in the map AND the map's private_original_sha256 ==
                          the manifest hash. The original bytes are NOT public; this is not a verification of them.
  UNAVAILABLE             file not published (git-ignored at the original commit, or a private run file: checkpoint,
                          feature cache, score packet, index set). Not counted as verified.
  FAIL                    anything else (a changed, missing or unlisted file): exit 1.

Tests: a test whose prerequisite is absent (private original evidence bytes, private commit objects, private run
directories) is NOT run and is reported UNAVAILABLE with the reason; it is never counted as passed. Tests are not edited:
the private suite (`python -m unittest discover -s tests -p 'test_r2*.py'`) stays the authenticated private check.
Exit 0 only if nothing FAILs, every declaration verifies and every test that was run passed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests")); sys.path.insert(0, str(ROOT / "scripts"))
ART = ROOT / "artifacts/r2/v4"
MAP = ART / "PUBLICATION_SANITIZATION_MAP.json"
EXP = ROOT.parents[1] / "experiments" / "r2-compositional"  # private experiment directory (absent on a public checkout)
EXP_V4 = EXP / "v4"
PRIVATE_COMMITS = ("d9ed896", "e96d471")  # private commits referenced by tests (amendment-4 pin diff; old Stage J evaluator)


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read_manifest(p: Path):
    for line in p.read_text().splitlines():
        h, _, rel = line.partition("  ")
        yield h, rel[2:] if rel.startswith("./") else rel


def ignored(p: Path) -> bool:
    return subprocess.run(["git", "check-ignore", "-q", str(p.relative_to(ROOT))], cwd=ROOT).returncode == 0


# ----------------------------------------------------------------------------- prerequisites
def has_private_commits() -> bool:
    return all(subprocess.run(["git", "cat-file", "-e", f"{c}^{{commit}}"], cwd=ROOT, capture_output=True).returncode == 0
               for c in PRIVATE_COMMITS)


def has_private_runs() -> bool:
    need = ["stage_j_phase1_20261114", "stage_j_phase2_20261115", "stage_hrt_20261116/runs", "stage_j4b_20261117/runs",
            "stage_j4b_confirm_20261117/runs", "features"]
    return all((EXP_V4 / n).is_dir() for n in need)


def has_stage_h_index_sets() -> bool:
    return (EXP_V4 / "stage_h_20261114/D1_synthComp_norm/indexsets").is_dir()


def has_v3_score_packets() -> bool:
    return (EXP / "v3").is_dir()


def has_original_evidence() -> bool:
    """False on the public tree: the 46 sanitized derivatives replace the original bytes."""
    return not MAP.exists()


PREREQS = {"private_original_evidence_bytes": has_original_evidence, "private_commit_objects": has_private_commits,
           "private_registered_runs": has_private_runs, "private_stage_h_index_sets": has_stage_h_index_sets,
           "private_v3_score_packets": has_v3_score_packets}
# test id (or class prefix) -> (prerequisite, reason). A listed test RUNS whenever its prerequisite is present; it is
# reported UNAVAILABLE (never passed) otherwise. Established by running the unchanged suite on a fresh single-branch
# clone of the public history, where exactly these tests fail, each on its missing private input.
PRIVATE_TESTS = {
    "test_r2v4_amendment4.TestAmendment3Preserved.test_historical_artifact_hashes_unchanged":
        ("private_original_evidence_bytes", "checks stage manifests against the ORIGINAL evidence bytes and the private run files"),
    "test_r2v4_amendment4.TestAmendment3Preserved.test_amendment_3_unchanged_and_still_verifies":
        ("private_commit_objects", "diffs the amendment-3 pinned files against private commit d9ed896"),
    "test_r2v4_amendment4.TestProductionRefusals.test_real_declaration_state":
        ("private_registered_runs", "compares the amendment-4 evidence pins with the registered J2 / J4b runs"),
    "test_r2v4_confirm.TestPreviousEvaluatorAcceptedInvalidFixtures":
        ("private_commit_objects", "loads the previous evaluator from private commit e96d471 to show what it accepted"),
    "test_r2v4_hrt.TestScopeAndFolds.test_index_set_must_reproduce_its_manifest":
        ("private_stage_h_index_sets", "verifies the saved Stage H matched-removal index sets against their manifest"),
    "test_r2v4_hrt.TestEndToEnd.test_both_arms_complete_and_evaluate":
        ("private_stage_h_index_sets", "the matched-random arm reuses the saved Stage H index sets"),
    "test_r2v4_hrt.TestVerificationRefusals.test_checkpoint_from_the_other_arm":
        ("private_stage_h_index_sets", "needs a completed matched-random arm (saved Stage H index sets)"),
    "test_r2v4_hrt.TestVerificationRefusals.test_feature_cache_from_the_other_arm":
        ("private_stage_h_index_sets", "needs a completed matched-random arm (saved Stage H index sets)"),
    "test_r2v4_j.TestSyntheticSixFoldDryRun.test_dry_run":
        ("private_v3_score_packets", "evaluates synthetic rows against the registered v3 D1 score packets"),
}


# ----------------------------------------------------------------------------- evidence
def verify_evidence():
    m = json.loads(MAP.read_text()) if MAP.exists() else {"files": {}}
    derived = m["files"]
    counts = {"ORIGINAL": 0, "SANITIZED_DERIVATIVE": 0, "UNAVAILABLE": 0, "FAIL": 0}
    detail, seen_derived = [], set()
    public = sorted(p for p in ART.rglob("*SHA256SUMS*") if not p.name.startswith("private_files") and "indexsets" not in p.name)
    for man in public:
        for h, rel in read_manifest(man):
            f = (man.parent / rel).resolve()
            key = str(f.relative_to(ROOT.resolve()))
            if key in derived:
                e = derived[key]; seen_derived.add(key)
                ok = f.exists() and sha(f) == e["public_sanitized_sha256"] and e["private_original_sha256"] == h
                st = "SANITIZED_DERIVATIVE" if ok else "FAIL"
            elif f.exists():
                st = "ORIGINAL" if sha(f) == h else "FAIL"
            else:
                st = "UNAVAILABLE" if ignored(f) else "FAIL"
            counts[st] += 1
            if st in ("FAIL", "UNAVAILABLE"):
                detail.append(f"{st}: {key}" + (" (not published: git-ignored)" if st == "UNAVAILABLE" else ""))
    for key in sorted(set(derived) - seen_derived):
        counts["FAIL"] += 1; detail.append(f"FAIL: map entry {key} is not listed by any stage manifest")
    # private run-file manifests: verified only where the private run directories exist
    priv = {"ORIGINAL": 0, "UNAVAILABLE": 0, "FAIL": 0}
    for man in sorted(p for p in ART.rglob("*SHA256SUMS*") if p.name.startswith("private_files") or "indexsets" in p.name):
        stage_rel = man.parent.relative_to(ART)
        bases = [EXP_V4 / stage_rel, EXP_V4 / stage_rel / "runs", EXP_V4 / stage_rel / "indexsets"]
        entries = list(read_manifest(man))
        base = next((b for b in bases if entries and (b / entries[0][1]).exists()), None)
        for h, rel in entries:
            if base is None or not (base / rel).exists():
                priv["UNAVAILABLE"] += 1
            elif sha(base / rel) == h:
                priv["ORIGINAL"] += 1
            else:
                priv["FAIL"] += 1; detail.append(f"FAIL: private run file {stage_rel}/{rel}")
    return {"published_evidence": counts, "private_run_files": priv, "map_entries": len(derived), "detail": detail}


# ----------------------------------------------------------------------------- declarations
def verify_declarations():
    out = {}
    py = sys.executable
    for name, args in (("base", []), ("amendment_1", ["--amendment", "1"]), ("amendment_2", ["--amendment", "2"]),
                       ("amendment_3", ["--amendment", "3"])):
        r = subprocess.run([py, "scripts/r2v4_declare.py", "--verify", *args], cwd=ROOT, capture_output=True, text=True)
        out[name] = {"exit": r.returncode, "output": r.stdout.strip()}
    r = subprocess.run([py, "scripts/r2v4_a4.py", "verify"], cwd=ROOT, capture_output=True, text=True)
    out["amendment_4"] = {"exit": r.returncode, "output": r.stdout.strip()}
    return out


# ----------------------------------------------------------------------------- tests
def _flatten(s):
    for t in s:
        if isinstance(t, unittest.TestSuite):
            yield from _flatten(t)
        else:
            yield t


def run_tests():
    avail = {k: f() for k, f in PREREQS.items()}
    loaded = unittest.defaultTestLoader.discover(str(ROOT / "tests"), pattern="test_r2*.py", top_level_dir=str(ROOT / "tests"))
    run, unavailable, broken = unittest.TestSuite(), [], []
    for t in _flatten(loaded):
        tid = t.id()
        if tid.startswith("unittest.loader._Failed"):
            broken.append(tid); run.addTest(t); continue
        req = next((v for k, v in PRIVATE_TESTS.items() if tid == k or tid.startswith(k + ".")), None)
        if req is not None and not avail[req[0]]:
            unavailable.append({"test": tid, "requires": req[0], "reason": req[1]})
        else:
            run.addTest(t)
    res = unittest.TextTestRunner(verbosity=2, stream=sys.stderr).run(run)
    return {"prerequisites_present": avail, "discovered": run.countTestCases() + len(unavailable), "run": res.testsRun,
            "passed": res.testsRun - len(res.failures) - len(res.errors) - len(res.skipped) - len(res.expectedFailures),
            "failures": [t.id() for t, _ in res.failures], "errors": [t.id() for t, _ in res.errors],
            "skipped_by_test": [{"test": t.id(), "reason": r} for t, r in res.skipped],
            "unavailable_not_run": unavailable, "load_errors": broken}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tests", action="store_true")
    ap.add_argument("--json")
    a = ap.parse_args()
    rep = {"tree": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
           "mode": "public tree (sanitized derivatives present)" if MAP.exists() else "private tree (original evidence)"}
    rep["evidence"] = verify_evidence()
    rep["declarations"] = verify_declarations()
    if a.tests:
        rep["tests"] = run_tests()
    ok = (rep["evidence"]["published_evidence"]["FAIL"] == 0 and rep["evidence"]["private_run_files"]["FAIL"] == 0
          and all(d["exit"] == 0 for d in rep["declarations"].values())
          and (not a.tests or not (rep["tests"]["failures"] or rep["tests"]["errors"] or rep["tests"]["load_errors"])))
    rep["result"] = "PASS" if ok else "FAIL"
    rep["note"] = ("SANITIZED_DERIVATIVE and UNAVAILABLE entries are not verifications of the private originals; "
                   "UNAVAILABLE tests were not run and are not counted as passed")
    if a.json:
        Path(a.json).write_text(json.dumps(rep, indent=1))
    t = rep.get("tests", {})
    print(json.dumps({k: v for k, v in rep.items() if k != "tests"} | ({"tests": {k: v for k, v in t.items() if k != "skipped_by_test"}
                                                                          | {"skipped_by_test": len(t.get("skipped_by_test", []))}} if t else {}),
                     indent=1))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
