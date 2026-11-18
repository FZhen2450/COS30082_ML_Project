"""PUBLIC-checkout verification of the Second Run v5 evidence (publication-only; fails closed).

  python scripts/r2v5_public_verify.py [--tests] [--json OUT]

Every entry of every PUBLIC v5 stage manifest (artifacts/r2/v5/**/SHA256SUMS) is reported as one of:
  ORIGINAL               published bytes == manifest hash (byte-identical to the private original)
  SANITIZED_DERIVATIVE   published bytes == public_derivative_sha256 in artifacts/r2/v5/PUBLICATION_MAP_v5.json and the map's
                         private_original_sha256 == the manifest hash (local paths or class-level official statistics removed;
                         the original bytes are NOT public)
  WITHHELD_PRIVATE       deliberately not published (class-level official statistics); the map records its original hash
  FAIL                   anything else (changed, missing, unlisted): exit 1
Manifests of private run / official-prediction files (private_files_SHA256SUMS.txt, official_output_SHA256SUMS.txt) are
reported UNAVAILABLE: those files (checkpoint, feature cache, per-image official predictions) are never published.
It also checks the frozen configuration (gamma 8.75, J2 configuration, candidate payload), the frozen checkpoint hash in
every public record, and recomputes the official H@1 / H@5 from the published counts. --tests runs
tests/test_r2v5_final.py, which needs only public files (training CSV, candidate payload, synthetic backend).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V5 = ROOT / "artifacts/r2/v5"
MAP = V5 / "PUBLICATION_MAP_v5.json"
CHECKPOINT = "e9dbbdf3fc95ba791e8e8129ef38bbd85352a0974c3091d6b7667bd633abb1c9"
PAYLOAD_SHA = "ba17e94119e8b2b7a4b18603e25405d2f52bf4a93396973f80491800697eb62f"
OFFICIAL = {"seen": (1142, 1754, 2000), "unseen": (1218, 1429, 1448), "H@1": 0.6802379489908098, "H@5": 0.928700476046502}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def manifest(p):
    for line in Path(p).read_text().splitlines():
        h, _, rel = line.partition("  ")
        yield h, rel[2:] if rel.startswith("./") else rel


def verify_evidence(m):
    derived, withheld = m["files"]["derived"], m["files"]["withheld"]
    counts = {"ORIGINAL": 0, "SANITIZED_DERIVATIVE": 0, "WITHHELD_PRIVATE": 0, "FAIL": 0}
    detail, seen = [], set()
    for man in sorted(V5.rglob("SHA256SUMS")):
        for h, rel in manifest(man):
            f = man.parent / rel
            key = str(f.resolve().relative_to(ROOT.resolve()))
            if key in derived:
                seen.add(key); e = derived[key]
                ok = f.exists() and sha(f) == e["public_derivative_sha256"] and e["private_original_sha256"] == h
                st = "SANITIZED_DERIVATIVE" if ok else "FAIL"
            elif key in withheld:
                seen.add(key)
                st = "WITHHELD_PRIVATE" if (not f.exists() and withheld[key]["private_original_sha256"] == h) else "FAIL"
            elif f.exists():
                st = "ORIGINAL" if sha(f) == h else "FAIL"
            else:
                st = "FAIL"
            counts[st] += 1
            if st != "ORIGINAL":
                detail.append(f"{st}: {key}")
    for key in sorted((set(derived) | set(withheld)) - seen):
        if key in derived and (ROOT / key).exists() and sha(ROOT / key) == derived[key]["public_derivative_sha256"] and key.endswith(".md"):
            counts["SANITIZED_DERIVATIVE"] += 1; detail.append(f"SANITIZED_DERIVATIVE (no manifest; map only): {key}")
        else:
            counts["FAIL"] += 1; detail.append(f"FAIL: map entry not covered: {key}")
    private = sum(1 for man in V5.rglob("*SHA256SUMS.txt") for _ in manifest(man))
    return {"public_manifest_entries": counts, "private_manifest_entries_UNAVAILABLE": private, "detail": detail}


def verify_frozen():
    p = []
    cfg = json.loads((ROOT / "research_r2_v5/final_config.json").read_text())
    if cfg["gamma"] != 8.75 or cfg["row"] != "J2_clipL_last4" or cfg["candidate_payload_sha256"] != PAYLOAD_SHA:
        p.append("final_config.json differs from the frozen candidate")
    if sha(V5 / "final_candidates_cos30082_v1.json") != PAYLOAD_SHA:
        p.append("candidate payload hash differs")
    if f'FROZEN_CHECKPOINT_SHA256 = "{CHECKPOINT}"' not in (ROOT / "research_r2_v5/final.py").read_text():
        p.append("final.py does not pin the frozen checkpoint")
    for rel, key in (("final_J2_full8000/receipt.json", "checkpoint_sha256"), ("final_J2_full8000/preflight_07e05ef/verification_07e05ef.json", "checkpoint_sha256_now"),
                     ("official_eval_ONE_SHOT_aggregate/official_evaluation.json", "checkpoint_sha256")):
        if json.loads((V5 / rel).read_text()).get(key) != CHECKPOINT:
            p.append(f"{rel}: checkpoint hash is not the frozen candidate")
    ev = json.loads((V5 / "official_eval_ONE_SHOT_aggregate/official_evaluation.json").read_text())
    rc = json.loads((V5 / "official_eval_ONE_SHOT_aggregate/independent_recomputation.json").read_text())
    if ev.get("gamma") != 8.75:
        p.append("official evaluation gamma is not 8.75")
    for part, (n1, n5, n) in (("seen", OFFICIAL["seen"]), ("unseen", OFFICIAL["unseen"])):
        r = rc[part]
        if (r["top1_correct"], r["top5_correct"], r["n"]) != (n1, n5, n) or ev[part]["count"] != n:
            p.append(f"{part}: published counts differ from the frozen result")
        if abs(ev[part]["top1"] - n1 / n) > 1e-15 or abs(ev[part]["top5"] - n5 / n) > 1e-15:
            p.append(f"{part}: published accuracies are inconsistent with the counts")
    s, u = OFFICIAL["seen"][0] / 2000, OFFICIAL["unseen"][0] / 1448
    s5, u5 = OFFICIAL["seen"][1] / 2000, OFFICIAL["unseen"][1] / 1448
    h1, h5 = 2 * s * u / (s + u), 2 * s5 * u5 / (s5 + u5)
    for name, val in (("H@1", h1), ("H@5", h5)):
        if abs(val - OFFICIAL[name]) > 1e-15 or abs(ev[name] - OFFICIAL[name]) > 1e-15 or abs(rc[f"{name}_recomputed"] - OFFICIAL[name]) > 1e-15:
            p.append(f"{name} differs from the frozen official result")
    return {"problems": p, "H@1_from_counts": h1, "H@5_from_counts": h5, "gamma": cfg["gamma"]}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tests", action="store_true"); ap.add_argument("--json")
    a = ap.parse_args()
    m = json.loads(MAP.read_text())
    rep = {"tree": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
           "evidence": verify_evidence(m), "frozen": verify_frozen()}
    ok = rep["evidence"]["public_manifest_entries"]["FAIL"] == 0 and not rep["frozen"]["problems"]
    if a.tests:
        r = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_r2v5_final.py"], cwd=ROOT, capture_output=True, text=True)
        rep["tests"] = {"command": "python -m unittest discover -s tests -p test_r2v5_final.py", "exit": r.returncode,
                        "summary": [l for l in r.stderr.splitlines() if l.startswith(("Ran ", "OK", "FAILED"))]}
        ok = ok and r.returncode == 0
    rep["result"] = "PASS" if ok else "FAIL"
    rep["note"] = "SANITIZED_DERIVATIVE, WITHHELD_PRIVATE and UNAVAILABLE entries are not verifications of the private originals"
    if a.json:
        Path(a.json).write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
