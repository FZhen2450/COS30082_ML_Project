"""R2 v4 feasibility: how much size-controlled source-shift evidence the v3 inner folds can support.

Pure metadata analysis (no features, no scores). For every v3 inner fold and every source s that has
fitting images:
  ablatable      removing all fitting images of s keeps every held-out crop and disease covered and
                 leaves every held-out composition still evaluable (same rule as r2_calibration_shift.py)
  novel_seen     seen-val images of s whose composition keeps >= 1 fitting image from another source
                 (the only images where "seen composition, novel source" is measurable)
  unseen_s       unseen-val images of s (held-out compositions)
  size control   a composition-stratified random removal of the SAME number of fitting images from
                 other sources, never emptying a composition: feasible iff enough removable images exist
Output: JSON with per-(fold, source) rows and totals.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research_r2.data import load_training_source  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="DPD_Small_train.csv")
    ap.add_argument("--bench", default="artifacts/r2/benchmark_v3.json")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    frame = load_training_source(a.csv)
    bench = json.loads(Path(a.bench).read_text())
    rows, tot = [], {"pairs": 0, "ablatable": 0, "size_control_feasible": 0, "novel_seen_images": 0,
                     "unseen_images_in_ablatable": 0}
    for fold in [f for f in bench["folds"] if f["role"] == "inner"]:
        fit = frame.loc[fold["fitting_rows"]]
        held = set(fold["heldout_composition_ids"])
        hsub = frame[frame.comp_id.isin(held)].drop_duplicates("comp_id")
        seen = frame.loc[fold["seen_val_rows"]]
        uns = frame.loc[fold["unseen_val_rows"]]
        for s in sorted(fit.source.unique()):
            kept = fit[fit.source != s]
            covered = set(hsub.crop) <= set(kept.crop) and set(hsub.disease) <= set(kept.disease)
            novel = seen[(seen.source == s) & seen.comp_id.isin(set(kept.comp_id))]
            n_rm = int((fit.source == s).sum())
            per_comp = kept.groupby("comp_id").size()
            removable = int((per_comp - 1).clip(lower=0).sum())
            r = {"fold": fold["name"], "source": s, "fitting_images_removed": n_rm, "ablatable": bool(covered),
                 "novel_seen_images": int(len(novel)), "novel_seen_compositions": int(novel.comp_id.nunique()),
                 "unseen_images_from_source": int((uns.source == s).sum()),
                 "compositions_emptied": int(len(set(fit.comp_id) - set(kept.comp_id))),
                 "size_control_feasible": bool(removable >= n_rm), "removable_other_source_images": removable,
                 "_novel_rows": [int(x) for x in novel.row], "_novel_comps": sorted(int(c) for c in novel.comp_id.unique())}
            rows.append(r)
            tot["pairs"] += 1
            if covered:
                tot["ablatable"] += 1
                tot["size_control_feasible"] += int(r["size_control_feasible"])
                tot["novel_seen_images"] += r["novel_seen_images"]
                tot["unseen_images_in_ablatable"] += r["unseen_images_from_source"]
    useful = [r for r in rows if r["ablatable"] and r["size_control_feasible"] and
              (r["novel_seen_images"] >= 10 or r["unseen_images_from_source"] >= 10)]
    tot["useful_pairs_(>=10_eval_images)"] = len(useful)
    tot["useful_novel_seen_images"] = sum(r["novel_seen_images"] for r in useful)
    tot["useful_unseen_images"] = sum(r["unseen_images_from_source"] for r in useful)
    # the global seen-holdout is shared by all inner folds, so the same novel-seen images recur across folds:
    # report unique images / compositions / sources as the effective sample size
    tot["unique_novel_seen_images_useful"] = len({x for r in useful for x in r["_novel_rows"]})
    tot["unique_novel_seen_compositions_useful"] = len({c for r in useful for c in r["_novel_comps"]})
    tot["unique_sources_useful"] = len({r["source"] for r in useful})
    for r in rows:
        r.pop("_novel_rows"), r.pop("_novel_comps")
    multi = frame.groupby("comp_id").source.nunique()
    tot["multi_source_compositions"] = int((multi > 1).sum())
    out = {"schema": "r2v4-source-feasibility-v1", "benchmark_semantic_sha256": bench["semantic_sha256"],
           "totals": tot, "useful_pairs": useful, "rows": rows}
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(tot, indent=1))
    for r in useful:
        print(r)


if __name__ == "__main__":
    main()
