# R2 v4 H_RT: implementation plan (written before any H_RT code; 16 Nov 2026)

**Registered design:** `amendment_2.json`, `source_co_endpoint.H_RT`. It is not changed here.

| Item | Registered | Verified now (Gate A) |
|---|---|---|
| Rows | three-seed passers | J1, J2, J3 (all confirmed in Phase 2) |
| Pairs | (I6, CDDM), (I1, MTID), (I2, CPDD): the three distinct sources with the most novel-source seen images among the 49 pairs, ties → lower fold | Re-derived from `source_feasibility.json`: identical |
| Seed | 0 | – |
| Arms | the row retrained on the source-ablated fitting rows, and on matched random control seed 0; base = the seed-0 fold model | Base models are the Phase 1 seed-0 fold models (receipts verify) |
| Matched control | the Stage H index set `<fold>_<source>_seed0.json` | sha256 equals `indexsets_SHA256SUMS`, and `matched_removal(seed=0)` reproduces `removed_rows` and `match_levels` exactly for all three |
| Status | diagnostic only | never a gate, never used to select or to change the J4–J7 parent |

**Pair facts (fitting rows of the fold):**

| Pair | Fitting | Removed (both arms) | Compositions emptied, ablated / random | Seen-val rows (base → ablated) | Novel-source seen | Source-s unseen | Universe base / ablated |
|---|---|---|---|---|---|---|---|
| I6 / CDDM | 5,062 | 1,950 | 25 / 0 | 878 → 676 | 135 | 253 | 301 / 276 |
| I1 / MTID | 5,042 | 381 | 1 / 0 | 874 → 856 | 53 | 0 | 301 / 300 |
| I2 / CPDD | 5,057 | 359 | 5 / 0 | 879 → 855 | 39 | 0 | 301 / 296 |

**Matched-random composition match levels** (composition / crop / disease / any):
- I6: 231 / 435 / 501 / 783.
- I1: 53 / 196 / 51 / 81.
- I2: 201 / 121 / 2 / 35.

The random arm therefore changes the class mix differently from the ablated arm, and this confound is reported.

**Lineage:**
- No lineage group is shared between fitting and evaluation rows in any fold (`r2v4_verify_inputs.py`).
- No novel-source evaluation image shares a group with any fitting row.
- **One registered-exclusion imperfection, reported and not altered:** in I1/MTID, one kept fitting image (`SCLD_sugarcane_viral_4947.jpg`, from another source and another crop) shares lineage group 33 with removed MTID maize images. The dhash ≤ 4 chain crosses crops, so this is very likely a hash collision, not a duplicate. Removing it would change the registered index set.

## Implementation (reuse, minimal surface)

**Training code:** `research_r2_v4/jrun.py` is not edited. The base and both arms run the identical `run_fold` (sha256 `433d4058…`).

**`research_r2_v4/hrt.py` (new):**
- `declared_pairs()`: the registered pairs, with the pair rule re-derived. Any difference is refused.
- `arm_fold()`: builds the arm's fitting rows.
  - Ablated: every source-s fitting row is removed.
  - Random: the saved index set, sha-checked and re-derived with `matched_removal(seed=0)`.
  - Then `derive_fold` (the v3 candidate policy) is applied and `verify_arm_fold()` checks the result.
- `verify_arm_fold()` refuses:
  - an incomplete source removal;
  - a random arm that removes source-s rows or empties a composition;
  - changed unseen or outer rows;
  - a universe outside the base universe;
  - a fitting-set mismatch.
- `ExclusionAccess(ImageAccess)`: refuses every removed row in every phase (train, probe, extract), on top of the phase allow-lists. The check happens before any image is opened, and it is inherited by DataLoader workers.
- **Arm identity:**
  - Row id: `<row>@HRT_<fold>_<source>_<arm>`.
  - Config: the declared row's config with only `row` changed.
  - Seed 0, `fold_index` = the fold's index in I1–I6, so `torch_seed` is the base model's.
  - The row id enters the checkpoint and feature-cache metadata, so a checkpoint or cache from another row, pair or arm is refused.
- **`run_arm()`:**
  - Output directory: `<out>/<row>/<fold>_<source>/<arm>/attemptN`, a new attempt each time, never overwritten.
  - Calls `jrun.run_fold`, then the H_RT post-checks, then writes `hrt_receipt.json` (the success marker) atomically, only if all checks pass.
  - On failure it writes `failure.json`.
- **`verify_arm()` (used by evaluation) requires:**
  - exactly one successful attempt;
  - the receipt hashes and the receipt-file hash recorded in `hrt_receipt.json`;
  - every `confirm.receipt_problems` check against the arm config;
  - a positive optimizer-step count;
  - the same pretrained initialisation as the base (identical frozen digest before training, weights file and torch seed);
  - a checkpoint different from the base's and the other arm's;
  - checkpoint and feature-cache metadata naming this arm;
  - the fitting and removed row hashes recomputed now;
  - a packet valid against the arm fold's contract.
- **`evaluate()`:**
  - Arms: base (Phase 1 seed-0 packet), ablated and random.
  - All three are restricted to the **ablated fold's** universe and evaluation rows (`common.restrict`, as H_RF). The universes are checked to nest: ablated ⊆ random ⊆ base.
  - γ ∈ {0, the row's seed-0 pooled LOFO γ}, as H_RF.
  - Outcomes are the Stage H set plus Top-5 / H@5, unseen crop and disease, within-crop and false-healthy.
  - Effects: source = ablated − random; size = random − base; total = ablated − base.
  - Paired cluster bootstrap within each pair (2,000 resamples, seed 0): novel-source seen Top-1 clustered by composition, and H_bal by seen and unseen composition.
  - Aggregates (equal weight per pair, stratified bootstrap) come only after the per-pair tables.
  - Side by side: the row's H_RF and D1's Stage H values for the same pair.

**Other code:**
- `scripts/r2v4_hrt.py` (new): `scope` (writes the Gate A record), `run`, `evaluate`.
- `research_r2_v4/confirm.py`: `receipt_problems` also requires a positive optimizer-step count for trained rows. This is a tightening only; Phase 1 and Phase 2 re-evaluation must stay identical.
- `tests/test_r2v4_hrt.py` (new, synthetic CPU backend, real folds and index sets): end-to-end runs of both arms, plus negative tests for every refusal above.

**Execution:**
- 3 rows × 3 pairs × 2 arms = 18 trainings, strictly sequential, into the new directory `experiments/.../v4/stage_hrt_20261116/`.
- GPU monitor every 30 s.
- Expected runtime about 3.5 h. The ablated I6 arm is shorter: 3,112 fitting rows.
