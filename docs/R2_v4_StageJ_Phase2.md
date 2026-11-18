# R2 v4 Stage J Phase 2: three-seed confirmation of J1–J3 (15–16 Nov 2026)

**Status:** post-freeze exploratory development cross-validation on the reused v3 inner folds I1–I6. Nothing is promoted, and D1 remains the provisional R2 candidate. No official-test file was opened, and no official inference was run.

**Scope held:**
- not run: J4–J8, H_RT, extra seeds, new model families;
- untouched: First Run, PR #34 and v1–v3 evidence;
- seed 0 was neither rerun nor rewritten;
- no merge, no tag, no push.

## 1. Git

| | |
|---|---|
| Branch | `research/r2-v4-ood-exploration` (worktree `workspace/worktrees/r2-v4`) |
| Start | `7f23ef1` (Phase 1 results; clean). GitHub PR #36 is still at `1b2fcdb`. |
| `8cd0b60` | fix(r2-v4): fail-closed three-seed confirmation in the Stage J evaluator. Committed **before** any Phase 2 training. |
| Results commit | this document, `scripts/r2v4_phase2_supplement.py` and `artifacts/r2/v4/stage_j_phase2_20261115/` |
| Worktree | clean at the start (the launcher refuses a dirty tree) and clean after the run |

## 2. Gate A: the evaluator repair

**Defects in the evaluator at `e96d471`** (`scripts/r2v4_stage_j.py`, `cmd_evaluate`):
1. It treated any set of ≥ 3 seed directories as a three-seed confirmation (`len(by_seed) >= 3`), so {0, 1, 3} or {0, 1, 2, 3} were accepted.
2. It took `min(seeds)` as the seed-0 screen, so with seed 0 missing, seed 1 silently stood in for it.
3. An incomplete seed was dropped, not refused, and the confirmation was then simply absent.
4. Seeds were not checked to be distinct runs of one declared configuration. A copied seed directory counted twice.
5. Only each seed's nested mean was recorded: no average, no held-fold results, no verdict.

**Files changed:**

| File | Change | Rationale |
|---|---|---|
| `research_r2_v4/confirm.py` (new) | Run discovery, receipt verification, cross-seed checks, three-seed statistics | Keeps the fix out of the training code. `jrun.py` is byte-identical (sha256 `433d4058…`), so seeds 1–2 ran the same training code as seed 0. |
| `scripts/r2v4_stage_j.py` | `evaluate` uses `confirm.*`, reads `--runs` (several run directories), and writes separate labelled outputs; `hrf` uses the same run verification | Training path (`cmd_run`) unchanged |
| `tests/test_r2v4_confirm.py` (new, 25 tests) | Negative tests for every refusal, an independent recomputation, and the old evaluator run on the same fixtures | Required regression evidence |

**Rules implemented, each taken from the existing declarations:**
- **Seeds:** exactly `common.seeds.confirmation_if_gate_passes = [0, 1, 2]` from `stage_j_v4.json`. Seed directory names must be canonical.
- **Folds:** exactly one successful attempt per fold, verified by receipt hashes.
- **Receipt provenance:**
  - row, seed and fold match the directory;
  - the config hash matches the recorded config, and the config equals the declared row (workers excluded, no `max_steps`);
  - `torch_seed = 1000·seed + fold_index`;
  - the checkpoint sha is present and equals the feature-cache metadata;
  - reload difference 0.0, frozen digests equal, all trainable tensors changed;
  - gradient checks all finite, AMP skips ≤ 5%;
  - code identity present, and the run came from a clean worktree.
- **Cross-seed:**
  - one config hash and one `jrun.py` hash;
  - distinct checkpoint, features, packet and torch seed per fold (otherwise it is a duplicate run);
  - identical candidate ids and truth;
  - a (row, seed) present in two run directories is refused.
- **Statistics (Amendment 2):**
  - each seed gets its own LOFO choice per fold;
  - the gate is applied to the per-fold seed mean of H_bal, AUSUC and healthy shortcut;
  - the nested estimate is computed per seed and averaged, with held-fold improvement counted on the per-fold seed mean;
  - verdict = gate pass **and** averaged nested ≥ +1.0 pp and ≥ 5/6 (protocol_v4 nested rule).
- **Precondition:** confirmation is refused unless the seed-0 screen passed.

**Unchanged:** metric definitions, the γ grid, the candidate universe, the thresholds, the tie rules and the selection procedure.

**Regression on real data:**
- Before training, the repaired evaluator was run on the Phase 1 runs (`gateA_regression/`).
- The seed-0 screen of all three rows, the stage nested estimate, the parent by rule and the parent audit are **identical** to the Phase 1 `evaluation.json`.
- All 18 seed-0 receipts pass the stricter checks.

**Tests:**
- Command: `python -m unittest discover -s tests -p 'test_r2*.py' -v`.
- Result: `Ran 117 tests in 767.777s … OK` (92 existing + 25 new). Log `logs/tests_pre_phase2.txt`, sha256 of the unredacted private log `517115f8…c30c`.
- Negative tests:
  - seed set {0, 1, 3}; an extra seed 3; seed 0 missing; the alias `seed01`;
  - a missing fold; a failed fold; two successful attempts;
  - a copied seed, with and without relabelled receipts; the same seed in two run directories;
  - an incompatible config; a config-hash mismatch; a declared-config mismatch; a smoke run;
  - different training code; a permuted candidate mapping; checkpoint provenance missing or mismatched;
  - ten invalid-receipt mutations; a dirty worktree;
  - a failed screen blocking confirmation.
- **The old evaluator wrongly accepted** three fixtures (`TestPreviousEvaluatorAcceptedInvalidFixtures`, run from `git show e96d471:…`):
  - for {0, 1, 3} it produced a three-seed block over seeds 0, 1, 3;
  - with seed 0 missing it used seed 1 as the screen;
  - a copied seed directory was counted twice (identical per-seed nested values).

## 3. Gate B: execution

- **Launcher** (`run_phase2.sh.txt`): `scripts/r2v4_stage_j.py run --row <J> --seed <1|2> --out <E>/stage_j_phase2_20261115` for J1 s1, J1 s2, J2 s1, J2 s2, J3 s1, J3 s2, strictly sequential. Then `hrf` for the same six, then `evaluate --runs <phase1> <phase2>`.
- **Settings:** the declared 5 epochs, optimiser, scheduler, preprocessing, flip augmentation, composition-balanced sampler, AMP and checkpoint policy, as recorded in the receipts. Every config hash equals seed 0's.
- **Timing:** start 16 Nov 00:17:59, end 10:06:01 (training 7 h 45 min, H_RF 2 h 00, evaluation 3 min).

**36/36 folds ok on their first attempt; 0 failures; 0 retries.**

| Row, seed | Wall time | s / fold | Peak GPU train / total (GiB) | AMP skipped / steps | Grad checks | Tensors changed | Reload diff |
|---|---|---|---|---|---|---|---|
| J1 s1 | 51.5 min | 510–518 | 2.84 / 3.85 | 1–2 / 785–790 | 18/18 finite | 27/27 | 0.0 |
| J1 s2 | 51.3 min | 510–515 | 2.84 / 3.85 | 1–2 / 785–790 | 18/18 | 27/27 | 0.0 |
| J2 s1 | 57.5 min | 570–579 | 3.96 / 4.33 | 1–2 / 785–790 | 18/18 | 51/51 | 0.0 |
| J2 s2 | 58.0 min | 574–584 | 3.96 / 4.33 | 1–2 / 785–790 | 18/18 | 51/51 | 0.0 |
| J3 s1 | 2 h 02 | 1,208–1,238 | 2.53 / 3.42 | 2 / 785–790 | 18/18 | 96/96 | 0.0 |
| J3 s2 | 2 h 04 | 1,232–1,241 | 2.53 / 3.42 | 2 / 785–790 | 18/18 | 96/96 | 0.0 |

- **Per-fold checks:** frozen digests unchanged in every fold. Every fold's feature cache metadata and packet validated before its receipt was written.
- **GPU monitor** (`gpu_monitor.csv`, 30 s, 1,176 samples): peak device memory 6.0 GiB of 8 GiB (including the display), mean utilisation 80% over training and H_RF, maximum temperature 89 °C.
- **Mean training loss by epoch (folds averaged):** J1 6.05 → 0.20; J2 5.79 → 0.13; J3 6.56 → 0.30. These are essentially identical across seeds (`logs/`, receipts).
- **Warnings:** the PyTorch "lr_scheduler.step() before optimizer.step()" warning from AMP's first skipped step appears as in Phase 1.

## 4. Gate C: results vs D1 (66.20)

LOFO γ throughout. pp = percentage points. Bootstrap = paired composition-cluster, 2,000 resamples. The three-seed bootstrap uses the same resampled compositions for every seed.

**Per seed:**

| Row | Seed | H_bal | Δ vs D1 | Folds better | AUSUC | Nested Δ (folds) | 95% CI |
|---|---|---|---|---|---|---|---|
| J1 | 0 | 69.00 | +2.80 | 5/6 | 59.50 | +2.80 (5/6) | [+0.78, +5.04] |
| J1 | 1 | 69.44 | +3.23 | 6/6 | 60.44 | +3.23 (6/6) | [+1.31, +5.45] |
| J1 | 2 | 69.01 | +2.81 | 5/6 | 60.38 | +2.42 (4/6) | [+0.89, +5.38] |
| J2 | 0 | 69.86 | +3.66 | 6/6 | 60.64 | +3.66 (6/6) | [+0.98, +6.53] |
| J2 | 1 | 69.02 | +2.82 | 6/6 | 61.36 | +2.82 (6/6) | [+0.04, +6.03] |
| J2 | 2 | 69.80 | +3.59 | 6/6 | 61.31 | +3.59 (6/6) | [+0.95, +7.02] |
| J3 | 0 | 69.51 | +3.31 | 5/6 | 60.60 | +3.31 (5/6) | [+1.34, +5.68] |
| J3 | 1 | 70.33 | +4.13 | 6/6 | 62.10 | +4.13 (6/6) | [+1.83, +7.01] |
| J3 | 2 | 69.87 | +3.67 | 6/6 | 61.85 | +3.67 (6/6) | [+1.35, +6.50] |

**Three-seed confirmation (gate on the per-fold seed mean; nested averaged over seeds):**

| | D1 | **J1** | **J2** | **J3** |
|---|---|---|---|---|
| H_bal | 66.20 | 69.15 (+2.95) | 69.56 (+3.36) | 69.90 (+3.70) |
| Folds better (per-fold seed mean) | – | 6/6 | 6/6 | 6/6 |
| AUSUC (Δ) | 58.02 | 60.11 (+2.09) | 61.11 (+3.08) | 61.51 (+3.49) |
| Healthy shortcut Δ (limit +5) | – | +0.39 | +1.04 | +0.63 |
| Nested vs D1, seed average (folds) | – | +2.82 (6/6) | +3.36 (6/6) | +3.70 (6/6) |
| Three-seed bootstrap Δ, 95% CI | – | +2.95 [+1.19, +5.11] | +3.36 [+0.83, +6.33] | +3.70 [+1.63, +6.26] |
| Seed SD / range of H_bal | – | 0.25 / 0.44 | 0.47 / 0.84 | 0.41 / 0.82 |
| **Verdict** | – | **confirmed** | **confirmed** | **confirmed** |

**Per-fold Δ H_bal, three-seed mean (pp):**

| Row | I1 | I2 | I3 | I4 | I5 | I6 |
|---|---|---|---|---|---|---|
| J1 | +3.54 | +0.46 | +3.37 | +4.88 | +3.24 | +2.19 |
| J2 | +4.20 | +0.82 | +4.47 | +6.65 | +1.92 | +2.09 |
| J3 | +5.81 | +3.22 | +4.27 | +6.18 | +0.27 | +2.47 |

**Regressions and outliers in single seeds:**
- J1 is −0.20 (s0) and −0.05 (s2) on I2.
- J3's I5 is −1.74 (s0), +0.07 (s1) and +2.49 (s2). That 4.2 pp spread is the largest fold spread observed.
- J2 seed 1 is the weakest J2 seed (+2.82; CI lower bound +0.04).
- No seed of any row fails the per-seed gate.

**Descriptive metrics (three-seed mean, LOFO γ):**

| | D1 | J1 | J2 | J3 |
|---|---|---|---|---|
| H_micro | 61.56 | 64.90 | 65.05 | 64.61 |
| Seen Top-1 | 71.62 | 73.73 | 74.64 | 72.81 |
| Unseen composition macro / micro | 62.89 / – | 66.16 / 59.14 | 66.33 / 58.82 | 68.45 / 59.48 |
| Unseen disease Top-1 | 58.37 | 61.30 | 60.91 | 61.25 |
| Unseen crop Top-1 | 87.66 | 87.76 | 88.96 | 89.34 |
| Within-crop disease (unseen macro) | 75.2 | 77.97 | 77.39 | 79.76 |
| H@5 (micro) | – | 96.81 | 97.07 | 96.87 |
| Selected γ (mean) | – | 8.24 | 8.15 | 8.12 |
| H_bal at γ = 0 (s0 / s1 / s2) | – | 7.5 / 8.1 / 7.0 | 7.1 / 6.8 / 5.3 | 7.7 / 8.0 / 8.8 |
| False-healthy, seen diseased (s0 / s1 / s2) | 2.36 | 2.28 / 2.11 / 2.26 | 2.23 / 2.11 / 1.98 | 2.31 / 2.16 / 2.31 |
| False-healthy, unseen (s0 / s1 / s2) | 1.41 | 2.02 / 1.69 / 1.69 | 2.44 / 2.48 / 2.43 | 2.18 / 1.91 / 2.01 |

Calibration curves for every row and seed are in `curves/`. Calibration stays essential: H_bal at γ = 0 is 5–9%.

**Stage-level nested selection** (D1 → best gate-passing of {J1, J2, J3}, every choice made on five folds):

| | Mean Δ | Held folds better | Selections |
|---|---|---|---|
| Seed-0 screen tables | +2.68 | 5/6 | J2 ×4, J3 ×2 |
| Seed 1 | +4.13 | 6/6 | J3 ×6 |
| Seed 2 | +3.27 | 6/6 | J2 ×2, J3 ×4 |
| **Three-seed average** | **+3.36** | **6/6** (per-fold mean: I1 +5.30, I2 +2.07, I3 +3.93, I4 +6.18, I5 +0.27, I6 +2.40) | meets thresholds |

**Paired comparisons, J2 vs J1 and J2 vs J3** (descriptive; positive = J2 better):

| | Mean Δ (3 seeds) | Per seed (s0 / s1 / s2) | Folds J2 better | Seeds J2 better | Three-seed bootstrap 95% CI | P(Δ ≤ 0) |
|---|---|---|---|---|---|---|
| J2 − J1 | +0.41 | +0.86 / −0.41 / +0.78 | 4/6 | 2/3 | [−0.89, +2.00] | 0.27 |
| J2 − J3 | −0.34 | +0.35 / −1.31 / −0.07 | 3/6 | 1/3 | [−2.51, +1.65] | 0.60 |

- **J2's seed-0 lead is not stable.** On the three-seed mean, J3 is higher than J2 by 0.34 pp and J1 is lower by 0.41 pp. Every pairwise interval contains 0, and J3 − J1 is +0.75 [−1.13, +3.10].
- **Under the available evidence the three variants are indistinguishable.** The confirmed finding is that partial fine-tuning of CLIP ViT-L/14 (two blocks, four blocks or LoRA-r8 MLP) improves H_bal over frozen-feature D1 by about 3–3.7 pp, reproducibly across seeds. The finding is not about which variant.
- **Independent calculation check** (`supplement.json`):
  - The three-seed gate was recomputed from the score packets without `RowTable` or `ev.gate`, using its own per-fold LOFO argmax with the smallest-γ tie rule and hand-written thresholds.
  - It matches the evaluator: |Δ mean gain| ≤ 7e-15 pp, |Δ AUSUC| ≤ 1.1e-16, shortcut and fold counts identical, verdicts identical.
  - The unit tests add an independent recomputation of the gate and the averaged nested estimate on the synthetic fixture.

**Parent rule for J4a–J7** (declared: best of J1–J3 by seed-0 screen H_bal):
- The parent stays **J2_clipL_last4**, unchanged.
- The inner parent is J3 for held folds I4 and I5, so J4a–J7 will be excluded from the nested estimate there.
- The three-seed means would favour J3. Changing the parent after seeing these results would be a post-hoc selection, so it is not done here. Any change needs an amendment before any J4 run.

## 5. Gate D: H_RF source diagnostic

H_RF keeps the representation fixed and refits D1 on the saved Stage H index sets:
- 49 pairs per row and seed; 9 sources for the novel-source metric, 16 for whole-fold metrics; all 6 × 49 pairs ok; 0 failed;
- every matched-control index set re-derived equal to Stage H.

**Novel-source seen Top-1, at each seed's selected γ (pp; cluster bootstrap over sources):**

| Row | Source effect, s0 / s1 / s2 | **Source effect, three-seed pooled [95% CI]** | Matched random-removal (size) effect, pooled |
|---|---|---|---|
| D1 (Stage H, γ 8) | −49.3 | −49.3 [−57.3, −40.5] | −0.2 |
| J1 | −37.9 / −38.2 / −38.7 | −38.3 [−47.1, −29.6] | +0.2 [−1.6, +2.1] |
| J2 | −31.2 / −30.1 / −31.2 | −30.8 [−39.9, −22.8] | −1.7 [−2.6, −0.9] |
| J3 | −40.2 / −38.0 / −41.9 | −40.0 [−50.8, −27.9] | −0.4 [−1.1, +0.5] |

**Other metrics, three-seed pooled:**
- Whole-fold H_bal source effect: −0.02 (J1), −0.04 (J2), −0.18 (J3); size effect −0.3 to −0.5.
- Source-s unseen Top-1: CIs span 0 (J1 +1.8, J2 +1.5, J3 −0.8).
- At γ 0 the novel-source seen effect is −34.0 (J1), −22.6 (J2) and −35.1 (J3).

**Limitation:** H_RF does **not** show source-independent generalisation. Each J backbone was fine-tuned **with** source s present, and only the D1 downstream is refitted without it. For D1 the frozen CLIP backbone never saw any DPD source, so its H_RF is a full measure. The smaller J magnitudes are therefore expected by construction, and H_RT was not run.

## 6. Limitations

- These are development cross-validation results on folds that already informed v1–v3, Stage I and Phase 1. The nested estimates (+2.8 to +3.7 single-row; +3.36 stage-level) are the defensible summary, but none of it is blind or independent.
- About 20 rows have been declared across v4. Three-seed confirmation and nested estimates address chance, not fold reuse.
- **Seed noise vs variant differences:** seed SD is 0.25–0.47 pp, and fold-level seed spread reaches 4.2 pp (J3, I5). Variant differences (≤ 0.75 pp) are within that noise.
- **Where the gain comes from:** unseen disease Top-1 rises by 2.5–2.9 pp and unseen crop by 0.1–1.7 pp, while the gain is largest on unseen composition macro (J3 +5.6) and seen Top-1 (J2 +3.0). The v3 bottleneck, disease transfer, still improves least.
- **False-healthy predictions on unseen images** rise from 1.41% to 1.7–2.5%, within the shortcut gate (+0.4 to +1.0 pp vs a limit of +5).
- The γ-0 H_bal is far below the calibrated value. All results depend on LOFO calibrated stacking, as in v3.

## 7. Evidence

`artifacts/r2/v4/stage_j_phase2_20261115/` contains:
- `evaluation.json`: seed-0 screen, three-seed confirmation, stage nested (seed 0 and three-seed), parent rule and paired parent comparisons;
- `supplement.json`: independent recomputation, J2 paired comparisons and three-seed H_RF pooling;
- `curves/` (9 row × seed curves) and `receipts/` (36);
- `hrf/` (pairs and summary for seeds 1–2; seed 0 is in Phase 1);
- `gpu_monitor.csv`, `code_start.txt`, `run_phase2.sh.txt`;
- `logs/` (redacted);
- `gateA_regression/`: the repaired evaluator on the Phase 1 runs;
- `SHA256SUMS`;
- `private_files_SHA256SUMS.txt`: 108 checkpoints, feature caches and packets, kept private in `workspace/experiments/r2-compositional/v4/stage_j_phase2_20261115/`.

## 8. Recommendation (registered evidence only)

1. **H_RT.** All three rows pass the three-seed gate, so all three qualify for H_RT under Amendment 2:
   - scope: pairs (I6, CDDM), (I1, MTID), (I2, CPDD); seed 0; ablated arm plus matched random control;
   - cost: 6 fold trainings per row, about 52 min (J1), 58 min (J2) and 2 h 04 (J3), so about 4 h for all three;
   - H_RT is the only declared test of whether the fine-tuned gain is source-robust;
   - **the retrain path is not implemented yet**, so it needs implementation and a fail-closed audit before any GPU run.
2. **J4a–J7.** Run on the declared parent **J2**, without re-choosing the parent from three-seed means. Record that the parent choice is not supported by the three-seed comparison (J2 vs J3 −0.34, CI [−2.51, +1.65]). Report the I4/I5 child exclusions. Any parent change needs an amendment before any J4 run.
3. **J8** (ConvNeXt stage-4 + frozen CLIP) is independent of the parent and can run under its declared settings against D1 and F10.
4. **Do not promote any J row to replace D1** on this evidence alone. The finding is a reproducible development-CV gain for partial fine-tuning, with source robustness untested (H_RT) and no blind evaluation.
