# R2 v4 — J4b three-seed confirmation vs J2 (amendment 4)

**Status:** post-freeze exploratory development cross-validation on the reused v3 inner folds I1–I6; adaptive; not blind.
No official-test image, label, score or inference. No First Run access. Amendments 1–3, v1–v3, J1–J3, H_RT and the J4b
seed-0 evidence are unchanged.

**Verdict: J4b FAILS the declared three-seed confirmation against J2** (mean H_bal +0.86 pp < +1.0; 4/6 folds improved < 5).
The seed-0 screen pass did not survive three-seed averaging. The negative result is kept; thresholds are unchanged.

## 1. Git

| Commit | Time (+08:00) | Content |
|---|---|---|
| `d9ed896` | 17 Nov 02:36 | reviewed checkpoint (J4b seed-0 screen) |
| `42d6474` | 17 Nov 13:33:42 | amendment-4 implementation and tests (before declaration) |
| `9122366` | 17 Nov 13:33:59 | **amendment 4 declared** (`amendment_4.json` `53c4b33a…faea`, `DECLARATION_AMENDMENT_4.json` `1e7ad336…cfd9`) |
| results commit | after the run | this report, the final decision report, `artifacts/r2/v4/stage_j4b_confirm_20261117/` |

J2 preflight started 13:35:48; J4b seed-1 training 13:36:54; seed 2 14:36:16; evaluation 15:35:39–15:39:14. Every receipt
of seeds 1–2 records `git_head = 9122366` and a clean worktree. The worktree stayed clean during the run.

## 2. Gate A — append-only mechanism

Amendment 3 pins 24 files (runner, evaluator, CLI, declaration tool, tests). It authorises seed 0 only, its declaration
tool has no amendment 4, and its evaluator takes one child directory. None of those files was edited. New files only:

* `research_r2_v4/amendment4.py`: authorisation chain, production runner entry, per-seed-base confirmation pair, six-fold
  decision, report;
* `scripts/r2v4_a4.py`: `build`, `declare`, `verify`, `run --seed`, `evaluate`;
* `tests/test_r2v4_amendment4.py`: 16 tests;
* `docs/R2_v4_Amendment4_Plan.md`: the plan (inspection findings and design);
* `research_r2_v4/amendment_4.json` + `DECLARATION_AMENDMENT_4.json`.

Every production entry runs `check_authorised(row, seed, purpose)`:

1. amendment 3 verifies with the **pinned** tool, files and 24 code pins;
2. `amendment_3.json` and `DECLARATION_AMENDMENT_3.json` keep their recorded sha256;
3. amendment 4 verifies, files and 28 code pins (the four new files plus all 24 amendment-3 pins);
4. the (row, seed, purpose) is listed: row J4b; training seeds `[1, 2]`; evaluation `confirm`, seeds `[0, 1, 2]`, folds I1–I6.

Then `check_pinned` compares the registered fold identities (ordered evaluation identity, contract, fitting rows), the J4b
and J2 configuration hashes, the single modification (`augment = photometric`), and the receipt hashes of J2 seeds 0–2 and
J4b seed 0 (plus seed-0 sidecars) with the amendment-4 pins. Only then does the runner call the unchanged amendment-3 shared
implementation `child_run._run`, writing to a fixed directory per seed. The amendment-3 runner still refuses seeds 1–2.

## 3. Gate B — tests

* `CUDA_VISIBLE_DEVICES='' python -m unittest discover -s tests -p 'test_r2*.py' -v` → **Ran 186 tests in 1753.7 s, OK**
  (170 earlier + 16 amendment-4). Log sha256 `5b57250b6103…acb98`.
* After the declaration, the real-file classes were re-run: 6 tests, OK (`tests_post_declaration.log`).
* Declarations: base and amendments 1–3 `--verify` OK; `r2v4_a4.py verify` OK before the run and after it.
* What the 16 tests cover:
  * seeds 1 and 2 pass only with a valid amendment 4;
  * seeds 0, 3, −1 and `"1"`, rows J5, J4a and J2, and an unknown purpose are rejected;
  * a widened authorisation, a changed code pin, a changed amendment 3, a failed pinned amendment-3 check, a missing
    declaration, re-declaration, or a wrong amendment-3 link are all refused;
  * amendment 3 is byte-identical to `d9ed896` and still refuses seeds 1–2;
  * the historical artifact SHA256SUMS hold for Phase 1/2, H_RT, both post-review folders and the J4b seed-0 stage, and so
    do the seed-0 private files;
  * J4b equals J2 except the photometric augmentation;
  * protected directories are refused;
  * on synthetic three-seed, two-base fixtures the following fail closed: a missing sidecar, a mismatched image identity,
    a stale feature cache, a wrong checkpoint, a changed packet, a missing receipt, a failed attempt, a missing fold, a
    duplicate attempt, a missing or extra seed directory in a base, a missing or extra seed, a mismatched configuration
    and a re-derivation mismatch;
  * H_bal is the only endpoint, a 5-fold or reordered subset is refused, and exactly seeds {0, 1, 2} are required;
  * forged decisions and seed-relabelled tables are refused;
  * the report is consistent with the gate.
* An earlier full-suite launch was stopped before completion, so that the evaluation report could first be made testable
  (moved into `amendment4.confirmation_report`) before anything was declared. Its partial log is kept as
  `tests_gateB_ABORTED_before_refactor.log`.

## 4. Gate C — J2 parent preflight (pinned `r2v4_child.py preflight`)

| Seed | Run dir | Folds | Max re-derivation diff | Result |
|---|---|---|---|---|
| 1 | `stage_j_phase2_20261115` | I1–I6 | 1.9073e-06 (≤ 1e-5) | pass (32.5 s) |
| 2 | `stage_j_phase2_20261115` | I1–I6 | 1.9073e-06 | pass (30.1 s) |

Receipts, file hashes, registered fold, contract, candidate ids, ordered seen/unseen file hashes, checkpoint, feature and
packet hashes are in `preflight_J2_seed{1,2}.json`. J2 was not retrained or modified, and its fingerprint was unchanged before
and after each seed run.

## 5. Gate D — J4b seeds 1–2 (12 fold runs)

| Seed | Folds ok | Attempts | Seconds per fold | Peak GPU train / total (GiB) | Steps | AMP skipped | Grad checks | Tensors changed | Reload diff | Torch seeds | Re-derivation |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 6/6 | 1 each | 582–591 | 3.96 / 4.33 | 785–790 | 1–2 | 18/18 | 51/51 | 0.0 | 1000–1005 | 1.9e-6 |
| 2 | 6/6 | 1 each | 578–598 | 3.96 / 4.33 | 785–790 | 1–2 | 18/18 | 51/51 | 0.0 | 2000–2005 | 1.9e-6 |

No failure, rejection or retry: 12 attempt directories, each with a sidecar. Training ran from 13:36:54 to 15:35:39 (≈ 2 h).
GPU: peak 6.39 GiB device memory, mean utilisation 91.6%, maximum temperature 90 °C. One GPU job at a time.

## 6. Gate E — three-seed confirmation (H_bal, each row's own LOFO γ)

| | Result | Threshold | Met |
|---|---|---|---|
| Mean H_bal change (per-fold seed means) | **+0.86 pp** (70.42 vs 69.56) | ≥ +1.0 | **no** |
| Folds improved | **4/6** (I2 −0.26, I6 −0.24) | ≥ 5 | **no** |
| AUSUC change | +0.83 pp | ≥ 0 | yes |
| Healthy-shortcut change | −0.11 pp | ≤ +5 | yes |
| **Decision** (`promotion_decision`, confirm) | **fail** | | |

Per seed and fold, H_bal change J4b − J2 (pp):

| Seed | I1 | I2 | I3 | I4 | I5 | I6 | Mean | Folds + | AUSUC Δ | Shortcut Δ |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | +2.75 | +0.87 | +1.36 | +1.66 | +1.98 | −0.15 | +1.41 | 5 | +1.87 | +0.60 |
| 1 | +3.29 | −0.78 | +1.39 | +2.12 | +2.89 | +0.41 | +1.55 | 5 | +0.78 | −0.74 |
| 2 | −1.43 | −0.86 | +0.11 | −0.12 | +1.03 | −0.98 | **−0.37** | 2 | −0.15 | −0.20 |
| **Mean** | +1.54 | −0.26 | +0.95 | +1.22 | +1.97 | −0.24 | **+0.86** | **4** | +0.83 | −0.11 |

Absolute H_bal (J4b / J2, %):
* seed 0: I1 61.58/58.83, I2 75.23/74.37, I3 69.01/67.65, I4 78.50/76.84, I5 73.35/71.37, I6 69.95/70.10
* seed 1: 61.68/58.39, 72.13/72.92, 70.71/69.31, 77.35/75.23, 71.39/68.50, 70.20/69.79
* seed 2: 57.82/59.25, 72.43/73.29, 69.65/69.54, 76.00/76.12, 72.15/71.12, 68.49/69.47

LOFO γ: J4b 7.75–8.75 and J2 7.75–8.75; the mean γ is 8.36 for J4b and 8.15 for J2.

**Uncertainty (exploratory only):**
* the paired composition-cluster bootstrap of the three-seed mean gives a 95% CI of [+0.07, +1.69] pp;
* the share of draws ≤ 0 is 0.018, which is descriptive and not a p-value;
* the CI excludes 0 but also excludes neither the +1 pp threshold nor values near 0, and it ignores seed variance (seed 2 is
  negative).

**Calibration-matched sensitivity** (J4b scored at J2's γ): +0.66 pp. Per fold: I1 +0.25, I2 −0.45, I3 +1.28, I4 +1.08,
I5 +1.70, I6 +0.12. That is 5/6 positive, but the mean is below the threshold.

## 7. Diagnostics (no pass/fail; cannot substitute for the gate)

Seed-mean, fold-mean, each row's own LOFO choice (J4b / J2 / Δ pp):

| Metric | J4b | J2 | Δ |
|---|---|---|---|
| H_bal | 70.42 | 69.56 | +0.86 |
| H_micro | 65.75 | 65.05 | +0.70 |
| Seen Top-1 (image-weighted) | 74.23 | 74.64 | −0.42 |
| Seen Top-5 | 97.32 | 96.97 | +0.35 |
| Unseen Top-1 (image-weighted, micro) | 60.13 | 58.82 | +1.31 |
| Unseen Top-1 (composition macro) | 68.09 | 66.33 | +1.76 |
| Unseen Top-5 (micro) | 96.81 | 97.22 | −0.41 |
| Unseen crop | 89.46 | 88.96 | +0.50 |
| Unseen disease | 62.16 | 60.91 | +1.24 |
| Unseen within-crop disease (macro) | 78.53 | 77.39 | +1.14 |
| Unseen healthy shortcut | 2.34 | 2.45 | −0.11 |
| AUSUC | 61.94 | 61.11 | +0.83 |

* **H_micro per fold:** I1 +1.82, I2 −1.26, I3 +1.65, I4 −0.45, I5 +3.03, I6 −0.58.
* **Unseen disease per fold:** I1 +2.50, I2 −0.65, I3 +2.76, I4 −2.02, I5 +5.73, I6 −0.86. The mean is +1.24, but the folds
  range from −2.0 to +5.7.
* **Healthy shortcut per fold (seed mean):** I1 −0.90, I2 −0.65, I3 −0.53, I4 −1.27, I5 +0.11, **I6 +2.59** (8.20% vs 5.61%).
  For I6 the seeds give +5.83, −0.97 and +2.91. The I6 increase seen at seed 0 is reproduced in direction by two of three
  seeds, with a smaller mean. It stays within the gate's mean limit.
* **False-healthy** (unseen images predicted healthy, mean over folds):
  * J4b 3.04 / 1.74 / 2.23% at seeds 0 / 1 / 2;
  * J2 2.44 / 2.48 / 2.43%;
  * diseased seen images predicted healthy: J4b 2.03 / 2.06 / 2.06%, J2 2.23 / 2.11 / 1.98%.

**Source strata (seen-val Top-1; seed mean, J4b / J2 / Δ pp; per-seed Δ):**

| Stratum | J4b | J2 | Δ | Seeds 0 / 1 / 2 | Image-folds | Unique images | Compositions | Sources | Composition-source clusters |
|---|---|---|---|---|---|---|---|---|---|
| Dominant fitting source | 76.42 | 76.58 | −0.16 | +0.63 / −0.02 / −1.09 | 4,608 | 809 | 195 | 39 | 195 |
| Other fitting source | 59.77 | 62.21 | **−2.44** | −1.43 / −1.27 / −4.62 | 628 | 111 | 30 | 15 | 40 |
| Novel source | 36.90 | 34.52 | +2.38 | +3.57 / +7.14 / −3.57 | 28 | **5** | 3 | 3 | 3 |

The non-dominant-source stratum gets worse with J4b in every seed. Its 628 image-fold instances are only 111 unique images
in 40 composition-source clusters. The novel-source stratum is 5 images, so its swings of ±3.6–7.1 pp are 1–2 image changes
and uninformative.

## 8. Answers to the review questions

1. **H_bal and H_micro:** +0.86 and +0.70 pp; neither is consistent across folds (2/6 and 3/6 negative).
2. **Seen / unseen Top-1 / Top-5:**
   * seen: Top-1 −0.42, Top-5 +0.35;
   * unseen: Top-1 +1.31 (micro) and +1.76 (macro), Top-5 −0.41;
   * the small gain is at Top-1 on unseen compositions, with a small seen cost.
3. **Unseen crop / disease / within-crop disease:** +0.50 / +1.24 / +1.14 pp, with large fold-to-fold variation in disease.
4. **Healthy shortcut:** −0.11 pp overall; I6 +2.59 pp (seed range −0.97 to +5.83).
5. **Non-dominant-source accuracy:** −2.44 pp (111 unique images, 40 composition-source clusters, negative in all three seeds).
6. **Does the seed-0 improvement survive three-seed averaging?** **No.**
   * Seed 1 reproduced it (+1.55, 5/6), but seed 2 reversed it (−0.37, 2/6).
   * The three-seed mean +0.86 pp fails the declared +1 pp and 5/6 rule.
   * The seed-to-seed spread of J4b − J2 (−0.37 to +1.55 pp) is larger than the effect.
7. **Is the registered mechanism supported?** **No.**
   * Photometric augmentation was registered to improve robustness to source and acquisition appearance. The stratum that
     tests this within the fitting distribution (other fitting source) is worse in every seed, and the novel-source
     stratum is too small to read.
   * No source-exclusion retraining was run for J4b, so no source-robustness claim of any kind can be made.

## 9. Limitations

* The folds are reused development folds (v1–v3, Stage I, J1–J3 and the J4b screen all saw them), so nothing here is
  blind.
* J4b was adaptively selected after H_RT. J2 was fixed as parent after seeing all six folds, so this is a paired
  exploratory comparison, not a nested estimate.
* Results depend on calibration: LOFO γ is about 8, and H_bal at γ 0 is 5–9%.
* Image-fold counts are not independent observations.
* The bootstrap ignores seed variance.

## 10. Evidence

* **Public:** `artifacts/r2/v4/stage_j4b_confirm_20261117/`. It holds:
  * `evaluation.json` (sha256 `21d1de61…28a1`);
  * 12 receipts and 12 sidecars;
  * the J2 preflights, statuses, GPU log, logs, run script and test logs;
  * `SHA256SUMS` (48 files) and `private_files_SHA256SUMS.txt` (60 run files);
  * `failures_and_rejections.txt` (none; 12 attempt directories).
* **Private:** checkpoints, features and packets under `experiments/r2-compositional/v4/stage_j4b_confirm_20261117/runs`.
* **Commands:**
  * `python scripts/r2v4_a4.py build`
  * `python scripts/r2v4_a4.py declare`
  * `python scripts/r2v4_child.py preflight --seed {1,2} --out …`
  * `python scripts/r2v4_a4.py run --seed {1,2}`
  * `python scripts/r2v4_a4.py evaluate`
  * all wrapped by `run_j4b_seeds12.sh`.
