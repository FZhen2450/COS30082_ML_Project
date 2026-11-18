# R2 v4 — Amendment 4 plan: J4b seeds 1–2 and three-seed confirmation vs J2 (append-only)

Status: post-freeze **exploratory** development cross-validation on the reused v3 inner folds I1–I6. Adaptive (J4b was
prioritised after H_RT). Nothing here is blind, independent or confirmatory in the statistical sense; "confirmation"
means the declared three-seed child-vs-J2 gate of amendment 3.

## Why an amendment 4 (Gate A inspection)

* `research_r2_v4/amendment_3.json` (sha256 `ba3f2442…3456`) authorises `J4b_best_photometric_aug`, **seed 0**, screen only.
  Its `code_pins` fix 24 files, including the runner (`child_run.py`), the evaluator (`child_inputs.py`, `child_gate.py`),
  the CLI (`scripts/r2v4_child.py`) and the declaration tool (`scripts/r2v4_declare.py`).
* `child_run.check_declaration` reads `authorised_now` from amendment 3 → seeds 1–2 are refused by the declared runner. That
  is correct and stays so.
* `scripts/r2v4_declare.py` knows amendments 1–3 only (`AMENDMENTS`); `--amendment 4` is a KeyError. Editing it would break
  amendment 3's code pin, so it is **not** edited.
* `child_inputs.verified_pair(child_base, "confirm")` takes ONE child run directory with all three seeds. Seed 0 lives in
  `stage_j4b_20261117/runs` (evidence, never written again); seeds 1–2 must go to a new directory. A per-seed child base is
  therefore needed for the confirmation evaluator.

Editing any pinned file would invalidate the amendment-3 verification, so everything below is **new files only**.

## Mechanism (smallest append-only change)

| New file | Role |
|---|---|
| `research_r2_v4/amendment4.py` | authorisation, runner entry, per-seed-base confirmation pair, confirmation decision, report |
| `scripts/r2v4_a4.py` | CLI: `build` (writes `amendment_4.json` once), `declare`, `verify`, `run --seed`, `evaluate` |
| `tests/test_r2v4_amendment4.py` | synthetic and real-file tests (Gate B) |
| `research_r2_v4/amendment_4.json` | the authorisation record (written by `build` from committed code, then never rewritten) |
| `research_r2_v4/DECLARATION_AMENDMENT_4.json` | sha256 of amendment_4.json, DECLARATION_AMENDMENT_3.json, amendment_3.json, DECLARATION.json |

Verification chain (every production entry, `amendment4.check_authorised(row, seed)`):

1. amendment 3 still verifies with the **pinned** tool (`r2v4_declare.py --verify --amendment 3`: its files and its 24 code pins);
2. `amendment_3.json` and `DECLARATION_AMENDMENT_3.json` still have the recorded sha256 (`ba3f2442…`, `95358ee8…`);
3. `DECLARATION_AMENDMENT_4.json` matches `amendment_4.json` and the earlier declarations; every amendment-4 code pin matches;
4. the (row, seed) is in amendment 4 `authorised_now` (row `J4b_best_photometric_aug`, seeds `[1, 2]`, mode `confirm`).

Runner (`amendment4.run_child`): the authorised row only; the declared configuration `child_inputs.declared_child_config`
(J2 plus `augment = photometric`, nothing else); then the **unchanged** amendment-3 shared implementation
`child_run._run` (jrun.run_fold, new attempt directory per fold, failures and rejections kept, post-run verification and
re-derivation, then `child_identity.json`). There is no configuration, backend or switch argument on the production entry;
`_run` is reached only after `check_authorised`. Output directory: `stage_j4b_confirm_20261117/runs`, refused if it overlaps
any amendment-3 protected directory or the seed-0 stage directory.

Confirmation evaluator (`amendment4.verified_confirmation_pair`): fixed child bases
`{0: stage_j4b_20261117/runs, 1: stage_j4b_confirm_20261117/runs, 2: stage_j4b_confirm_20261117/runs}` and the registered J2
bases (`child_inputs.PARENT_BASES`). Each base must hold exactly the seed directories it is declared for (no extra or
duplicate seeds). Tables are loaded with `child_inputs.load_verified` (mandatory re-derivation, sidecar, ordered identity,
contract, lineage, sealed), then the same pair checks as `child_inputs._pair` (cross-seed distinctness, same training code,
same identity per fold, distinct child/parent checkpoints). The decision is `child_gate.promotion_decision(C, P, "confirm")`
on all six folds (a fold subset is refused); H_bal is the only promotion endpoint.

## Frozen experiment (unchanged recipe)

`J4b_best_photometric_aug` = J2 (`clip_l14`, last-4 blocks, AdamW lr 1e-5 / 1e-3, wd 0.05, cosine with 1 warm-up epoch, AMP,
5 epochs, batch 32, composition-balanced sampler) + registered photometric augmentation; seed-derived initialisation from the
original pretrained weights, torch seed `1000·seed + fold index`; folds I1–I6 as registered. `jrun.py` is unchanged.

## Gate (amendment 3, unchanged)

Exactly seeds {0, 1, 2}; per-fold seed means; mean H_bal gain ≥ +1 pp; ≥ 5/6 folds improved (strict); mean AUSUC change ≥ 0;
mean healthy-shortcut change ≤ +5 pp; each row's own LOFO (config, γ) choice. Diagnostics (no pass/fail): H_micro, seen /
unseen Top-1 and Top-5, unseen crop / disease / within-crop disease, per-fold shortcut, false-healthy, source strata with
unique-image and composition-source-cluster counts, calibration-matched H_bal, exploratory paired bootstrap.

## Order of work

1. implement + tests (this commit, before the declaration); full R2/v4 test suite;
2. `python scripts/r2v4_a4.py build` → `declare` → commit (before any new scoring);
3. J2 seeds 1–2 preflight with the existing pinned `r2v4_child.py preflight` (re-derivation; stop on any failure);
4. J4b seeds 1, then 2 (one GPU job at a time; no worktree writes during the run);
5. `python scripts/r2v4_a4.py evaluate`; stop and report regardless of outcome.

Not authorised: J4a, J5–J8, H_RT arms for J4b, other seeds, other models, official-test data, First Run access, pushes.
