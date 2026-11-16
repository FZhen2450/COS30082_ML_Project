# PROPOSAL v3 (declaration-ready draft; NOT declared): amendment 3, the J4a–J7 child-versus-parent gate

**Supersedes** v2 (`R2_v4_Amendment3_PROPOSAL_v2.md`) and v1. Both are kept unchanged.

**Status:**
- No amendment is declared.
- No child exists, and no real-data child evaluation has been run.
- Code: `research_r2_v4/child_inputs.py` (verified-input boundary) and `research_r2_v4/child_gate.py` (gate).
- Tests: `tests/test_r2v4_child_inputs.py` and `tests/test_r2v4_child_gate.py`.

**Changes from v2** (the four findings of the independent review):

| Finding | v3 |
|---|---|
| A. Label equality does not establish image identity or order | Verified-input boundary (section 1). The gate refuses any table without its identity record. A same-label image exchange is refused, both in the fold order and inside a packet. |
| B. `p_le_0` is not a calibrated p-value | Renamed `bootstrap_sign_proportion_le_0` and labelled descriptive. Holm removed. An optional formal test is proposed (section 4). |
| C. Primary endpoint must be locked | `H_bal` only. Every other quantity is a diagnostic without pass/fail. `promotion_decision` accepts only the primary result. |
| D. Statistical interpretation | Section 3: an exploratory paired comparison with a fixed comparator, not a nested or independent selection estimate. The claim that comparator bias is conservative is withdrawn. |

## 1. Verified inputs (`child_inputs.load_verified` / `verified_pair`)

Each check below is enforced before any arithmetic; a failure refuses the comparison.

1. **Ordered evaluation image identity.**
   - `identity_sha256` covers the ordered seen-val and unseen-val row ids and training-CSV file names of the **registered** benchmark fold, plus fitting rows, contract, candidate ids and the lineage groups of the evaluation rows.
   - A child attempt must carry `child_identity.json`. Its runner writes it from the fold object actually trained and scored. It names the identity, its own receipt hash, its checkpoint and packet hashes, and the parent row.
   - Old parent receipts store neither the evaluation order (the contract hashes only labels and candidates) nor anything but a sorted row set. So **every** packet, parent and child, is **re-derived**: the unchanged D1 downstream is run on the attempt's verified feature cache in the registered fold order and must reproduce the packet (|Δ| ≤ 1e-5).
   - An exchange of two same-label images changes the scores and is refused (tested).
2. **Fold and lineage.**
   - The registered fold is used, and the receipt contract hash must equal `fold_contract(registered fold)`.
   - The feature-cache row set must equal the registered fold's.
   - No lineage group may be shared between fitting and evaluation rows.
   - A child runner must also pass `check_registered_fold` (ordered lists).
3. **Candidates:** ids, order, crop, disease, healthy mapping and fitting mask, via the contract and `check_pair`.
4. **Recipe and receipts:** `confirm.verified_row_seed`.
   - exactly one successful attempt; file hashes;
   - the declared configuration; torch seed 1000·s + fold index;
   - gradient, AMP (> 0 steps), frozen, reload and update checks;
   - code identity and a clean worktree.
5. **Seeds and checkpoints:**
   - exact seed sets (screen {0}; confirmation exactly {0, 1, 2} as seed directories);
   - `confirm.cross_seed_problems` (distinct checkpoints, features, packets and torch seeds across seeds; one configuration and code);
   - a child checkpoint different from the parent's in every fold and seed.
6. **One registered modification:** the child configuration equals the declared J4a–J7 row on J2 (`jrun.j_rows("J2_clipL_last4")`) and differs from J2's in exactly one field:
   - J4a augment geometric;
   - J4b augment photometric;
   - J5 source_balanced;
   - J6 extra supcon;
   - J7 extra adversary.
7. **Feature caches:** metadata must match on load (row, seed, fold, config, checkpoint, row set). A stale or foreign cache is refused.
8. **Code:** child and parent must have the same `jrun.py` hash. At declaration, the hashes of `research_r2/methods.py` and `runner.py` used by re-derivation must also be recorded.

**Residual:** only the re-derivation verifies a parent packet's row order. It depends on the unchanged downstream code and the 1e-5 tolerance. This is the R2 reproduction standard (the J0 and J8 audits reproduced packets at 0.0).

## 2. Primary gate (endpoint locked: H_bal)

    for s ∈ S, k ∈ K = (I1..I6):
        each of child C_s and parent J2_s takes its own LOFO choice (config, γ) on K∖{k} by mean H_bal
        (ties → lower config, smaller γ);  H_bal, AUSUC and healthy shortcut are read at that choice
        d_s(k) = H_bal(C_s,k) − H_bal(J2_s,k);   a_s(k) = ΔAUSUC;   h_s(k) = Δshortcut
    S = {0} (screen);  S = {0,1,2} exactly (confirmation)
    D(k), A(k), H(k) = means over s ∈ S
    PASS ⇔ 100·mean_k D(k) ≥ 1.0  ∧  #{k : D(k) > 0} ≥ 5  ∧  mean_k A(k) ≥ 0  ∧  100·mean_k H(k) ≤ 5.0

- **Unchanged** from the registered v3 gate: +1 pp, ≥ 5/6, AUSUC non-decrease, shortcut ≤ +5 pp, the seed-0 screen, and confirmation on exactly seeds 0, 1 and 2.
- **Null child:** a child identical to J2 fails (D ≡ 0).
- **Promotion:** `promotion_decision` accepts only `{"role": "primary", "endpoint": "H_bal"}`.
  - A screen pass leads to seeds 1–2.
  - A confirmation pass means "confirmed vs J2", exploratory. It is never "above D1" without review.

**Diagnostics (no pass/fail; can never substitute for the primary endpoint):**
- unseen disease, crop, composition-macro and within-crop Top-1; seen Top-1; H_micro;
- the calibration-matched H_bal difference (child at J2's γ);
- source-stratified seen Top-1;
- H_RT-population accuracies;
- the paired composition-cluster bootstrap CI of the H_bal difference.

**Secondary pipeline estimate (optional, H_bal only, cannot promote):**
- It compares the child on each held fold's inner parent π(k) (from seed-0 screen tables on K∖{k}: J2 for I1, I2, I3, I6; J3 for I4, I5) against that parent.
- It requires the child trained on **J2 and J3 on all six folds** per seed: about 2 h 05 extra per seed for the J3 variants.
- A missing variant is refused; there is no fallback.

## 3. Statistical interpretation

- Each row's calibration (γ) for fold k is chosen without fold k.
- J2 is fixed as the comparator before any child experiment.
- **But J2 itself was chosen** by the seed-0 screen after examining all six development folds, including every held fold. The comparator is therefore not independent of the held-fold outcomes.
- **So the primary gate is an exploratory, paired child-versus-fixed-J2 comparison on reused development folds.** It is not a fully nested or independent model-selection estimate, and it supports no claim about performance on new data.
- **The direction of any bias from choosing J2 is not assumed.** It depends on how J2's seed-0 advantage relates to the child's own randomness. The three-seed confirmation reduces, but does not remove, seed-specific effects.

## 4. Uncertainty and multiplicity

- **Reported:** the paired composition-cluster bootstrap 95% CI of the seed-mean, fold-mean H_bal difference.
  - It uses the same resampled compositions for child and parent and for every seed in a fold, each row at its own chosen γ.
  - `bootstrap_sign_proportion_le_0` is the descriptive share of bootstrap draws ≤ 0. It is **not a p-value** and is never adjusted or called significant.
- **Previously published values are unchanged.** Earlier v4 reports showed a `p_le_0` / "P(Δ ≤ 0)" column, which this proposal characterises the same way.
- **Optional formal test (proposed, not implemented, never part of the gate):** a paired cluster-randomisation test per child row.
  - Statistic: T = the seed-mean, fold-mean H_bal difference at each row's fixed LOFO choice.
  - Null: within each fold, the child's and parent's prediction vectors are exchangeable for every composition cluster (the modification changes nothing).
  - Resampling: for each (fold, composition), independently swap the child and parent hit vectors, jointly across seeds. Recompute T; p = (1 + #{T* ≥ T}) / (1 + B).
  - Holm across the child rows **actually run** is then a valid family-wise adjustment under that null.
  - Caveats:
    - it is conditional on the fixed γ choices;
    - it treats compositions as exchangeable units, so it ignores source clustering and may be anti-conservative where source effects dominate;
    - it addresses chance only, not fold reuse or adaptive choice.
- **Always disclosed:** the adaptive choice of children after H_RT, the reused folds, and every child row attempted, with its outcome.
- **Counts:** image-fold counts are **not** independent observations. For example, the 628 other-source seen image-fold instances are 111 unique images in 30 compositions, 15 sources and 40 composition × source clusters (`strata_units.json`).

## 5. What declaration would fix

- the gate and diagnostics above;
- the verified-input requirements (section 1), with the code hashes of `child_inputs.py`, `child_gate.py`, `jrun.py`, `research_r2/methods.py` and `runner.py`;
- the parent J2 and its registered Phase 1 and Phase 2 runs;
- the child rows to be attempted, in order (see `R2_v4_Amendment3_v3_Review.md` §6);
- the mechanism diagnostics per child;
- that nothing promotes above D1 without separate review.
