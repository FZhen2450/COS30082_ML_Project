# PROPOSAL v2 (not declared, not applied): amendment 3, the J4a–J7 child-versus-parent gate

**Supersedes** `docs/R2_v4_Amendment3_PROPOSAL.md` (v1, kept unchanged).

**Status:** a proposal for owner and independent review.
- It changes nothing in `protocol_v4.json`, `amendment_1.json`, `amendment_2.json` or any recorded evaluation.
- No child exists and none has been trained or evaluated.
- The evaluator is `research_r2_v4/child_gate.py`, validated on synthetic tables only (`tests/test_r2v4_child_gate.py`).
- If accepted, it must be declared (`scripts/r2v4_declare.py --amendment 3`) before any J4–J7 run.

## 1. Why v1 was ambiguous

- **v1 had two primary rules:**
  - P1 gated the child against the global parent J2.
  - P2 gated it against each held fold's inner parent, with J3 children trained only for I4/I5.
- **They can disagree.** v1 did not say which decides.
- **P2 as written was also infeasible without leakage-free calibration.** The γ of the I4 J3-child must be chosen on the other five folds, which needs J3-children on **all six** folds, not only on I4/I5.

## 2. Resolution: one primary gate, one optional secondary estimate

### Primary gate: child vs the global parent, paired on every fold

**1. Parent per fold:** **J2** in every fold.
- It is the declared parent, fixed by the registered seed-0 rule (best of J1–J3 by screen H_bal) before any child exists.
- It is not re-chosen from Phase 2 or H_RT.
- Each child is trained from J2's configuration with one declared modification.

**2. Held-fold isolation:**
- No parent is chosen per fold, and no child is selected. Each child row is judged alone; there is no "best child".
- For each seed s and fold k, child and parent each get their own LOFO choice on folds ≠ k: γ by mean H_bal, ties to the smaller γ (`RowTable.choose`, the registered calibration rule).
- Fold k's own data never enter its γ (tested).
- J2's selection used all six folds, but this selects the **comparator**, not the child. It can only make J2 a harder baseline (regression to the mean), so the bias is conservative.

**3. Global or nested parent:** the **global parent** is primary. Section 3 explains why.

**4. Per-fold improvement** (endpoint E = H_bal for every child; read at each row's own LOFO choice):

    d_s(k) = E(child, s, k) − E(J2, s, k)
    a_s(k) = AUSUC(child, s, k) − AUSUC(J2, s, k)          (AUSUC is γ-free per fold)
    h_s(k) = hs(child, s, k) − hs(J2, s, k)                 (hs = unseen healthy-shortcut rate at the chosen γ)
    screen:        seeds S = {0};       confirmation: S = {0, 1, 2} exactly (refused otherwise)
    D(k) = mean_{s∈S} d_s(k),  A(k) = mean_{s∈S} a_s(k),  H(k) = mean_{s∈S} h_s(k)

**5–6. Thresholds** (the v3 thresholds, with J2 as the control):

    pass  ⇔  100·mean_k D(k) ≥ 1.0   ∧   #{k : D(k) > 0} ≥ 5   ∧   mean_k A(k) ≥ 0   ∧   100·mean_k H(k) ≤ 5.0

- Improvement is strict (D > 0), so a child numerically identical to J2 has D ≡ 0 and **fails** (0 pp, 0/6).
- The fold count uses the per-fold seed mean, not any single seed.

**7. Three seeds:**
- A seed-0 screen pass triggers seeds 1 and 2.
- The confirmation applies the same rule to the seed means.
- Child and parent must both have exactly seeds {0, 1, 2}, verified by `confirm.verified_row_seed` and the cross-seed checks.
- The parent's seeds are the registered Phase 1 and Phase 2 runs.

**8. Attribution:**
- The gate is a child-minus-own-parent difference on every fold, so there is **no fallback**. A parent's gain over D1 can never enter it.
- Reports also give the decomposition child − D1 = (child − J2) + (J2 − D1) per fold; only the first term is the child's.

**9. Uncertainty and multiplicity (disclosure, not gate):**
- A paired composition-cluster bootstrap of the seed-mean, fold-mean H_bal difference: 2,000 resamples; the same resampled compositions for child and parent and for every seed in a fold.
- A calibration-matched sensitivity: the child scored at J2's γ in each fold. This separates representation from calibration effects.
- Holm-adjusted bootstrap P(Δ ≤ 0) across all child rows evaluated.
- The number of child rows attempted, and every result, pass or fail.

**Input checks (fail closed):**
- identical fold identities and order;
- one configuration;
- identical candidate order and identity (crop, disease, healthy, fitting mask) and identical truth in every fold;
- exact seed sets.

### Secondary estimate (optional): the pipeline estimate

This answers a different question: does "choose the parent by the rule, then apply the modification" beat "choose the parent by the rule"?

- π(k) = the parent chosen on folds ≠ k from the seed-0 screen tables (each inner fold at its inner LOFO γ; ties → J1 < J2 < J3).
- On the registered tables π = J2 for I1, I2, I3 and I6, and J3 for I4 and I5.
- d_s(k) = E(child on π(k), s, k) − E(π(k), s, k), with the same thresholds.
- It requires the child trained on **every** parent that occurs in π, on **all six folds**: J2 and J3 here, 12 fold trainings per child at screen. A missing variant is refused; there is no fallback.
- Gains from switching parents (J2 → J3) cancel inside d. The decomposition reports them as parent-attributable.
- It is reported only if the owner funds the extra J3 training. It cannot rescue a child that fails the primary gate.

## 3. Why the global parent is primary

- **Every fold is attributable:** the gate reaches 6/6, where the registered rule allowed at most 4/6.
- **Nothing is ever credited to the child that it did not do:** no parent switching and no fallback.
- **It needs only the six J2-child folds,** where the pipeline estimate doubles the cost (J3 folds take about 20.8 min vs about 9.7 min for J2).
- **The parent choice is not what J4–J7 test.** Phase 2 already showed that J1–J3 are indistinguishable (J2 − J3 −0.34 [−2.51, +1.65]), so a "pipeline" question adds cost without a clear scientific gain.
- **Leakage:** none from the held fold enters the child's γ. J2's own selection used all folds, which is a conservative comparator bias, disclosed.

## 4. Mechanism endpoints (pre-declared per child; diagnostic, never a gate)

**Endpoints:**
- **J4b (photometric):**
  - Novel-source seen Top-1 in H_RT source-excluded arms, child minus J2, on the three registered pairs, seed 0. This needs 3 child trainings; the J2 ablated arms already exist.
  - Seen Top-1 on other-fitting-source images, six folds.
- **J5 (source-balanced):**
  - Seen Top-1 on other-fitting-source images, six folds (628 image-fold instances; currently 61–65% vs 75–77% for dominant-source images).
  - The H_RF novel-source effect, child vs J2.
- **J7 (crop adversary):** unseen disease Top-1, child minus J2, with unseen crop Top-1 non-inferiority (≥ −1 pp).

**Outcome categories** (reported as found; none promotes a model above D1):
- gate pass + mechanism improved;
- gate pass only;
- mechanism only (a robustness finding without a development H_bal gain);
- neither.

## 5. Remaining risks

1. **Fold reuse:** the development folds have informed v1–v3, Stage I and J1–J3, so child results stay exploratory.
2. **Adaptive choice of children:** they are prioritised after seeing H_RT. That choice is adaptive and must be disclosed, as must the number of children tried.
3. **Conservative comparator:** J2 was selected on all folds, which biases its seed-0 screen upwards; the three-seed confirmation reduces this. The child screen uses seed 0 against J2 seed 0, which is the same comparison as Phase 1.
4. **Calibration:** child and parent each re-choose γ. The calibration-matched sensitivity shows whether a gain is representational.
5. **H_RT mechanism endpoint:** 3 pairs, one seed, composition and source confounding (see `R2_v4_HRT_ReviewSummary.md`). It is descriptive only.
6. **Not real-data tested:** the evaluator is synthetic-validated only. On first real use it must also pass the existing run verification (`confirm.verified_row_seed`, cross-seed checks), and the child receipts must name J2's configuration plus exactly one declared change.
