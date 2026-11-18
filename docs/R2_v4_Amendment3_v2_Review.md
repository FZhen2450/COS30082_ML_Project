# R2 v4: H_RT disposition and amendment 3 v2 repair (review package, 17 Nov 2026)

**Scope:** methodological repair and planning only.
- No GPU training.
- No J4–J8 run.
- No real-data child evaluation.
- No official-test access.
- No First Run change.
- No historical result change.
- Amendment 3 is **not declared**.

## 1. Git state and files inspected

- **Start:** `9659270`, clean.
  - Ancestry `1b2fcdb → … → 9659270` verified. PR #36 is still at `1b2fcdb`.
  - Review ZIP sha256 `a30400d8…468e` verified.
  - H_RT artifacts: 106/106 `SHA256SUMS` and 99/99 private hashes match.
- **Inspected:**
  - `research_r2_v4/amendment_2.json` (`nested_parent_rule`, `three_seed_gate`, `row_details`);
  - `research_r2_v4/confirm.py`;
  - `research_r2_v4/evaluate.py`;
  - `scripts/r2v4_stage_j.py` (`nested_with_parent_rule`, `cmd_evaluate`);
  - `scripts/r2v4_parent_rule_audit.py`;
  - `docs/R2_v4_Amendment3_PROPOSAL.md` (v1);
  - `research_r2/metrics.py` (`FoldScorer`);
  - `research_r2/methods.py` (`within_concept_source_weight`);
  - `docs/R2_v3_Candidate_Policy_and_Phase2.md` (F1–F7);
  - Stage I `results.json`;
  - `matrices/stage_j_v4.json`.
- **New (this commit):**

| File | Purpose |
|---|---|
| `docs/R2_v4_HRT_ReviewSummary.md` | Task A: reviewer summary (development gains, H_RT, whole-fold vs excluded-source, confounding, observation vs cause, limits) |
| `docs/R2_v4_Amendment3_PROPOSAL_v2.md` | Task B: one coherent child-vs-parent protocol. Supersedes v1, which is kept unchanged. |
| `research_r2_v4/child_gate.py` | Task C: proposal evaluator (primary gate, optional pipeline estimate, bootstrap, Holm). Not used by any registered stage. |
| `tests/test_r2v4_child_gate.py` | Task C: 14 synthetic tests |
| `docs/R2_v4_Priorities_PostHRT.md` | Task D: J4b / J5 / J7 reassessment |
| `scripts/r2v4_priority_stats.py` | Task D: descriptive data facts (no model) |
| `artifacts/r2/v4/stage_hrt_20261116_post_review/` | Priority stats, null-child demonstration, test log, SHA256SUMS |

## 2. H_RT disposition

- **Accepted as a diagnostic**, with the reviewer's framing (see the summary document):
  - the effects compare two removal arms and are not causal source-style effects;
  - per-pair CIs are wide and there is one seed per arm;
  - the I6/CDDM comparable population excludes 25 emptied compositions, and the surviving compositions differ in supervision coverage between arms;
  - no general robustness or non-robustness claim is made.
- **No result changed.**

## 3. The proposed gate (precise definition)

**Inputs:**
- Child tables C_s and parent tables P_s (parent = J2, fixed in advance), for seeds s ∈ S.
- S = {0} for the screen; S = {0, 1, 2} exactly for confirmation.
- Folds K = (I1, …, I6).
- Endpoint E = H_bal.

```
for s in S, k in K:
    (c_ci, c_gi) = argmax_{config, γ} mean_{f∈K∖{k}} H_bal(C_s, f, config, γ)   # ties: lower config, smaller γ
    (p_ci, p_gi) = argmax_{config, γ} mean_{f∈K∖{k}} H_bal(P_s, f, config, γ)
    d_s(k) = E(C_s, k, c_ci, c_gi) − E(P_s, k, p_ci, p_gi)
    a_s(k) = AUSUC(C_s, k, c_ci) − AUSUC(P_s, k, p_ci)
    h_s(k) = hs(C_s, k, c_ci, c_gi) − hs(P_s, k, p_ci, p_gi)
D(k) = mean_s d_s(k);  A(k) = mean_s a_s(k);  H(k) = mean_s h_s(k)
PASS ⇔ 100·mean_k D(k) ≥ 1.0  ∧  |{k : D(k) > 0}| ≥ |K| − 1  ∧  mean_k A(k) ≥ 0  ∧  100·mean_k H(k) ≤ 5.0
```

**Refused before any arithmetic:**
- different fold identities or order;
- more than one configuration;
- any difference in candidate order or identity (crop, disease, healthy, fitting mask) or in truth;
- inexact seed sets.

**Reported, not gating:**
- the calibration-matched sensitivity (child at the parent's γ);
- the paired composition-cluster bootstrap (same resamples for child, parent and all seeds);
- Holm-adjusted P(Δ ≤ 0) across the child rows tried;
- the decomposition child − D1 = (child − J2) + (J2 − D1).

**Optional pipeline estimate:**
- π(k) is chosen on K∖{k} from the seed-0 screen tables (registered π: J2 for I1, I2, I3, I6; J3 for I4, I5).
- d_s(k) = E(child on π(k)) − E(π(k)).
- It requires child variants on every π parent on **all six** folds (no fallback, refused otherwise).
- Gains from switching parents cancel inside d.

## 4. Synthetic tests (`tests/test_r2v4_child_gate.py`; 14/14 pass)

- **Null child (identical to its parent):** fails at screen and confirmation, with 0.0 pp, 0/6 folds and a degenerate bootstrap [0, 0], P(Δ ≤ 0) = 1.
- **Genuinely improving child:** passes at screen and confirmation with 6/6 folds and a bootstrap lower bound above 0.
- **Fold-count threshold:** 5/6 can pass; 4/6 fails.
- **Constraints:** an AUSUC decrease fails; a shortcut increase above +5 pp fails; improvement counts use the per-fold seed mean.
- **Seeds:** {0, 1}, {0, 1, 3}, a screen without seed 0, and a parent missing seed 2 are all refused.
- **Candidates and folds:**
  - a reordered packet cannot be scored (`PacketError`);
  - an altered candidate identity is refused;
  - swapped fold order, a missing fold and fold data under another fold's key are refused.
- **Held-fold isolation:**
  - changing only the held fold never changes its γ;
  - making a parent perfect only on the held fold never changes that fold's inner parent.
- **Missing parent variant:** refused, with no fallback.
- **J2/J3 switching:**
  - null children on both parents give 0 child-attributable gain in every fold, while the parent-attributable part is positive;
  - a child that helps only on J2 is credited only where J2 is the held fold's parent (1 fold), and fails.
- **Fallback:** the registered stage rule selects a parent in some held folds. The proposed estimate always compares the child row with its own parent row.
- **Disclosure:** Holm adjustment; the endpoint switch; refusal of an unknown endpoint; the calibration-matched sensitivity.

**Full R2 suite:** `Ran 148 tests in 1135.066s … OK` (134 earlier + 14). Log sha256 `81acf718…c46b`.

## 5. Null-child demonstration (`null_child_demo.json`, synthetic)

| Rule | Null child |
|---|---|
| Registered (child vs D1, single-row nested) | +58.04 pp, 6/6, `meets_gate_thresholds: true` (accepted) |
| Proposed (child vs J2), screen | 0.00 pp, 0/6, AUSUC Δ 0, shortcut Δ 0, `pass: false` |
| Proposed, confirmation | 0.00 pp, 0/6, bootstrap [0.00, 0.00], `pass: false` |
| Proposed, an improving child (control) | +32.6 pp, 6/6, bootstrap [+31.9, +33.3], `pass: true` |

The magnitudes are synthetic. Only the accept/reject behaviour is the point.

## 6. Remaining leakage and statistical risks

1. **Fold reuse:** the development folds are reused, so every child result is exploratory.
2. **Adaptive selection:** children are prioritised after H_RT. This must be disclosed, with the number of children tried.
3. **Comparator bias:** J2 was selected on all six folds, which biases the comparator upward (conservative for the child). It is disclosed and reduced by three-seed confirmation.
4. **Calibration:** γ is re-chosen per row; the calibration-matched sensitivity separates calibration from representation.
5. **H_RT mechanism endpoints:** 3 pairs, 1 seed, confounded; descriptive only.
6. **No real-data test yet:** the evaluator is validated on synthetic tables only. Real use additionally requires child receipts naming J2's configuration plus exactly one declared change, verified by `confirm.verified_row_seed`.

## 7. Next experiments (not executed)

See `docs/R2_v4_Priorities_PostHRT.md`. At most two are recommended:
1. **J4b** (photometric) first. It is the only candidate whose mechanism could act on the H_RT failure mode. Mechanism endpoint: H_RT source-excluded novel-source Top-1, child − J2.
2. **J5** (within-concept source balancing) second. Mechanism endpoint: other-fitting-source seen Top-1 (628 image-folds), child − J2. Expectations are low: its frozen analogue was null, and it cannot address absent sources.

**J7 is not recommended:**
- the registered design feeds z, not Pd(z), downstream;
- four prior disentanglement attempts failed.

**Resources:** about 1.5–5.5 h of Katana per child. Before that: declaring amendment 3 and implementing it on real data (about half a day of CPU work). First Run's 27 Nov submission comes first.

## 8. Commands

    python scripts/r2v4_priority_stats.py --out <E>/stage_hrt_20261116/post_review/priority_stats.json
    python <post_review>/null_child_demo.py.txt <E>/stage_hrt_20261116/post_review/null_child_demo.json   # synthetic only
    python -m unittest discover -s tests -p 'test_r2v4_child_gate.py' -v     # 14 OK
    python -m unittest discover -s tests -p 'test_r2*.py' -v                  # 148 OK

**Stop:** owner review is needed before declaring amendment 3 or starting any training.
