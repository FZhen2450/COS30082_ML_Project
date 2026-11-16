# PROPOSAL (not declared, not applied): amendment 3, the J4a–J7 child evaluation rule

**Status:** a proposal for independent review only.
- Nothing here changes `amendment_2.json`, the declared parent (J2) or any registered output.
- No J4–J8 model has been trained.
- If accepted, it must be declared (`r2v4_declare.py --amendment 3`) **before** any J4–J7 run.

## 1. The defect

The registered rule (`amendment_2.json` → `nested_parent_rule`) and the evaluator together have three problems:

1. **A child is gated against D1, not against its parent.** A child identical to J2 passes its single-row nested gate vs D1: +3.66 pp, 6/6, `meets_gate_thresholds: true`. It has no effect of its own; the gain is entirely J2's (`parent_rule_audit.json`, scenario `null_copy_of_J2`).
2. **A J2 child can be credited in at most 4 of 6 held folds.** On the seed-0 tables the inner parent is J3 for held folds I4 and I5, so J2 children are excluded there. Even a perfect child is selected only in I1, I2, I3 and I6 (`perfect` scenario). A child-attributable 5/6 requirement can therefore never be met.
3. **Fallback is credited to the stage.** In I4 and I5 the stage nested estimate falls back to J3. The stage then reports, for example, +24.18 pp and 5/6, of which +0.75 pp comes from the J3 fallback folds. A stage nested estimate with a child included cannot be read as the child's effect.

## 2. Proposed rule (prospective)

**P1. Gate a child against its parent.**
- A child passes only if, on the per-fold seed mean, it beats **its parent** by ≥ +1.0 pp, on ≥ 5/6 folds, with AUSUC not lower and healthy-shortcut increase ≤ +5 pp.
- These are the v3 thresholds, with the parent replacing D1.
- Screening uses seed 0; a screen pass triggers seeds 1 and 2, as for J1–J3.
- The comparison with D1 is still reported, but it is not the child's gate.

**P2. Nested-complete training.**
- For every held fold whose inner parent (from the seed-0 screen tables, the declared rule) differs from J2, also train the child on **that inner parent**. Currently that means J3 for folds I4 and I5.
- Then every held fold has the child of its own inner parent.
- The child's nested estimate compares, in each held fold, the inner-parent child against the inner parent. The threshold is ≥ +1.0 pp and ≥ 5/6, with every fold attributable.

**P3. Attribution in every report.**
- Report the folds where a child is selected, and the gain coming from child-selected folds vs fallback folds.
- Report child vs parent paired differences per fold and per seed, with a paired composition bootstrap.

**P4. No post-hoc parent change.**
- The global parent stays J2 under the seed-0 rule.
- Phase 2 (where J3 leads by 0.34 pp, CI [−2.51, +1.65]) and H_RT results must not be used to change it.
- Under P2 the parent question no longer decides the outcome, because each fold uses its own inner parent.

**P5. Source co-endpoint.** Any child that passes P1 and P2 also gets H_RF, plus H_RT on the same three registered pairs. This is diagnostic only, as for J1–J3.

## 3. Cost (Katana, one GPU job at a time)

| | Per child at seed 0 | Five children |
|---|---|---|
| Six folds on J2 (≈ 9.7 min / fold) | ≈ 58 min | ≈ 4.9 h |
| Two folds on J3, I4 and I5 (≈ 20.8 min / fold) | ≈ 42 min | ≈ 3.5 h |
| **Total screen** | **≈ 1 h 40** | **≈ 8.3 h** |

- Three-seed confirmation adds about 3 h 20 per passing child.
- With the 27 Nov deadline, a reviewer may prefer to restrict the stage to the children aimed at the failure modes found so far: **J5** (source-balanced weights, for source sensitivity) and **J7** (crop adversary, for disease transfer). That costs about 3 h 20 at screen.
- The order would be: declare amendment 3, then J5, then J7, then the others only if time remains.

## 4. What reviewers should check

- P1–P3 are fixed before any child result exists.
- The inner parents for P2 come from the seed-0 screen tables only (I4, I5 → J3), so there are no held-fold data in the parent choice.
- The evaluator implements P1–P3 with negative tests, including:
  - a null child must fail P1;
  - a child missing its I4/I5 J3 version must be refused for the P2 nested estimate.
