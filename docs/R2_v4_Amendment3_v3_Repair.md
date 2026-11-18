# Amendment 3 v3: final repairs before declaration (17 Nov 2026)

This addendum to `R2_v4_Amendment3_PROPOSAL_v3.md` covers the defects the independent review found in the v3 code. The declared content is `research_r2_v4/amendment_3.json`.

| Review finding | Repair |
|---|---|
| **A1.** `verified_pair(..., rederive=False)` returned "verified" tables after a same-label packet swap | **Re-derivation is mandatory.** `load_verified` and the production `verified_pair` have no `rederive` parameter (passing one is a `TypeError`). The shared `_pair(..., rederive=False)` path exists only for tests. Its tables carry `rederived: False`, and `child_gate` refuses them at `primary_gate` and `promotion_decision`. |
| **A2.** `promotion_decision({"role":"primary", ..., "gate":{"pass":True}})` returned a pass | **`promotion_decision(child_by_seed, parent_by_seed, mode)` recomputes the primary H_bal gate from the tables.** Any dictionary or non-table input is refused. Tables must carry an identity record **sealed** by `child_inputs`: an HMAC with a per-process key over the identity record and the table's own score, truth and candidate arrays. A copied identity, an edited table or an unsealed table is refused. |
| **A3.** Parent and configuration overrides were reachable | **Production `verified_pair(child_base, mode, frame, F, child_row)` fixes the parent to `J2_clipL_last4`.** The parent runs are the registered Phase 1 and Phase 2 directories. The child row must be in `AUTHORISED_CHILDREN = ("J4b_best_photometric_aug",)` with its declared configuration, and the modification must be exactly `augment = photometric`. Synthetic configurations exist only through `_pair` and `child_run._run` in tests. |
| **B.** Child runner | **`research_r2_v4/child_run.py` and `scripts/r2v4_child.py run` are added.** They: verify the declaration and authorisation; refuse a dirty worktree and output directories that overlap protected evidence; train each registered fold with `jrun.run_fold` in a new attempt directory; and write `child_identity.json` only after receipt checks, lineage and re-derivation pass. Otherwise they keep `failure.json` or `child_rejected.json`. J2 evidence is re-verified before and after a run. |
| **C.** J2 parent preflight | **`scripts/r2v4_child.py preflight --seed s`** re-verifies one J2 seed (receipts, hashes, registered fold, contract, lineage, identity) and re-derives every packet against the 1e-5 tolerance. It writes a receipt. It is required per seed before any child scoring with that seed. |

**Unchanged:** metric definitions, thresholds (+1 pp, ≥ 5/6, AUSUC non-decrease, shortcut ≤ +5 pp), seeds (screen {0}, confirmation exactly {0, 1, 2}), H_bal as the sole promotion endpoint, and all historical outputs.

**Residual limits:**
- Python cannot hide module internals. A deliberately malicious caller with code access could call the test-only `_seal`.
- The seal is an integrity boundary against accidental misuse and forged result objects. It is not a security boundary.
- The production CLI and evaluation use only `verified_pair`.
