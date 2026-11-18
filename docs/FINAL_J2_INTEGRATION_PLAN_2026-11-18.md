# Final submission: integrating Second Run J2 as the proposed approach (plan, 18 Nov 2026)

> **Status (18 Nov 2026):** the team adopted J2. The integration commit was pushed and opened as draft PR #39
> (stacked on #38). The plan below is kept as written; the decisions it lists as pending have since been taken.

## Current submission state (inspected)

- **`main`** is at `bbdabfd`. Nothing has been merged since 8 Nov.
- **The real submission code is the unmerged stack:**
  - PR #32 `feature/wp1-wp4-final333-20261111` (`90e70c4`): WP1 full-8,000 release and the WP4 final-333 GUI;
  - PR #37 `review/wp4-full8000-m6-20261114` (`841120b`): the full-8,000 M6 GUI integration;
  - PR #38 `review/jonathan-wp4-top5-layout-stacked-20261114` (`1891e34`): the Top-5 layout fix, approved by Fa Zhen and mergeable.
- **GUI:** `wp4/`, a Tkinter app launched with `python gui.py`.
  - A model is a `MODEL_SPECS` entry with hash-pinned asset discovery (`wp4/asset_discovery.py`), loaded through `wp4/model_loader.load_model`.
  - A predictor returns 333 ids in protocol order with 1 × 333 scores. `wp4/service.present_prediction` ranks them (ascending-id ties) and fails closed on any mismatch.
  - Reference images come only from the validated full-8,000 training pool (`Catalog.references`).
- **Final protocol:** `artifacts/data/final_full8000_333_v1.json`, schema `dpd-full8000-final333-v1`, candidate sha256 `ba17e941…`.
  - Its 333 rows are identical to J2's candidate payload. J2 uses exactly the GUI's candidate contract.
- **WP5 four-model official results** (WP1, WP2, frozen CLIP, M6) are recorded in the 12–13 Nov official run and in Jonathan's report master. They are kept as they are.
- **Team decision needed:** Fa Zhen's 14 Nov handoff says "Round 2 is excluded". Adopting J2 changes that decision, so the whole team has to agree to it (see Blocker 1).

## Smallest safe integration

1. **New branch** `feature/final-j2-integration-20261118`, from `1891e34` (top of the approved stack). No Second Run history is merged. No teammate branch is modified.
2. **J2 inference asset** (private, hash-pinned, about 1 MB):
   - Run the frozen J2 scoring pipeline once on the frozen training features, to capture its three fitted downstream heads (crop, disease, composition) and the feature standardisation.
   - The capture records the heads the real pipeline fits; it does not refit them separately. The asset changes neither the scientific method nor γ (8.75).
   - It is stored next to the private checkpoint and never committed.
   - Builder: `scripts/j2_build_inference_asset.py`. It reads the private Second Run code read-only through `--r2-root`, so the Second Run code stays on its research branches (`research/r2-v5-final-model`) rather than in this branch.
3. **`wp4/j2_predictor.py`:**
   - **Loading:**
     - the pinned CLIP ViT-L/14 through the existing `load_native_clip`;
     - the frozen J2 checkpoint (sha256 `e9dbbdf3…` pinned), with the 51 trainable tensors checked, plus its metadata;
     - the pinned inference asset.
   - **Scoring:** CLIP preprocessing, image feature, then standardise, crop and disease log-probabilities, composition log-probability, fusion, and γ = 8.75 for the 323 seen candidates.
   - **Output:** 333 scores in the GUI contract. There is no silent fallback: any failed check raises.
4. **Registration:**
   - `MODEL_SPECS["j2"]` and `SHORT_LABELS["j2"]` added;
   - hash-pinned discovery for the J2 checkpoint and asset;
   - one branch in `load_model`.
   - No GUI redesign. Retrieval is unchanged, training-only.
5. **Verification:**
   - unit and negative tests;
   - score parity against the frozen J2 pipeline on training images (the saved 3,448-image preflight rehearsal scores);
   - Top-5 stability, CPU and GPU latency;
   - a real GUI launch driven end to end by `scripts/j2_gui_smoke.py` (automatic screen capture does not work under WSLg, so a person checks the window visually).
   - No official-test inference.
6. **Documentation:** a README section with launch instructions and how to obtain the authorised assets; J2 report text and figures for the report master; a presentation and video outline.
7. **Stop before any push, PR or merge.**

## Team split (proposed 17 Nov; each member confirms their own task)

| Member | Task | Why them |
|---|---|---|
| **All (blocking)** | Agree to adopt J2 as the proposed approach. WP1 and WP2 stay the required baselines; frozen CLIP and M6 stay reported as earlier attempts. | Reverses the 14 Nov "Round 2 excluded" decision |
| **Fa Zhen** | Review the J2 GUI integration (the WP4 owner); run it on their own machine; record the GUI demo segment for the video | Owns WP4 and the GUI stack |
| **Jonathan** | Merge the J2 sections into the report master (Word), keep it to 8 pages, add the J2 row and seen confusion-matrix figure next to the existing four-model table | Owns the report master and WP5 |
| **Andy** | Independently verify the J2 checkpoint and asset hashes and GUI score parity, like the WP1 provenance sign-off in PR #32 | Independent verifier role |
| **Frederick** | Human GUI rehearsal with J2 (the pending acceptance); slides and narration for the "results and limitations" segment | Pending GUI rehearsal owner |
| **Yuen Kai (owner)** | Approves pushes and PRs; presents the J2 method and Second Run research in the video and Q&A; checks the Who-did-what entries | Did the Second Run research |
| **Claude (AI assistant)** | Builds the asset, adapter, tests and parity evidence; drafts the J2 report text and README | Must be disclosed as AI assistance, if the unit's policy requires it |

## Blockers and risks

1. **Team agreement** to switch the proposed approach to J2. *Resolved 18 Nov 2026: adopted.*
2. **Asset distribution.** The checkpoint (196 MiB) and the asset are private. Teammates need authorised private access (the owner shares them, as with the M6 handoff).
3. **The report must include J2's seen confusion matrix** (assignment requirement). That figure is generated privately for the report and is not published on GitHub.
4. **Stacked base.** The integration sits on unmerged PRs #32, #37 and #38; merge order is the team's decision.
5. **Who-did-what accuracy.** Second Run work was the owner's research with AI assistance. No teammate is credited for it unless they actually contribute.
