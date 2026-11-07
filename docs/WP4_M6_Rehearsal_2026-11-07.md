# M6 model-picker rehearsal — 7 November 2026

The normal LeafLens model picker now loads the selected real M6 adapter from a
fresh application instance. A native CLIP image encode, Top-5 presentation,
four fitting-only references and matching JSON export passed on this machine.
No predictor or cached image embedding was injected. No official images were used.

## Frozen inputs

- M6: seed 45, epoch 1, residual cap 0.20; adapter SHA-256
  `8835a18b9eb933ae224bd17866f18bc56406e24d8612743f4e7c88848a8135cc`.
- Official ViT-L/14 JIT base SHA-256
  `b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836`.
- Native OpenAI CLIP revision
  `d05afc436d78f1c48dc0dbf8e5980a9d471f35f6`.
- Actual private adapter uses the published `wp3b-visual-adapter-m4-v1` envelope.
  Its contract/text bank and all 323 candidate IDs, pairs, names and v2 prompts
  passed exact checks. Contract SHA-256:
  `1cdef457c4b3331d2f266e04d14cb6d173079506327bb349e7c51bc0946bd33d`.
- Runtime: existing Python 3.13.13 / torch 2.9.1+cpu /
  torchvision 0.24.1+cpu, pinned official CLIP 1.0. No package changes.

## Real image result and limitations

Predetermined seen-development query: `PWV2_peach_leaf_curl_6128.jpg`.
Truth: original global ID 219, peach / curl.

| Rank | Global ID | Prediction | Raw cosine score |
| --- | ---: | --- | ---: |
| 1 | 15 | apple / scab | 0.28392773802444327 |
| 2 | 10 | apple / rust | 0.2816434275329758 |
| 3 | 293 | tea / red_scab | 0.28145408078090806 |
| 4 | 292 | tea / red_leaf_spot | 0.27974795439206257 |
| 5 | 4 | apple / brown_spot | 0.27573247331688727 |

The truth is absent from Top-5. This is a successful engineering rehearsal with
an incorrect prediction, not evidence of classification accuracy. Raw ranking
scores are not confidence percentages. All 323 scores are finite float64.

Four distinct references were rendered for the predicted apple/scab pair:
`APPD_apple_scab_1359.jpg`, `CDDM_apple_scab_39832.jpg`,
`CDDM_apple_scab_40260.jpg`, `APPD_apple_scab_1181.jpg`.
All belong to the 4,868 fitting identities and exclude the query. The GUI labels
them "Same crop and disease. Fitting images only. Closest colours first."
They illustrate the predicted class; they do not assert the query is apple/scab.

## Reproduce

With the pinned CLIP package and verified local assets, launch normally:

```powershell
python gui.py --protocol development_seed42_reconciled.json --images DPD_Small_dataset
```

Use Connect a model -> M6, choose the verified base and adapter, select CPU, load
the predetermined query, Analyze leaf and Export result.

The same flow can be recorded using the explicit harness:

```powershell
python -B scripts/rehearse_wp4_m6.py --clip PATH_TO_VIT_L_14 --adapter PATH_TO_M6 --protocol development_seed42_reconciled.json --images DPD_Small_dataset --output NEW_PRIVATE_EVIDENCE_DIRECTORY --screenshots
```

Only file dialogs and error reporting are automated. The harness instantiates
the real app and model picker, verifies fitting/development query membership,
uses the real worker/loader/model, compares predictor scores with the presenter,
checks fitting-only references and reads back the actual JSON export.

Local evidence is preserved under ignored
`runs/wp4-issue13-20261107/real-m6-rehearsal-final/`: `evidence.json`,
`canonical_scores.json`, `prediction.json`, `model-picker.png`, `m6-result.png`.
The JSON records exact current source-byte hashes and runtime provenance.
Weights, screenshots, images and prediction artifacts are excluded from Git.
The prior Chia rehearsal remains separate historical evidence.

The existing real WP1 picker regression also passed on its required
torch 2.7.1+cu118 / torchvision 0.22.1+cu118 environment. Preserve that loader
guard; this CLIP environment is not a claim of cross-version WP1 reconstruction.
Full automatic regressions are recorded in the PR validation notes.

Shared-333 policy, official-test inference, model quality and the team's final
demo recording remain separate. Frederick can reproduce this procedure with
the same private assets before recording the presentation.
