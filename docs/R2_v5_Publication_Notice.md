# Second Run v5: publication notice

## What is published

* The six commits after `1a2f4b0` are a **sanitized replay** of the private v5 commits `aec9d39` … `27d4739`. Each has
  the same message (plus a replay trailer), the same author date and the same tree, except for the files listed in
  `artifacts/r2/v5/PUBLICATION_MAP_v5.json`.
* A final public-only commit adds the map, this notice and `scripts/r2v5_public_verify.py`.
* **The chronology is preserved:** implementation, final training, preflight fix, preflight evidence, the one-shot
  official evaluation, then the limitations.

## Public derivatives and withheld files

| File | Public form |
|---|---|
| `artifacts/r2/v5/final_J2_full8000/run_final_train.sh.txt`, `logs/tests_final_targeted.log.txt` | local machine paths replaced (`<WORKSPACE>`, `<CONDA_ENV>`, `<TMP>`) |
| `official_eval_ONE_SHOT_aggregate/official_evaluation.json` | per-crop and per-disease official accuracies removed; all aggregate values unchanged |
| `official_eval_ONE_SHOT_aggregate/independent_recomputation.json` | per-unseen-class counts and the list of largest class confusions removed; all counts, accuracies, H@1 and H@5 unchanged |
| `docs/R2_v5_Official_Evaluation_Result.md` | the per-unseen-class table and the largest-confusion list removed |
| `official_eval_ONE_SHOT_aggregate/seen_confusion_333x333.npy` | **withheld** (class-level official statistics); its original sha256 is in the map and the stage manifest |

* **Derivatives are not byte-identical** to the private originals. The map records the private-original and public
  sha256 of each one.
* **Stage manifests are unchanged** and still record the original hashes.
* **Never published:** no per-image official prediction, official CSV, official image, checkpoint or feature cache.
  Their sha256 values appear only in the private-file manifests.
* **The candidate payload** `final_candidates_cos30082_v1.json` is byte-identical to the file already on `origin/main`.

## Verification on a public checkout

```bash
python scripts/r2v5_public_verify.py --tests
```

It reports each manifest entry as `ORIGINAL`, `SANITIZED_DERIVATIVE`, `WITHHELD_PRIVATE` or `FAIL`, and reports the
private-file manifests as `UNAVAILABLE`. It also checks the frozen configuration (γ = 8.75, J2, payload hash), checks the
frozen checkpoint hash in every public record, and recomputes the official H@1 and H@5 from the published counts. It
then runs `tests/test_r2v5_final.py`, which needs only public files.

**Not verifiable publicly:** the checkpoint, the per-image predictions and the private originals of the derivatives.

## The frozen result (unchanged)

| | Top-1 | Top-5 |
|---|---|---|
| Seen | 1,142 / 2,000 = 57.10 % | 1,754 / 2,000 = 87.70 % |
| Unseen | 1,218 / 1,448 = 84.12 % | 1,429 / 1,448 = 98.69 % |

Official H@1 is **68.02379489908098 %** and H@5 is 92.8700476046502 %. This was a single owner-authorised evaluation
(17 Nov 2026) of the frozen model: checkpoint `e9dbbdf3…abb1c9`, γ = 8.75, 333 candidates. First Run's best official
H@1 is 9.945951 %. Development used only the 8,000 training images, and the comparison is descriptive (see the result
report for limitations).
