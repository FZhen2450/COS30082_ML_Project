# Second Run final model: one-shot official evaluation procedure (prepared, NOT executed)

**Status:** ready to run only after explicit owner authorisation. Nothing in this procedure has been run, and no
official-test image, label, CSV or score has been read.

## Frozen candidate (must not change before or after the run)

| Item | Value |
|---|---|
| Architecture | CLIP ViT-L/14, last 4 visual blocks fine-tuned (v4 row J2), crop and disease heads during training |
| Downstream | unchanged D1 `synth_composition` (`fuse = 1.0`, `normalise = true`), fitted on all 8,000 training images |
| Training data | all 8,000 DPD-Small training-CSV images; seed 0 (torch seed 0) |
| Candidate universe | 333 global composition ids 0–332 (pinned First Run payload `ba17e941…`); the 10 unseen ids 39, 84, 107, 124, 133, 136, 144, 166, 276, 303 |
| Calibration | γ = 8.75 subtracted from the 323 seen candidates. γ was chosen on development folds only and is never re-tuned |
| Ranking | descending calibrated score; ties broken by ascending global id (First Run tie rule) |
| Metrics | image-weighted Top-1 and Top-5 for seen and unseen; H@1 and H@5 = 2·s·u/(s+u); seen confusion matrix (333 × 333); per-crop and per-disease accuracy |

Hashes of the checkpoint (pinned in code), feature cache, receipt, configuration, code, payload and training CSV are
recorded in `verification_<head>.json`. The evaluation command refuses to run if any of them has changed.

## Checks before any inference (fail closed)

All of these run on **both** CSVs before any model is loaded. Any failure stops the evaluation with nothing scored.

1. **Authorisation:** `R2V5_OFFICIAL_EVAL_AUTHORIZED=1` and `--owner-authorization` are both given, and the output
   directory does not already exist.
2. **Frozen identities:**
   - the checkpoint sha256 equals the pinned `e9dbbdf3…`;
   - the feature cache, checkpoint and packet equal the receipt;
   - the receipt, configuration, code, payload and training CSV equal the verification record.
3. **CSV structure:** each row has exactly 7 columns and integer labels; no header row; file names have no path
   separators and don't start with ".".
4. **Counts:** exactly 2,000 seen rows and 1,448 unseen rows.
5. **Labels:**
   - every seen label is one of the 323 seen compositions, and every unseen label one of the 10 unseen ids;
   - every label is in the 333-candidate universe;
   - each row's crop and disease labels agree with the candidate payload.
6. **Image identity:**
   - file names are unique within each part;
   - no file is shared between the seen and unseen parts, or with the training CSV;
   - every image file exists.

After inference, each score matrix is checked to be finite and n × 333, Top-1 must equal the argmax, and the seen
confusion matrix must be 333 × 333 and count each image once.

## Command (only after owner authorisation)

```bash
R2V5_OFFICIAL_EVAL_AUTHORIZED=1 python scripts/r2v5_final.py official-eval \
  --run <E>/v5/final_J2_full8000 --verification <E>/v5/final_J2_full8000/verification_<head>.json \
  --seen-csv <PROJECT>/DPD_Small_test_seen.csv --unseen-csv <PROJECT>/DPD_Small_test_unseen.csv \
  --out <E>/v5/official_eval_ONE_SHOT \
  --owner-authorization "<owner name, date and approval reference>"
```

## Rules
1. Run once. The command refuses if the output directory exists. A crash before results are written is recorded and
   re-run only with fresh owner approval.
2. No calibration, architecture, threshold or training change may follow the result. The candidate is fixed above.
3. Report Top-1, Top-5, H@1, H@5, the seen confusion matrix, and per-crop and per-disease diagnostics alongside First
   Run's historical official H@1 (WP1 9.945951 %, frozen CLIP 6.617868 %, M6 adapter 5.958218 %, WP2 1.757603 %).
4. Disclose that First Run's official scores were already known to the project, so the broader project does not have a
   completely blind test. Second Run development never used them.
