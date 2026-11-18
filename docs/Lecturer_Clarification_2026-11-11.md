# Lecturer clarification and final-release actions - 11 November 2026

Source: the user relayed five answers from a Teams call with Dr. Lee in this
conversation on 11 November 2026. This is a student-reported call record, not a
recording, independently verified transcript, or replacement assessment rubric.
The answers below take precedence over earlier pending-status notes for these
five questions. Historical experiment records remain unchanged.

## Reported answers

| Question | Reported clarification | Required update |
| --- | --- | --- |
| Evaluation splits | Report Total (Overall), Seen and Unseen together. Every model ranks against all 333 candidates. | Use the same ordered 333 global IDs for every fold and model. Show Overall, Seen and Unseen Top-1/Top-5, counts and denominators, plus H@1/H@5 across Seen and Unseen. |
| Training scope | Retrain final models using all 8,000 training images. | Prepare a separate full-training release for trainable WP1, WP2 and the proposed adapter. Frozen CLIP remains a frozen reference. |
| GUI scope and design | The layout and features are sufficient. Expand the candidate space to 333, with labelled fallbacks for unseen references. | Keep the current layout. Add a final-333 catalog and compatible model/scoring paths; retain honest training-only reference labels. |
| Grading priority | Official-test generalization matters more than validation scores. | Lead the final results section with official results; retain development evidence as model-selection history. |
| Proposed approach | The adapter should show clear value by exceeding Frozen CLIP's reported 6.62% official H@1. | Evaluate the final adapter against the same frozen baseline, candidate task, images and metric definition. This is a target, not an achieved result. |

The previously outstanding shared-versus-restricted candidate question and
4,868-versus-8,000 final training question are now clarified by this call report.
No additional epoch count, seed budget, loss, architecture or augmentation rule
was supplied in these answers.

## Reporting: updates possible without new model inference

Use three explicit columns/groups for each model:

- Overall: all 3,448 official images, combining 2,000 Seen and 1,448 Unseen.
- Seen: 2,000 official images, ranked against all 333 candidates.
- Unseen: 1,448 official images, ranked against the same 333 candidates.

For Top-k, Overall accuracy is total correct predictions divided by 3,448:

`Overall@k = (Seen_correct@k + Unseen_correct@k) / 3448`

It is neither the unweighted mean of the two fold accuracies nor H@k.
H@k remains the harmonic mean of Seen@k and Unseen@k. Fractions are converted
to percentages only for presentation. Use full-precision saved values and actual
hit counts, not rounded percentages, when reconstructing summary tables.

`evaluation/reporting.py::build_report` already exposes `seen`, `unseen`,
`combined_micro`, `harmonic_mean_top1` and `harmonic_mean_top5`. The report can
label `combined_micro` as Overall without changing the historical metric schema.
Other diagnostics must state their population and denominator; an H value is
not a separate per-fold accuracy.

Keep two visibly distinct experiment groups:

1. Historical official shared-333 results from the original frozen releases.
2. New official shared-333 results from explicitly identified full-8,000 releases.

The [independent four-model verification](https://github.com/FZhen2450/COS30082_ML_Project/issues/8#issuecomment-6732629416)
still supports the historical table. Its rounded H@1 values are WP1 13.19%,
WP2 1.61%, Frozen CLIP 6.62% and M6 5.83%. The historical proposed M6 therefore
does not meet the newly stated adapter target. WP1's higher result does not
establish that the proposed adapter has beaten Frozen CLIP.

## Separate full-8,000 training release

- Preserve the original development partition, selected models and all saved
  official results. Do not modify the reconciled development manifest to make
  it appear that an old checkpoint trained on 8,000 images.
- Use exactly the 8,000 course training-source images for the final fitting pool;
  retain the approved source identity and a new ordered full-training ledger.
  Official images and labels do not enter optimization or augmentation design.
- Full training contains the 323 course training compositions. The ten official
  unseen compositions remain absent from this fitting pool, even though all
  333 compositions compete during evaluation.
- Record the training recipe, number of epochs/steps, checkpoint rule, seed and
  comparison budget before executing each new release. The call does not select
  those settings. Development-selected settings are the starting point, with
  any declared changes documented separately.
- Keep the original development runs for selection evidence. A model fitted on
  all 8,000 images cannot describe the old 1,238/1,894 folds as held-out validation.
- Retrain trainable systems under their owners' reviewed recipes. Do not retrain
  Frozen CLIP. Keep the baseline preprocessing, prompts and native scorer fixed
  for the primary matched adapter comparison unless a separate experiment is
  explicitly declared.
- Export new checkpoints with new hashes, training-scope metadata and runtime
  provenance. Historical pinned GUI assets must remain identifiable.
- Freeze each final release before its official evaluation. Existing official
  outcomes have already been observed; disclose that history and do not call
  the next comparison a pristine label-blind first test. Do not repeatedly tune
  against official errors until the adapter crosses 6.62%.
- Compare unrounded adapter and Frozen CLIP H@1 values. The 6.62% figure is a
  rounded reference, not a substitute for the baseline's exact saved value.

## GUI 333 integration: requirements and initial implementation gap

The published three-column GUI at `27c4030` was a development-323 implementation.
The subsequent implementation and actual execution receipts are tracked in the
[WP1/WP4 final-release record](WP1_WP4_Final_Release_2026-11-11.md). The gaps below
describe the starting point at this call record, rather than the later source.
Changing a displayed number or simply appending labels is insufficient:

- `wp4/service.py::Catalog` validates a development partition and restricts
  references to its 4,868 fitting records.
- `independent_baseline.py::IndependentPredictor` validates development
  checkpoint and protocol identities before constructing candidate scores.
- `wp4/clip_predictor.py` pins the 323-ID contract and 323-row text bank.
- `wp4/asset_discovery.py` recognizes the historical selected checkpoint hashes.

Implement a separate final release contract with original ordered 333 global
IDs, component/name mappings and explicit candidate/prompt/training identities.
Bind GUI inference to the same final candidate policy as offline evaluation.
Keep development support and its original hashes as a separate mode; do not
disable guards or overwrite historical fingerprints to accept new files.

Build final reference retrieval from the new release's declared 8,000-image
training pool. For an unseen composition, clearly distinguish an exact training
pair from same-disease/different-crop fallback. If no suitable training reference
exists, state that none is available. Never use official-test images as
references or label a fallback as an exact match. Continue excluding the query
and pixel-identical copies. Show the final training scope and 333-candidate scope
in the GUI and exported provenance.

The layout does not require another redesign. Final asset discovery must verify
the new checkpoints while preserving the historical release entries.

## Owner checklist

| Work | Coordination | Next concrete output |
| --- | --- | --- |
| Call record and shared roadmap | Team / Issue #8 | Record these five answers with their student-reported source. |
| WP1 final training | Fa Zhen, with Andy's data/augmentation checks | Reviewed full-8,000 recipe, new checkpoint pair and export provenance. |
| WP2 and proposed adapter | Yuen Kai, with Jonathan's evaluation coordination | Reviewed full-training recipes and immutable final releases. |
| WP5 reporting | Jonathan / Frederick | Overall/Seen/Unseen table and paired adapter-versus-Frozen-CLIP comparison. |
| WP4 final inference | Fa Zhen / Frederick | Separate final-333 catalog, scorer/text-bank integration and labelled references. |
| Report, deck and video | Team | Replace stale pending-policy text; keep historical and new results separate. |

This documentation update does not execute training, build a new GUI catalog,
change model files, run official inference, or claim that the target is met.
