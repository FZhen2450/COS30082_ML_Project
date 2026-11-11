# Draft Issue #8 update - not posted

Historical draft from before the implementation work. See the current
[WP1/WP4 release record](WP1_WP4_Final_Release_2026-11-11.md) before posting a
completion update; the original no-execution status below describes that draft.

## Dr. Lee Teams-call clarification - 11 November 2026

The following answers were relayed from today's call with Dr. Lee. This is a
student-reported call record, rather than a formal written rubric.

1. **Evaluation:** report Total (Overall), Seen and Unseen simultaneously;
   all models and folds rank against the full **333 candidates**. Show the
   counts/denominators, Top-1/Top-5 and Seen/Unseen H@1/H@5. Overall means the
   combined correct count divided by 3,448, not H or an unweighted fold average.
2. **Final fitting:** retrain trainable final models on the complete **8,000
   course training images**, rather than only the 4,868 development fitting
   images. Keep the original development runs and results as history. Frozen
   CLIP remains the frozen reference.
3. **GUI:** the current layout/features are sufficient. Expand inference to
   **333 classes** and use clearly labelled training-image fallbacks for unseen
   compositions. The currently pushed GUI is still development-323, so this
   requires a separate final contract, text bank/scorer and checkpoint handoff.
4. **Reporting priority:** emphasize official-test generalization; development
   scores remain selection evidence and must be reported separately.
5. **Proposed approach:** demonstrate value by exceeding Frozen CLIP's reported
   **6.62% official H@1**. Historical M6 is 5.83%, so this target is not yet met.
   Compare full-precision saved metrics for the actual final decision.

Next actions: agree and freeze the full-training recipes; issue new checkpoint
identities; update WP5 Overall/Seen/Unseen tables; integrate final-333 GUI
inference and training-only references; then update the report/deck and rehearse.
Preserve all historical checkpoints/results. No repeated official-test tuning
or guaranteed improvement is implied by the target.

No retraining, new official evaluation or GUI-333 implementation has been
performed as part of recording this clarification.
