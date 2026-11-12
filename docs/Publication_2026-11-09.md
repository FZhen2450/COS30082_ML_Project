# WP1 and WP4 source publication on 9 November 2026

The project owner explicitly authorized pushing local updates to GitHub.
Two new source branches were pushed without force and their remote refs were
independently read back from GitHub:

| Branch | Verified source commit | Scope |
|---|---|---|
| `fix/wp4-completion-20261109` | `5c9d1f9b05f7de9f9c86082ecd6bf1f676048d21` | WP4 lifecycle, provenance, type validation, real-model rehearsal and runbooks |
| `fix/wp1-final-handoff-20261109` | `73146444ace8637426042e3bfb76aa9bb3b8585c` | Includes WP4 plus WP1 audit/replay tools, reference aliases, tests and handoff documentation |

The working checkout is on the combined branch. The original local changes
were preserved in named Git stashes before integration. Root-level development
reference aliases were verified against the tracked reconciled artifact,
including exact filename order. Existing source/scorer/checkpoint identities
remain unchanged. No force push or merge into main was performed.

The combined branch passed **95/95 applicable CPU tests, no skips**, in
52.905 seconds using the existing producer-compatible runtime. Scope: WP4
presentation/picker/native adapter, independent baseline, data/shared protocol,
WP1 audit/export/replay and native GUI parity guards. Actual Tk event paths were
included. This does not replace the earlier 48-WP1, 79-combined-WP4 or historical
621/642 Katana receipts, and does not claim a full repository Windows pass.

Jonathan's independent saved-score audit is separately recorded in
[Issue #8](https://github.com/FZhen2450/COS30082_ML_Project/issues/8#issuecomment-6732629416):
212/212 three-model checks and 86/86 WP1 checks, reproducing 13,792 model-image
Top-5 rows. WP1's H@1 is the observed official leader; its concentrated unseen
hits and exploratory comparison uncertainty remain disclosed. No actual model
inference, real-data training, retuning or post-test reselection was performed
for this publication.

Updated report, presentation, personal speaking notes and prepared issue replies
are under `docs/consultation/2026-11-09/`. Their subsequent documentation commit
does not change either pinned source commit above. The original report/deck
copies and historical result packages remain unchanged.

Raw images, checkpoint/model files, runtime environments, prediction-score
binaries, local handoff ZIPs and temporary dependency caches are excluded from
Git. The consultation PPTX is a small editable document, not a model artifact.
The original report is editable Markdown; figures referenced by its source ZIP
must still be supplied by the report owner's normal publishing pipeline.

Issue replies are prepared text only. This task does not post comments, tick
another member's checklist, invent review approval or mark the team video and
signatures complete. Main remains the separately sealed integration checkpoint.
