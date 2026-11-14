# LeafLens full-8,000 M6 rehearsal — 12 November 2026

> 14 November source handoff: see
> [publication, fresh verification and independent rehearsal setup](LeafLens_M6_Publication_2026-11-14.md).
> The local/uncommitted status below records the original 12 November state.

The supplied private ZIP and checkpoint match the hashes published on issue #29.
This is a Codex-assisted local owner-side integration and automated GUI rehearsal.
Frederick's independent human operation and sign-off remain pending. Changes are
local working-tree changes, not a published or merged release.

Posting the result to #29 was attempted but the GitHub connector returned HTTP
403, `Resource not accessible by integration`. GitHub CLI is not installed.
The ready-to-post comment is saved in
[`ISSUE_29_M6_Full8000_2026-11-12_DRAFT.md`](ISSUE_29_M6_Full8000_2026-11-12_DRAFT.md).

## Integration

- Final-mode discovery selects the full-8,000 adapter by content hash. Development
  discovery retains its historical adapter. Changing protocols clears stale adapter paths.
- Loading checks the full-training identity, fixed terminal selection, recipe,
  original 323-class training contract, frozen encoder, finite state and architecture.
- Final inference uses the separate original 333-row float64 official text bank,
  without re-encoding or renormalizing its stored rows. Ranking arithmetic is unchanged.
- The loaded model/export is labelled `M6 full-8000 / final-333 (CPU)`.

## Verified identities

| Item | SHA-256 |
| --- | --- |
| Handoff ZIP | `1f79d37bfadb4c96aeef5590f66791532a31f01f49dedd45d42a7b9eae7da798` |
| Adapter | `a6bd8980bd091df53e567f9e67601d94d1738c998cfd137fd1270cbec9dc07f0` |
| Adapter contract | `46e3373dd15f9ff59209ed9029f607293f2e95302dc9495325a88a7a098efe99` |
| Ordered training filenames | `7f94ac00c89b212ec30a646985b24f1a61fca19cf64b6f25757547868e3e4870` |
| Original official bank file | `1e1183371b7ad3b1687f7a140af1fea89e1fa6744ac98f1a30ad6fdae19be22b` |
| Final candidate semantics | `ba17e94119e8b2b7a4b18603e25405d2f52bf4a93396973f80491800697eb62f` |

Adapter: seed 45, epoch 1, cap 0.20, 99,136 parameters,
`wp3b-visual-adapter-m4-v1`. Runtime: Python 3.13.13, Torch 2.7.1+cu118,
Torchvision 0.22.1+cu118, CPU; official CLIP revision
`d05afc436d78f1c48dc0dbf8e5980a9d471f35f6` and pinned ViT-L/14 weights.

## Rehearsal result

Original training query: `PWV2_peach_leaf_curl_6128.jpg`. The native GUI scorer
returned finite `[1,333]` float64 scores with ascending global IDs 0–332.
Stable Top-5, presentation and JSON export matched the canonical scorer exactly.

| Rank | ID | Prediction | Cosine score |
| --- | ---: | --- | ---: |
| 1 | 10 | apple / rust | 0.3052368953991595 |
| 2 | 15 | apple / scab | 0.2974595987349973 |
| 3 | 4 | apple / brown_spot | 0.28909786980415364 |
| 4 | 292 | tea / red_leaf_spot | 0.28858730997137605 |
| 5 | 7 | apple / frog_eye_leaf_spot | 0.28837828751940664 |

This peach leaf curl query is misclassified; a successful integration rehearsal
does not imply correct classification or establish official accuracy. Four
retrieved references were training-only and excluded the query. Browsing candidate
39 displayed labelled same-disease/different-crop references. Normal and 1280×840
screenshots were visually inspected. No official test images were opened.

WP1/WP4 regression suite: **63 passed, no skips**. After the final selector-label
change, its **11 model-picker tests passed again, no skips**, and native GUI
inference/export was repeated. Tk required execution outside the sandbox.

Evidence index with full receipt, source hashes and companion hashes:
[`leaflens_full8000_m6_rehearsal_20261112.json`](../artifacts/results/leaflens_full8000_m6_rehearsal_20261112.json).
Private scores, export and screenshots are in `runs/wp4-full8000-m6-final-20261112/`.
Private checkpoint/setup files are in `runs/m6-full8000-handoff-20261112/`.

Reproduce with the existing local runtime and pinned CLIP package:

```powershell
$env:PYTHONPATH = (Resolve-Path 'runs/wp4-completion-20261109/clip_packages').Path
& 'D:/COS30082_WP1_Runtime/venv/Scripts/python.exe' -B scripts/rehearse_wp4_final333.py --model m6 --output runs/NEW_M6_REHEARSAL --screenshots
```

## Team actions still open

- Publish/review these local integration changes before another workstation uses them.
- Frederick: perform independent GUI rehearsal and record his receipt on #29 by 15 Nov MYT.
- Jonathan: replay the 11 November WP1 saved-score packet and record acceptance on #30.
- WP5/model owners: finish remaining score packets, encoder/runtime/scoring parity,
  and the common four-model comparison by 17 Nov MYT. Preserve historical
  4,868-trained results separately. This rehearsal does not complete #30.
