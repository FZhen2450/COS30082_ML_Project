# Fa Zhen owner review: small-window Top-5 correction

Decision: **take Jonathan's supplied correction forward as the single layout
candidate**, in a separate stacked PR against #37. Do not combine it with Yuen
Kai's independently prepared correction. This is a code/test review decision,
not a merge or Frederick's independent real-laptop acceptance.

## Source and scope

- Issue #29 review package:
  https://github.com/FZhen2450/COS30082_ML_Project/issues/29#issuecomment-6878455930
- ZIP SHA-256:
  `30feb6cfbd552b19130edb1c8c5ca88d5a8bd57a973487bfdc421d156a538f66`.
- Supplied patch commit: `04afd4e5b57032afa4a042aea8caf91afd591e72`.
- Original patch base: `1af9ef54b1a1f7cd63d4b1b482faa2311abd6117`.
- Owner review base: `f1ccc8e26e52a04c68116f44c9a902704ddfac07`, which adds
  documentation only after the original base. The patch applies cleanly.
- Exact diff: three files, +49/-3: `wp4/layout.py`, `wp4/widgets.py` and
  `tests/test_wp4_model_picker.py`.

The archive SHA and all **34 manifest-listed file hashes** matched. Supplied
documents were treated as evidence rather than instructions; the probe script
was read before running. No package notebook or model was executed.

## Findings and independent checks

No blocking defect was found in the reviewed three-file change. It reserves
the results card's required height from the heading, status text, five compact
rows, score caveat and padding. It reuses the row-height constants for sizing
and rendering. Predictor, M6 integration, score arithmetic, mapping and export
code are unchanged.

The new test was first run with the unmodified production layout. It failed
at **all three** client sizes: 1366×688, 1280×640 and 1280×560. After applying
the production fix, **43/43 WP4 tests passed, no skips**, in 23.009 seconds.
Runtime: Windows, Python 3.13.13, Torch 2.7.1+cu118, torchvision 0.22.1+cu118,
CPU, using the existing desktop Tk installation. No full-suite pass is claimed.

The owner then ran the reviewed real-app probe with the existing training
query and a labelled synthetic predictor. Both checks confirmed:

| Emulated screen | Actual client | Ranks 1–5 visible | Score note visible | Preview height |
| --- | --- | --- | --- | ---: |
| 1366×768 | 1366×688 | All 100% | 100% | 183 px |
| 1280×640 | 1280×560 | All 100% | 100% | 55 px |

At both sizes, reference scrolling, rank ordering against synthetic scores,
JSON export/state equality and the full-8000 label passed. The Load model
button remains reachable by scrolling the existing sidebar. At the shortest
size the model chooser is partly below the sidebar viewport until scrolling.
The owner inspected the shortest-screen screenshot and confirmed that all
five results and the score caveat are visible.

## Selection rationale and limits

Jonathan's small, supplied patch directly addresses the reproduced defect and
passed the owner's independent regression and real-app geometry checks. This
is sufficient to select it for the next review stage. Yuen Kai's source was
not supplied in this review, so no comparative quality claim or rejection of
his implementation is made. He independently corroborated the original defect.

At 1280×640 the image becomes a **55 px thumbnail**. This is a disclosed
tradeoff, not a full responsive redesign. The harness emulates window sizes
on the owner's desktop; it does not establish physical-laptop or 175–200%
display-scaling acceptance. The screenshots use synthetic ranking scores and
must not be presented as actual M6 predictions or accuracy evidence.

## Next owner actions

Jonathan can open his original correction as a stacked PR against
`review/wp4-full8000-m6-20261114`, preserving attribution and the #32 → #37 →
layout order. This review did not apply the fix to the published #37 branch
or merge either candidate. Frederick's native full-8,000 M6 laptop rehearsal
and personal receipt remain his responsibility.

Private owner evidence: `runs/top5-review-20261114/`, including the original
archive, baseline failure log, patched pass log and two probe directories.
No official images, model weights or WP5 score packets were used in this review.
