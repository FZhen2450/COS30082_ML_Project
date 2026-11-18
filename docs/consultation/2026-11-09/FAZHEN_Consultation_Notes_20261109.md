# Fa Zhen consultation speaking notes

Updated 9 November 2026 for the 10 November 11:30am consultation.

## Slide 3 speech

Slot: 1:05-2:00. 95 words, approximately 50-55 seconds at a measured pace.

For WP1, we trained separate ResNet50 classifiers for crop and disease. We selected crop epoch two and disease epoch three on development validation, then froze both models before official testing. Their native float32 log-probability scores rank valid crop-disease pairs. Under the team's shared-333 protocol, WP1 achieved 32.35 percent seen Top-1, 8.29 percent unseen Top-1 and 13.19 percent H-at-one. Jonathan independently verified the saved outputs. WP1 is the observed leader, with concentrated unseen hits and uncertain comparisons that Jonathan will explain. WP2 uses one ConvNeXt-Base backbone with two classification heads. Our original model selections remain unchanged.

## If asked about the official result

WP1 has the highest observed official H@1 among the four frozen reporting
systems. Its exact counts are 647/2,000 seen Top-1 and 120/1,448 unseen Top-1,
with H@1 13.19% and H@5 39.01%. Of its 120 correct unseen predictions,
49 are pepper bacterial spot, 46 grape black rot and 19 sugarcane rust.
These total 114, or 95% of its correct unseen predictions. This is not 95%
accuracy on the unseen fold.

Jonathan's independent saved-score verification passed 86/86 checks. Together
with his earlier 212 checks, it reproduced 13,792 model-image Top-5 rows.
Those are independent WP5 results; do not describe them as my own audit.
Only ten official unseen compositions are available. The exploratory H@1
difference intervals are [-4.12,+15.01] percentage points versus CLIP and
[-4.42,+16.10] versus M6. They do not establish reliable superiority over either.
The development and official tasks differ in candidates and composition mix.
We preserve the first outputs and original model selections.

## WP4 technical backup

The published LeafLens interface supports WP1, frozen CLIP and M6. It uses real
native image inference, fitting-only references, original IDs and labelled
different-crop same-disease fallback. Scores rank matches; they are not calibrated
confidence. The published hardening passed 79/79 applicable isolated tests
and all three real picker/inference/export rehearsals. It adds WP1 file/input
provenance and fixes ID validation and Tk teardown. The combined publication
suite subsequently passed 95/95 without skips. Source is published at
[7314644](https://github.com/FZhen2450/COS30082_ML_Project/commit/73146444ace8637426042e3bfb76aa9bb3b8585c), including WP4 commit
[5c9d1f9](https://github.com/FZhen2450/COS30082_ML_Project/commit/5c9d1f9b05f7de9f9c86082ecd6bf1f676048d21); the branches are not merged into main.
Main's historical 621-test receipt retains its separate scope.

On the fixed peach/curl development example (truth 219), WP1 returns
102/97/104/103/219, M6 15/10/293/292/4 and frozen CLIP 293/292/53/291/41.
Every Top-1 is wrong. The true class is fifth for WP1 and absent for CLIP/M6.
The references illustrate a predicted label; they do not validate correctness.
Frederick retains the actual GUI demonstration and independent-operation role.

## Avoid repeated consultation questions

- I explain WP1's separate classifiers, selected epochs, compatible runtime,
  native FP32 scorer and frozen one-time run.
- Jonathan explains independent verification, composition concentration,
  intervals and scientific interpretation.
- Yuen Kai asks about shared versus restricted candidates and final fitting scope.
- Andy covers data and augmentation checks; Frederick demonstrates LeafLens.

If Dr. Lee asks why ResNet won, describe WP1 as the observed leader on this
task. The evidence does not isolate architecture from pretraining, fitting
budgets, candidate policy or composition distribution. Do not attribute a single
cause or choose a new primary model from the official scores.

Source: https://github.com/FZhen2450/COS30082_ML_Project/issues/8#issuecomment-6732629416
