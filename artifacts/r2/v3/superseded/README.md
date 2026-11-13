These two files were produced by `scripts/r2_calibration_shift.py` before a bug fix and are kept only as evidence.

The bug: under the v3 candidate universe, a source-ablated fold can lose compositions (they move from F to E), which shifts column indices. The unseen-side comparison scored ablated-fold predictions against **base-fold** truth indices. As a result, "unseen_top1_source_novel_*" is wrong in these files (e.g. 8.5% instead of the correct value).

The seen-side comparison was already aligned (`ts2`, `sm2`) and is unaffected. The fix uses the ablated fold's own truth indices (`tu2`). The corrected outputs are `../source_stress_D1.json` and `../source_stress_F10.json`. The fixed script also reproduces the committed v2 stress results exactly on the legacy benchmark (`../stress_v2_repro_check.txt`).

Do not cite these files.
