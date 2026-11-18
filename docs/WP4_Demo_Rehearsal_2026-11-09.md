# WP4 demo and peer rehearsal — 9 November 2026

Use the existing compatible environment and verified local assets. No official
test image, retraining, model selection or package upgrade is needed.

## Launch on the current Windows machine

From the project root:

```powershell
./scripts/launch_wp4.ps1 -Python 'D:/COS30082_WP1_Runtime/venv/Scripts/python.exe' -ClipPackages runs/wp4-completion-20261109/clip_packages -CheckOnly
./scripts/launch_wp4.ps1 -Python 'D:/COS30082_WP1_Runtime/venv/Scripts/python.exe' -ClipPackages runs/wp4-completion-20261109/clip_packages
```

`-ClipPackages` reuses existing copied package files; it is optional when pinned
CLIP is already installed in the chosen environment. The launcher refuses an
optional package folder containing Torch/Torchvision overrides and restores the
parent's PYTHONPATH afterwards. On another machine supply its actual compatible
Python path and approved private files. Never replace version strings to bypass
the WP1 loader's torchvision check. GUI startup itself does not load weights.

## Ninety-second demonstration

1. **0:00–0:15 — Scope and input.** Show the 323-development-candidate scope;
   open `PWV2_peach_leaf_curl_6128.jpg` from the local dataset. State that this is
   a predeclared seen-development example with peach/curl truth, not an official
   test or an arbitrary OOD detector demonstration.
2. **0:15–0:35 — Real model.** Connect M6 through the normal picker, supplying
   the pinned ViT-L/14 base and selected seed-45/epoch-1 adapter. Select CPU and
   Analyze leaf. Explain that the app performs native image encoding.
3. **0:35–0:50 — Honest failure.** Show IDs 15, 10, 293, 292 and 4. The top
   match is apple/scab; truth 219 is absent. Say: “This example is wrong. These
   are ranking scores, not calibrated confidence. The adapter's development
   gains did not establish uniform disease transfer, and healthy predictions
   can also be wrong.” Do not present a successful GUI call as model accuracy.
4. **0:50–1:10 — Reference semantics.** Select a result and explain that its
   references come only from fitting images, excluding the query and pixel
   copies. Browse apple/alternaria leaf spot (ID 1): fitting examples are cabbage
   and cauliflower with the same disease, labelled different-crop fallback and
   “Class browser (not a prediction).” Do not call them exact apple examples.
5. **1:10–1:30 — Export and support.** Export JSON and point out selected model,
   full-precision rankings, candidate/split hashes and provenance. Show the three
   supported model choices. Mention that normal EXIF-corrected CPU inference is
   distinct from the historical batch-8 CUDA parity check. Finish with the actual
   team recording, not a simulated speaker or fabricated recording URL.

## Automated real-model rehearsal

All commands require a new output directory. The default query is the declared
development image; the harness rejects filenames outside the development folds.
It uses the actual picker/loader/predictor and checks reference membership and
JSON export. It does not supply a fake predictor or calculate dataset accuracy.

```powershell
python -B scripts/rehearse_wp4_m6.py --model wp1 --crop PATH_TO_CROP_E2.pt --disease PATH_TO_DISEASE_E3.pt --protocol artifacts/data/development_seed42_reconciled_v1.json --images DPD_Small_dataset --output NEW_WP1_REHEARSAL --screenshots
python -B scripts/rehearse_wp4_m6.py --model frozen_clip --clip PATH_TO_VERIFIED_VIT_L14.pt --protocol artifacts/data/development_seed42_reconciled_v1.json --images DPD_Small_dataset --output NEW_CLIP_REHEARSAL --screenshots
python -B scripts/rehearse_wp4_m6.py --model m6 --clip PATH_TO_VERIFIED_VIT_L14.pt --adapter PATH_TO_SELECTED_M6.pt --protocol artifacts/data/development_seed42_reconciled_v1.json --images DPD_Small_dataset --output NEW_M6_REHEARSAL --screenshots
```

On Windows screenshots target only the application's HWND, avoiding unrelated
foreground windows. The recorded normal/minimum-size layouts and fallback view
must be visually inspected before sharing. A class with one fitting reference
correctly displays one; four references are a maximum, not a fabricated minimum.

## Frederick's independent receipt

Fill this in only after actual independent operation:

- Operator and date/time:
- Source commit and launch command:
- Python / Torch / Torchvision / pinned CLIP revision:
- Base/adapter (and WP1 if used) file hashes verified:
- Query filename and development-fold membership:
- Ranked IDs/scores and exported JSON checked:
- Fitting-only references, query-copy exclusion and fallback labels checked:
- Failed load retains the previous model; successful switch clears stale output:
- Minimum-size keyboard/scroll/Analyze/Export usability:
- Recording file/URL, actual speakers, reviewer and remaining observations:

Leave fields blank until performed. Do not fill them with another member's
automated evidence or claim that a prepared script is a recorded presentation.
