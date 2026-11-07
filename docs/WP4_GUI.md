# WP4 desktop GUI

This initial WP4 implementation assumes no existing Frederick GUI/retrieval code.
It implements the brief's image input, crop-disease result display and relevant
training-image display in a local English Tkinter application.

## Run

Install the existing `requirements.txt` in Python 3.11-3.13, including Pillow.
Tkinter is included in the standard Windows Python installer (enable Tcl/Tk).
From the repository root:

```powershell
python gui.py
```

On this machine, `python` points to Python 2.7. Use the verified Python 3 path:

```powershell
& "$env:LOCALAPPDATA/Programs/Python/Python313/python.exe" gui.py
```

No extra GUI packages or web server are needed. WP1 uses the existing project
dependencies. Frozen CLIP/M6 additionally require the pinned official CLIP
package and explicitly supplied local model files (see below). No model files
are downloaded by the app.
The app resolves default paths relative to the repository, not the current shell
directory. Alternate inputs are supported:

```powershell
python gui.py --images D:/DPD_Small_dataset --protocol artifacts/data/development_seed42_reconciled.json
```

## Available now

- Open and preview a local image, with EXIF orientation corrected.
- Browse labelled training references immediately without loading a model.
  Class browsing is explicitly separate from recognition and creates no prediction.
- Load two real WP1 crop/disease `.pt` checkpoints on CPU or CUDA. The existing
  loader validates checkpoint tasks, preprocessing, source and protocol digests.
- The model picker also offers frozen CLIP ViT-L/14 and the selected M6 capped
  adapter (seed 45, epoch 1). Each choice displays only its required local files.
  Native image encoding runs for each uploaded image; saved image-feature caches
  are not used as an encoder.
- Identify with the existing `IndependentPredictor`: show Top-5 crop/disease
  pairs using original global IDs, and display their ranking scores.
- Select a predicted pair to see up to four fitting-only training references.
  If no fitting example exists for the pair, same-disease examples from other
  crops are clearly labelled as such. No exact-pair match is invented.
- Matching examples are ordered by a simple RGB histogram distance to the query;
  manual class browsing uses filenames. This is illustrative reference retrieval,
  not learned semantic retrieval or a disease-classification algorithm.
- Exclude the input path and pixel-identical query images from references.
- Export real predictions and protocol fingerprints to JSON.
- CLIP/M6 exports retain the verified catalog/model candidate hash namespaces,
  prompt hash, CLIP identity and adapter provenance alongside the ranking scores.
- Image decoding, checkpoint loading, inference and retrieval run in background
  threads. Tk widgets are updated only on the UI thread. Errors are shown without
  substituting example predictions.

## Boundaries and next integration

No trained model checkpoints are committed. Identification remains disabled
until valid local weights are loaded; image preview and class browsing work now.
Historical split reconciliation remains pending unless a reconciled protocol is
supplied. Browsing the supplied unreconciled WP1 development manifest does not
approve it for training or establish equivalence to WP2/WP3 historical filenames.

References are restricted to `fitting_images` from the validated protocol and
read metadata only from `DPD_Small_train.csv`. The code does not load either
official-test CSV. Do not use the GUI to tune models on official-test images.
The default 323 development candidates cannot recognise final official unseen
classes; final candidates and permissible label metadata require a separate
approved protocol. The UI displays this scope.

Scores are not calibrated confidence. The GUI does not compute dataset accuracy,
infer that an arbitrary uploaded image is OOD, or claim official-test results.

WP2 is not selectable. Frozen CLIP/M6 use their own native preprocessing and
direct float64 cosine scoring, then the existing `present_prediction` and
fitting-only retrieval path. WP5 metrics remain in the offline evaluator.
Learned feature retrieval and packaged executables are future work.

## CLIP / M6 private asset handoff

Install the existing project dependencies in a separate environment, then the
official CLIP package at the exact revision below. This is an explicit setup
step; the GUI never installs or upgrades packages.

```powershell
python -m pip install "git+https://github.com/openai/CLIP.git@d05afc436d78f1c48dc0dbf8e5980a9d471f35f6"
```

Use an environment compatible with the selected checkpoints. In particular,
the WP1 loader checks the exact torchvision version saved by each WP1 run;
the private WP1 release here used torch 2.7.1+cu118 / torchvision 0.22.1+cu118.

The connection dialog requires:

| Choice | Local files |
| --- | --- |
| WP1 | Crop checkpoint and disease checkpoint |
| Frozen CLIP | Official ViT-L/14 JIT weights |
| M6 | Official ViT-L/14 JIT weights and the selected adapter checkpoint |

Pinned SHA-256 values from Issue #13:

- ViT-L/14: `b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836`.
- M6 seed 45 / epoch 1 / cap 0.20: `8835a18b9eb933ae224bd17866f18bc56406e24d8612743f4e7c88848a8135cc`.

The loader currently implements the published `wp3b-visual-adapter-m4-v1`
adapter-only envelope: `adapter_spec`, `adapter_state`, `contract`,
`contract_sha256`, `text_state` and `selection`. It checks the selected M6
hash before safe CPU tensor loading, then validates its architecture, exact
ordered IDs/pairs/component names, original source/split hashes, prompt policy,
native preprocessing and saved direct text bank. The historical model digest
is retained separately from the catalog digest after equivalence is verified.
An unknown private M6 envelope is rejected; obtain the actual checkpoint/schema
handoff rather than renaming/repacking it or bypassing the pinned hash.

The CLIP loader accepts only the verified official JIT file, reconstructs the
pinned native model/preprocessing, freezes it, and has no arbitrary-pickle or
download fallback. Inference applies the accepted bounded 768→64→768 residual
and canonical CPU float64 cosine arithmetic. Stored training-image embeddings
are never used to predict a new image.

All remote branch trees were checked on 6 November 2026: no model weight files
were present. Issue #3 states that M6 remains local to its owner. The supplied
report, demo script and roadmap contain experiment evidence, not model assets.
On 7 November 2026, the owner's private M6 handoff was downloaded and its hash,
embedded contract and text bank verified. The official pinned ViT-L/14 base was
obtained separately and its full SHA-256 verified before loading. A new local
clean-launch rehearsal passed through the normal model picker with real native
image encoding, Top-5 display, four fitting-only references and matching JSON.
See `docs/WP4_M6_Rehearsal_2026-11-07.md`. The earlier owner smoke and this new
cross-machine check remain distinct; neither establishes official performance.

## Clean-launch rehearsal

1. Obtain the two pinned private files from Chia. The published checkpoint
   envelope embeds its direct text bank; if the real M6 export differs, obtain
   its approved interface and any companion files before integration acceptance.
2. Launch `python gui.py --protocol /path/to/development_seed42_reconciled.json
   --images /path/to/DPD_Small_dataset --source /path/to/DPD_Small_train.csv`.
   An existing reconciled protocol in `artifacts/data/` or the project root is
   preferred automatically. No protocol is silently regenerated or declared
   reconciled by the app.
3. Select **Connect a model → M6 capped adapter (seed 45, epoch 1)**, choose
   both supplied files, select CPU or available CUDA and load the model.
4. Open the predetermined fitting/seen-development example; run Analyze leaf.
   Verify five ranked global IDs and full-precision scores against the canonical
   scorer, the selected model label, accurately labelled fitting-only references,
   query/identical-copy exclusion and exported JSON provenance.
5. Repeat model selection with WP1 and frozen CLIP as applicable. A failed load
   must retain the previous model's label/results; a successful model change
   clears old predictions and disables export until new inference completes.
6. Record the launch command, environment, file hashes, query filename, output
   IDs/scores, reference filenames and export. Do not present the one-image demo
   or any synthetic test as an official-test result.

The optional `scripts/rehearse_wp4_m6.py` harness automates the same picker and
file dialogs with supplied real local assets; it never injects a predictor.
Use `--clip`, `--adapter`, `--protocol`, `--images`, and a new `--output` directory.
`--screenshots` records the app window for a private evidence bundle. Its default
query is the predetermined seen-development peach/curl example. No official
test input is accepted by the harness's filename membership check.

## Verification

```powershell
python -m unittest discover -s tests -v
```

WP4 tests exercise retrieval boundaries, same-disease fallback, missing images,
self-match exclusion, non-contiguous IDs, exact ties and incompatible model
output rejection. Synthetic software checks do not establish real model quality.

Initial verification on 31 October 2026: 39/39 automated tests passed. Actual
Tkinter window checks exercised image opening, real fitting-image browsing and
the unloaded-model state. Default and minimum window sizes were checked under
Windows display scaling. No trained model or official-test evaluation was run.

## Visual design update

The desktop interface uses native Tk widgets with an off-white botanical palette,
rounded cards, consistent pixel-based typography and a single primary action.
The Top-5 view uses selectable ranking cards: the first match is highlighted,
with numbered badges and right-aligned raw ranking scores. Arrow keys, Enter
and Space support ranking selection; buttons are keyboard accessible.
The compact layout activates below 860 pixels in height. The minimum window is
1100 x 800; input settings and reference galleries scroll independently while
the Analyze and Export actions stay visible. No GUI dependencies were added.

Local `runs/wp4-modern-preview.png` shows real image browsing with no prediction.
`runs/wp4-modern-ranking-synthetic.png` is explicitly a synthetic layout check,
not a trained model result. These previews remain outside Git.
