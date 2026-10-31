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

No extra GUI pip packages, web server, online API or model download are needed.
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
- Image decoding, checkpoint loading, inference and retrieval run in background
  threads. Tk widgets are updated only on the UI thread. Errors are shown without
  substituting example predictions.

## Boundaries and next integration

The repository has no trained WP1 checkpoints. Identification remains disabled
until real checkpoints are loaded; image preview and class browsing work now.
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

WP2 and WP3 inference adapters are not in this repository yet and cannot be
selected. When available, integrate their native preprocessing and candidate
scoring through the same PIL-image `predict` contract, then pass output through
`present_prediction`; never route CLIP similarities through independent-head
scoring. Display the actual adapter name and score semantics. WP5 metrics remain
in the offline evaluator. Learned feature retrieval and packaged executables are
future work.

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
