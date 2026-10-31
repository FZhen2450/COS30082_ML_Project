# WP1 integration status — 31 October 2026

The WP1 patch was applied on `feature/wp1-shared-data-contract`, based on main
`cead983` (including merged WP5 PR #1). No DPD training or official-test model
evaluation was run. Backbone selection and training remain with the model owners;
WP4 GUI/retrieval and final evaluation policy are deferred.

## Current verification

- 34/34 automated tests passed in Python 3.13.13, torch 2.9.1+cpu,
  torchvision 0.24.1, pandas 2.3.3 and Pillow 12.0.0.
- CSV checkout verification passed: all three original Git blob IDs, historical
  CRLF source hashes and existing split-manifest source hashes match.
- Fresh Git checkouts passed under `core.autocrlf=false`, `true` and `input`.
  The checksum list has an explicit LF rule; the historical archive disables
  text conversion to retain its original bytes across platforms.
- The training-only protocol validates: 4,868 fitting / 1,238 seen validation /
  1,894 internally unseen validation; 323 ascending original global candidates.
- The synthetic WP5 bridge passed against the merged `evaluation.metrics`,
  including non-contiguous IDs, Top-1/5, H@1/H@5 and per-pair counts.
- All 11,448 metadata-listed real images decoded successfully, with no missing
  or unreadable files. See [the audit](../artifacts/data/wp1_image_audit_2026-10-31.json).
- EDA statistics and all six figures were regenerated locally in `runs/wp1-eda/`
  using seaborn 0.13.2. The [summary](../artifacts/data/wp1_eda_summary_2026-10-31.txt)
  is retained; figures, including the training-image grid, remain local.
- A fitting-only augmentation preview was generated at
  `runs/wp1-augmentation-review/augmentation_preview.png`. An assistant visual
  inspection found visible symptoms retained in the sample; Andy/Fa Zhen's human
  acceptance remains pending in the local `review.json`. Previews stay out of Git.

## Issues #2/#3 and split reconciliation

Read both issue bodies and all available comments on 31 October 2026:
[WP2 #2](https://github.com/FZhen2450/COS30082_ML_Project/issues/2) and
[WP3 #3](https://github.com/FZhen2450/COS30082_ML_Project/issues/3).
Their published counts, seed, component coverage and candidate ordering agree
with the WP1 protocol. The issues contain results and digests, but do not attach
the three complete ordered filename lists or the original hash serialization.

Preserve the reported historical candidate digest
`4cb0a24a96cf647f32311817fa7b082a46d3cf00c20ae61f4287e83cc898dfaf`
and split digest
`f0c7ba212449fd68f1d19c0f84cc91bf4cc1af29ae405128e5198bce89f25f85`.
The WP1 canonical-JSON digests use separate namespaces and are not replacements.
Matching counts cannot confirm exact membership/order. Training remains blocked
until an accepted WP2/WP3 ordered filename export is supplied and reconciled:

```powershell
python shared_protocol.py --reference accepted_wp2_wp3_filenames.json --output artifacts/data/development_seed42_reconciled.json
```

The required JSON keys are `fitting_images`, `seen_validation_images` and
`unseen_validation_images`, each containing the original ordered filenames.
Do not generate this reference from WP1's own output and call it independent
reconciliation. See the [runbook](WP1_Shared_Contract_and_Runbook.md).

## Dataset sharing

The local dataset contains 11,448 files totaling 3,455,601,743 bytes. By owner
choice, teammates obtain the images from Canvas; no raw images are committed.
GitHub retains the decoding audit and the per-file
[SHA-256 manifest](../artifacts/data/DPD_Small_images.sha256).
To compare a downloaded copy from the project root in Python 3.11+:

```python
import hashlib
from pathlib import Path

root = Path("DPD_Small_dataset")
failures = []
for line in Path("artifacts/data/DPD_Small_images.sha256").read_text().splitlines():
    expected, name = line.split("  ", 1)
    path = root / name
    if not path.is_file():
        failures.append(name)
        continue
    with path.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
            failures.append(name)
print("Missing or different:", failures)
raise SystemExit(bool(failures))
```

## Handoff folder retirement

The transport folder `WP1_Handoff_2026-10-31/` is redundant after application.
Its 12 files are accounted for in the [inventory](wp1-handoff-2026-10-31/inventory.json).
The original README, status, runbook and all verification evidence are preserved
byte-for-byte in `docs/wp1-handoff-2026-10-31/`; `.log` evidence uses `.txt` names
so normal Git ignore rules do not hide it. The patch itself is represented by
the applied repository changes and its recorded SHA-256; reverse-application
checking passed immediately after application. Original archive statements refer
to the earlier preparation environment and are not current completion claims.

The active runbook is `docs/WP1_Shared_Contract_and_Runbook.md`. Remaining inputs
are explicitly recorded above; deleting the transport folder does not mark
training, historical reconciliation or human augmentation acceptance complete.
