# WP1 implementation handoff

This package contains a patch, not a published GitHub branch or trained model.
Upstream base: FZhen2450/COS30082_ML_Project main at 38afc2c071876e768f961b50577702a0311578ed.
Local review branch: feature/wp1-shared-data-contract.
Local prepared code commit: d1ea5402be4ca264a1f8db85eae9de15bfcc76d2.
The local Git baseline was reconstructed from the exact upstream archive; its
commit ancestry is not the upstream ancestry. Apply wp1.patch to your real clone;
do not push the reconstructed repository as a replacement for upstream.

## Included

- wp1.patch: scoped CSV attributes, source verifier, development protocol,
  separate classifiers, screening configs, native transforms, checkpoint reload,
  PIL prediction adapter, WP5 development bridge, augmentation preview, tests/docs.
- WP1_Shared_Contract_and_Runbook.md: complete English execution instructions.
- WP1_Status_Update.md: accurate shareable team status; no message was sent.
- evidence/: actual verification outputs and limitations.

## Apply to the existing project

Your project copy was located at:
C:/Users/USER/OneDrive/Desktop/COS30082 ML/Group assignment/github/COS30082_ML_Project
It was not modified by this task. Open that folder in VS Code, then use its terminal.
Keep your current uncommitted work intact; inspect git status before proceeding.

    git status --short
    git switch -c feature/wp1-shared-data-contract
    git apply --check C:/path/to/extracted/WP1_Handoff/wp1.patch
    git apply --index C:/path/to/extracted/WP1_Handoff/wp1.patch
    git diff --cached --stat

If apply reports conflicts, stop and inspect differences; do not force replacement.
The patch includes no CSV row edits and no changes to the historical manifests.
After reviewing the staged diff, commit these staged changes using your own Git identity:

    git commit -m 'feat(wp1): add shared development contract and independent baselines'
    python scripts/check_fresh_checkout.py

The last command tests new Git checkouts without touching your current CSV files.
An existing worktree may retain LF files even after adding attributes. Use a fresh
checkout of your committed review branch for protocol export/training, and verify
its bytes with python scripts/verify_csv_checkout.py. GitHub ZIP archives do not
perform checkout conversion. Do not regenerate historical manifests to bypass a
source-hash failure.

Run python -m unittest discover -s tests -v in the pinned environment. For the
Windows Codex sandbox, TEMP/TMP/TMPDIR may need an explicitly writable workspace
folder; the delivered tests themselves were not weakened to bypass that issue.

No remote push, code PR, merge or teammate message was performed. After applying
and reviewing in the real clone, you can push the review branch and open a PR.

## Blocking external inputs

1. On the accepted WP2/WP3 VS Code/WSL machine, export the exact three ordered
   filename lists and reconcile them as described in the runbook. Counts match,
   but historical identity remains unverified. Training fails closed until reconciliation.
2. On the machine with Canvas images, run full decoding/EDA and review the
   fitting-leaf augmentation preview. The actual project copy inspected here
   contained no DPD_Small_dataset directory.
3. On a CUDA GPU machine, run the crop and disease tasks for all four screening
   configs, then development comparison. No real training or GPU benchmark is claimed.
4. Jonathan must make WP5 PR #1 metrics available for the real evaluator. The
   isolated compatibility test used its pinned actual code, not a WP1 copy.
5. Frederick must connect and test the actual GUI/retrieval implementation.
6. Freeze final candidate/unseen-metadata policy with the team/lecturer before official tests.
