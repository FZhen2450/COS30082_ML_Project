# R2 v4 public history: what was published and how it differs from the private research record

## What this branch is

* The public commits after `1b2fcdb` are a **sanitized replay** of the 17 private R2 v4 commits `ca63114` … `06ccf43`
  (frozen research head `06ccf43`, 17 Nov 2026).
* Each public commit has its private commit's exact tree, message (plus a replay trailer) and author date. The one
  exception is the evidence files listed below, whose local machine paths were replaced.
* The final public commit adds this notice, the hash map and `docs/R2_Second_Run_Final_Summary.md`. Those three files do
  not exist in the private history.

## Sanitized derivatives (not byte-identical)

* `artifacts/r2/v4/PUBLICATION_SANITIZATION_MAP.json` lists **46 evidence files**: training and test logs, run scripts,
  failure receipts with tracebacks, and the J4b three-seed `evaluation.json`.
* In these files only local path strings were replaced, by `<WORKSPACE>`, `<PROJECT>`, `<CONDA_ENV>`, `<HOME>` or
  `<TMP>`. No metric, hash or other content changed.
* For each file the map gives:
  * the stage manifest that lists it;
  * the **private original sha256**, checked against that manifest;
  * the **public sanitized sha256**;
  * the replacement counts.
* **These public copies are not byte-identical to the originals.**
  * The stage `SHA256SUMS` manifests are published unchanged and record the private original hashes. On a public
    checkout, `sha256sum -c SHA256SUMS` therefore reports exactly these 46 files as changed.
  * Verify them with the map instead.
  * The original bytes are kept in the private research history and the owner's review archives.

## Unchanged and verifiable from the public tree

* All code, tests and docs (other than the 46 files), all declarations, amendments 1–4 and all **code-pinned** files are
  byte-identical to the private commits.
* `python scripts/r2v4_declare.py --verify [--amendment N]` and `python scripts/r2v4_a4.py verify` pass. No declaration
  pin was changed, and no pinned file was sanitized.

## Not published

* Private score packets (`*.npz`), feature caches and checkpoints: only their sha256 values in
  `private_files_SHA256SUMS.txt`.
* Image files and official-test files or labels. No official-test data was ever used. Every image name quoted in the
  published evidence is a DPD-Small training-CSV image.

## Tests that need private evidence

These tests can pass only against the private originals and data:

* `test_r2v4_amendment4.TestAmendment3Preserved.test_historical_artifact_hashes_unchanged` checks the manifests against
  the original evidence bytes and private run files. On the public tree it fails, by design, for the sanitized files.
* Tests that compare with private commit objects (for example `d9ed896`) or that load the training images, feature caches
  or registered runs need the owner's local data.

All results are development cross-validation on reused training-derived folds, not official-test results.
