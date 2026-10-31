# WP1 status update — 31 October 2026

The proposed scoped CRLF implementation is prepared and technically verified.
Only the three DPD-Small CSV paths receive text eol=crlf attributes. Fresh Git
checkouts with core.autocrlf=false, true and input reproduced all three expected
historical CRLF source hashes; committed LF blob IDs and dataset rows remain
unchanged. Both historical split-manifest source hashes were verified. These
checks used initialized repositories with copied Git objects and actual Git
read-tree/checkout-index conversion. Remote clone transport was blocked by this
Windows sandbox and was not claimed as verified.

The local review implementation adds a training-source-only three-fold protocol
(4,868 fitting / 1,238 seen validation / 1,894 internally unseen validation), 323
ordered global candidates, validation against ordered historical filename lists,
and separate crop/disease classifiers for ResNet50, EfficientNet-B3, ConvNeXt-Base
and ViT-B/16. Screening configs, native weight preprocessing, run provenance,
checkpoint reconstruction, a PIL-image inference adapter and a WP5 development
evaluation bridge are provided.

Verification: 28/28 CPU automated tests passed. All four backbone builders and
output shapes were exercised without pretrained downloads. Separate-parameter
updates, validation behavior, checkpoint reconstruction, synthetic end-to-end
training/reload/inference, protocol leakage/tampering rejection and the original
data tests passed. A separate synthetic bridge check against actual WP5 PR #1
code passed non-contiguous IDs, scoring/ranking, Top-1/5, H@1/H@5 and per-pair
counts. These are software checks, not DPD experiment results.

The three-fold counts match Issues #2/#3, but exact historical filename order
and original hash serialization still require the accepted WP2/WP3 artifacts.
The new export uses explicitly separate digest namespaces and does not replace
original hashes. DPD training is blocked until ordered filenames are reconciled.

Remaining: real-image decoding and augmentation acceptance; GPU screening and
full independent baseline runs; WP5 integration; actual WP4 GUI/retrieval wiring;
and team/lecturer approval of final official-test candidates. No official-test
evaluation has been performed.

Code is prepared locally on feature/wp1-shared-data-contract as a reviewable patch
against main 38afc2c. There is no published code PR yet, and no claim of WP1 full
completion. Existing WP2/WP3 experiments and records were not modified.
