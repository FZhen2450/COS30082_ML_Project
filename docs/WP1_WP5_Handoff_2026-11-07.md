# WP1 named development handoff — 7 November 2026

The urgent filename-level WP1 handoff is complete locally. No training,
checkpoint reselection, official-test access, commit, push or external posting
was performed. The selected private weights remain unchanged.

Delivery: `runs/deliverables/WP1_WP5_Development_Handoff_2026-11-07.zip`

Size: 8,602,009 bytes. Raw SHA-256:

```text
753cba68d9e1b0e6ebb7c4864321c54b93d69ca04c173e1fa187b87222bb5815
```

The ZIP contains full named canonical float32 scores for all 1,238 seen and
1,894 internally unseen development images, Top-5 CSVs, the full 8,000-identity
manifest, selected-pair provenance, original training evidence, matching original
loader sources, exact dependency declarations and the pinned WP5 reporting source.
Weights and images are unnecessary for importing the predictions and are excluded.

Both weight hashes and all four original training source-byte hashes were checked.
Names/truths were recorded with the actual decoded input batches. New inference
reproduced the original results exactly: seen 661/1,238 Top-1 and 960/1,238 Top-5;
unseen 13/1,894 Top-1 and 164/1,894 Top-5. H@1 1.3553%; H@5 15.5783%.

Decoded JSON passed PR #18's ModelOutputAdapter/build_report at commit
7d2852bd679e0365c494f91992e68d930925904b. ZIP CRC, all packaged file hashes,
standalone verification from extraction and nine negative validation cases passed.
Evidence is beside the ZIP as `.verification.json` and `.sha256.txt`.

Portability decision: retain the original strict loader guard and use an isolated
torch 2.7.1+cu118 / torchvision 0.22.1+cu118 environment for model reconstruction.
No compatibility with Katana's 2.5.1 / 0.20.1 loader is claimed. Saved-output
reporting was separately verified in the existing CPU environment without loading
the weights. No shared packages were changed.

This is a single-role WP1 handoff. Jonathan should verify identity equivalence
with the independently received WP3 manifest and agree the common comparison
budget declaration before composing the four-model outer handoff. Final-333
approval and M6 GUI rehearsal remain separate.

## Acceptance and portable verification follow-up

The owner independently accepted this ZIP in Issue #2 comment 6702503751.
Its full identity manifest is content-identical to the WP2/WP3 manifest.
Issue #8 comment 6702596251 confirms the complete real four-model import passes
the unchanged PR #18 implementation. The shared reporting declaration explicitly
does not claim equal historical training/tuning budgets.

The original ZIP receipt contains Windows backslashes in its WP5 source keys.
PR #19's updated verifier accepts either path separator style, resolves entries
within the explicit root and preserves exact byte-hash checks. Future exports
write POSIX-style source keys. The delivered ZIP, packets and original receipt
are unchanged; its previously published SHA-256 remains valid. Use the current
repository verifier externally against the unchanged extracted bundle:

```powershell
python -B scripts/verify_wp1_development_export.py --bundle PATH_TO_ORIGINAL_EXTRACTION --wp5-root PATH_TO_BYTE_IDENTICAL_WP5_SOURCE
```

Separator normalization does not normalize source content or relax file hashes.
On Linux retain the original received raw source bytes for this integrity check.

## Copyable Issue #2 / #8 update

Post only after attaching or privately sharing the ZIP. Insert the actual private
location; no upload has been performed by this task.

```html
<html>
<body>
<h2>WP1 filename-level development handoff ready — 7 November 2026</h2>
<p>The selected independent ResNet50 crop epoch-2 and disease epoch-3 pair
now has complete original-filename development exports: 1,238 seen and
1,894 internally unseen records, each with all 323 canonical float32 scores.
Named rank1–5 CSVs, the reconciled identity manifest, matching original loader
source, selected-checkpoint provenance and checksums are included.</p>
<p>This was frozen-model inference only. No retraining, checkpoint reselection
or official-test evaluation was performed. The original weight bytes and native
preprocessing/scoring were preserved.</p>
<p>The named results exactly reproduce the original release: seen Top-1
661/1,238 and Top-5 960/1,238; internally unseen Top-1 13/1,894 and Top-5
164/1,894. H@1 = 1.3553%; H@5 = 15.5783%.</p>
<p>The decoded packets passed PR #18's WP1 adapter and complete report builder.
ZIP extraction, file integrity, report reproduction and nine malformed/missing/
duplicate-record rejection checks passed. This is real WP1 development evidence,
not a completed four-model comparison.</p>
<p>Loader portability: retain the original torchvision version guard and use
the matching isolated torch 2.7.1+cu118 / torchvision 0.22.1+cu118 environment.
No Katana shared packages were changed or cross-version reconstruction claimed.
Jonathan can report these saved predictions without reconstructing the models.</p>
<p>Private delivery file:
<code>WP1_WP5_Development_Handoff_2026-11-07.zip</code> (8,602,009 bytes).</p>
<p>ZIP SHA-256:</p>
<pre><code>753cba68d9e1b0e6ebb7c4864321c54b93d69ca04c173e1fa187b87222bb5815</code></pre>
<p>Jonathan: please verify the supplied original identity manifest against
the WP3 handoff and coordinate the common comparison-budget declaration before
combining all four roles. This ZIP is a single-role handoff, not the complete
four-model CLI bundle. Shared-333 protocol approval and official evaluation
remain separately pending.</p>
</body>
</html>
```

## Copyable WhatsApp notification

```text
Hi Jonathan/Chia, the WP1 filename-level development handoff is ready: all 1,238 seen + 1,894 internally unseen images, full canonical float32 scores, Top-5 CSVs, matching loader source and provenance. Results exactly match the original release; WP5 import/report and ZIP-extraction checks passed. No retraining or official tests. I'll share WP1_WP5_Development_Handoff_2026-11-07.zip privately. Please verify the identity manifest against the WP3 handoff and coordinate the common comparison-budget declaration before the four-model import.
```
