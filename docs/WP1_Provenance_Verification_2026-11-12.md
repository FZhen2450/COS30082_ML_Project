# WP1 Training Image Order and Checkpoint Provenance Verification

**Date:** 12 November 2026  
**Auditor / Reviewer:** Andy Wei Jian CHU (WP1 Co-owner)  
**Branch:** `feature/wp1-wp4-final333-20261111`  
**Tracking Issues:** [#26](https://github.com/FZhen2450/COS30082_ML_Project/issues/26), [#30](https://github.com/FZhen2450/COS30082_ML_Project/issues/30)

---

## 1. Executive Summary

As co-owner of WP1 (Data + Independent Baseline), I have independently verified the **WP1 training image order** and the **final crop and disease checkpoint provenance** submitted by Fa Zhen in the 11 November full-8,000 release.

All 8,000 training images, candidate IDs, manifest hashes, training recipes, and serialization-independent state-dictionary identities match the frozen project contract. All relevant automated test suites passed without skips or errors.

---

## 2. Training Data & Image Order Audit

| Metric / Check | Specification / Expected | Observed / Verified | Status |
| :--- | :--- | :--- | :---: |
| **Training CSV Source** | `DPD_Small_train.csv` | CRLF, 645,555 bytes | **PASS** |
| **Source File SHA-256** | `9f1b729ab08f120c9e8d3a5681aa1ccbdf1d9210f9aaa6143959363dcb106a7d` | `9f1b729ab08f120c9e8d3a5681aa1ccbdf1d9210f9aaa6143959363dcb106a7d` | **PASS** |
| **Training Records Count** | 8,000 images | 8,000 images | **PASS** |
| **Training Image Ordering** | Exactly matches `DPD_Small_train.csv` and `WP2_Original_Ordered_Filenames.json` union | Row-for-row identical | **PASS** |
| **Training Manifest SHA-256** | `7f94ac00c89b212ec30a646985b24f1a61fca19cf64b6f25757547868e3e4870` | `7f94ac00c89b212ec30a646985b24f1a61fca19cf64b6f25757547868e3e4870` | **PASS** |
| **Represented Compositions** | 323 training classes (10 official unseen withheld) | 323 represented classes | **PASS** |
| **Global Candidate Space** | 333 candidate classes (0..332) | 333 candidate classes | **PASS** |
| **Candidate Hash** | `ba17e94119e8b2b7a4b18603e25405d2f52bf4a93396973f80491800697eb62f` | `ba17e94119e8b2b7a4b18603e25405d2f52bf4a93396973f80491800697eb62f` | **PASS** |
| **Test Image Leakage** | 0 test images in training pool | 0 test images consumed | **PASS** |

---

## 3. Checkpoint Provenance Audit

Both supervised classifiers were trained independently from verified pretrained ImageNet V2 weights (`resnet50-11ad3fa6.pth`) following the frozen recipe in `configs/wp1/resnet50_full8000.json`.

* **Training Recipe & Hyperparameters:**
  * Seed: 42
  * Batch Size: 2
  * Optimizer: AdamW (learning rate = 0.0001, weight decay = 0.0001)
  * Transform: Native ImageNet V2 (224 x 224, RandomResizedCrop + RandomHorizontalFlip)
* **Epoch Budgets:**
  * **Crop:** 2 epochs (fixed terminal checkpoint, no validation-guided reselection)
  * **Disease:** 3 epochs (fixed terminal checkpoint, no validation-guided reselection)

### Checkpoint File & State Dictionary Identities

| Task | Checkpoint Relative Path | Checkpoint File SHA-256 | State-Dict SHA-256 (`cos30082-state-dict-sha256-v1`) | Tensor Count | Status |
| :--- | :--- | :--- | :--- | :---: | :---: |
| **Crop** | `runs/wp1-full8000-20261111/crop/final.pt` | `16c8f8ee1933e9333d49f8e8e33b88dd27c914d0b839920b35d9b3996f4cd6f5` | `19917af4b944c2c03c05547e749867035a06665e6a07008b7b6f2fbe08ae5d04` | 320 | **VERIFIED** |
| **Disease** | `runs/wp1-full8000-20261111/disease/final.pt` | `f4c05a331794e4dab06f52877780a76a1ff35229eb5d157f9597c8c1719429e4` | `f9a82d429cc51c1c4f9c7bfe2c84745b7e945906ca20640bba806b5bde59c8a9` | 320 | **VERIFIED** |

*Both state dictionaries exclude container metadata and pickling variance, verifying raw parameter tensors directly.*

---

## 4. Test Suite Execution Results

Automated regression and contract test suites executed locally on 12 November 2026:

1. `tests/test_wp1_full8000.py` & `tests/test_wp1_state_identity.py`:
   * 7/7 tests passed in 2.450s.
   * Confirms 8,000-record contract, fixed epoch-2 crop saving, 333 candidate ranking, and serialization invariance.
2. `tests/test_wp1_full8000_evaluation.py`:
   * 5/5 tests passed in 0.279s.
   * Confirms evaluation metric integrity, missing-fold guards, and score replay validation.

---

## 5. Verification Sign-off

The WP1 full-8,000 dataset ordering and ResNet50 checkpoint provenance satisfy all WP1 handoff criteria. I formally accept and sign off on this provenance for WP5 four-model integration and reporting.

**Signed:** Andy Wei Jian CHU  
**Role:** Co-owner, WP1 Data + Independent Baseline  
