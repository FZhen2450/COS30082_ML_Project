# COS30082 – Applied Machine Learning (Project AS02)
## Out-of-Distribution Plant Disease Identification
**Swinburne University of Technology**

---

## 📌 Project Overview
This repository contains the dataset pipeline, exploratory data analysis (EDA), and project documentation for **Out-of-Distribution (OOD) and Unseen Plant Disease Identification** using the **DPD-Small** dataset.

The system addresses the challenge of identifying unseen crop-disease categories that are absent during training, achieving a balance between **discriminative capability** on observed classes and **generalization robustness** on zero-shot unseen classes and real-world field conditions.

---

## 👥 Team & Workload Allocation (Balanced 20% Contribution)

| Member Name | Student ID | Role | Key Focus Area | Contribution |
| :--- | :---: | :--- | :--- | :---: |
| **Andy Wei Jian CHU** | **104401283** | **Data Engineering & Robustness Lead (M1)** | Data pipeline, stratified splits, augmentations, field perturbation suite | **20%** |
| **Fa Zhen NG** | **104400141** | **Multi-Task & Baseline Lead (M2)** | Git setup, Baseline 1 (Independent CNNs), Baseline 2 (Multi-Task CNN) | **20%** |
| **Team Member 3** | *[Student ID]* | **Proposed Deep Learning Lead (M3)** | Proposed novel model, zero-shot transfer loss, unseen class evaluation | **20%** |
| **Jonathan Zheng Li CHAI** | *104400675* | **Evaluation & Metrics Lead (M4)** | Evaluation harness (Top-1/5, Harmonic Mean $H$), confusion matrix, benchmark tables | **20%** |
| **Team Member 5** | *[Student ID]* | **UI & Retrieval Specialist (M5)** | Interactive GUI (Streamlit/Gradio), vector similarity image retrieval, presentation video | **20%** |

---

## 📥 1. Dataset Setup Instructions (For All Team Members)

> [!IMPORTANT]
> **Raw image files (`DPD_Small_dataset/`) and the 3.4 GB `.7z` archive are intentionally excluded from this Git repository** using `.gitignore` because GitHub has a strict 100 MB file limit.
> 
> Follow these two quick steps to set up the images locally on your machine:

1. **Download `DPD_Small.7z`**: Download the archive from the link provided on Canvas.
2. **Extract Images to Root Folder**: Extract `DPD_Small_dataset/` directly into this project root folder, so your project structure looks like this:
   ```text
   COS30082_ML_Project/
   ├── DPD_Small_dataset/          <-- [EXTRACT HERE: 11,448 image files]
   │   ├── APPD_apple_healthy_138.jpg
   │   └── ...
   ├── DPD_Small_train.csv         <-- [Already committed in Git: 8,000 rows]
   ├── DPD_Small_test_seen.csv     <-- [Already committed in Git: 2,000 rows]
   ├── DPD_Small_test_unseen.csv   <-- [Already committed in Git: 1,448 rows]
   ├── dataset.py                  <-- [TensorFlow data pipeline]
   ├── eda.py                      <-- [EDA analysis script]
   ├── eda_figures/                <-- [6 generated publication charts]
   ├── eda_summary.txt             <-- [Full dataset statistics]
   └── README.md
   ```

---

## 🚀 2. How to Load Data in Your Models (Members 2, 3, 4, 5)

All data ingestion, stratified train/validation splitting (80% / 20%), resizing to $224 \times 224$, and data augmentations are already handled for you in **`dataset.py`**.

In your model training scripts, simply import `get_dataloaders`:

```python
from dataset import get_dataloaders

# Retrieve high-performance tf.data.Dataset objects
train_ds, val_ds, test_seen_ds, test_unseen_ds, mappings = get_dataloaders(
    batch_size=32,          # Adjust based on your GPU/CPU memory
    image_size=224,         # Standard input resolution
    as_dict_targets=True,   # Returns multi-task targets as a dictionary
    augment_train=True      # Applies RandomFlip, Rotation, Translation, Zoom
)

# -------------------------------------------------------------
# 1. Inspect Class Cardinality
# -------------------------------------------------------------
print(f"Num Crops: {mappings['num_crops']}")                      # 55
print(f"Num Diseases: {mappings['num_diseases']}")                # 175
print(f"Num Seen Compositions: {mappings['num_seen_compositions']}") # 323
print(f"Num Unseen (OOD): {mappings['num_unseen_compositions']}")    # 10

# -------------------------------------------------------------
# 2. Batch Format
# -------------------------------------------------------------
# images: Tensor of shape (batch_size, 224, 224, 3), float32 normalized
# targets['crop']: Tensor of shape (batch_size,), integer labels (0 to 54)
# targets['disease']: Tensor of shape (batch_size,), integer labels (0 to 174)
# targets['composition']: Tensor of shape (batch_size,), integer labels (0 to 322)
for images, targets in train_ds.take(1):
    print("Batch images:", images.shape)
    print("Batch crop IDs:", targets['crop'])
    print("Batch disease IDs:", targets['disease'])
    print("Batch composition IDs:", targets['composition'])

# -------------------------------------------------------------
# 3. Direct Keras Multi-Task Model Training
# -------------------------------------------------------------
# model.fit(
#     train_ds,
#     validation_data=val_ds,
#     epochs=25
# )
```

---

## 📊 3. Exploratory Data Analysis (EDA) Highlights

Run `python eda.py` anytime to reproduce the full statistical analysis and visualizations:

- **Total Images**: **11,448** (8,000 train, 2,000 seen test, 1,448 unseen test) — **0 corrupted / missing files**.
- **Class Cardinality**: 55 crops, 175 diseases, 323 seen compositions, and 10 unseen compositions.
- **Severe Class Imbalance**: Training samples per composition class range from **1** to **544** (Mean: 24.8, Median: 7.0).
- **Most Common Crops**: Tomato (1,164 images), Cassava (1,160), Apple (1,016), Corn (910), Paddy (576).
- **The 10 Unseen Zero-Shot (OOD) Classes**:
  1. `tomato_early_blight` (472 images)
  2. `grape_black_rot` (271 images)
  3. `pepper_bell_bacterial_spot` (220 images)
  4. `cotton_bacterial_blight` (125 images)
  5. `mango_anthracnose` (95 images)
  6. `sugarcane_rust` (89 images)
  7. `cherry_aphis_spp` (72 images)
  8. `cucumber_powdery_mildew` (37 images)
  9. `corn_brown_spot` (35 images)
  10. `cucumber_downy_mildew` (32 images)

All high-resolution figures for the report are ready in the **`eda_figures/`** directory.

---

## 🗓️ 4. Project Roadmap (Weeks 5 – 12)
- **Sprint 1 (Weeks 5 – 7)**: Data Pipeline, Baseline 1 (Independent CNNs), Baseline 2 (Multi-Task CNN), Baseline Benchmarking.
- **Sprint 2 (Weeks 8 – 10)**: Proposed Novel Model (Zero-Shot / Vision-Language), Image Retrieval Engine, Interactive GUI.
- **Review Back & Submission (Weeks 11 – 12)**: Cross-model review, 8-page academic report writing, 8-minute presentation video recording, and final Canvas submission (due Friday, 27/11/2026 at 11:59 pm).
