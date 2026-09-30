# Out-of-Distribution Plant Disease Identification
### COS30082 – Applied Machine Learning (Project AS02)
**Swinburne University of Technology**

---

## 📌 Project Overview
This repository contains the official codebase and data engineering pipeline for **Out-of-Distribution (OOD) and Unseen Plant Disease Identification** using the **DPD-Small** dataset.

In real-world agricultural environments, automated visual diagnosis models frequently encounter novel disease variations, unobserved crop-disease compositions, and severe environmental shifts (lighting, background foliage, viewpoints). The primary objective of this project is to develop deep learning architectures that balance **discriminative ability** on seen classes with **generalization capability** to zero-shot unseen disease categories.

---

## 👥 Team & Workload Allocation (Balanced 20% Contribution)

| Member Name | Core Role | Primary Responsibility | Contribution |
| :--- | :---: | :--- | :--- | :---: |
| **Andy**  | **Data Engineering & Robustness Lead** | Data pipeline, stratified splits, augmentations, field perturbation suite | **20%** |
| **Fa Zhen NG**  | **Multi-Task & Baseline Lead** | Git setup, Baseline 1 (Independent CNNs), Baseline 2 (Multi-Task CNN) | **20%** |
| **Team Member 3** | **Proposed Deep Learning Researcher** | Proposed novel architecture, zero-shot transfer loss, unseen class evaluation | **20%** |
| **Team Member 4**  | **Evaluation & Metrics Lead** | Evaluation harness (Top-1/5, Harmonic Mean $H$), confusion matrix, benchmark tables | **20%** |
| **Team Member 5**  | **UI & Retrieval Specialist** | Interactive GUI (Streamlit/Gradio), image similarity retrieval engine, video presentation | **20%** |

---

## 📊 Dataset Structure (DPD-Small)

The dataset consists of **11,448 real-world images** divided into three splits:

| Split Name | Image Count | Metadata File | Purpose |
| :--- | :---: | :--- | :--- |
| **Train Set** | 8,000 (69.9%) | `DPD_Small_train.csv` | Training seen composition classes (323 compositions across 55 crops and 175 diseases) |
| **Test Seen Set** | 2,000 (17.5%) | `DPD_Small_test_seen.csv` | In-distribution evaluation to measure discriminative accuracy |
| **Test Unseen Set** | 1,448 (12.6%) | `DPD_Small_test_unseen.csv` | Out-of-distribution evaluation on 10 unseen composition categories (Zero-Shot) |
| **Total** | **11,448** | | All images verified on disk with 0 corrupted files |

### Metadata Columns
Each CSV contains 7 comma-separated columns (without a header line):
1. `image_name`: Image filename (e.g. `APPD_apple_healthy_138.jpg`).
2. `crop_id`: Integer label for crop category (0 to 54).
3. `disease_id`: Integer label for disease category (0 to 174).
4. `composition_id`: Integer label for composite crop-disease class (0 to 322 for seen; 10 unseen classes).
5. `crop_name`: Text name of the crop (e.g. `apple`).
6. `disease_name`: Text name of the disease (e.g. `healthy`).
7. `composition_name`: Composite text name (e.g. `apple_healthy`).

---

## 🛠️ Installation & Environment Setup

### 1. Prerequisites
- Python 3.10+ (Python 3.12 recommended)
- Git

### 2. Clone the Repository
```bash
git clone https://github.com/FZhen2450/COS30082_ML_Project.git
cd COS30082_ML_Project
```

### 3. Create & Activate Virtual Environment
```bash
python -m venv .venv

# On Windows (PowerShell):
.\.venv\Scripts\Activate.ps1

# On Linux/macOS:
source .venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install tensorflow pandas scikit-learn matplotlib seaborn pillow python-docx
```

---

## 🚀 How to Run

### 1. Run Data Pipeline (`dataset.py`)
Tests and builds TensorFlow `tf.data.Dataset` pipelines with stratified 80/20 train/validation splits:
```bash
python dataset.py
```

#### Usage in Model Code:
```python
from dataset import get_dataloaders

# Retrieve high-performance tf.data pipelines
train_ds, val_ds, test_seen_ds, test_unseen_ds, mappings = get_dataloaders(
    batch_size=32,
    image_size=224,
    as_dict_targets=True, # Returns targets as {'crop': y_crop, 'disease': y_disease, 'composition': y_comp}
    augment_train=True
)

# Seamless Keras multi-output training
# model.fit(train_ds, validation_data=val_ds, epochs=25)
```

### 2. Run Exploratory Data Analysis (`eda.py`)
Generates distribution plots and statistical verification reports:
```bash
python eda.py
```
Outputs:
- Figures saved to `eda_figures/`:
  - `01_split_overview.png`: Sample distribution across train, seen, and unseen splits.
  - `02_crop_distribution.png`: Frequency ranking of top 20 crop categories.
  - `03_disease_distribution.png`: Frequency ranking of top 20 disease categories.
  - `04_composition_imbalance_histogram.png`: Class imbalance histogram across the 323 seen compositions.
  - `05_unseen_classes_distribution.png`: Distribution of the 10 unseen zero-shot compositions.
  - `06_sample_images_grid.png`: Visual collage showing real-world field lighting and background conditions.
- Text summary saved to `eda_summary.txt`.

---

## 📈 Key EDA Findings
- **High Class Imbalance**: Training samples per composition class range from **1** to **544** (Mean: 24.8, Median: 7.0), indicating severe long-tail distribution.
- **Top Crops**: Tomato (1,164 images), Cassava (1,160 images), Apple (1,016 images), Corn (910 images).
- **Zero-Shot Classes (10 Unseen Compositions)**:
  `tomato_early_blight` (472), `grape_black_rot` (271), `pepper_bell_bacterial_spot` (220), `cotton_bacterial_blight` (125), `mango_anthracnose` (95), `sugarcane_rust` (89), `cherry_aphis_spp` (72), `cucumber_powdery_mildew` (37), `corn_brown_spot` (35), `cucumber_downy_mildew` (32).

---

## 🗓️ Project Roadmap (Weeks 5 – 12)
- **Sprint 1 (Weeks 5 – 7)**: Data Pipeline, Baseline 1 (Independent CNNs), Baseline 2 (Multi-Task CNN), Baseline Benchmarking.
- **Sprint 2 (Weeks 8 – 10)**: Proposed Novel Deep Learning Model (Zero-Shot / Vision-Language), Image Retrieval Engine, Interactive GUI (Streamlit/Gradio).
- **Review Back & Finalization (Weeks 11 – 12)**: Cross-model error analysis, 8-page academic report writing, 8-minute presentation video recording, and final Canvas submission.
