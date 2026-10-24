"""
Exploratory Data Analysis (EDA) Script for DPD-Small Dataset
COS30082 Applied Machine Learning - AS02 Plant Disease Identification
WP1 Data + independent baseline (Andy Wei Jian CHU / Fa Zhen NG)

Generates:
1. Dataset verification and sanity checks (corrupt/missing files).
2. Class cardinality, frequency distributions, and imbalance statistics.
3. Publication-quality visualization figures in 'eda_figures/' directory.
4. Formal statistical summary report in 'eda_summary.txt' for the final report.
"""

import argparse
import json
import os
import random
import pandas as pd
from PIL import Image

from data_utils import audit_images, validate_metadata

def configure_plotting():
    """Load visualization dependencies only for the full EDA command."""
    global plt, np, sns
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    sns.set_theme(style="whitegrid", font="sans-serif")
    plt.rcParams.update({
        "font.size": 11, "axes.labelsize": 12, "axes.titlesize": 13,
        "xtick.labelsize": 10, "ytick.labelsize": 10, "figure.titlesize": 14,
    })

COLUMN_NAMES = [
    "image_name",
    "crop_id",
    "disease_id",
    "composition_id",
    "crop_name",
    "disease_name",
    "composition_name",
]

OUTPUT_DIR = "eda_figures"


def load_datasets():
    train_df = pd.read_csv("DPD_Small_train.csv", header=None, names=COLUMN_NAMES)
    test_seen_df = pd.read_csv("DPD_Small_test_seen.csv", header=None, names=COLUMN_NAMES)
    test_unseen_df = pd.read_csv("DPD_Small_test_unseen.csv", header=None, names=COLUMN_NAMES)
    return train_df, test_seen_df, test_unseen_df


def verify_image_files(train_df, test_seen_df, test_unseen_df, img_dir="DPD_Small_dataset"):
    print("Fully decoding every metadata-listed image...")
    failures, sizes = audit_images([train_df, test_seen_df, test_unseen_df], img_dir)
    print(f"Decoded {len(sizes)} images. Corrupt or missing: {len(failures)}")
    return failures, sizes


def plot_split_overview(train_df, test_seen_df, test_unseen_df):
    splits = ["Train (Seen)", "Test Seen", "Test Unseen (OOD)"]
    counts = [len(train_df), len(test_seen_df), len(test_unseen_df)]
    colors = ["#1B365D", "#2969B0", "#D9534F"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(splits, counts, color=colors, width=0.55, edgecolor="black", linewidth=0.8)
    
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:,}\n({height/sum(counts)*100:.1f}%)",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    ax.set_title("DPD-Small Dataset Split Distribution (Total: 11,448 Images)", pad=15)
    ax.set_ylabel("Number of Images")
    ax.set_ylim(0, max(counts) * 1.18)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "01_split_overview.png"), dpi=300)
    plt.close()


def plot_crop_distribution(train_df):
    crop_counts = train_df["crop_name"].value_counts()
    top_crops = crop_counts.head(20)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=top_crops.values, y=top_crops.index, palette="Blues_r", ax=ax, edgecolor="black", linewidth=0.5)
    ax.set_title("Top 20 Crop Categories by Sample Count (Train Set)", pad=12)
    ax.set_xlabel("Number of Training Images")
    ax.set_ylabel("Crop Category")
    
    for i, v in enumerate(top_crops.values):
        ax.text(v + 3, i, str(v), color="black", va="center", fontweight="semibold")
        
    ax.set_xlim(0, max(top_crops.values) * 1.1)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "02_crop_distribution.png"), dpi=300)
    plt.close()


def plot_disease_distribution(train_df):
    disease_counts = train_df["disease_name"].value_counts()
    top_diseases = disease_counts.head(20)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=top_diseases.values, y=top_diseases.index, palette="Oranges_r", ax=ax, edgecolor="black", linewidth=0.5)
    ax.set_title("Top 20 Disease Categories by Sample Count (Train Set)", pad=12)
    ax.set_xlabel("Number of Training Images")
    ax.set_ylabel("Disease Category")

    for i, v in enumerate(top_diseases.values):
        ax.text(v + 3, i, str(v), color="black", va="center", fontweight="semibold")

    ax.set_xlim(0, max(top_diseases.values) * 1.1)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "03_disease_distribution.png"), dpi=300)
    plt.close()


def plot_composition_imbalance(train_df):
    comp_counts = train_df["composition_id"].value_counts()

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(comp_counts.values, bins=35, kde=True, color="#2969B0", edgecolor="black", ax=ax)
    
    mean_val = np.mean(comp_counts.values)
    median_val = np.median(comp_counts.values)
    
    ax.axvline(mean_val, color="red", linestyle="--", linewidth=1.5, label=f"Mean: {mean_val:.1f}")
    ax.axvline(median_val, color="green", linestyle="-.", linewidth=1.5, label=f"Median: {median_val:.1f}")
    
    ax.set_title("Class Imbalance Histogram: Images per Composition Class (Train)", pad=12)
    ax.set_xlabel("Samples per Composition Category")
    ax.set_ylabel("Frequency (Number of Classes)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "04_composition_imbalance_histogram.png"), dpi=300)
    plt.close()


def plot_unseen_distribution(test_unseen_df):
    unseen_counts = test_unseen_df["composition_name"].value_counts()

    fig, ax = plt.subplots(figsize=(11, 5))
    sns.barplot(x=unseen_counts.values, y=unseen_counts.index, palette="Reds_r", ax=ax, edgecolor="black", linewidth=0.5)
    ax.set_title("Test Unseen (Zero-Shot / OOD) 10 Composition Categories (Total: 1,448)", pad=12)
    ax.set_xlabel("Number of Test Images")
    ax.set_ylabel("Unseen Crop-Disease Composition")

    for i, v in enumerate(unseen_counts.values):
        ax.text(v + 2, i, f"{v} ({v/len(test_unseen_df)*100:.1f}%)", color="black", va="center", fontweight="semibold")

    ax.set_xlim(0, max(unseen_counts.values) * 1.15)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "05_unseen_classes_distribution.png"), dpi=300)
    plt.close()


def plot_sample_grid(train_df, img_dir="DPD_Small_dataset"):
    random.seed(42)
    sample_indices = random.sample(range(len(train_df)), 12)
    
    fig, axes = plt.subplots(3, 4, figsize=(14, 10))
    axes = axes.flatten()

    for idx, sample_idx in enumerate(sample_indices):
        row = train_df.iloc[sample_idx]
        img_path = os.path.join(img_dir, str(row["image_name"]))
        try:
            img = Image.open(img_path)
            axes[idx].imshow(img)
            title = f"{row['crop_name']}\n{row['disease_name']}"
            axes[idx].set_title(title, fontsize=10, pad=4)
        except Exception:
            axes[idx].text(0.5, 0.5, "Image Error", ha="center")
        axes[idx].axis("off")

    plt.suptitle("Sample Visual Variations from DPD-Small Dataset", fontsize=15, y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "06_sample_images_grid.png"), dpi=300)
    plt.close()


def generate_summary_report(train_df, test_seen_df, test_unseen_df, sample_sizes,
                            failures=None, images_checked=False, figures_generated=False):
    comp_train = train_df["composition_id"].value_counts()
    crops_train = train_df["crop_name"].value_counts()
    diseases_train = train_df["disease_name"].value_counts()
    audit = validate_metadata(train_df, test_seen_df, test_unseen_df)
    image_status = (f"{len(sample_sizes)} decoded; {len(failures or [])} failed"
                    if images_checked else "NOT RUN (metadata-only; image integrity unverified)")

    summary_text = f"""======================================================================
COS30082 APPLIED MACHINE LEARNING - DPD-SMALL DATASET EDA REPORT
WP1 Data + independent baseline (Andy Wei Jian CHU / Fa Zhen NG)
======================================================================

1. DATASET SPLIT CARDINALITY
----------------------------------------------------------------------
- Training Images (Seen):        {len(train_df):,}
- Test Seen Images:              {len(test_seen_df):,}
- Test Unseen Images (OOD):      {len(test_unseen_df):,}
- TOTAL METADATA ROWS:           {len(train_df) + len(test_seen_df) + len(test_unseen_df):,}
- Image integrity audit:         {image_status}
- Filename overlap across official splits: none (metadata checked)

2. CLASS CARDINALITY
----------------------------------------------------------------------
- Unique Crop Categories:        {audit['num_crops']}
- Unique Disease Categories:     {audit['num_diseases']}
- Seen Composition Categories:   {audit['num_seen_compositions']}
- Unseen Composition Categories: {audit['num_unseen_compositions']} (disjoint from seen)
- Singleton training compositions: {audit['singleton_compositions']} (kept in training)
- Seen composition IDs are non-contiguous global IDs; use explicit index mappings.

3. CLASS IMBALANCE & STATISTICS (TRAIN SET)
----------------------------------------------------------------------
- Samples per Composition Class:
  * Minimum:                     {comp_train.min()}
  * Maximum:                     {comp_train.max()}
  * Mean:                        {comp_train.mean():.2f}
  * Median:                      {comp_train.median():.2f}
  * Standard Deviation:          {comp_train.std():.2f}

- Top 5 Most Common Crops:
{crops_train.head(5).to_string()}

- Top 5 Least Common Crops:
{crops_train.tail(5).to_string()}

- Top 5 Most Common Diseases:
{diseases_train.head(5).to_string()}

- Top 5 Least Common Diseases:
{diseases_train.tail(5).to_string()}

4. ZERO-SHOT / OUT-OF-DISTRIBUTION (UNSEEN) COMPOSITIONS
----------------------------------------------------------------------
The 10 Unseen Test Compositions:
{test_unseen_df['composition_name'].value_counts().to_string()}

5. RESOLUTION INSPECTION
----------------------------------------------------------------------
- Decoded Image Dimensions:      {sorted(set(sample_sizes)) if images_checked else 'Not checked'}
- Target Input Resolution:       224 x 224 (Standardized via torchvision transforms)
- Color Channels:                RGB (3 channels)

======================================================================
Figures: {'regenerated in ' + OUTPUT_DIR if figures_generated else 'not regenerated; existing figures retained'}.
======================================================================
"""
    with open("eda_summary.txt", "w", encoding="utf-8") as f:
        f.write(summary_text)

    print("\n" + summary_text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-only", action="store_true",
                        help="Regenerate statistics without claiming image verification or new figures")
    args = parser.parse_args()
    print("Loading DPD-Small CSV files...")
    train_df, test_seen_df, test_unseen_df = load_datasets()
    validate_metadata(train_df, test_seen_df, test_unseen_df)
    if args.metadata_only:
        generate_summary_report(train_df, test_seen_df, test_unseen_df, [])
        return 0
    if not os.path.isdir("DPD_Small_dataset"):
        raise FileNotFoundError("Extract DPD_Small_dataset first, or use --metadata-only")
    corrupt, sample_sizes = verify_image_files(train_df, test_seen_df, test_unseen_df)
    with open("eda_image_audit.json", "w", encoding="utf-8") as stream:
        json.dump({"decoded": len(sample_sizes), "failures": corrupt}, stream, indent=2)
    if corrupt:
        generate_summary_report(train_df, test_seen_df, test_unseen_df, sample_sizes,
                                corrupt, images_checked=True)
        print("Image verification failed; see eda_image_audit.json. Figures were not regenerated.")
        return 1
    
    print("Generating EDA figures...")
    configure_plotting()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plot_split_overview(train_df, test_seen_df, test_unseen_df)
    plot_crop_distribution(train_df)
    plot_disease_distribution(train_df)
    plot_composition_imbalance(train_df)
    plot_unseen_distribution(test_unseen_df)
    plot_sample_grid(train_df)

    generate_summary_report(train_df, test_seen_df, test_unseen_df, sample_sizes,
                            corrupt, images_checked=True, figures_generated=True)
    print("EDA pipeline successfully executed!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
