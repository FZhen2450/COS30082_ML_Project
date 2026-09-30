"""
DPD-Small TensorFlow Dataset Pipeline for COS30082 Applied Machine Learning
Project: Out-Of-Distribution Plant Disease Identification
Member 1: Data Engineering & Robustness Lead (Andy Wei Jian CHU)

Features:
- Built with TensorFlow 2.x / tf.data API for high-performance input pipelines.
- Handles DPD-Small train (8,000), test seen (2,000), and test unseen (1,448) splits.
- Stratified train/val split (80/20) preserving seen composition distributions.
- Configurable tf.keras image preprocessing and data augmentation layers.
- Supports multi-task dictionary targets {'crop': crop_id, 'disease': disease_id, 'composition': comp_id}
  for seamless Keras model.fit() integration.
"""

import os
from typing import Dict, Optional, Tuple, Union

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Standard Column Names for DPD-Small Metadata
COLUMN_NAMES = [
    "image_name",
    "crop_id",
    "disease_id",
    "composition_id",
    "crop_name",
    "disease_name",
    "composition_name",
]

# Standard ImageNet normalization parameters
IMAGENET_MEAN = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
IMAGENET_STD = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)


def get_augmentation_layer(image_size: int = 224) -> tf.keras.Sequential:
    """Keras Sequential pipeline for training data augmentation."""
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomTranslation(0.05, 0.05),
        tf.keras.layers.RandomZoom(0.05),
    ], name="data_augmentation")


def load_and_preprocess_image(
    file_path: tf.Tensor,
    image_size: int = 224,
    normalize_imagenet: bool = True,
) -> tf.Tensor:
    """
    Reads an image from disk, decodes JPEG, resizes, and normalizes.
    """
    img_bytes = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, [image_size, image_size], method="bilinear")
    
    # Scale to [0, 1]
    img = img / 255.0

    if normalize_imagenet:
        img = (img - IMAGENET_MEAN) / IMAGENET_STD

    return img


def build_class_mappings(
    train_csv: str = "DPD_Small_train.csv",
    unseen_csv: str = "DPD_Small_test_unseen.csv",
) -> Dict[str, Union[Dict, int]]:
    """
    Extracts complete ID-to-Name and Name-to-ID dictionaries for crops, diseases,
    seen compositions, and unseen compositions.
    """
    train_df = pd.read_csv(train_csv, header=None, names=COLUMN_NAMES)
    unseen_df = pd.read_csv(unseen_csv, header=None, names=COLUMN_NAMES)

    crop_id_to_name = dict(zip(train_df["crop_id"].astype(int), train_df["crop_name"]))
    disease_id_to_name = dict(zip(train_df["disease_id"].astype(int), train_df["disease_name"]))
    seen_comp_id_to_name = dict(zip(train_df["composition_id"].astype(int), train_df["composition_name"]))
    unseen_comp_id_to_name = dict(zip(unseen_df["composition_id"].astype(int), unseen_df["composition_name"]))

    return {
        "crop_id_to_name": crop_id_to_name,
        "disease_id_to_name": disease_id_to_name,
        "seen_comp_id_to_name": seen_comp_id_to_name,
        "unseen_comp_id_to_name": unseen_comp_id_to_name,
        "num_crops": len(crop_id_to_name),
        "num_diseases": len(disease_id_to_name),
        "num_seen_compositions": len(seen_comp_id_to_name),
        "num_unseen_compositions": len(unseen_comp_id_to_name),
    }


def create_tf_dataset_from_dataframe(
    df: pd.DataFrame,
    img_dir: str,
    image_size: int = 224,
    batch_size: int = 32,
    is_training: bool = False,
    shuffle_buffer: int = 1024,
    as_dict_targets: bool = True,
    augment: bool = False,
) -> tf.data.Dataset:
    """
    Constructs an optimized tf.data.Dataset from a pandas DataFrame.
    """
    # Construct absolute file paths
    file_paths = [os.path.join(img_dir, str(f)) for f in df["image_name"]]
    crops = df["crop_id"].astype(int).values
    diseases = df["disease_id"].astype(int).values
    compositions = df["composition_id"].astype(int).values

    if as_dict_targets:
        targets_ds = tf.data.Dataset.from_tensor_slices({
            "crop": crops,
            "disease": diseases,
            "composition": compositions,
        })
    else:
        targets_ds = tf.data.Dataset.from_tensor_slices((crops, diseases, compositions))

    paths_ds = tf.data.Dataset.from_tensor_slices(file_paths)
    images_ds = paths_ds.map(
        lambda p: load_and_preprocess_image(p, image_size=image_size),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    dataset = tf.data.Dataset.zip((images_ds, targets_ds))

    if is_training:
        dataset = dataset.shuffle(buffer_size=min(len(df), shuffle_buffer), seed=42)

    dataset = dataset.batch(batch_size)

    if is_training and augment:
        aug_layer = get_augmentation_layer(image_size)
        dataset = dataset.map(
            lambda x, y: (aug_layer(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset


def get_dataloaders(
    data_dir: str = ".",
    img_dir: str = "DPD_Small_dataset",
    train_csv: str = "DPD_Small_train.csv",
    test_seen_csv: str = "DPD_Small_test_seen.csv",
    test_unseen_csv: str = "DPD_Small_test_unseen.csv",
    val_split: float = 0.2,
    batch_size: int = 32,
    image_size: int = 224,
    seed: int = 42,
    as_dict_targets: bool = True,
    augment_train: bool = True,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, Dict]:
    """
    Loads DPD-Small CSVs, performs stratified train/val split, and returns tf.data.Datasets.

    Returns:
        (train_ds, val_ds, test_seen_ds, test_unseen_ds, class_mappings)
    """
    train_csv_path = os.path.join(data_dir, train_csv)
    test_seen_path = os.path.join(data_dir, test_seen_csv)
    test_unseen_path = os.path.join(data_dir, test_unseen_csv)
    full_img_dir = os.path.join(data_dir, img_dir)

    train_df = pd.read_csv(train_csv_path, header=None, names=COLUMN_NAMES)
    test_seen_df = pd.read_csv(test_seen_path, header=None, names=COLUMN_NAMES)
    test_unseen_df = pd.read_csv(test_unseen_path, header=None, names=COLUMN_NAMES)

    # Stratified Split on composition_id
    counts = train_df["composition_id"].value_counts()
    singletons = counts[counts < 2].index

    if len(singletons) == 0:
        train_sub_df, val_sub_df = train_test_split(
            train_df,
            test_size=val_split,
            stratify=train_df["composition_id"],
            random_state=seed,
        )
    else:
        normal_mask = ~train_df["composition_id"].isin(singletons)
        normal_df = train_df[normal_mask]
        singleton_df = train_df[~normal_mask]

        train_norm, val_norm = train_test_split(
            normal_df,
            test_size=val_split,
            stratify=normal_df["composition_id"],
            random_state=seed,
        )
        train_sub_df = pd.concat([train_norm, singleton_df]).sample(frac=1.0, random_state=seed)
        val_sub_df = val_norm.sample(frac=1.0, random_state=seed)

    # Build tf.data Datasets
    train_ds = create_tf_dataset_from_dataframe(
        train_sub_df,
        full_img_dir,
        image_size=image_size,
        batch_size=batch_size,
        is_training=True,
        as_dict_targets=as_dict_targets,
        augment=augment_train,
    )

    val_ds = create_tf_dataset_from_dataframe(
        val_sub_df,
        full_img_dir,
        image_size=image_size,
        batch_size=batch_size,
        is_training=False,
        as_dict_targets=as_dict_targets,
        augment=False,
    )

    test_seen_ds = create_tf_dataset_from_dataframe(
        test_seen_df,
        full_img_dir,
        image_size=image_size,
        batch_size=batch_size,
        is_training=False,
        as_dict_targets=as_dict_targets,
        augment=False,
    )

    test_unseen_ds = create_tf_dataset_from_dataframe(
        test_unseen_df,
        full_img_dir,
        image_size=image_size,
        batch_size=batch_size,
        is_training=False,
        as_dict_targets=as_dict_targets,
        augment=False,
    )

    class_mappings = build_class_mappings(train_csv_path, test_unseen_path)
    return train_ds, val_ds, test_seen_ds, test_unseen_ds, class_mappings


if __name__ == "__main__":
    print("Testing TensorFlow DPD-Small Dataset Pipeline...")
    train_ds, val_ds, test_seen_ds, test_unseen_ds, mappings = get_dataloaders(
        batch_size=16, as_dict_targets=True, augment_train=True
    )

    print(f"Num Crops: {mappings['num_crops']}")
    print(f"Num Diseases: {mappings['num_diseases']}")
    print(f"Num Seen Compositions: {mappings['num_seen_compositions']}")
    print(f"Num Unseen Compositions: {mappings['num_unseen_compositions']}")

    # Inspect one batch
    for images, targets in train_ds.take(1):
        print(f"Batch image shape: {images.shape}")
        print(f"Batch image dtype: {images.dtype}")
        print(f"Batch crop targets shape: {targets['crop'].shape}")
        print(f"Batch disease targets shape: {targets['disease'].shape}")
        print(f"Batch composition targets shape: {targets['composition'].shape}")
        print(f"First 5 crop IDs: {targets['crop'][:5].numpy()}")
        print(f"First 5 disease IDs: {targets['disease'][:5].numpy()}")
        print(f"First 5 composition IDs: {targets['composition'][:5].numpy()}")

    print("TensorFlow Dataset pipeline successfully verified!")
