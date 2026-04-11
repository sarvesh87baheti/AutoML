# Updated version with tf.data + augmentation (without changing I/O behavior)

import os
import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from collections import Counter

# ==============================
# CONFIG
# ==============================
IMG_SIZE = (224, 224)
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
RANDOM_SEED = 42
BATCH_SIZE = 32

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"}

# Determine the processed data directory based on the main module location
_MAIN_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = str(_MAIN_DIR / "processed_data")

# ==============================
# LOAD DATA
# ==============================

def extract_zip(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(extract_to)


def discover_images(root_dir):
    data = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        dirnames[:] = [d for d in dirnames if not d.startswith('.') and d != '__MACOSX']

        for f in filenames:
            if f.startswith('.'):
                continue
            if Path(f).suffix.lower() not in SUPPORTED_EXTS:
                continue

            label = Path(dirpath).name
            full_path = os.path.join(dirpath, f)
            data.append((full_path, label))

    return data

# ==============================
# PREPROCESS
# ==============================

def decode_and_resize(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    return img

# 🔥 AUGMENTATION (only for training later)
def augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, 0.9, 1.1)
    return image, label


def preprocess(zip_path):
    print("Loading data...")

    tf.random.set_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    tmp_dir = tempfile.mkdtemp()

    extract_zip(zip_path, tmp_dir)
    collected = discover_images(tmp_dir)

    if not collected:
        shutil.rmtree(tmp_dir)
        raise ValueError("No images found!")

    paths, labels = zip(*collected)
    class_names = sorted(set(labels))

    print("Classes:", class_names)
    print("Total images:", len(paths))

    # distribution
    print("Class distribution:")
    dist = Counter(labels)
    for k, v in dist.items():
        print(f"  {k}: {v}")

    # label encoding
    le = LabelEncoder()
    le.fit(class_names)
    y = le.transform(labels)

    # load images (as tensors now, not numpy)
    X = []
    valid_indices = []

    for i, p in enumerate(paths):
        if i % 100 == 0:
            print(f"Processing {i}/{len(paths)}")

        try:
            img = decode_and_resize(p)
            X.append(img)
            valid_indices.append(i)
        except Exception as e:
            print(f"Skipping image {p}: {e}")

    if len(X) == 0:
        shutil.rmtree(tmp_dir)
        raise ValueError("All images failed to load!")

    X = tf.stack(X)
    y = np.array([y[i] for i in valid_indices], dtype="int32")

    num_classes = len(class_names)
    if num_classes > 2:
        y = tf.keras.utils.to_categorical(y, num_classes=num_classes)

    # split
    test_val_size = VAL_SPLIT + TEST_SPLIT

    strat = np.argmax(y, axis=1) if num_classes > 2 else y

    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X.numpy(), y, test_size=test_val_size,
        stratify=strat, random_state=RANDOM_SEED
    )

    relative_test = TEST_SPLIT / test_val_size

    strat_tmp = np.argmax(y_tmp, axis=1) if num_classes > 2 else y_tmp

    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=relative_test,
        stratify=strat_tmp, random_state=RANDOM_SEED
    )

    # ==============================
    # 🔥 BUILD TF.DATA (NEW)
    # ==============================

    train_ds = (
        tf.data.Dataset.from_tensor_slices((X_train, y_train))
        .shuffle(1000, seed=RANDOM_SEED)
        .map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    val_ds = (
        tf.data.Dataset.from_tensor_slices((X_val, y_val))
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    test_ds = (
        tf.data.Dataset.from_tensor_slices((X_test, y_test))
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    # ==============================
    # SAVE (UNCHANGED)
    # ==============================

    dataset_name = os.path.splitext(os.path.basename(zip_path))[0]
    dataset_folder = os.path.join(PROCESSED_DIR, dataset_name)
    os.makedirs(dataset_folder, exist_ok=True)

    np.save(os.path.join(dataset_folder, "X_train.npy"), X_train)
    np.save(os.path.join(dataset_folder, "X_val.npy"), X_val)
    np.save(os.path.join(dataset_folder, "X_test.npy"), X_test)

    np.save(os.path.join(dataset_folder, "y_train.npy"), y_train)
    np.save(os.path.join(dataset_folder, "y_val.npy"), y_val)
    np.save(os.path.join(dataset_folder, "y_test.npy"), y_test)

    np.save(os.path.join(dataset_folder, "classes.npy"), np.array(class_names))

    print(f"\nSaved processed data to: {dataset_folder}")

    shutil.rmtree(tmp_dir)

    # 🔥 RETURN DATASETS FOR TRAINING
    return train_ds, val_ds, test_ds


# ==============================
# RUN
# ==============================
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python imageclassification_preprocess.py <path_to_dataset.zip>")
        sys.exit(1)

    zip_path = sys.argv[1]
    preprocess(zip_path)
