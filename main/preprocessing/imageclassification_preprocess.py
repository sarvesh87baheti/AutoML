# Updated version with tf.data + augmentation (without changing I/O behavior)

import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from PIL import Image

import json
from collections import Counter

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ==============================
# CONFIG
# ==============================
IMG_SIZE = (224, 224)
VAL_SPLIT = 0.15
TEST_SPLIT = 0.4
RANDOM_SEED = 42
BATCH_SIZE = 16
SHUFFLE_BUFFER_SIZE = 512

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


def sanitize_image_if_needed(img_path):
    """
    Re-encode PNGs with malformed ICC profiles so TensorFlow/libpng does not
    spam stderr with iCCP warnings while decoding.
    """
    suffix = Path(img_path).suffix.lower()
    if suffix != ".png":
        return

    with Image.open(img_path) as image:
        sanitized = image.convert("RGBA" if image.mode in {"RGBA", "LA", "P"} else "RGB")
        sanitized.save(img_path, format="PNG")


def validate_and_prepare_image(img_path):
    """
    Make sure an image can be opened before training starts. PNGs are sanitized
    in the temp extraction directory to avoid libpng profile warnings.
    """
    sanitize_image_if_needed(img_path)
    with Image.open(img_path) as image:
        image.verify()

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


def build_dataset(paths, labels, num_classes, training=False):
    path_ds = tf.data.Dataset.from_tensor_slices(paths)
    label_ds = tf.data.Dataset.from_tensor_slices(labels)
    ds = tf.data.Dataset.zip((path_ds, label_ds))

    def _load(path, label):
        image = decode_and_resize(path)
        label = tf.cast(label, tf.int32)
        if num_classes > 2:
            label = tf.one_hot(label, depth=num_classes, dtype=tf.float32)
        else:
            label = tf.cast(label, tf.float32)
        return image, label

    options = tf.data.Options()
    options.experimental_deterministic = not training
    ds = ds.with_options(options)
    ds = ds.map(_load, num_parallel_calls=1)

    if training:
        ds = ds.shuffle(min(len(paths), SHUFFLE_BUFFER_SIZE), seed=RANDOM_SEED)
        ds = ds.map(augment, num_parallel_calls=1)

    return ds.batch(BATCH_SIZE).prefetch(1)


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

    valid_paths = []
    valid_labels = []

    for i, p in enumerate(paths):
        if i % 100 == 0:
            print(f"Processing {i}/{len(paths)}")

        try:
            validate_and_prepare_image(p)
            valid_paths.append(p)
            valid_labels.append(y[i])
        except Exception as e:
            print(f"Skipping image {p}: {e}")

    if len(valid_paths) == 0:
        shutil.rmtree(tmp_dir)
        raise ValueError("All images failed to load!")

    num_classes = len(class_names)
    y = np.array(valid_labels, dtype="int32")

    # split
    test_val_size = VAL_SPLIT + TEST_SPLIT

    strat = y

    train_paths, tmp_paths, y_train, y_tmp = train_test_split(
        np.array(valid_paths, dtype=object), y, test_size=test_val_size,
        stratify=strat, random_state=RANDOM_SEED
    )

    relative_test = TEST_SPLIT / test_val_size

    strat_tmp = y_tmp

    val_paths, test_paths, y_val, y_test = train_test_split(
        tmp_paths, y_tmp, test_size=relative_test,
        stratify=strat_tmp, random_state=RANDOM_SEED
    )

    train_ds = build_dataset(train_paths.tolist(), y_train.tolist(), num_classes, training=True)
    val_ds = build_dataset(val_paths.tolist(), y_val.tolist(), num_classes, training=False)
    test_ds = build_dataset(test_paths.tolist(), y_test.tolist(), num_classes, training=False)

    dataset_name = os.path.splitext(os.path.basename(zip_path))[0]
    dataset_folder = os.path.join(PROCESSED_DIR, dataset_name)
    os.makedirs(dataset_folder, exist_ok=True)

    np.save(os.path.join(dataset_folder, "classes.npy"), np.array(class_names))
    metadata = {
        "problem_type": "image_classification",
        "class_names": class_names,
        "num_classes": num_classes,
        "train_size": int(len(train_paths)),
        "val_size": int(len(val_paths)),
        "test_size": int(len(test_paths)),
        "batch_size": BATCH_SIZE,
        "img_size": list(IMG_SIZE),
    }
    with open(os.path.join(dataset_folder, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved processed data to: {dataset_folder}")

    return train_ds, val_ds, test_ds, tmp_dir


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
