import os
import random
import shutil
from pathlib import Path
import math
import tensorflow as tf

# New cell (index 0) - split PetImages_224 into train/val/test and create augmented TF datasets

try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()

SRC_DIR = Path(BASE_DIR) / 'data' / 'processed' / 'PetImages_224'
OUT_DIR = SRC_DIR.parent / (SRC_DIR.name + "_split")
SPLITS = {"train": 0.8, "val": 0.1, "test": 0.1}
SEED = 42
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

random.seed(SEED)

# Create directory structure and copy files (stratified by class)
def make_dirs(path):
    path.mkdir(parents=True, exist_ok=True)

if not SRC_DIR.exists():
    raise FileNotFoundError(f"Source directory not found: {SRC_DIR}")

make_dirs(OUT_DIR)
for split in SPLITS:
    make_dirs(OUT_DIR / split)

classes = [p.name for p in SRC_DIR.iterdir() if p.is_dir()]
if not classes:
    raise RuntimeError("No class subdirectories found in source directory.")

for cls in classes:
    src_class_dir = SRC_DIR / cls
    files = [p for p in src_class_dir.iterdir() if p.suffix.lower() in IMG_EXTS and p.is_file()]
    random.shuffle(files)
    n = len(files)
    n_train = math.floor(n * SPLITS["train"])
    n_val = math.floor(n * SPLITS["val"])
    # remainder goes to test
    train_files = files[:n_train]
    val_files = files[n_train:n_train + n_val]
    test_files = files[n_train + n_val:]
    for split_name, split_files in [("train", train_files), ("val", val_files), ("test", test_files)]:
        dest_dir = OUT_DIR / split_name / cls
        make_dirs(dest_dir)
        for f in split_files:
            dest = dest_dir / f.name
            if not dest.exists():
                shutil.copy2(f, dest)

# Create tf.data datasets from directories
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    OUT_DIR / "train",
    labels="inferred",
    label_mode="categorical",
    batch_size=BATCH_SIZE,
    image_size=IMAGE_SIZE,
    seed=SEED,
    shuffle=True,
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    OUT_DIR / "val",
    labels="inferred",
    label_mode="categorical",
    batch_size=BATCH_SIZE,
    image_size=IMAGE_SIZE,
    seed=SEED,
    shuffle=False,
)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    OUT_DIR / "test",
    labels="inferred",
    label_mode="categorical",
    batch_size=BATCH_SIZE,
    image_size=IMAGE_SIZE,
    seed=SEED,
    shuffle=False,
)

# Data augmentation pipeline
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.08),
    tf.keras.layers.RandomZoom(0.08),
    tf.keras.layers.RandomTranslation(0.05, 0.05),
    tf.keras.layers.RandomContrast(0.08),
])

# Normalization (rescale to [0,1])
normalization_layer = tf.keras.layers.Rescaling(1./255)

# Apply augmentation to training data and normalization to all
def prepare(ds, augment=False):
    ds = ds.map(lambda x, y: (normalization_layer(x), y), num_parallel_calls=AUTOTUNE)
    if augment:
        ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)
    return ds.shuffle(1000).prefetch(AUTOTUNE) if augment else ds.prefetch(AUTOTUNE)

train_ds = prepare(train_ds, augment=True)
val_ds = prepare(val_ds, augment=False)
test_ds = prepare(test_ds, augment=False)

# Summary
def count_images(directory):
    return sum(1 for _ in (directory.rglob("*") if directory.exists() else [] ) if _.is_file() and _.suffix.lower() in IMG_EXTS)

print("Output split directory:", OUT_DIR)
print("Classes:", classes)
print("Counts -> train:", count_images(OUT_DIR / "train"), "val:", count_images(OUT_DIR / "val"), "test:", count_images(OUT_DIR / "test"))