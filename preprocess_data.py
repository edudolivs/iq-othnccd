"""
preprocess_data.py
==================
Preprocesses lung CT images for MedSigLIP fine-tuning:
  1. Reads JPG images from data/{benign, normal, malignant}/
  2. Resizes every image to 448×448 (MedSigLIP native resolution)
  3. Splits stratified into train (70%), val (15%), test (15%)
  4. Saves processed images into processed_data/{train,val,test}/{class}/

Usage:
    python preprocess_data.py
"""

import os
import shutil
from pathlib import Path

from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ──────────────────────── Configuration ────────────────────────
SOURCE_DIR = Path("data")
OUTPUT_DIR = Path("processed_data")
TARGET_SIZE = (448, 448)

CLASS_NAMES = ["benign", "normal", "malignant"]

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42
# ───────────────────────────────────────────────────────────────


def collect_image_paths(source_dir: Path, class_names: list[str]):
    """Collect all image paths and their corresponding labels."""
    paths: list[Path] = []
    labels: list[str] = []

    for class_name in class_names:
        class_dir = source_dir / class_name
        if not class_dir.is_dir():
            print(f"[WARNING] Directory not found: {class_dir}")
            continue

        for img_file in sorted(class_dir.iterdir()):
            if img_file.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"):
                paths.append(img_file)
                labels.append(class_name)

    return paths, labels


def stratified_split(paths, labels, train_ratio, val_ratio, test_ratio, seed):
    """
    Split data into train / val / test preserving class proportions.
    First split: train vs (val+test)
    Second split: val vs test (50/50 of the remainder)
    """
    val_test_ratio = val_ratio + test_ratio  # 0.30

    paths_train, paths_temp, labels_train, labels_temp = train_test_split(
        paths, labels,
        test_size=val_test_ratio,
        stratify=labels,
        random_state=seed,
    )

    # Split the remaining 30% evenly into val (15%) and test (15%)
    relative_test = test_ratio / val_test_ratio  # 0.5

    paths_val, paths_test, labels_val, labels_test = train_test_split(
        paths_temp, labels_temp,
        test_size=relative_test,
        stratify=labels_temp,
        random_state=seed,
    )

    return (
        (paths_train, labels_train),
        (paths_val, labels_val),
        (paths_test, labels_test),
    )


def resize_and_save(src_path: Path, dst_path: Path, target_size: tuple[int, int]):
    """Open an image, resize to target_size, convert to RGB, and save as JPG."""
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src_path) as img:
        img = img.convert("RGB")
        img = img.resize(target_size, Image.LANCZOS)
        img.save(dst_path, "JPEG", quality=95)


def process_split(split_name: str, paths, labels, output_dir: Path, target_size):
    """Resize and save all images for a given split."""
    print(f"\n🔄 Processing {split_name} split ({len(paths)} images)...")
    for src_path, label in tqdm(zip(paths, labels), total=len(paths), desc=f"  {split_name}"):
        dst_path = output_dir / split_name / label / src_path.name
        resize_and_save(src_path, dst_path, target_size)


def main():
    # Clean output directory if it exists
    if OUTPUT_DIR.exists():
        print(f"⚠️  Removing existing output directory: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)

    # 1. Collect all image paths
    print("📂 Collecting images...")
    paths, labels = collect_image_paths(SOURCE_DIR, CLASS_NAMES)
    print(f"   Total images found: {len(paths)}")
    for cls in CLASS_NAMES:
        count = labels.count(cls)
        print(f"   - {cls}: {count}")

    if len(paths) == 0:
        print("❌ No images found. Check your source directory.")
        return

    # 2. Stratified split
    print("\n✂️  Splitting data (70 / 15 / 15)...")
    (train_data, val_data, test_data) = stratified_split(
        paths, labels, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_SEED
    )

    for name, (p, l) in [("train", train_data), ("val", val_data), ("test", test_data)]:
        print(f"   {name}: {len(p)} images", end="  →  ")
        for cls in CLASS_NAMES:
            print(f"{cls}={l.count(cls)}", end="  ")
        print()

    # 3. Resize and save
    for name, (p, l) in [("train", train_data), ("val", val_data), ("test", test_data)]:
        process_split(name, p, l, OUTPUT_DIR, TARGET_SIZE)

    print(f"\n✅ Done! Processed data saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
