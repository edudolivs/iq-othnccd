"""
dataset_loader.py
=================
Custom Dataset and DataLoader for MedSigLIP fine-tuning on lung CT images.

Features:
  - Online data augmentation (slight translation + slight zoom) for training.
  - No augmentation for validation/test (only resize + normalize).
  - WeightedRandomSampler to handle class imbalance (weights inversely
    proportional to class frequencies).

Usage:
    from dataset_loader import get_dataloaders

    train_loader, val_loader, test_loader, class_names, num_classes = get_dataloaders(
        data_dir="processed_data",
        batch_size=16,
    )
"""

import os
from pathlib import Path
from collections import Counter

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image


# ──────────────────────── Configuration ────────────────────────
IMAGE_SIZE = 448

# ImageNet normalization (commonly used as fallback for vision models)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
# ───────────────────────────────────────────────────────────────


class LungCTDataset(Dataset):
    """
    PyTorch Dataset for lung CT images organized in
    processed_data/{split}/{class_name}/ folders.
    """

    def __init__(self, root_dir: str, transform=None):
        """
        Args:
            root_dir: Path to split folder (e.g. 'processed_data/train').
            transform: torchvision transforms to apply.
        """
        self.root_dir = Path(root_dir)
        self.transform = transform

        # Discover classes from subdirectory names (sorted for determinism)
        self.class_names = sorted(
            [d.name for d in self.root_dir.iterdir() if d.is_dir()]
        )
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}

        # Collect all (image_path, label) pairs
        self.samples: list[tuple[Path, int]] = []
        for class_name in self.class_names:
            class_dir = self.root_dir / class_name
            for img_file in sorted(class_dir.iterdir()):
                if img_file.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    self.samples.append((img_file, self.class_to_idx[class_name]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

    def get_labels(self) -> list[int]:
        """Return list of all labels (used to compute sampler weights)."""
        return [label for _, label in self.samples]


def build_train_transforms() -> transforms.Compose:
    """
    Training transforms with LIGHT online augmentation:
      - Slight translation (up to 5% of image size)
      - Slight zoom (scale 0.90 to 1.10)
      - Random horizontal flip
      - Normalize
    """
    return transforms.Compose([
        # Slight zoom: crop between 90%-100% of the area, then resize back
        transforms.RandomResizedCrop(
            size=IMAGE_SIZE,
            scale=(0.90, 1.0),   # slight zoom (up to 10%)
            ratio=(0.95, 1.05),  # nearly square crops
            interpolation=transforms.InterpolationMode.LANCZOS,
        ),
        # Slight translation: translate up to 5% in x and y
        transforms.RandomAffine(
            degrees=0,                   # no rotation
            translate=(0.05, 0.05),      # max 5% translation
            fill=0,                      # fill with black
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def build_eval_transforms() -> transforms.Compose:
    """
    Evaluation transforms: only resize + normalize (no augmentation).
    """
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def compute_sample_weights(labels: list[int]) -> torch.DoubleTensor:
    """
    Compute per-sample weights inversely proportional to class frequency.

    Example:
        class 0 (benign):    120 samples → weight = N / (num_classes * 120)
        class 1 (malignant): 561 samples → weight = N / (num_classes * 561)
        class 2 (normal):    416 samples → weight = N / (num_classes * 416)

    Each sample gets the weight of its class.
    """
    counter = Counter(labels)
    num_samples = len(labels)
    num_classes = len(counter)

    # Weight per class: inversely proportional to frequency
    class_weights = {
        cls: num_samples / (num_classes * count)
        for cls, count in counter.items()
    }

    # Assign each sample the weight of its class
    sample_weights = [class_weights[label] for label in labels]

    print("  📊 Class weights for WeightedRandomSampler:")
    for cls, weight in sorted(class_weights.items()):
        print(f"     class {cls}: count={counter[cls]}, weight={weight:.4f}")

    return torch.DoubleTensor(sample_weights)


def get_dataloaders(
    data_dir: str = "processed_data",
    batch_size: int = 16,
    num_workers: int = 4,
    pin_memory: bool = True,
):
    """
    Build train, val, and test DataLoaders.

    The train DataLoader uses:
      - Online augmentation (slight translation + zoom)
      - WeightedRandomSampler for class balancing

    The val/test DataLoaders use:
      - No augmentation
      - Sequential loading

    Returns:
        train_loader, val_loader, test_loader, class_names, num_classes
    """
    data_path = Path(data_dir)

    # ── Datasets ──
    print("📦 Building datasets...")
    train_dataset = LungCTDataset(data_path / "train", transform=build_train_transforms())
    val_dataset   = LungCTDataset(data_path / "val",   transform=build_eval_transforms())
    test_dataset  = LungCTDataset(data_path / "test",  transform=build_eval_transforms())

    class_names = train_dataset.class_names
    num_classes = len(class_names)

    print(f"   Classes: {class_names}")
    print(f"   Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

    # ── Weighted Sampler for training ──
    print("\n⚖️  Computing weighted sampler...")
    train_labels = train_dataset.get_labels()
    sample_weights = compute_sample_weights(train_labels)

    train_sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_dataset),
        replacement=True,  # required for oversampling minority classes
    )

    # ── DataLoaders ──
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,  # replaces shuffle=True
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader, class_names, num_classes


# ──────────────────────── Quick Smoke Test ─────────────────────
if __name__ == "__main__":
    train_loader, val_loader, test_loader, class_names, num_classes = get_dataloaders(
        data_dir="processed_data",
        batch_size=8,
        num_workers=0,  # 0 for debugging
    )

    print(f"\n🧪 Smoke test — fetching one training batch...")
    images, labels = next(iter(train_loader))
    print(f"   Batch shape : {images.shape}")    # [B, 3, 448, 448]
    print(f"   Labels      : {labels.tolist()}")
    print(f"   Label names : {[class_names[l] for l in labels.tolist()]}")

    # Check class distribution in a few batches
    print(f"\n📊 Class distribution over 10 batches:")
    counts = Counter()
    for i, (_, batch_labels) in enumerate(train_loader):
        if i >= 10:
            break
        for l in batch_labels.tolist():
            counts[class_names[l]] += 1
    for name, count in sorted(counts.items()):
        print(f"   {name}: {count}")

    print("\n✅ DataLoader smoke test passed!")
