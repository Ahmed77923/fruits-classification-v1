"""
Fruit image classification pipeline built on PyTorch.

Includes:
    - FruitsDataset: an ImageFolder-style dataset with corrupted-image handling
    - CNNModel: a small CNN classifier
    - train_one_epoch / evaluate: training and validation loops
    - main(): a CLI entry point that wires everything together with
      checkpointing, logging, and reproducibility controls.

Example
-------
    python fruits_classifier.py --data-dir ./dataset --epochs 20 --batch-size 32
"""

from __future__ import annotations

import argparse
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image, UnidentifiedImageError

logger = logging.getLogger(__name__)

IMAGE_SIZE = (100, 100)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
DEFAULT_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
def set_seed(seed: int = 42) -> None:
    """Seed all relevant RNGs for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class FruitsDataset(Dataset):
    """
    An ImageFolder-style dataset: expects `root_dir/<class_name>/<image files>`.

    Corrupted or unreadable images are detected and discarded during
    `__init__` (not silently replaced with a black image at train time),
    so every sample returned by `__getitem__` is guaranteed to be a real,
    correctly labeled image.
    """

    def __init__(
        self,
        root_dir: str | Path,
        transform: Optional[Callable] = None,
        mode: str = "train",
        image_extensions: Optional[tuple[str, ...]] = None,
    ) -> None:
        super().__init__()

        if mode not in ("train", "val", "test"):
            raise ValueError(f"mode must be 'train', 'val', or 'test', got {mode!r}")
        self.mode = mode

        self.image_extensions = tuple(
            ext.lower() for ext in (image_extensions or DEFAULT_EXTENSIONS)
        )

        self.root_dir = Path(root_dir)
        if not self.root_dir.is_dir():
            raise ValueError(f"Root directory {self.root_dir} does not exist.")

        self.class_names = sorted(
            f.name for f in self.root_dir.iterdir() if f.is_dir()
        )
        if not self.class_names:
            raise ValueError(f"No class directories found in {self.root_dir}.")

        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}

        self.samples: list[dict] = self._build_and_verify_samples()
        if not self.samples:
            raise ValueError("No valid images found in the dataset.")

        self.transform = transform or (
            self.default_train_transform()
            if self.mode == "train"
            else self.default_val_transform()
        )

    # -- construction helpers ------------------------------------------------ #
    def _iter_image_paths(self, class_dir: Path):
        """Yield image paths under class_dir, matching extensions case-insensitively."""
        for path in class_dir.iterdir():
            if path.is_file() and path.suffix.lower() in self.image_extensions:
                yield path

    def _build_and_verify_samples(self) -> list[dict]:
        """
        Scan every class directory, verify each image can actually be opened,
        and drop any file that is missing/corrupted (with a logged warning)
        instead of deferring the failure to training time.
        """
        samples = []
        skipped = 0

        for class_name in self.class_names:
            class_dir = self.root_dir / class_name
            for image_path in self._iter_image_paths(class_dir):
                if self._is_readable(image_path):
                    samples.append(
                        {"path": image_path, "label": self.class_to_idx[class_name]}
                    )
                else:
                    skipped += 1
                    logger.warning("Skipping unreadable image: %s", image_path)

        if skipped:
            logger.info("Skipped %d corrupted/unreadable image(s) during indexing.", skipped)
        return samples

    @staticmethod
    def _is_readable(path: Path) -> bool:
        try:
            with Image.open(path) as img:
                img.verify()  # cheap structural check, doesn't decode pixels
            return True
        except (UnidentifiedImageError, OSError):
            return False

    # -- default transforms --------------------------------------------------- #
    @staticmethod
    def default_train_transform() -> transforms.Compose:
        """Training-time transforms: light augmentation + normalization."""
        return transforms.Compose(
            [
                transforms.RandomRotation(10),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                ),
                transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    @staticmethod
    def default_val_transform() -> transforms.Compose:
        """Validation/test-time transforms: deterministic resize + normalization."""
        return transforms.Compose(
            [
                transforms.Resize(IMAGE_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    # -- Dataset protocol ------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        sample = self.samples[idx]
        image = Image.open(sample["path"]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, sample["label"]


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class CNNModel(nn.Module):
    """
    A compact CNN classifier.

    Uses AdaptiveAvgPool2d before the classifier head, so it accepts any
    input spatial resolution (e.g. 100x100 or 224x224) without needing
    architecture changes.
    """

    def __init__(self, num_classes: int) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


# --------------------------------------------------------------------------- #
# Train / eval loops
# --------------------------------------------------------------------------- #
@dataclass
class EpochMetrics:
    loss: float
    accuracy: float


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> EpochMetrics:
    """Run one training epoch and return average loss and accuracy."""
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    return EpochMetrics(loss=running_loss / total, accuracy=correct / total)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> EpochMetrics:
    """Run inference over a dataloader (no gradient updates) and return metrics."""
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    return EpochMetrics(loss=running_loss / total, accuracy=correct / total)


# --------------------------------------------------------------------------- #
# CLI / orchestration
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a fruit image classifier.")
    parser.add_argument("--data-dir", type=Path, required=True,
                         help="Root directory containing 'train' and 'val' subfolders, "
                              "each with one subfolder per class.")
    parser.add_argument("--output-dir", type=Path, default=Path("./checkpoints"),
                         help="Where to save the best model checkpoint.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=5,
                         help="Epochs to wait for val-loss improvement before early stopping.")
    return parser.parse_args()











def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    train_dataset = FruitsDataset(args.data_dir / "train", mode="train")
    val_dataset = FruitsDataset(args.data_dir / "val", mode="val")
    num_classes = len(train_dataset.class_names)
    logger.info("Found %d classes: %s", num_classes, train_dataset.class_names)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )

    model = CNNModel(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_metrics.loss)

        logger.info(
            "Epoch %d/%d | train_loss=%.4f train_acc=%.4f | val_loss=%.4f val_acc=%.4f",
            epoch, args.epochs,
            train_metrics.loss, train_metrics.accuracy,
            val_metrics.loss, val_metrics.accuracy,
        )

        if val_metrics.loss < best_val_loss:
            best_val_loss = val_metrics.loss
            epochs_without_improvement = 0
            checkpoint_path = args.output_dir / "best_model.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_to_idx": train_dataset.class_to_idx,
                    "epoch": epoch,
                    "val_loss": val_metrics.loss,
                    "val_accuracy": val_metrics.accuracy,
                },
                checkpoint_path,
            )
            logger.info("Saved new best checkpoint to %s", checkpoint_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                logger.info("Early stopping: no val_loss improvement for %d epochs.", args.patience)
                break


if __name__ == "__main__":
    main()