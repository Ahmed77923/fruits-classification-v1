from __future__ import annotations

import argparse
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import logging
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image, UnidentifiedImageError


from fruits.config.config import (
    Config,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    DEFAULT_EXTENSIONS,
)

# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class FruitsDataset(Dataset): 
    """
    A custom PyTorch Dataset for loading fruit image datasets organized
    in a class-per-directory structure.

    The dataset automatically discovers class names from the subdirectories
    inside ``root_dir``, assigns a unique integer label to each class,
    filters unsupported or corrupted images, and returns image-label pairs
    suitable for training, validation, or testing deep learning models.

    Directory structure:
        root_dir/
        ├── Apple/
        │   ├── img1.jpg
        │   ├── img2.jpg
        │   └── ...
        ├── Banana/
        │   ├── img1.jpg
        │   └── ...
        └── Orange/
            ├── img1.jpg
            └── ...

    Args:
        root_dir (str | Path):
            Path to the dataset root directory containing one subdirectory
            per class.

        transform (Callable | None, optional):
            Transformations applied to each image before it is returned.
            If ``None``, default transforms are selected automatically
            based on the dataset mode.

        mode (str):
            Dataset split to use. Must be one of:
            - ``"train"``
            - ``"val"``
            - ``"test"``

        image_size (int, optional):
            Target image size used by the default transforms.

        image_extensions (tuple[str, ...] | None, optional):
            Supported image file extensions.
            If ``None``, the default extensions are used.

    Raises:
        ValueError:
            If ``mode`` is not one of ``"train"``, ``"val"``, or ``"test"``.

        ValueError:
            If ``root_dir`` does not exist or is not a directory.

        ValueError:
            If no class subdirectories are found inside ``root_dir``.

    Notes:
        - Images with unsupported extensions are skipped.
        - Corrupted or unreadable images are ignored during indexing.
        - Labels are assigned automatically based on the alphabetical
          order of the class directory names.
    """
    def __init__(self, root_dir: str | Path, transform: Optional[Callable] = None,mode = None,image_size=IMAGE_SIZE,image_extensions: Optional[tuple[str, ...]] = None) -> None:
        super().__init__()
        self.transform = transform
        self.mode = mode
        if self.mode not in ("train", 'val', "test"):
            raise ValueError(f"mode must be 'train', 'val', or 'test', got {mode!r}")
    
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
        self.samples: list[dict] = []
        
        self.skipped = 0
        
        for class_name in self.class_names:
            class_dir = self.root_dir / class_name
            for image_path in class_dir.iterdir():
                if self.is_readable(image_path)  and image_path.suffix.lower() in self.image_extensions and image_path.is_file():
                    self.samples.append({
                        "path": image_path, 
                         "label": self.class_to_idx[class_name]
                        })
                else:
                    self.skipped += 1
                    logging.warning("Skipping invalid file: %s", image_path)
                    
        logging.info("Skipped %d corrupted/unreadable image(s) during indexing.", self.skipped)
        if self.transform is None:
            if self.mode == "train":
                self.transform = self.default_train_transform() 
            else:
                self.transform = self.default_val_transform()
    @staticmethod
    def is_readable(path: Path) -> bool:
        try:
            with Image.open(path) as img:
                img.verify()  # cheap structural check, doesn't decode pixels
            return True
        except (UnidentifiedImageError, OSError):
            return False
    @staticmethod
    def default_train_transform():
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize( IMAGE_SIZE),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
            ])
    @staticmethod
    def default_val_transform():
        return transforms.Compose([
                transforms.Resize( IMAGE_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
            ])
        
                
    
    def __len__(self)-> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)
    
    
    def __getitem__(self, idx: int )-> tuple[torch.Tensor, int]:
        """Get an image and its label by index."""
        sample = self.samples[idx]
        with Image.open(sample["path"]) as img:
            image = img.convert("RGB")
        if self.transform:  
            image = self.transform(image)
        return image, sample["label"]   
    
    
