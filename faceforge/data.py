"""FFHQ image discovery, validation, preprocessing, and dataloader construction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass(frozen=True)
class DatasetReport:
    root: Path
    files_found: int
    valid_images: int
    invalid_images: tuple[str, ...]
    image_size: int


def discover_images(root: Path) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"FFHQ dataset directory does not exist: {root}")
    images = sorted(path for path in root.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)
    if not images:
        raise FileNotFoundError(f"No supported face images found below: {root}")
    return images


def validate_dataset(root: Path, image_size: int, max_checks: int = 128) -> DatasetReport:
    """Check a representative subset before an expensive training session."""
    files = discover_images(root)
    invalid: list[str] = []
    for path in files[:max_checks]:
        try:
            with Image.open(path) as image:
                image.verify()
        except (OSError, UnidentifiedImageError):
            invalid.append(str(path))
    return DatasetReport(
        root=root,
        files_found=len(files),
        valid_images=min(len(files), max_checks) - len(invalid),
        invalid_images=tuple(invalid),
        image_size=image_size,
    )


def build_transform(image_size: int, training: bool = False) -> transforms.Compose:
    """Resize aligned FFHQ faces and normalize them to the generator's Tanh range."""
    operations: list[object] = [
        transforms.Resize(image_size, antialias=True),
        transforms.CenterCrop(image_size),
    ]
    # Mirroring is a safe augmentation for aligned face crops and costs almost nothing.
    if training:
        operations.append(transforms.RandomHorizontalFlip(p=0.5))
    operations.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    return transforms.Compose(operations)


class FaceImageDataset(Dataset):
    def __init__(self, root: Path, image_size: int = 64, training: bool = False) -> None:
        self.files = discover_images(root)
        self.transform = build_transform(image_size, training=training)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int):
        path = self.files[index]
        with Image.open(path) as image:
            return self.transform(image.convert("RGB"))


def build_dataloader(
    root: Path,
    image_size: int,
    batch_size: int,
    workers: int = 2,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
    seed: Optional[int] = None,
) -> DataLoader:
    """Build a Windows-safe, pinned-memory loader for local GPU training."""
    if workers < 0:
        raise ValueError("workers must be zero or a positive integer")
    if prefetch_factor < 1:
        raise ValueError("prefetch_factor must be at least one")

    dataset = FaceImageDataset(root, image_size=image_size, training=True)
    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)

    loader_options: dict[str, object] = {
        "batch_size": batch_size,
        "shuffle": True,
        "num_workers": workers,
        "pin_memory": pin_memory,
        "persistent_workers": workers > 0,
        "drop_last": True,
        "generator": generator,
    }
    if workers > 0:
        loader_options["prefetch_factor"] = prefetch_factor
    return DataLoader(
        dataset,
        **loader_options,
    )
