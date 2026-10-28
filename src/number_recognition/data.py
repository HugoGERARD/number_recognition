"""MNIST loading and reproducible train/validation preparation."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import datasets, transforms
from torchvision.datasets import MNIST


def load_mnist(data_dir: str | Path = "data", download: bool = True) -> tuple[MNIST, MNIST]:
    """Load the official MNIST training and test partitions."""
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(
        root=str(data_dir), train=True, download=download, transform=transform
    )
    test_dataset = datasets.MNIST(
        root=str(data_dir), train=False, download=download, transform=transform
    )
    return train_dataset, test_dataset


def split_train_validation(
    train_dataset: Dataset,
    validation_size: int = 5_000,
    seed: int = 42,
) -> tuple[Subset, Subset]:
    """Create a deterministic validation split from the training partition."""
    train_size = len(train_dataset) - validation_size
    if train_size <= 0:
        raise ValueError("validation_size must be smaller than the training dataset")

    return random_split(
        train_dataset,
        [train_size, validation_size],
        generator=torch.Generator().manual_seed(seed),
    )


def create_data_loaders(
    train_subset,
    validation_subset,
    test_dataset,
    batch_size: int = 128,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Wrap train, validation and test datasets in DataLoaders."""
    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    validation_loader = DataLoader(
        validation_subset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    return train_loader, validation_loader, test_loader
