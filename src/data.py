# Create datasets and dataloaders with transforms

from pathlib import Path

import torch
from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data import DataLoader

from . import config


def get_transforms():
    """
    Returns training and validation transforms.
    """
    train_transform = v2.Compose(
        [
            v2.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            v2.RandomHorizontalFlip(),
            v2.RandomRotation(config.RANDOM_ROTATION),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
        ]
    )

    val_transform = v2.Compose(
        [
            v2.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
        ]
    )

    return train_transform, val_transform


def create_datasets():
    """
    Creates ImageFolder datasets.
    """
    train_transform, val_transform = get_transforms()

    train_dataset = datasets.ImageFolder(
        root=Path(config.DATA_DIR) / "train", transform=train_transform
    )

    val_dataset = datasets.ImageFolder(
        root=Path(config.DATA_DIR) / "valid", transform=val_transform
    )

    return train_dataset, val_dataset


def create_dataloaders():
    """
    Creates PyTorch DataLoaders.
    """
    train_dataset, val_dataset = create_datasets()

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_dataloader, val_dataloader, train_dataset.classes


if __name__ == "__main__":
    train_dataloader, val_dataloader, classes = create_dataloaders()

    images, labels = next(iter(train_dataloader))

    print(f"Image shape: {images.shape}")
    print(f"Labels shape: {labels.shape}]")
    print(f"Number of classes: {len(classes)}")
