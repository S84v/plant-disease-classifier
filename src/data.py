# Create datasets and dataloaders with transforms

from pathlib import Path
from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from config import *


def get_transforms(image_size=IMAGE_SIZE):
    train_transform = v2.Compose(
        [
            v2.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            v2.RandomHorizontalFlip(),
            v2.RandomRotation(10),
            v2.ToTensor(),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    val_transform = v2.Compose(
        [
            v2.Resize(IMAGE_SIZE, IMAGE_SIZE),
            v2.ToTensor(),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    return train_transform, val_transform


def create_datasets(data_dir, image_size=IMAGE_SIZE):
    train_transform, val_transform = get_transforms(IMAGE_SIZE)

    train_dataset = datasets.ImageFolder(
        root=Path(DATA_DIR) / "train", transform=train_transform
    )

    val_dataset = datasets.ImageFolder(
        root=Path(DATA_DIR) / "test", transform=val_transform
    )
