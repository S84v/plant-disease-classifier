# Load a trained model and perform inference on unseen images

import config
from model import build_model

import torch
from torchvision.transforms import v2
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt


def load_model():
    model = build_model(
        num_classes=config.NUM_CLASSES, pretrained=True, freeze_backbone=False
    )
    model.load_state_dict(
        state_dict=torch.load(Path(config.MODEL_DIR) / "best_model.pth")
    )

    model.eval()

    return model


def get_inference_transform():
    inference_transform = v2.Compose(
        [
            v2.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
        ]
    )

    return inference_transform


def load_image(image_path):
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        raise ValueError(f"Failed to load image: {image_path}") from e

    return image


def preprocess_image(image, transform):
    image = transform(image)
    image = image.unsqueeze(0)

    return image
