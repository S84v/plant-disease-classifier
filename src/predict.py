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
