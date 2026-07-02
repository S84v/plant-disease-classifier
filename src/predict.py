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
