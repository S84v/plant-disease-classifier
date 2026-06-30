# Model evaluation logic
import config
from data import create_dataloaders
from model import build_model

import torch
from torch import nn

from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path


def load_model():
    model = build_model(config.NUM_CLASSES, pretrained=True, freeze_backbone=False)
    model.load_state_dict(torch.load(Path(config.MODEL_DIR) / "best_model.pth"))
    model.eval()

    return model


def collect_predictions(model, dataloader, loss_fn):
    running_loss = 0.0
    y_true = []
    y_pred = []
    total_samples = 0

    with torch.no_grad():
        for xb, yb in dataloader:
            logits = model(xb)
            running_loss = loss_fn(logits, yb)

            y_pred = torch.argmax(logits, dim=1)
            y_true.extend(yb)

            total_samples = xb.size(0)
        avg_loss = running_loss / total_samples

    return avg_loss, y_true, y_pred
