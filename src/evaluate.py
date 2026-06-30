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


