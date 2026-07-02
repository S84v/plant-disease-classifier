# Load a trained model and perform inference on unseen images

import config
from model import build_model

import torch
from torchvision.transforms import v2
from PIL import Image
import pathlib
import matplotlib.pyplot as plt

