# Create datasets and dataloaders with transforms

from pathlib import Path
from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from config import *

def get_transforms(image_size = 224):
    train_transform = v2.Compose([
        v2.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        v2.RandomHorizontalFlip(),
        v2.RandomRotation(10),
        v2.ToTensor(),
        v2.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    ])