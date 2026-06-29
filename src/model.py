# Build and return a PyTorch model

from torchvision.models import resnet18, ResNet18_Weights
from torch import nn


def build_model(num_classes, pretrained=True, freeze_backbone=True):

    weights = ResNet18_Weights.DEFAULT if pretrained else None

    model = resnet18(weights=weights)
    in_features = model.fc.in_features

    model.fc = nn.Linear(in_features, num_classes)

    if pretrained and freeze_backbone:

        for param in model.parameters():
            param.requires_grad = False

        for param in model.fc.parameters():
            param.requires_grad = True

    elif pretrained and not freeze_backbone:

        for param in model.parameters():
            param.requires_grad = False

        for param in model.layer4.parameters():
            param.requires_grad = True

        for param in model.fc.parameters():
            param.requires_grad = True

    else:
        pass

    return model


if __name__ == "__main__":
    import torch

    model = build_model(38, pretrained=True)

    print("Model architecture (layer 4 + fc layer):")
    print()
    print(model.layer4)
    print(model.fc)

    exp_tensor = torch.rand((1, 3, 224, 224))
    print()
    print("Dummy tensor shape: ", model(exp_tensor).shape)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total trainable parameters: ", trainable_params)
