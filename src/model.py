# Build and return a PyTorch model

from torchvision.models import resnet18, ResNet18_Weights
from torch import nn


def build_model(
    num_classes: int, pretrained: bool = True, freeze_backbone: bool = True
) -> nn.Module:
    """
    Build a ResNet18 model for image classification.

    Args:
        num_classes: Number of output classes.
        pretrained: Load ImageNet pretrained weights.
        freeze_backbone: Freeze all backbone layers. If False,
            only layer4 and the classifier remain trainable.

    Returns:
        Configured PyTorch model.
    """
    weights = ResNet18_Weights.DEFAULT if pretrained else None

    model = resnet18(weights=weights)
    in_features = model.fc.in_features

    model.fc = nn.Linear(in_features, num_classes)  # Change final layer to (512, 38)

    if pretrained and freeze_backbone:

        for param in model.parameters():  # Freeze entire model
            param.requires_grad = False

        for param in model.fc.parameters():  # Unfreeze fc layer
            param.requires_grad = True

    elif pretrained and not freeze_backbone:

        for param in model.parameters():  # Freeze entire model
            param.requires_grad = False

        for param in model.layer4.parameters():  # Unfreeze layer 4
            param.requires_grad = True

        for param in model.fc.parameters():  # Unfreeze fc layer
            param.requires_grad = True

    else:  # Freeze nothing. Train all layers.
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
