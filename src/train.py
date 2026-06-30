# Model training logic

import config
from data import create_dataloaders
from model import build_model
from engine import train_one_epoch, validate_one_epoch

import torch
from torch import nn

from pathlib import Path


def main():
    train_dl, val_dl, classes = create_dataloaders()
    model = build_model(num_classes=len(classes), pretrained=True, freeze_backbone=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        params=filter(lambda p: p.requires_grad, model.parameters()), lr=config.LR
    )

    history = {
        "train_loss": [],
        "valid_loss": [],
        "train_accuracy": [],
        "valid_accuracy": [],
    }

    best_valid_accuracy = 0.0

    for epoch in range(1, config.EPOCHS + 1):
        train_epoch_results = train_one_epoch(
            model=model,
            dataloader=train_dl,
            criterion=criterion,
            optimizer=optimizer,
        )

        val_epoch_results = validate_one_epoch(
            model=model, dataloader=val_dl, criterion=criterion
        )

        history["train_loss"].append(train_epoch_results["loss"])
        history["valid_loss"].append(val_epoch_results["loss"])
        history["train_accuracy"].append(train_epoch_results["accuracy"])
        history["valid_accuracy"].append(val_epoch_results["accuracy"])

        current_val_accuracy = val_epoch_results["accuracy"]

        if current_val_accuracy > best_valid_accuracy:
            best_valid_accuracy = current_val_accuracy
            torch.save(model.state_dict(), Path(config.MODEL_DIR) / "best_model.pth")
            print(
                f"\nNew best model saved. Validation accuracy: {current_val_accuracy:.4f}\n"
            )

        print(
            f"Epoch: {epoch:03d}/{config.EPOCHS:03d} | "
            f"Train loss: {history['train_loss'][-1]:.4f} | "
            f"Valid loss: {history['valid_loss'][-1]:.4f} | "
            f"Train accuracy: {history['train_accuracy'][-1]:.4f} | "
            f"Valid accuracy: {history['valid_accuracy'][-1]:.4f}"
        )

    return history


if __name__ == "__main__":
    main()
