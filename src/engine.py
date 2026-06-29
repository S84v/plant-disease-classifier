# Train/validation logic

import torch
from torch import nn


def train_one_epoch(model, dataloader, criterion, optimizer):
    model.train()

    running_loss = 0.0
    running_correct = 0
    total_samples = 0

    for xb, yb in dataloader:

        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)

        loss.backward()
        optimizer.step()

        predictions = torch.argmax(logits, dim=1)

        running_loss += loss.item() * xb.size(0)
        running_correct += (predictions == yb).sum().item()
        total_samples += xb.size(0)

    epoch_loss = running_loss / total_samples
    epoch_accuracy = running_correct / total_samples

    return {"loss": epoch_loss, "accuracy": epoch_accuracy}


def validate_one_epoch(model, dataloader, criterion):
    model.eval()

    running_loss = 0.0
    running_correct = 0
    total_samples = 0

    with torch.no_grad():
        for xb, yb in dataloader:
            logits = model(xb)
            loss = criterion(logits, yb)

            predictions = torch.argmax(logits, dim=1)

            running_loss += loss.item() * xb.size(0)
            running_correct += (predictions == yb).sum().item()
            total_samples += xb.size(0)
        epoch_loss = running_loss / total_samples
        epoch_accuracy = running_correct / total_samples

    return {"loss": epoch_loss, "accuracy": epoch_accuracy}


if __name__ == "__main__":
    from torch.utils.data import DataLoader, Subset
    from torch import nn
    from torch.optim import Adam

    from data import create_dataloaders
    from model import build_model

    # Load full datasets
    train_loader, valid_loader, class_names = create_dataloaders()

    # Keep only the first 4 images
    train_subset = Subset(train_loader.dataset, indices=range(4))
    valid_subset = Subset(valid_loader.dataset, indices=range(4))

    # Create tiny dataloaders
    train_loader = DataLoader(
        train_subset,
        batch_size=2,
        shuffle=False
    )

    valid_loader = DataLoader(
        valid_subset,
        batch_size=2,
        shuffle=False
    )

    # Build model
    model = build_model(num_classes=len(class_names))

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-3)

    print("Testing train_one_epoch...")
    train_metrics = train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer,
    )
    print(train_metrics)

    print("\nTesting validate_one_epoch...")
    valid_metrics = validate_one_epoch(
        model,
        valid_loader,
        criterion,
    )
    print(valid_metrics)
