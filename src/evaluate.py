# Model evaluation logic
from . import config
from .data import create_dataloaders
from .model import build_model

import torch
from torch import nn

from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from pathlib import Path
import json


def load_model():
    model = build_model(config.NUM_CLASSES, pretrained=True, freeze_backbone=False)
    model.load_state_dict(torch.load(Path(config.MODEL_DIR) / "best_model.pth", weights_only=True),)
    model.eval()

    return model


def collect_predictions(model, dataloader, loss_fn):
    running_loss = 0.0
    total_samples = 0

    y_true = []
    y_pred = []

    with torch.no_grad():

        for xb, yb in dataloader:

            logits = model(xb)

            loss = loss_fn(logits, yb)

            running_loss += loss.item() * xb.size(0)
            total_samples += xb.size(0)

            predictions = torch.argmax(logits, dim=1)
            y_true.extend(yb.tolist())
            y_pred.extend(predictions.tolist())

        avg_loss = running_loss / total_samples

    return avg_loss, y_true, y_pred


def compute_metrics(y_true, y_pred):
    accuracy = metrics.accuracy_score(y_true, y_pred)
    precision = metrics.precision_score(y_true, y_pred, average="weighted")
    recall = metrics.recall_score(y_true, y_pred, average="weighted")
    f1 = metrics.f1_score(y_true, y_pred, average="weighted")
    class_report = metrics.classification_report(y_true, y_pred)

    return {
        "Accuracy score": accuracy,
        "Precision score": precision,
        "Recall score": recall,
        "F1 Score": f1,
    }, class_report


def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = metrics.confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    heatmap = sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.5,
        linecolor="gray",
    )
    heatmap.set_ylabel("True labels")
    heatmap.set_xlabel("Predicted labels")
    heatmap.set_title("Model Performance Confusion Matrix")

    plt.savefig(Path(config.FIG_DIR) / "heatmap.png", bbox_inches="tight")
    plt.show()
    plt.close()

    return cm


def plot_per_class_accuracy(cm, class_names):
    correct = np.diag(cm)
    total = cm.sum(axis=1)

    per_class_accuracy = np.divide(
        correct, total, out=np.zeros_like(correct, dtype=float), where=total != 0
    )
    plt.figure(figsize=(14, 6))
    plt.bar(class_names, per_class_accuracy)
    plt.xticks(rotation=90)
    plt.xlabel("Class")
    plt.ylabel("Accuracy")
    plt.title("Per class accuracy")
    plt.tight_layout()
    plt.savefig(Path(config.FIG_DIR) / "per_class_accuracy.png")
    plt.show()
    plt.close()


def show_misclassified_images(model, dataloader, class_names, max_images=16):
    model.eval()
    misclassified = []

    with torch.no_grad():
        for xb, yb in dataloader:
            logits = model(xb)
            predictions = torch.argmax(logits, dim=1)
            incorrect = predictions != yb

            for image, true_label, pred_label, is_wrong in zip(
                xb, yb, predictions, incorrect
            ):
                if is_wrong:
                    misclassified.append((image, true_label, pred_label))

                if len(misclassified) >= max_images:
                    break
            if len(misclassified) >= max_images:
                break
        if len(misclassified) == 0:
            print("No misclassification found.")
            return
        rows = 4
        cols = 4

        plt.figure(figsize=(12, 12))

        mean = torch.tensor(config.IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(config.IMAGENET_STD).view(3, 1, 1)

        for i, (image, true_label, pred_label) in enumerate(misclassified):
            plt.subplot(rows, cols, i + 1)

            image = image * std + mean
            image = image.clamp(0, 1)

            image = image.permute(1, 2, 0)

            image = image.numpy()

            plt.imshow(image)
            plt.title(
                f"True: {class_names[true_label]}\nPredicted: {class_names[pred_label]}",
                fontsize=8,
            )

            plt.axis("off")
        plt.tight_layout()

        plt.savefig(Path(config.FIG_DIR) / "misclassification.png")
        plt.close()


def save_results(metrics, classification_report, output_dir):
    output_dir = Path(output_dir)

    with open(output_dir / "metrics.json", "w") as file:
        json.dump(metrics, file, indent=4)

    with open(output_dir / "classification_report.txt", "w") as file:
        file.write(classification_report)
    print(f"Results saved to {output_dir}")


def main():
    _, valid_loader, class_names = create_dataloaders()
    model = load_model()

    criterion = nn.CrossEntropyLoss()

    loss, y_true, y_pred = collect_predictions(
        model=model, dataloader=valid_loader, loss_fn=criterion
    )

    metrics, report = compute_metrics(y_true, y_pred)
    metrics["Loss"] = loss

    print("\nEvaluation Results")
    print("=" * 20)

    for key, value in metrics.items():
        print(f"{key}: {value}")

    print("\nClassification report")
    print("=" * 25)
    print(report)

    cm = plot_confusion_matrix(y_true, y_pred, class_names)
    plot_per_class_accuracy(cm=cm, class_names=class_names)
    show_misclassified_images(
        model=model, dataloader=valid_loader, class_names=class_names
    )

    save_results(
        metrics=metrics, classification_report=report, output_dir=Path(config.FIG_DIR)
    )


if __name__ == "__main__":
    main()
