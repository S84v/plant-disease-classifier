# Model evaluation logic
import config
from data import create_dataloaders
from model import build_model

import torch
from torch import nn

from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
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


def compute_metrics(y_true, y_pred):
    accuracy = metrics.accuracy_score(y_true, y_pred)
    precision = metrics.precision_score(y_true, y_pred)
    recall = metrics.recall_score(y_true, y_pred)
    f1 = metrics.f1_score(y_true, y_pred)
    class_report = metrics.classification_report(y_true, y_pred)

    return {
        "Accuracy score": accuracy,
        "Precision score": precision,
        "Recall score": recall,
        "F1 Score": f1,
        "Classification Report": class_report,
    }


def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = metrics.confusion_matrix(y_true, y_pred)
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


def plot_per_class_accuracy(cm, class_names):
    correct = np.diag(cm)
    total = cm.sum(axis=1)

    per_class_accuracy = np.divide(
        correct, total, out=np.zeros_like(correct, dtype=float), where=total != 0
    )

    plt.bar(class_names, per_class_accuracy)
    plt.xticks(rotation=90)
    plt.xlabel("Class")
    plt.ylabel("Accuracy")
    plt.title("Per class accuracy")
    plt.tight_layout()
    plt.savefig(Path(config.FIG_DIR) / "per_class_accuracy.png")
    plt.show()


def show_misclassified_images(model, dataloader, class_names, fig_path, max_images=16):
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
                    misclassified.append((xb, true_label, pred_label))

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

        mean = torch.tensor([config.IMAGENET_MEAN]).view(3, 1, 1)
        std = torch.tensor([config.IMAGENET_STD]).view(3, 1, 1)

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
