# Load a trained model and perform inference on unseen images

from . import config
from .model import build_model

import torch
from torchvision.transforms import v2
from PIL import Image
from pathlib import Path


def load_model():
    model = build_model(
        num_classes=config.NUM_CLASSES, pretrained=True, freeze_backbone=False
    )
    model.load_state_dict(
        state_dict=torch.load(Path(config.MODEL_DIR) / "best_model.pth")
    )

    model.eval()

    return model


def get_inference_transform():
    inference_transform = v2.Compose(
        [
            v2.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
        ]
    )

    return inference_transform


def load_image(image_path):
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        raise ValueError(f"Failed to load image: {image_path}") from e

    return image


def preprocess_image(image, transform):
    image = transform(image)
    image = image.unsqueeze(0)

    return image


def predict_image(model, image_tensor, class_names, top_k=5):

    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = torch.softmax(logits, dim=1)

        top_probs, top_indices = torch.topk(probabilities, k=top_k, dim=1)
        top_probs = top_probs.squeeze(0).tolist()
        top_indices = top_indices.squeeze(0).tolist()

        top_predictions = []

        for index, probability in zip(top_indices, top_probs):
            top_predictions.append(
                {"class_name": class_names[index], "confidence": probability}
            )
        return {
            "predicted_class": top_predictions[0]["class_name"],
            "confidence": top_predictions[0]["confidence"],
            "top_k_predictions": top_predictions,
        }


def main():
    model = load_model()
    transform = get_inference_transform()
    image = load_image(config.TEST_IMAGE)

    image_tensor = preprocess_image(image=image, transform=transform)

    prediction = predict_image(
        model=model, image_tensor=image_tensor, class_names=config.CLASS_NAMES, top_k=5
    )
    print(f"Prediction : {prediction['predicted_class']}")
    print(f"Confidence : {prediction['confidence']:.2%}")

    print("\nTop predictions:")
    for pred in prediction["top_k_predictions"]:
        print(f"{pred['class_name']:<30} {pred['confidence']:.2%}")


if __name__ == "__main__":
    main()
