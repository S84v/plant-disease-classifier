# tests/test_api.py

from pathlib import Path


def test_root(client):
    """
    Test the root endpoint.
    """
    response = client.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")

    assert response.json() == {"message": "API running."}


def test_health(client):
    """
    Test the health endpoint.
    """
    response = client.get("/health")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")

    data = response.json()

    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_predict_success(client, healthy_image):
    """
    Test successful prediction on a valid image.
    """
    with open(healthy_image, "rb") as image_file:
        response = client.post(
            "/predict",
            files={
                "file": (
                    Path(healthy_image).name,
                    image_file,
                    "image/jpeg",
                )
            },
        )

    assert response.status_code == 200

    data = response.json()

    assert "predicted_class" in data
    assert "confidence" in data
    assert "top_k_predictions" in data

    assert isinstance(data["predicted_class"], str)
    assert isinstance(data["confidence"], float)

    assert 0.0 <= data["confidence"] <= 1.0

    assert isinstance(data["top_k_predictions"], list)
    assert len(data["top_k_predictions"]) == 5

    for prediction in data["top_k_predictions"]:
        assert "class_name" in prediction
        assert "confidence" in prediction

        assert isinstance(prediction["class_name"], str)
        assert isinstance(prediction["confidence"], float)

        assert 0.0 <= prediction["confidence"] <= 1.0


def test_predict_invalid_file_type(client, text_file):
    """
    Test uploading a non-image file.
    """
    with open(text_file, "rb") as file:
        response = client.post(
            "/predict",
            files={
                "file": (
                    Path(text_file).name,
                    file,
                    "text/plain",
                )
            },
        )

    assert response.status_code == 415
    assert response.json()["detail"] == "Unsupported image format."


def test_predict_corrupted_image(client, fake_image):
    """
    Test uploading a corrupted image.
    """
    with open(fake_image, "rb") as file:
        response = client.post(
            "/predict",
            files={
                "file": (
                    Path(fake_image).name,
                    file,
                    "image/jpeg",
                )
            },
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid image file."


def test_predict_missing_file(client):
    """
    Test request with no uploaded file.
    """
    response = client.post("/predict")

    assert response.status_code == 422


def test_predict_wrong_method(client):
    """
    Test using GET instead of POST.
    """
    response = client.get("/predict")

    assert response.status_code == 405
