# FastAPI wrapper for predict.py
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from contextlib import asynccontextmanager
from predict import load_model, get_inference_transform, preprocess_image, predict_image
import config
from io import BytesIO
from PIL import Image, UnidentifiedImageError


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading model...")
    app.state.model = load_model()
    app.state.transform = get_inference_transform()
    print("Model loaded successfully.")
    yield
    print("Shutting down API...")


app = FastAPI(
    title="Plant Disease Classifier API",
    description="REST API for predicting plant disease from leaf images.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def root():
    return {"message": "API running."}


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    allowed_types = {
        "image/jpeg",
        "image/png",
        "image/webp",
    }

    model = request.app.state.model
    transform = request.app.state.transform

    if file.content_type not in allowed_types:
        raise HTTPException(status_code=415, detail="Unsupported image format.")

    try:
        image_bytes = await file.read()

        image = Image.open(BytesIO(image_bytes))
        image.load()
        image = image.convert("RGB")

        image_tensor = preprocess_image(image=image, transform=transform)
        prediction = predict_image(
            model=model,
            image_tensor=image_tensor,
            class_names=config.CLASS_NAMES,
            top_k=5,
        )

    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Invalid image file.")
    return prediction
