# FastAPI wrapper for predict.py
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from .predict import (
    load_model,
    get_inference_transform,
    preprocess_image,
    predict_image,
)


from . import config
from io import BytesIO
from PIL import Image, UnidentifiedImageError
import logging
from pydantic import BaseModel

from .exceptions import register_exception_handlers


class Prediction(BaseModel):
    class_name: str
    confidence: float


class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    top_k_predictions: list[Prediction]


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading model...")
    app.state.model = load_model()
    app.state.transform = get_inference_transform()
    logger.info("Model loaded successfully.")
    yield
    logger.info("Shutting down API...")


app = FastAPI(
    title="Plant Disease Classifier API",
    description="REST API for predicting plant disease from leaf images.",
    version="1.0.0",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory="static"), name="static")
register_exception_handlers(app)

allowed_types = {
    "image/jpeg",
    "image/png",
    "image/webp",
}


@app.get("/")
def root():
    return {"message": "API running."}


@app.get("/health")
def health_check(request: Request):
    model_loaded = hasattr(request.app.state, "model")
    return {"status": "healthy", "model_loaded": model_loaded}


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: Request, file: UploadFile = File(...)):

    model = request.app.state.model
    transform = request.app.state.transform

    if file.content_type not in allowed_types:
        raise HTTPException(status_code=415, detail="Unsupported image format.")

    image_bytes = await file.read()
    try:
        image = Image.open(BytesIO(image_bytes))
        image.load()
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Invalid image file.")
    image = image.convert("RGB")

    image_tensor = preprocess_image(image=image, transform=transform)
    prediction = predict_image(
        model=model,
        image_tensor=image_tensor,
        class_names=config.CLASS_NAMES,
        top_k=5,
    )
    return prediction
