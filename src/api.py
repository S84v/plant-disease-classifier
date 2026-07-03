# FastAPI wrapper for predict.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from contextlib import asynccontextmanager
from predict import load_model
from io import BytesIO
from PIL import Image, UnidentifiedImageError


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading model...")
    app.state.model = load_model()
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
async def predict(file: UploadFile = File(...)):
    allowed_types = {
        "image/jpeg",
        "image/png",
        "image/webp",
    }

    if file.content_type not in allowed_types:
        raise HTTPException(status_code=415, detail="Unsupported image format.")

    try:
        image_bytes = await file.read()

        image = Image.open(BytesIO(image_bytes))
        image.load()
        image = image.convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Invalid image file.")
    return {
        "predicted_class": None,
        "confidence": None,
        "top_predictions": [],
    }
