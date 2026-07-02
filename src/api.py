# FastAPI wrapper for predict.py
from fastapi import FastAPI

app = FastAPI(
    title="Plant Disease Classifier API",
    description="REST API for prediction plant disease from leaf images.",
    version="1.0.0",
)


@app.get("/")
def root():
    return {"message": "API running."}


@app.get("/health")
def health_check():
    return {"status": "healthy"}
