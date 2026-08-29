from pathlib import Path
from time import perf_counter

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel

app = FastAPI(
    title="ML Stack Lab API",
    version="0.1.0",
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_PATH = PROJECT_ROOT / "artifacts" / "baseline_model.joblib"

model = joblib.load(MODEL_PATH)

PREDICTION_REQUESTS = Counter(
    "ml_stack_predict_requests",
    "Total number of prediction requests",
)

PREDICTION_LATENCY = Histogram(
    "ml_stack_prediction_latency_seconds",
    "Prediction processing time in seconds",
)


class PredictionRequest(BaseModel):
    feature: float


class PredictionResponse(BaseModel):
    prediction: float


@app.get("/")
def root():
    return {"message": "ML Stack Lab API"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    return Response(
        content=generate_latest(),
        headers={"Content-Type": CONTENT_TYPE_LATEST},
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    start_time = perf_counter()

    PREDICTION_REQUESTS.inc()

    features = pd.DataFrame(
        [{"feature": request.feature}],
    )

    prediction = model.predict(features)[0]

    PREDICTION_LATENCY.observe(perf_counter() - start_time)

    return PredictionResponse(prediction=float(prediction))
