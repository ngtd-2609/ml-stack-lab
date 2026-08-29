from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(
    title="ML Stack Lab API",
    version="0.1.0",
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_PATH = PROJECT_ROOT / "artifacts" / "baseline_model.joblib"

model = joblib.load(MODEL_PATH)


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


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    features = pd.DataFrame(
        [{"feature": request.feature}],
    )

    prediction = model.predict(features)[0]

    return PredictionResponse(prediction=float(prediction))
