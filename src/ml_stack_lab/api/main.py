from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(
    title="ML Stack Lab API",
    version="0.1.0",
)


class PredictionRequest(BaseModel):
    features: list[float]


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
    prediction = sum(request.features)

    return PredictionResponse(prediction=prediction)
