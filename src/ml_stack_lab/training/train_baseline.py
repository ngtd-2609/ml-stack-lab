import os
from pathlib import Path

import joblib
import mlflow
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

DATA_PATH = Path("data/raw/sample.csv")
MODEL_PATH = Path("artifacts/baseline_model.joblib")

TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5000",
)


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment("ml-stack-lab-training")

    data = pd.read_csv(DATA_PATH)

    x = data[["feature"]]
    y = data["target"]

    with mlflow.start_run():
        model = LinearRegression()
        model.fit(x, y)

        predictions = model.predict(x)

        mse = mean_squared_error(y, predictions)
        r2 = r2_score(y, predictions)

        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, MODEL_PATH)

        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_param("n_samples", len(data))

        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2", r2)

        mlflow.log_artifact(MODEL_PATH, artifact_path="model")

        print(f"Model saved to: {MODEL_PATH}")
        print(f"Coefficient: {model.coef_[0]}")
        print(f"Intercept: {model.intercept_}")
        print(f"MSE: {mse}")
        print(f"R2: {r2}")
        print(f"MLflow tracking: {mlflow.get_tracking_uri()}")


if __name__ == "__main__":
    main()
