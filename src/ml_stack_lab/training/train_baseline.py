from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression

DATA_PATH = Path("data/raw/sample.csv")
MODEL_PATH = Path("artifacts/baseline_model.joblib")


def main():
    data = pd.read_csv(DATA_PATH)

    x = data[["feature"]]
    y = data["target"]

    model = LinearRegression()
    model.fit(x, y)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print(f"Model saved to: {MODEL_PATH}")
    print(f"Coefficient: {model.coef_[0]}")
    print(f"Intercept: {model.intercept_}")


if __name__ == "__main__":
    main()
