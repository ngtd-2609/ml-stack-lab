from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression

MODEL_PATH = Path("artifacts/baseline_model.joblib")


def pytest_sessionstart(session):
    if MODEL_PATH.exists():
        return

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    x = pd.DataFrame({"feature": [1.0, 2.0, 3.0]})
    y = [10.0, 20.0, 30.0]

    model = LinearRegression()
    model.fit(x, y)

    joblib.dump(model, MODEL_PATH)
