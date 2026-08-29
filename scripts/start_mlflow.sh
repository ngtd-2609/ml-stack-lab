#!/usr/bin/env bash
set -e

export MLFLOW_TRACKING_URI="http://127.0.0.1:5000"

uv run mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --artifacts-destination ./mlartifacts \
  --host 127.0.0.1 \
  --port 5000
