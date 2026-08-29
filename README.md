# ML Stack Lab

End-to-end MLOps lab demonstrating a reproducible machine learning workflow with:

- Python 3.12 + uv
- scikit-learn
- PyTorch + NVIDIA CUDA
- DVC
- MLflow
- FastAPI
- Docker Compose
- Prometheus
- Grafana
- Ruff + Pytest
- pre-commit
- GitHub Actions CI

## Architecture

```text
                     ┌─────────────────────┐
                     │      Raw Data       │
                     │        DVC          │
                     └──────────┬──────────┘
                                │
                                ▼
                     ┌─────────────────────┐
                     │   Model Training    │
                     │   scikit-learn      │
                     └──────┬───────┬──────┘
                            │       │
                    model   │       │ metrics
                            ▼       ▼
                 ┌──────────────┐ ┌──────────────┐
                 │  Artifacts   │ │    MLflow    │
                 │     DVC      │ │    :5000     │
                 └──────┬───────┘ └──────────────┘
                        │
                        ▼
                 ┌──────────────┐
                 │   FastAPI    │
                 │    :8000     │
                 └──────┬───────┘
                        │ /metrics
                        ▼
                 ┌──────────────┐
                 │ Prometheus   │
                 │    :9090     │
                 └──────┬───────┘
                        │
                        ▼
                 ┌──────────────┐
                 │   Grafana    │
                 │    :3000     │
                 └──────────────┘
```

## Project Structure

```text
ml-stack-lab/
├── .github/
│   └── workflows/
│       └── ci.yml
├── artifacts/
│   └── baseline_model.joblib.dvc
├── data/
│   ├── external/
│   ├── interim/
│   ├── processed/
│   └── raw/
│       └── sample.csv.dvc
├── docker/
│   ├── Dockerfile
│   └── Dockerfile.ci
├── monitoring/
│   └── prometheus.yml
├── scripts/
│   └── start_mlflow.sh
├── src/
│   └── ml_stack_lab/
│       ├── api/
│       └── training/
├── tests/
├── .dockerignore
├── .dvcignore
├── .gitignore
├── .pre-commit-config.yaml
├── .python-version
├── compose.yaml
├── pyproject.toml
├── README.md
└── uv.lock
```

## Requirements

Recommended development environment:

- Linux or WSL2
- Python 3.12
- Docker
- Docker Compose
- NVIDIA GPU
- NVIDIA Driver
- NVIDIA Container Toolkit
- uv

Check NVIDIA GPU:

```bash
nvidia-smi
```

## Environment Setup

Clone the repository:

```bash
git clone https://github.com/ngtd-2609/ml-stack-lab.git
cd ml-stack-lab
```

Synchronize the Python environment:

```bash
uv sync
```

Verify Python:

```bash
uv run python --version
```

## GPU Verification

Verify PyTorch CUDA support:

```bash
uv run python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"
```

Verify GPU access from Docker:

```bash
docker run --rm --gpus all nvidia/cuda:13.0.3-base-ubuntu24.04 nvidia-smi
```

## DVC

DVC is used to version datasets and model artifacts.

Tracked files include:

```text
data/raw/sample.csv
artifacts/baseline_model.joblib
```

Check DVC status:

```bash
uv run dvc status
```

Pull DVC data/artifacts:

```bash
uv run dvc pull
```

The development environment currently uses a local DVC remote.

A different machine must configure an accessible DVC remote before using `dvc pull`.

## Model Training

The baseline example uses scikit-learn `LinearRegression`.

Run training:

```bash
uv run python -m ml_stack_lab.training.train_baseline
```

Generated model:

```text
artifacts/baseline_model.joblib
```

Training runs also log experiment information to MLflow.

## MLflow

Start the MLflow tracking server:

```bash
./scripts/start_mlflow.sh
```

Open:

```text
http://127.0.0.1:5000
```

MLflow is used for:

- experiment tracking
- parameters
- metrics
- model artifacts

## FastAPI

The model is exposed through FastAPI.

Available endpoints:

```text
GET  /
GET  /health
POST /predict
GET  /metrics
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Example prediction:

```bash
curl -X POST \
  http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"feature":4}'
```

Expected prediction is approximately:

```json
{
  "prediction": 40.0
}
```

## Docker Compose

Start the stack:

```bash
docker compose up -d
```

Check running containers:

```bash
docker compose ps
```

Stop the stack:

```bash
docker compose down
```

The application container is configured to access the NVIDIA GPU.

## Services

| Service | Address | Purpose |
|---|---|---|
| FastAPI | http://localhost:8000 | Model inference API |
| Prometheus | http://localhost:9090 | Metrics collection |
| Grafana | http://localhost:3000 | Monitoring dashboard |
| MLflow | http://127.0.0.1:5000 | Experiment tracking |

## Prometheus

Prometheus scrapes application metrics from:

```text
http://app:8000/metrics
```

Custom application metrics include:

```text
ml_stack_predict_requests_total
ml_stack_prediction_latency_seconds
```

Check whether the API target is healthy:

```bash
curl -sG http://127.0.0.1:9090/api/v1/query \
  --data-urlencode 'query=up{job="ml-stack-api"}'
```

A healthy target returns a value of:

```text
1
```

## Grafana

Grafana uses Prometheus as its data source.

Dashboard:

```text
ML Stack Monitoring
```

Current dashboard panels:

```text
Total Prediction Requests
Average Inference Latency
```

Grafana data is persisted in the Docker volume:

```text
ml-stack-lab-grafana-data
```

This allows dashboards and Grafana settings to survive container recreation.

## Testing

Run all tests:

```bash
uv run pytest
```

Current tests cover:

- package smoke test
- API root endpoint
- health endpoint
- prediction endpoint
- metrics endpoint

## Code Quality

Run Ruff:

```bash
uv run ruff check src tests
```

Format code:

```bash
uv run ruff format src tests
```

Run all pre-commit hooks:

```bash
uv run pre-commit run --all-files
```

## Continuous Integration

GitHub Actions runs automatically on pushes and pull requests to `main`.

CI pipeline:

```text
Checkout
   ↓
Python 3.12
   ↓
Install uv
   ↓
Install CI dependencies
   ↓
Ruff
   ↓
Pytest
   ↓
Build lightweight Docker CI image
```

CI workflow:

```text
.github/workflows/ci.yml
```

Lightweight CI Docker image:

```text
docker/Dockerfile.ci
```

GPU development/runtime Docker image:

```text
docker/Dockerfile
```

## Final Verification

Useful commands for verifying the complete development environment:

```bash
docker compose ps

uv run pytest

uv run ruff check src tests

uv run dvc status

docker compose exec app python -c \
  "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"

curl -s http://127.0.0.1:8000/health
```

Expected healthy state:

```text
FastAPI       UP
Prometheus    UP
Grafana       UP
CUDA          True
Pytest        Passed
Ruff          Passed
DVC           Up to date
GitHub CI     Success
```

## Workflow

```text
Environment Setup
       ↓
Data Versioning
       ↓
Experiment Tracking
       ↓
Model Training
       ↓
API Serving
       ↓
GPU Docker Runtime
       ↓
Automated Testing
       ↓
Continuous Integration
       ↓
Metrics Collection
       ↓
Monitoring Dashboard
```
