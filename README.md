# NYC Taxi Demand MLOps

Build a reproducible MLOps system that forecasts hourly NYC taxi demand and can
detect, retrain, and recover from data or model problems.

## Project requirements

- Download NYC TLC trip records and aggregate demand by pickup zone and hour.
- Create recent-hour and calendar features; split data chronologically, not randomly.
- Define reference and later drift periods.
- Version data with DVC. Do not commit raw or processed datasets to Git.
- Track parameters, metrics, Git commit, data fingerprint, and artifacts with MLflow.
- Register models and serve predictions through a tested FastAPI service.
- Monitor service health, input drift, prediction behavior, and model quality when
  delayed labels are available.
- Demonstrate drift alerting, a retraining trigger, and recovery by model rollback.
- Document one end-to-end incident and a complete reproduction procedure.

The model's score is less important than reproducibility, monitoring, automation,
and recovery.

## Development plan

1. Establish the data source, reproducible download, hourly aggregation, and
   chronological splits.
2. Add DVC and an MLflow training workflow with a simple baseline model.
3. Register the model and build the FastAPI prediction service and integration tests.
4. Containerize the service and add CI build and smoke tests.
5. Add drift monitoring, alerts, retraining, and rollback.
6. Document reproduction and incident recovery; prepare a live demo and backup.

## Team and sharing

This is being built solo for now. The project repository is private; add the teacher
as a collaborator once their GitHub account is known. The project requirements call
for a three-person group, so confirm with the instructor that solo work is acceptable.

## Development environment

Python 3.11 is the initial target. Setup and run commands will be added alongside
the first working pipeline so they can be verified end to end.

## Data pipeline

The initial dataset uses official NYC TLC Yellow Taxi trip records for January-April
2024. The split is chronological: January-February for training, March for reference
evaluation, and April as the later drift period. Trip records are aggregated by
pickup zone and pickup hour. Calendar fields and per-zone lags at 1, 24, and 168
hours are created; the first week is excluded from the splits until all lags exist.

TLC publishes the monthly Parquet files at
`https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_YYYY-MM.parquet`.
Pickup timestamps are treated as the naive NYC local times supplied by TLC.

```powershell
uv sync
uv run python scripts/download_data.py
uv run python scripts/build_dataset.py
uv run dvc status
uv run pytest
```

Generated Parquet files go under `data/raw/` and `data/processed/`. They are excluded
from Git; `data/raw.dvc` and `data/processed.dvc` record their DVC fingerprints. The
public download source and build script let a fresh checkout recreate the data.

## Experiment tracking

Start the local MLflow tracking server in one terminal:

```powershell
uv run mlflow server --host 127.0.0.1 --port 5000 `
  --backend-store-uri sqlite:///mlflow.db `
  --artifacts-destination ./mlartifacts
```

In another terminal, train on January-February and evaluate on March:

```powershell
uv run python scripts/train.py
```

The run records model parameters, MAE/RMSE/WAPE, a lag-24 baseline, Git state,
and DVC fingerprints for the raw and processed data. Compare parameter variations
with overrides such as `uv run python scripts/train.py max_leaf_nodes=15`. April is
held out for the later drift and incident workflow.

After comparing runs in the MLflow UI, register the selected run under the candidate
alias:

```powershell
uv run python scripts/register_model.py <RUN_ID>
```

MLflow stores model artifacts with `skops`. The training code explicitly trusts the
single internal tree type required by its locally trained scikit-learn estimator;
never load artifacts from an untrusted source.

## Prediction API

Start the MLflow server first, then run the service in another terminal. It loads
the current model assigned to the `candidate` registry alias once at startup.

```powershell
uv run mlflow server --host 127.0.0.1 --port 5000 `
  --backend-store-uri sqlite:///mlflow.db `
  --artifacts-destination ./mlartifacts
```

```powershell
uv run uvicorn taxi_demand_mlops.api:app --host 127.0.0.1 --port 8000
```

Check readiness and open the generated API documentation:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
Start-Process http://127.0.0.1:8000/docs
```

Submit the feature values for one pickup zone and target hour:

```powershell
$request = @{
  pickup_zone_id = 161
  hour = 14
  day_of_week = 2
  month = 3
  is_weekend = $false
  lag_1 = 8
  lag_24 = 12
  lag_168 = 10
} | ConvertTo-Json -Compress

Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/v1/predict `
  -ContentType "application/json" -Body $request
```

The response contains the estimated hourly demand and the model version used. Invalid
features return `422`; if the model registry cannot supply a model, `/health` and
prediction return `503`. The service also attaches request ID and processing-time
headers. Run the integration tests with `uv run pytest`.

## Container and CI

Build the production image from the repository root:

```powershell
docker build -t nyc-taxi-mlops:local .
```

For Docker Desktop on Windows, run MLflow on the host with a host that Docker can
reach. This exposes MLflow on local network interfaces; use it only on a trusted
network and stop it when finished:

```powershell
uv run mlflow server --host 0.0.0.0 `
  --allowed-hosts host.docker.internal:5000,localhost:5000,127.0.0.1:5000 `
  --port 5000 --backend-store-uri sqlite:///mlflow.db `
  --artifacts-destination ./mlartifacts
```

Run the API image in a second terminal:

```powershell
docker run --rm -p 8000:8000 `
  -e MLFLOW_TRACKING_URI=http://host.docker.internal:5000 `
  nyc-taxi-mlops:local
```

The image runs as a non-root user and excludes data, local MLflow databases,
artifacts, virtual environments, and `.env` files from its build context. GitHub
Actions runs the tests and lint checks, builds the image, then starts it without an
MLflow registry to verify that the API reports degraded readiness rather than
crashing when the model registry is unavailable.