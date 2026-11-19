# Live demo and recovery runbook

This runbook uses the controlled 3x lag-shift scenario in
[the incident report](incident-report.md). Run commands from the repository root.
Use a trusted local network only: the MLflow command below makes its server
reachable to Docker on the host network.

## Before the demo

1. Start Docker Desktop and confirm the DVC datasets exist:

   ```powershell
   uv run dvc status
   ```

2. Start MLflow in Terminal 1:

   ```powershell
   uv run mlflow server --host 0.0.0.0 `
     --allowed-hosts host.docker.internal:5000,localhost:5000,127.0.0.1:5000 `
     --port 5000 --backend-store-uri sqlite:///mlflow.db `
     --artifacts-destination ./mlartifacts
   ```

3. If this is a fresh registry, train and register the baseline as described in
   the README. Confirm the `nyc-taxi-demand` model has a `@champion` alias:

   ```powershell
   uv run python scripts/initialize_champion.py
   ```

4. Build and start the API in Terminal 2:

   ```powershell
   docker build -t nyc-taxi-mlops:local .
   docker run --rm -p 8000:8000 `
     -e MLFLOW_TRACKING_URI=http://host.docker.internal:5000 `
     nyc-taxi-mlops:local
   ```

   Verify `http://127.0.0.1:8000/health` reports `"status":"ok"`.

5. Start Prometheus in Terminal 3 and verify its `taxi-api` target is **UP**:

   ```powershell
   docker compose -f monitoring/docker-compose.yml up -d
   Start-Process http://127.0.0.1:9090/targets
   ```

## Run the incident and recovery

1. Start the alert watcher in Terminal 4 before sending traffic:

   ```powershell
   uv run python scripts/retrain_if_drift.py `
     --prometheus-url http://127.0.0.1:9090 --lag-scale 3 --watch
   ```

2. In Terminal 5, replay 600 April rows with demand lags scaled by 3:

   ```powershell
   uv run python scripts/simulate_drift.py `
     --count 600 --rate 30 --lag-scale 3
   ```

   The replay writes matched prediction and delayed-label files under
   `.monitoring-output/`. Wait for Prometheus's 30-second alert hold and for the
   watcher to print its completed retraining and promotion decision.

3. Score the replay's delayed outcomes and inspect the resulting metrics in MLflow:

   ```powershell
   uv run python scripts/evaluate_outcomes.py
   Start-Process http://127.0.0.1:5000
   ```

   The quality gate compares the candidate against the champion on the chronological
   holdout. An alert does not guarantee promotion; a rejected candidate leaves the
   champion unchanged.

4. To demonstrate recovery, roll back the champion and restart the API process so
   it reloads the restored alias:

   ```powershell
   uv run python scripts/rollback_model.py `
     --reason "Demo recovery after a delayed-label quality regression"
   ```

   Stop the API container with Ctrl+C, then rerun the `docker run` command from
   Terminal 2. Verify `/health` reports the restored model version. For a live
   presentation, do this only after recording the promoted version and the
   rollback reason in MLflow.

5. Stop the watcher with Ctrl+C, then stop Prometheus and the MLflow server:

   ```powershell
   docker compose -f monitoring/docker-compose.yml down
   ```

## Backup and cleanup

- If alerting or Docker is unavailable, run `uv run pytest` and show this
  incident report plus the local MLflow experiment and registry history. Do not
  claim that a unit test is a live alert demonstration.
- Preserve `.monitoring-output/` during the demo if you need to rerun delayed-label
  evaluation. The files are generated locally and are not committed.
- Stop the API container, watcher, Prometheus stack, and MLflow server after the
  session. The local SQLite registry and model artifacts remain under `mlflow.db`
  and `mlartifacts/`.
- The project still needs the instructor's approval for solo work or additional
  teammates, and a shared DVC remote must be configured before another machine can
  pull the data rather than recreate it.
