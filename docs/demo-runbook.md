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

4. To demonstrate recovery, perform a controlled rollback after promotion and restart
   the API process so it reloads the restored alias:

   ```powershell
   uv run python scripts/rollback_model.py `
     --reason "Controlled demo rollback after promotion; no live impact"
   ```

   Stop the API container with Ctrl+C, then rerun the `docker run` command from
   Terminal 2. Verify `/health` reports the restored model version. For a live
   presentation, record the promoted version and rollback reason in MLflow. The
   current replay labels only requests served by v1; they do not establish a
   regression in v3.

5. Stop the watcher with Ctrl+C, then stop Prometheus and the MLflow server:

   ```powershell
   docker compose -f monitoring/docker-compose.yml down
   ```

## 10-15 minute presentation outline

| Time | Show and explain |
| --- | --- |
| 0-1 min | Forecasting goal, success criteria, and the controlled nature of this incident. |
| 1-3 min | TLC ingestion, chronological splits, DVC fingerprints, and MLflow baseline. |
| 3-5 min | FastAPI health/prediction endpoints, champion version, and Prometheus metrics. |
| 5-7 min | Start the replay; show PSI crossing 0.2 and `TaxiInputDrift` entering firing. |
| 7-10 min | Show the alert fingerprint, retraining run, chronological holdout metrics, and promotion decision in MLflow. |
| 10-12 min | Join the delayed labels by request ID and explain the v1-only cohort results. |
| 12-14 min | Roll back, restart the API, verify the restored version, then summarize limits and next steps. |

## Technical-choice defense

- **Chronological rather than random splitting:** this is a time-series forecast, so
  future hours must not leak into training. The final April holdout measures the
  candidate on later hours than its training data.
- **Zone-level lags at 1, 24, and 168 hours:** these represent recent demand and
  daily/weekly repetition. The rolling API monitor compares only these lag features
  to avoid alerting on expected calendar or zone-mix changes.
- **PSI as a drift signal, not a quality score:** PSI detects a changed input
  distribution; it cannot establish that forecasts are inaccurate. Delayed MAE and
  WAPE provide outcome evidence once labels arrive.
- **Champion/candidate aliases and a quality gate:** the API continues serving the
  champion while a candidate is evaluated. Promotion requires a 1% MAE improvement
  and limits WAPE regression to 2%; the previous champion version is recorded for
  rollback.
- **Alert fingerprinting and delayed-label request IDs:** repeated watcher polls
  must not launch duplicate retraining for one alert, and delayed outcomes must
  match the exact prediction/model version they evaluate.
- **Controlled lag scaling:** multiplying replay lags by 3 makes the incident
  reproducible on demand. It is a demo fault injection, not evidence that natural
  April demand drifted or that the deployed model is production-ready.
- **Current operational limits:** MLflow and its registry are local, and there is
  no shared DVC remote yet. The data can be recreated from the public TLC source;
  another machine cannot pull the tracked data until a remote is provisioned.

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
