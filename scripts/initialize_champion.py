from __future__ import annotations

import os

from mlflow import MlflowClient

from taxi_demand_mlops.registry import MODEL_NAME, ensure_champion_initialized


def main() -> None:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    client = MlflowClient(tracking_uri=tracking_uri)
    version = ensure_champion_initialized(client, MODEL_NAME)
    print(f"{MODEL_NAME} v{version} is assigned to @champion")


if __name__ == "__main__":
    main()
