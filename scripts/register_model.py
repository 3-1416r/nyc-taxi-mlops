from __future__ import annotations

import argparse
import os

import mlflow
from mlflow import MlflowClient

from taxi_demand_mlops.registry import (
    CANDIDATE_ALIAS,
    MODEL_NAME,
    ensure_champion_initialized,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Register a logged demand model and assign an alias"
    )
    parser.add_argument("run_id")
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--alias", default=CANDIDATE_ALIAS)
    args = parser.parse_args()

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri)
    run = client.get_run(args.run_id)
    model_id = run.data.tags.get("logged_model_id")
    if not model_id:
        raise ValueError(
            f"Run {args.run_id} has no logged_model_id; train it with scripts/train.py"
        )

    registered_model = mlflow.register_model(
        f"models:/{model_id}",
        args.model_name,
    )
    client.set_registered_model_alias(
        args.model_name,
        args.alias,
        str(registered_model.version),
    )
    champion_version = None
    if args.alias == CANDIDATE_ALIAS:
        champion_version = ensure_champion_initialized(client, args.model_name)
    print(
        f"{args.model_name} v{registered_model.version} is now "
        f"@{args.alias} (source run {args.run_id})"
    )
    if champion_version is not None:
        print(f"Serving champion is v{champion_version}")


if __name__ == "__main__":
    main()
