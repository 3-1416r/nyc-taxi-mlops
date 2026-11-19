from __future__ import annotations

import argparse
import os

from mlflow import MlflowClient

from taxi_demand_mlops.registry import MODEL_NAME, rollback_champion


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Restore the previous champion model version"
    )
    parser.add_argument("--reason", required=True)
    parser.add_argument("--model-name", default=MODEL_NAME)
    args = parser.parse_args()
    if not args.reason.strip():
        parser.error("--reason must not be empty")

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    client = MlflowClient(tracking_uri=tracking_uri)
    current_version, restored_version = rollback_champion(
        client,
        reason=args.reason,
        model_name=args.model_name,
    )
    print(
        f"Rolled champion back from v{current_version} to v{restored_version}."
    )


if __name__ == "__main__":
    main()
