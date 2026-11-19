from __future__ import annotations

import argparse
import os

from mlflow import MlflowClient

from taxi_demand_mlops.registry import (
    MODEL_NAME,
    promote_candidate,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Promote a registered candidate if it passes validation guardrails"
    )
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--minimum-mae-improvement", type=float, default=0.01)
    parser.add_argument("--maximum-wape-regression", type=float, default=0.02)
    args = parser.parse_args()

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    client = MlflowClient(tracking_uri=tracking_uri)
    result = promote_candidate(
        client,
        model_name=args.model_name,
        minimum_mae_improvement=args.minimum_mae_improvement,
        maximum_wape_regression=args.maximum_wape_regression,
    )
    print(
        f"Promotion {result.status}: candidate v{result.candidate_version}, "
        f"champion v{result.champion_version}. {result.reason}"
    )


if __name__ == "__main__":
    main()
