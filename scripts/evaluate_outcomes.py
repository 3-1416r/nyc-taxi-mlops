from __future__ import annotations

import argparse
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import mlflow
import pandas as pd
from mlflow import MlflowClient

from taxi_demand_mlops.outcomes import evaluate_delayed_outcomes
from taxi_demand_mlops.registry import MODEL_NAME

DEFAULT_PREDICTIONS = Path(".monitoring-output/predictions.jsonl")
DEFAULT_LABELS = Path(".monitoring-output/labels.csv")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Join delayed taxi-demand labels to prediction request IDs"
    )
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--model-name", default=MODEL_NAME)
    args = parser.parse_args()

    if not args.predictions.is_file():
        raise FileNotFoundError(f"Prediction log not found: {args.predictions}")
    if not args.labels.is_file():
        raise FileNotFoundError(f"Delayed labels not found: {args.labels}")
    predictions = pd.read_json(args.predictions, lines=True)
    labels = pd.read_csv(args.labels)
    if predictions.empty or labels.empty:
        raise ValueError("Prediction and label files must both contain records")
    if predictions["request_id"].duplicated().any():
        raise ValueError("Prediction request IDs must be unique")
    if labels["request_id"].duplicated().any():
        raise ValueError("Label request IDs must be unique")
    if set(predictions["request_id"]) != set(labels["request_id"]):
        raise ValueError("Prediction and label request IDs do not match")
    if predictions["model_version"].isna().any():
        raise ValueError("Prediction records contain missing model versions")

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    evaluated_at = datetime.now(UTC).isoformat()
    for model_version, version_predictions in predictions.groupby(
        "model_version",
        sort=True,
    ):
        version_labels = labels[
            labels["request_id"].isin(version_predictions["request_id"])
        ]
        metrics = evaluate_delayed_outcomes(version_predictions, version_labels)
        registered_version = client.get_model_version(
            args.model_name,
            str(model_version),
        )
        with mlflow.start_run(run_id=registered_version.run_id):
            mlflow.log_metrics(metrics)
            mlflow.set_tags(
                {
                    "delayed_label_evaluation_at": evaluated_at,
                    "delayed_label_predictions_file": str(args.predictions),
                    "delayed_label_labels_file": str(args.labels),
                }
            )
        print(
            f"Model v{model_version}: "
            f"MAE {metrics['delayed_mae']:.3f}, "
            f"WAPE {metrics['delayed_wape']:.3f}, "
            f"{int(metrics['delayed_labeled_predictions'])} labeled predictions"
        )


if __name__ == "__main__":
    main()
