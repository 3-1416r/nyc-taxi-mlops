from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from mlflow.models import infer_signature
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

FEATURE_COLUMNS = [
    "pickup_zone_id",
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "lag_1",
    "lag_24",
    "lag_168",
]
TARGET_COLUMN = "demand"
TRAIN_PATH = Path("data/processed/train.parquet")
REFERENCE_PATH = Path("data/processed/reference.parquet")
RAW_DVC_PATH = Path("data/raw.dvc")
PROCESSED_DVC_PATH = Path("data/processed.dvc")


def load_params(overrides: list[str]) -> dict[str, int | float | str]:
    with Path("params.yaml").open(encoding="utf-8") as file:
        params = yaml.safe_load(file)
    if not isinstance(params, dict):
        raise TypeError("params.yaml must contain a parameter mapping")

    for override in overrides:
        key, separator, value = override.partition("=")
        if not separator or key not in params:
            raise ValueError(f"Unknown or invalid parameter override: {override!r}")
        parsed_value = yaml.safe_load(value)
        if isinstance(parsed_value, bool) or not isinstance(
            parsed_value, type(params[key])
        ):
            raise TypeError(
                f"Parameter {key!r} must have type {type(params[key]).__name__}"
            )
        params[key] = parsed_value
    return params


def fit(
    params: dict[str, int | float | str],
    train: pd.DataFrame,
    reference: pd.DataFrame,
) -> tuple[HistGradientBoostingRegressor, dict[str, float], pd.DataFrame]:
    """Fit on the earlier period and evaluate on the immediately following period."""
    model = HistGradientBoostingRegressor(**params)
    train_features, train_target = _split_features_target(train)
    reference_features, reference_target = _split_features_target(reference)
    model.fit(train_features, train_target)

    prediction = pd.Series(
        model.predict(reference_features), index=reference_target.index
    )
    baseline_prediction = reference_features["lag_24"]
    metrics = {
        "reference_mae": float(mean_absolute_error(reference_target, prediction)),
        "reference_rmse": float(
            root_mean_squared_error(reference_target, prediction)
        ),
        "reference_wape": _weighted_absolute_percentage_error(
            reference_target, prediction
        ),
        "lag_24_baseline_mae": float(
            mean_absolute_error(reference_target, baseline_prediction)
        ),
    }
    return model, metrics, reference_features


def _split_features_target(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    missing_columns = {*FEATURE_COLUMNS, TARGET_COLUMN}.difference(frame.columns)
    if missing_columns:
        raise ValueError(f"Training data is missing columns: {sorted(missing_columns)}")
    features = frame[FEATURE_COLUMNS]
    target = frame[TARGET_COLUMN]
    if features.isna().any().any() or target.isna().any():
        raise ValueError("Training data contains missing feature or target values")
    if (target < 0).any():
        raise ValueError("Poisson regression requires non-negative demand")
    return features, target


def _weighted_absolute_percentage_error(
    actual: pd.Series,
    prediction: pd.Series,
) -> float:
    denominator = float(actual.sum())
    if denominator <= 0:
        raise ValueError("WAPE is undefined when the evaluation demand sums to zero")
    return float((actual - prediction).abs().sum() / denominator)


def _dvc_output_md5(path: Path) -> str:
    with path.open(encoding="utf-8") as file:
        metadata = yaml.safe_load(file)
    outputs = metadata.get("outs") if isinstance(metadata, dict) else None
    if not outputs or not isinstance(outputs[0], dict) or "md5" not in outputs[0]:
        raise ValueError(f"Missing DVC MD5 fingerprint in {path}")
    return str(outputs[0]["md5"])


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def main() -> None:
    params = load_params(sys.argv[1:])
    train = pd.read_parquet(TRAIN_PATH)
    reference = pd.read_parquet(REFERENCE_PATH)

    mlflow.set_tracking_uri(
        os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    )
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT", "nyc-taxi-demand"))

    with mlflow.start_run() as run:
        model, metrics, reference_features = fit(params, train, reference)
        mlflow.log_params(params)
        mlflow.log_params(
            {"train_rows": len(train), "reference_rows": len(reference)}
        )
        mlflow.log_metrics(metrics)
        mlflow.set_tags(
            {
                "git_commit": _git("rev-parse", "HEAD"),
                "git_dirty": str(bool(_git("status", "--porcelain"))),
                "data_md5": _dvc_output_md5(PROCESSED_DVC_PATH),
                "raw_data_md5": _dvc_output_md5(RAW_DVC_PATH),
                "training_data": str(TRAIN_PATH),
                "reference_data": str(REFERENCE_PATH),
            }
        )
        mlflow.sklearn.log_model(
            model,
            name="model",
            signature=infer_signature(
                reference_features.head(10),
                model.predict(reference_features.head(10)),
            ),
            input_example=reference_features.head(3),
        )

    print(
        f"Reference MAE {metrics['reference_mae']:.3f} | "
        f"RMSE {metrics['reference_rmse']:.3f} | "
        f"WAPE {metrics['reference_wape']:.3f}"
    )
    print(f"Lag-24 baseline MAE {metrics['lag_24_baseline_mae']:.3f}")
    print(f"run_id = {run.info.run_id}")


if __name__ == "__main__":
    main()
