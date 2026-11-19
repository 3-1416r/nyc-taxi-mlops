from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import mlflow
import mlflow.pyfunc
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.metrics import mean_absolute_error

from taxi_demand_mlops.registry import (
    CANDIDATE_ALIAS,
    CHAMPION_ALIAS,
    MODEL_NAME,
    promote_candidate,
)
from taxi_demand_mlops.training import (
    FEATURE_COLUMNS,
    PROCESSED_DVC_PATH,
    RAW_DVC_PATH,
    REFERENCE_PATH,
    TRAIN_PATH,
    _dvc_output_md5,
    _git,
    _weighted_absolute_percentage_error,
    fit,
    load_params,
    prepare_model_features,
)

LAG_FEATURES = ("lag_1", "lag_24", "lag_168")
TIMESTAMP_COLUMN = "pickup_hour"
TARGET_COLUMN = "demand"
DRIFT_PATH = Path("data/processed/drift.parquet")
DEFAULT_VALIDATION_FRACTION = 0.2
MODEL_TRUSTED_TYPES = [
    "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor"
]


@dataclass(frozen=True)
class RetrainingSplit:
    training: pd.DataFrame
    validation: pd.DataFrame
    validation_start: pd.Timestamp


def split_retraining_data(
    train: pd.DataFrame,
    reference: pd.DataFrame,
    drift: pd.DataFrame,
    *,
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
    lag_scale: float = 1.0,
) -> RetrainingSplit:
    if (
        not math.isfinite(validation_fraction)
        or validation_fraction <= 0
        or validation_fraction >= 1
    ):
        raise ValueError("validation_fraction must be between 0 and 1")
    if not math.isfinite(lag_scale) or lag_scale <= 0:
        raise ValueError("lag_scale must be a positive finite number")

    frames = {"train": train, "reference": reference, "drift": drift}
    for name, frame in frames.items():
        if TIMESTAMP_COLUMN not in frame:
            raise ValueError(f"{name} data is missing {TIMESTAMP_COLUMN!r}")
        if TARGET_COLUMN not in frame:
            raise ValueError(f"{name} data is missing {TARGET_COLUMN!r}")
        missing_features = set(FEATURE_COLUMNS).difference(frame.columns)
        if missing_features:
            raise ValueError(
                f"{name} data is missing features: {sorted(missing_features)}"
            )

    scaled_drift = drift.copy()
    if lag_scale != 1:
        scaled_drift.loc[:, list(LAG_FEATURES)] = (
            scaled_drift.loc[:, list(LAG_FEATURES)].astype("float64") * lag_scale
        )
    drift_timestamps = pd.to_datetime(
        scaled_drift[TIMESTAMP_COLUMN],
        errors="raise",
    )
    if drift_timestamps.isna().any():
        raise ValueError("Drift data contains invalid pickup timestamps")
    unique_hours = pd.DatetimeIndex(drift_timestamps.unique()).sort_values()
    validation_hour_count = math.ceil(len(unique_hours) * validation_fraction)
    if validation_hour_count >= len(unique_hours):
        raise ValueError("Drift data has too few unique hours for a holdout split")
    validation_start = unique_hours[-validation_hour_count]
    validation_mask = drift_timestamps >= validation_start
    early_drift = scaled_drift.loc[~validation_mask]
    validation = scaled_drift.loc[validation_mask]
    if early_drift.empty or validation.empty:
        raise ValueError("Chronological retraining split produced an empty partition")

    training = pd.concat([train, reference, early_drift], ignore_index=True)
    training_timestamps = pd.to_datetime(training[TIMESTAMP_COLUMN], errors="raise")
    if training_timestamps.max() >= validation_start:
        raise ValueError("Retraining split leaks validation timestamps into training")
    return RetrainingSplit(
        training=training,
        validation=validation.reset_index(drop=True),
        validation_start=validation_start,
    )


@dataclass(frozen=True)
class RetrainingResult:
    status: str
    run_id: str | None
    candidate_version: str | None
    champion_version: str | None
    reason: str


def _validate_alert_id(alert_id: str | None) -> None:
    if alert_id is not None and (
        len(alert_id) != 64 or any(char not in "0123456789abcdef" for char in alert_id)
    ):
        raise ValueError("alert_id must be a 64-character lowercase SHA-256 digest")


def _previous_alert_run(
    client: MlflowClient,
    experiment_id: str,
    alert_id: str | None,
) -> RetrainingResult | None:
    if alert_id is None:
        return None
    runs = client.search_runs(
        [experiment_id],
        filter_string=f"tags.drift_alert_id = '{alert_id}'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not runs:
        return None
    run = runs[0]
    tags = run.data.tags
    if tags.get("retraining_status") != "completed":
        raise RuntimeError(
            f"Alert {alert_id} already has an incomplete retraining run "
            f"{run.info.run_id}; inspect it in MLflow before retrying"
        )
    return RetrainingResult(
        status="already_processed",
        run_id=run.info.run_id,
        candidate_version=tags.get("candidate_model_version"),
        champion_version=tags.get("resulting_champion_version"),
        reason="this Prometheus alert already has a completed retraining run",
    )


def run_retraining(
    *,
    alert_id: str | None = None,
    lag_scale: float = 1.0,
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
) -> RetrainingResult:
    _validate_alert_id(alert_id)
    if not math.isfinite(lag_scale) or lag_scale <= 0:
        raise ValueError("lag_scale must be a positive finite number")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    experiment_name = os.getenv("MLFLOW_EXPERIMENT", "nyc-taxi-demand")
    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.set_experiment(experiment_name)
    client = MlflowClient(tracking_uri=tracking_uri)
    previous_run = _previous_alert_run(client, experiment.experiment_id, alert_id)
    if previous_run is not None:
        return previous_run

    champion_version = client.get_model_version_by_alias(
        MODEL_NAME,
        CHAMPION_ALIAS,
    )
    champion_model = mlflow.pyfunc.load_model(
        f"models:/{MODEL_NAME}@{CHAMPION_ALIAS}"
    )
    split = split_retraining_data(
        pd.read_parquet(TRAIN_PATH),
        pd.read_parquet(REFERENCE_PATH),
        pd.read_parquet(DRIFT_PATH),
        validation_fraction=validation_fraction,
        lag_scale=lag_scale,
    )
    params = load_params([])
    candidate_model, validation_metrics, _ = fit(
        params,
        split.training,
        split.validation,
    )
    validation_features = prepare_model_features(
        split.validation[FEATURE_COLUMNS]
    )
    validation_target = split.validation[TARGET_COLUMN]
    champion_prediction = champion_model.predict(validation_features)
    if len(champion_prediction) != len(validation_target):
        raise ValueError("Champion returned an invalid number of validation forecasts")
    champion_mae = float(
        mean_absolute_error(validation_target, champion_prediction)
    )
    champion_wape = _weighted_absolute_percentage_error(
        validation_target,
        pd.Series(champion_prediction, index=validation_target.index),
    )
    comparison_metrics = {
        "candidate_validation_mae": validation_metrics["reference_mae"],
        "candidate_validation_rmse": validation_metrics["reference_rmse"],
        "candidate_validation_wape": validation_metrics["reference_wape"],
        "candidate_lag_24_baseline_mae": validation_metrics["lag_24_baseline_mae"],
        "champion_validation_mae": champion_mae,
        "champion_validation_wape": champion_wape,
    }

    with mlflow.start_run() as run:
        mlflow.log_params(params)
        mlflow.log_params(
            {
                "retraining_lag_scale": lag_scale,
                "validation_fraction": validation_fraction,
                "training_rows": len(split.training),
                "validation_rows": len(split.validation),
            }
        )
        mlflow.log_metrics(comparison_metrics)
        tags = {
            "git_commit": _git("rev-parse", "HEAD"),
            "git_dirty": str(bool(_git("status", "--porcelain"))),
            "data_md5": _dvc_output_md5(PROCESSED_DVC_PATH),
            "raw_data_md5": _dvc_output_md5(RAW_DVC_PATH),
            "training_period": "2024-01 through early 2024-04",
            "validation_period_start": split.validation_start.isoformat(),
            "champion_version_before_retraining": str(champion_version.version),
            "retraining_status": "started",
            "promotion_status": "pending",
        }
        if alert_id is not None:
            tags["drift_alert_id"] = alert_id
        mlflow.set_tags(tags)
        validation_predictions = candidate_model.predict(validation_features)
        model_info = mlflow.sklearn.log_model(
            candidate_model,
            name="model",
            skops_trusted_types=MODEL_TRUSTED_TYPES,
            signature=infer_signature(validation_features, validation_predictions),
            input_example=validation_features.head(3),
        )
        mlflow.set_tag("logged_model_id", model_info.model_id)
        run_id = run.info.run_id

    registered_model = mlflow.register_model(
        f"models:/{model_info.model_id}",
        MODEL_NAME,
    )
    candidate_version = str(registered_model.version)
    client.set_registered_model_alias(
        MODEL_NAME,
        CANDIDATE_ALIAS,
        candidate_version,
    )
    client.set_tag(run_id, "candidate_model_version", candidate_version)

    promotion = promote_candidate(client, model_name=MODEL_NAME)
    client.set_tag(run_id, "retraining_status", "completed")
    client.set_tag(run_id, "promotion_status", promotion.status)
    client.set_tag(run_id, "promotion_reason", promotion.reason)
    client.set_tag(run_id, "resulting_champion_version", promotion.champion_version)
    return RetrainingResult(
        status=promotion.status,
        run_id=run_id,
        candidate_version=candidate_version,
        champion_version=promotion.champion_version,
        reason=promotion.reason,
    )
