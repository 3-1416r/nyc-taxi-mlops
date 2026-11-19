from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error


def evaluate_delayed_outcomes(
    predictions: pd.DataFrame,
    labels: pd.DataFrame,
) -> dict[str, float]:
    prediction_columns = {"request_id", "model_version", "forecast_demand"}
    label_columns = {"request_id", "actual_demand"}
    missing_predictions = prediction_columns.difference(predictions.columns)
    missing_labels = label_columns.difference(labels.columns)
    if missing_predictions:
        raise ValueError(
            f"Prediction log is missing columns: {sorted(missing_predictions)}"
        )
    if missing_labels:
        raise ValueError(f"Labels are missing columns: {sorted(missing_labels)}")
    if predictions.empty or labels.empty:
        raise ValueError("Predictions and labels must both be non-empty")
    if predictions["request_id"].isna().any() or labels["request_id"].isna().any():
        raise ValueError("Prediction and label request IDs must not be missing")
    if predictions["request_id"].duplicated().any():
        raise ValueError("Prediction request IDs must be unique")
    if labels["request_id"].duplicated().any():
        raise ValueError("Label request IDs must be unique")

    joined = predictions.merge(
        labels[["request_id", "actual_demand"]],
        on="request_id",
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    unmatched = joined["_merge"] != "both"
    if unmatched.any():
        raise ValueError(
            f"Prediction/label request IDs do not match ({int(unmatched.sum())} unmatched)"
        )
    if joined["model_version"].isna().any():
        raise ValueError("Model versions must not be missing")

    forecast = pd.to_numeric(joined["forecast_demand"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    actual = pd.to_numeric(joined["actual_demand"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    if not np.isfinite(forecast).all() or not np.isfinite(actual).all():
        raise ValueError("Forecasts and actual labels must be finite numbers")
    if (forecast < 0).any() or (actual < 0).any():
        raise ValueError("Forecasts and actual demand must be non-negative")
    actual_total = float(actual.sum())
    if actual_total <= 0:
        raise ValueError("WAPE is undefined when total actual demand is zero")
    return {
        "delayed_mae": float(mean_absolute_error(actual, forecast)),
        "delayed_wape": float(np.abs(actual - forecast).sum() / actual_total),
        "delayed_labeled_predictions": float(len(joined)),
    }
