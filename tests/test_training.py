import pandas as pd
import pytest

from taxi_demand_mlops.training import _split_features_target, fit


def _example_rows(count: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "pickup_zone_id": [1 + (index % 2) for index in range(count)],
            "hour": [index % 24 for index in range(count)],
            "day_of_week": [index % 7 for index in range(count)],
            "month": [1] * count,
            "is_weekend": [int(index % 7 >= 5) for index in range(count)],
            "lag_1": [float(index % 4) for index in range(count)],
            "lag_24": [float(index % 5) for index in range(count)],
            "lag_168": [float(index % 3) for index in range(count)],
            "demand": [float(index % 6) for index in range(count)],
        }
    )


def test_fit_trains_only_earlier_data_and_reports_reference_metrics() -> None:
    params = {
        "loss": "poisson",
        "learning_rate": 0.1,
        "max_iter": 5,
        "max_leaf_nodes": 7,
        "min_samples_leaf": 2,
        "l2_regularization": 1.0,
        "random_state": 42,
    }
    train = _example_rows(40)
    reference = _example_rows(10)

    model, metrics, features = fit(params, train, reference)

    assert len(features) == len(reference)
    assert set(metrics) == {
        "reference_mae",
        "reference_rmse",
        "reference_wape",
        "lag_24_baseline_mae",
    }
    assert all(value >= 0 for value in metrics.values())
    assert len(model.predict(features)) == len(reference)


def test_fit_rejects_negative_count_targets() -> None:
    data = _example_rows(5)
    data.loc[0, "demand"] = -1

    with pytest.raises(ValueError, match="non-negative demand"):
        _split_features_target(data)
