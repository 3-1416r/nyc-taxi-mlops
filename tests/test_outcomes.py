import pandas as pd
import pytest

from taxi_demand_mlops.outcomes import evaluate_delayed_outcomes


def test_delayed_outcomes_join_by_request_id_and_compute_quality_metrics() -> None:
    predictions = pd.DataFrame(
        {
            "request_id": ["r1", "r2"],
            "model_version": ["1", "1"],
            "forecast_demand": [2.0, 3.0],
        }
    )
    labels = pd.DataFrame(
        {
            "request_id": ["r2", "r1"],
            "actual_demand": [5, 1],
        }
    )

    assert evaluate_delayed_outcomes(predictions, labels) == {
        "delayed_mae": 1.5,
        "delayed_wape": 0.5,
        "delayed_labeled_predictions": 2.0,
    }


def test_delayed_outcomes_reject_unmatched_request_ids() -> None:
    predictions = pd.DataFrame(
        {
            "request_id": ["prediction-id"],
            "model_version": ["1"],
            "forecast_demand": [2.0],
        }
    )
    labels = pd.DataFrame(
        {"request_id": ["different-id"], "actual_demand": [1]}
    )

    with pytest.raises(ValueError, match="do not match"):
        evaluate_delayed_outcomes(predictions, labels)


def test_delayed_outcomes_reject_zero_actual_demand_for_wape() -> None:
    predictions = pd.DataFrame(
        {
            "request_id": ["r1"],
            "model_version": ["1"],
            "forecast_demand": [0.0],
        }
    )
    labels = pd.DataFrame({"request_id": ["r1"], "actual_demand": [0]})

    with pytest.raises(ValueError, match="WAPE is undefined"):
        evaluate_delayed_outcomes(predictions, labels)
