import pandas as pd
import pytest

from taxi_demand_mlops.retraining import split_retraining_data
from taxi_demand_mlops.training import FEATURE_COLUMNS


def _period(start: str, hours: int) -> pd.DataFrame:
    rows = []
    for pickup_hour in pd.date_range(start, periods=hours, freq="h"):
        for zone_id in (1, 2):
            rows.append(
                {
                    "pickup_hour": pickup_hour,
                    "pickup_zone_id": zone_id,
                    "demand": float(zone_id),
                    "hour": pickup_hour.hour,
                    "day_of_week": pickup_hour.day_of_week,
                    "month": pickup_hour.month,
                    "is_weekend": int(pickup_hour.day_of_week >= 5),
                    "lag_1": 1.0,
                    "lag_24": 2.0,
                    "lag_168": 3.0,
                }
            )
    return pd.DataFrame(rows, columns=["pickup_hour", *FEATURE_COLUMNS, "demand"])


def test_retraining_holdout_is_later_than_all_training_hours() -> None:
    split = split_retraining_data(
        _period("2024-01-01", 4),
        _period("2024-03-01", 4),
        _period("2024-04-01", 10),
        validation_fraction=0.2,
        lag_scale=3,
    )

    assert split.validation_start == pd.Timestamp("2024-04-01 08:00")
    assert split.training["pickup_hour"].max() < split.validation_start
    assert split.validation["pickup_hour"].min() == split.validation_start
    april_training = split.training.loc[
        split.training["pickup_hour"] >= pd.Timestamp("2024-04-01")
    ]
    assert len(april_training) == 16
    assert (april_training["lag_24"] == 6).all()
    assert (split.validation["lag_168"] == 9).all()


@pytest.mark.parametrize("fraction", [0, 1, -0.1, 1.1])
def test_retraining_rejects_invalid_validation_fraction(fraction: float) -> None:
    with pytest.raises(ValueError, match="validation_fraction"):
        split_retraining_data(
            _period("2024-01-01", 2),
            _period("2024-03-01", 2),
            _period("2024-04-01", 2),
            validation_fraction=fraction,
        )


def test_retraining_rejects_too_few_unique_drift_hours() -> None:
    one_hour = _period("2024-04-01", 1)

    with pytest.raises(ValueError, match="too few unique hours"):
        split_retraining_data(
            _period("2024-01-01", 2),
            _period("2024-03-01", 2),
            one_hour,
        )
