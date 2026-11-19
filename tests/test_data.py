import pandas as pd
import pytest

from taxi_demand_mlops.data import (
    aggregate_trip_records,
    month_range,
    split_dataset,
)


def test_aggregation_fills_hours_and_creates_per_zone_lags() -> None:
    records = pd.DataFrame(
        {
            "tpep_pickup_datetime": [
                "2024-01-01 00:10:00",
                "2024-01-01 00:45:00",
                "2024-01-01 01:15:00",
                "2024-01-08 00:10:00",
                "2024-01-01 01:00:00",
                "2023-12-31 23:00:00",
            ],
            "PULocationID": [1, 1, 2, 1, None, 1],
        }
    )

    dataset = aggregate_trip_records(records, "2024-01", "2024-01")
    zone_one_start = dataset.loc[
        (dataset["pickup_hour"] == pd.Timestamp("2024-01-01 00:00:00"))
        & (dataset["pickup_zone_id"] == 1)
    ].iloc[0]
    zone_two_empty_hour = dataset.loc[
        (dataset["pickup_hour"] == pd.Timestamp("2024-01-01 00:00:00"))
        & (dataset["pickup_zone_id"] == 2)
    ].iloc[0]
    zone_two_next_hour = dataset.loc[
        (dataset["pickup_hour"] == pd.Timestamp("2024-01-01 01:00:00"))
        & (dataset["pickup_zone_id"] == 2)
    ].iloc[0]
    zone_one_week_later = dataset.loc[
        (dataset["pickup_hour"] == pd.Timestamp("2024-01-08 00:00:00"))
        & (dataset["pickup_zone_id"] == 1)
    ].iloc[0]

    assert len(dataset) == 31 * 24 * 2
    assert zone_one_start["demand"] == 2
    assert zone_two_empty_hour["demand"] == 0
    assert zone_two_next_hour["demand"] == 1
    assert zone_one_week_later["lag_168"] == 2


def test_month_range_is_inclusive_across_year_boundary() -> None:
    assert month_range("2024-11", "2025-02") == [
        "2024-11",
        "2024-12",
        "2025-01",
        "2025-02",
    ]


def test_split_dataset_has_ordered_non_overlapping_windows() -> None:
    hours = pd.date_range(
        "2024-01-01", "2024-05-01", freq="h", inclusive="left"
    )
    dataset = pd.DataFrame(
        {
            "pickup_hour": hours,
            "demand": 1,
            "lag_1": 1.0,
            "lag_24": 1.0,
            "lag_168": 1.0,
        }
    )

    splits = split_dataset(dataset)

    assert splits["train"]["pickup_hour"].max() < pd.Timestamp("2024-03-01")
    assert splits["reference"]["pickup_hour"].min() >= pd.Timestamp("2024-03-01")
    assert splits["reference"]["pickup_hour"].max() < pd.Timestamp("2024-04-01")
    assert splits["drift"]["pickup_hour"].min() >= pd.Timestamp("2024-04-01")


def test_split_dataset_rejects_reversed_periods() -> None:
    with pytest.raises(ValueError, match="before reference_end_month"):
        split_dataset(pd.DataFrame(), "2024-03", "2024-02")
