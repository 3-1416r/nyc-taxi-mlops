from __future__ import annotations

import calendar
import shutil
from datetime import datetime
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

TLC_TRIP_DATA_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data"
PICKUP_DATETIME_COLUMN = "tpep_pickup_datetime"
PICKUP_ZONE_COLUMN = "PULocationID"
REQUIRED_TRIP_COLUMNS = (PICKUP_DATETIME_COLUMN, PICKUP_ZONE_COLUMN)
LAG_HOURS = (1, 24, 168)


def month_range(start_month: str, end_month: str) -> list[str]:
    """Return inclusive YYYY-MM values, rejecting invalid or reversed ranges."""
    start = _parse_month(start_month)
    end = _parse_month(end_month)
    if start > end:
        raise ValueError("start_month must be on or before end_month")

    months: list[str] = []
    year, month = start.year, start.month
    while (year, month) <= (end.year, end.month):
        months.append(f"{year:04d}-{month:02d}")
        if month == 12:
            year, month = year + 1, 1
        else:
            month += 1
    return months


def download_months(
    start_month: str,
    end_month: str,
    output_dir: Path,
    *,
    force: bool = False,
) -> list[Path]:
    """Download official NYC TLC Yellow Taxi Parquet files for an inclusive range."""
    months = month_range(start_month, end_month)
    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[Path] = []

    for month in months:
        destination = output_dir / f"yellow_tripdata_{month}.parquet"
        partial = destination.with_suffix(destination.suffix + ".part")
        if destination.exists() and not force:
            if destination.stat().st_size == 0:
                raise ValueError(f"Existing data file is empty: {destination}")
            print(f"Using existing {destination}")
            downloaded.append(destination)
            continue

        url = f"{TLC_TRIP_DATA_URL}/yellow_tripdata_{month}.parquet"
        try:
            with urlopen(url, timeout=120) as response, partial.open("wb") as target:
                shutil.copyfileobj(response, target)
            if partial.stat().st_size == 0:
                raise ValueError(f"Downloaded data file is empty: {url}")
            partial.replace(destination)
        finally:
            if partial.exists():
                partial.unlink()

        print(f"Downloaded {url} -> {destination}")
        downloaded.append(destination)

    return downloaded


def aggregate_trip_records(
    records: pd.DataFrame,
    start_month: str,
    end_month: str,
) -> pd.DataFrame:
    """Aggregate TLC trips to a complete hourly pickup-zone demand grid."""
    months = month_range(start_month, end_month)
    missing_columns = set(REQUIRED_TRIP_COLUMNS).difference(records.columns)
    if missing_columns:
        raise ValueError(f"Trip data is missing columns: {sorted(missing_columns)}")

    start = pd.Timestamp(f"{months[0]}-01")
    end_month_year, end_month_number = map(int, months[-1].split("-"))
    end = pd.Timestamp(
        f"{end_month_year:04d}-{end_month_number:02d}-"
        f"{calendar.monthrange(end_month_year, end_month_number)[1]:02d}"
    ) + pd.Timedelta(days=1)

    pickup_time = pd.to_datetime(records[PICKUP_DATETIME_COLUMN], errors="coerce")
    if pickup_time.dt.tz is not None:
        raise ValueError("TLC pickup timestamps must be naive NYC local time")

    zone = pd.to_numeric(records[PICKUP_ZONE_COLUMN], errors="coerce")
    valid = (
        pickup_time.notna()
        & zone.notna()
        & zone.between(1, 265)
        & pickup_time.ge(start)
        & pickup_time.lt(end)
    )
    trips = pd.DataFrame(
        {
            "pickup_hour": pickup_time.loc[valid].dt.floor("h"),
            "pickup_zone_id": zone.loc[valid].astype("int16"),
        }
    )
    if trips.empty:
        raise ValueError("No valid pickup records found in the selected date range")

    zones = sorted(trips["pickup_zone_id"].unique())
    hours = pd.date_range(start, end, freq="h", inclusive="left")
    complete_grid = pd.MultiIndex.from_product(
        [hours, zones], names=["pickup_hour", "pickup_zone_id"]
    )
    counts = trips.groupby(["pickup_hour", "pickup_zone_id"]).size()
    hourly = counts.reindex(complete_grid, fill_value=0).rename("demand").reset_index()
    hourly["demand"] = hourly["demand"].astype("int32")

    hourly["hour"] = hourly["pickup_hour"].dt.hour.astype("int8")
    hourly["day_of_week"] = hourly["pickup_hour"].dt.day_of_week.astype("int8")
    hourly["month"] = hourly["pickup_hour"].dt.month.astype("int8")
    hourly["is_weekend"] = (hourly["day_of_week"] >= 5).astype("int8")

    for lag in LAG_HOURS:
        hourly[f"lag_{lag}"] = (
            hourly.groupby("pickup_zone_id", sort=False)["demand"]
            .shift(lag)
            .astype("float32")
        )

    return hourly


def build_hourly_dataset(
    input_paths: list[Path],
    start_month: str,
    end_month: str,
) -> pd.DataFrame:
    """Read the selected TLC Parquet files and create the hourly feature table."""
    expected_months = month_range(start_month, end_month)
    if len(input_paths) != len(expected_months):
        raise ValueError(
            f"Expected {len(expected_months)} monthly files, got {len(input_paths)}"
        )
    records = pd.concat(
        [
            pd.read_parquet(path, columns=list(REQUIRED_TRIP_COLUMNS))
            for path in input_paths
        ],
        ignore_index=True,
    )
    return aggregate_trip_records(records, start_month, end_month)


def split_dataset(
    dataset: pd.DataFrame,
    train_end_month: str = "2024-02",
    reference_end_month: str = "2024-03",
) -> dict[str, pd.DataFrame]:
    """Create chronological train, reference, and later drift datasets."""
    train_end = _parse_month(train_end_month)
    reference_end = _parse_month(reference_end_month)
    if train_end >= reference_end:
        raise ValueError("train_end_month must be before reference_end_month")
    if "pickup_hour" not in dataset.columns:
        raise ValueError("Dataset must include a pickup_hour column")

    lag_columns = [f"lag_{lag}" for lag in LAG_HOURS]
    missing_lags = set(lag_columns).difference(dataset.columns)
    if missing_lags:
        raise ValueError(f"Dataset is missing lag features: {sorted(missing_lags)}")

    eligible = dataset.dropna(subset=lag_columns).copy()
    timestamps = pd.to_datetime(eligible["pickup_hour"])
    train_boundary = pd.Timestamp(
        train_end.replace(day=1) + pd.DateOffset(months=1)
    )
    reference_boundary = pd.Timestamp(
        reference_end.replace(day=1) + pd.DateOffset(months=1)
    )
    splits = {
        "train": eligible.loc[timestamps < train_boundary].copy(),
        "reference": eligible.loc[
            (timestamps >= train_boundary) & (timestamps < reference_boundary)
        ].copy(),
        "drift": eligible.loc[timestamps >= reference_boundary].copy(),
    }

    empty_splits = [name for name, frame in splits.items() if frame.empty]
    if empty_splits:
        raise ValueError(
            f"Date range must populate every split; empty: {', '.join(empty_splits)}"
        )
    return {name: frame.reset_index(drop=True) for name, frame in splits.items()}


def _parse_month(value: str) -> datetime:
    try:
        parsed = datetime.strptime(value, "%Y-%m")
    except ValueError as error:
        raise ValueError(f"Invalid month {value!r}; expected YYYY-MM") from error
    if parsed.strftime("%Y-%m") != value:
        raise ValueError(f"Invalid month {value!r}; expected YYYY-MM")
    return parsed
