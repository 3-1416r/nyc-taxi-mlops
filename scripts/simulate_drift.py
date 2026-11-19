from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pandas as pd

from taxi_demand_mlops.training import FEATURE_COLUMNS

DRIFT_DATA = Path("data/processed/drift.parquet")
DEFAULT_OUTPUT_DIR = Path(".monitoring-output")
LAG_FEATURES = ("lag_1", "lag_24", "lag_168")


def _request_payload(row: pd.Series, lag_scale: float) -> dict[str, int | float | bool]:
    payload: dict[str, int | float | bool] = {}
    for feature in FEATURE_COLUMNS:
        value = row[feature]
        if feature == "is_weekend":
            payload[feature] = bool(value)
        elif feature in LAG_FEATURES:
            payload[feature] = float(value) * lag_scale
        else:
            payload[feature] = int(value)
    return payload


async def _send_requests(
    frame: pd.DataFrame,
    *,
    url: str,
    count: int,
    rate: float,
    lag_scale: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    started = time.monotonic()
    predictions: list[dict[str, object]] = []
    labels: list[dict[str, object]] = []
    async with httpx.AsyncClient(timeout=20) as client:
        for index in range(count):
            row = frame.iloc[index]
            payload = _request_payload(row, lag_scale)
            response = await client.post(url, json=payload)
            response.raise_for_status()
            request_id = response.headers.get("x-request-id")
            body = response.json()
            if not request_id or not isinstance(body, dict):
                raise ValueError("Prediction response is missing its request ID or body")
            forecast = float(body["forecast_demand"])
            model_version = str(body["model_version"])
            if not math.isfinite(forecast) or forecast < 0:
                raise ValueError("Prediction response contains an invalid forecast")
            if not model_version:
                raise ValueError("Prediction response contains an empty model version")
            predictions.append(
                {
                    "request_id": request_id,
                    "predicted_at": datetime.now(UTC).isoformat(),
                    "model_version": model_version,
                    "forecast_demand": forecast,
                }
            )
            labels.append(
                {
                    "request_id": request_id,
                    "pickup_hour": pd.Timestamp(row["pickup_hour"]).isoformat(),
                    "pickup_zone_id": int(row["pickup_zone_id"]),
                    "actual_demand": int(row["demand"]),
                }
            )
            if (index + 1) % 100 == 0:
                print(f"Sent {index + 1}/{count} drifted predictions")
            delay = started + (index + 1) / rate - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
    return predictions, labels


def _write_replay_records(
    output_dir: Path,
    predictions: list[dict[str, object]],
    labels: list[dict[str, object]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.jsonl"
    predictions_path.write_text(
        "".join(
            json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
            for record in predictions
        ),
        encoding="utf-8",
    )
    labels_path = output_dir / "labels.csv"
    with labels_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(labels[0]))
        writer.writeheader()
        writer.writerows(labels)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay April taxi features with intentionally shifted demand lags"
    )
    parser.add_argument("--url", default="http://127.0.0.1:8000/v1/predict")
    parser.add_argument("--data", type=Path, default=DRIFT_DATA)
    parser.add_argument("--count", type=int, default=600)
    parser.add_argument("--rate", type=float, default=30)
    parser.add_argument("--lag-scale", type=float, default=3)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be positive")
    if not math.isfinite(args.rate) or args.rate <= 0:
        parser.error("--rate must be a positive finite number")
    if not math.isfinite(args.lag_scale) or args.lag_scale <= 0:
        parser.error("--lag-scale must be a positive finite number")
    if not args.data.is_file():
        raise FileNotFoundError(
            f"Missing {args.data}; download and build the DVC datasets first"
        )

    frame = pd.read_parquet(args.data)
    if len(frame) < args.count:
        raise ValueError(
            f"Requested {args.count} rows, but drift data has only {len(frame)}"
        )
    missing_features = set(FEATURE_COLUMNS).difference(frame.columns)
    if missing_features:
        raise ValueError(f"Drift data is missing features: {sorted(missing_features)}")
    missing_outcome_columns = {"pickup_hour", "demand"}.difference(frame.columns)
    if missing_outcome_columns:
        raise ValueError(
            f"Drift data is missing outcome columns: {sorted(missing_outcome_columns)}"
        )
    predictions, labels = asyncio.run(
        _send_requests(
            frame,
            url=args.url,
            count=args.count,
            rate=args.rate,
            lag_scale=args.lag_scale,
        )
    )
    _write_replay_records(args.output_dir, predictions, labels)
    print(
        f"Completed {args.count} predictions at about {args.rate:g} requests/s; "
        f"scaled demand lags by {args.lag_scale:g}. "
        f"Prediction and label records are in {args.output_dir}."
    )


if __name__ == "__main__":
    main()
