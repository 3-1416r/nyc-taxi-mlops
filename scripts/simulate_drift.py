from __future__ import annotations

import argparse
import asyncio
import math
import time
from pathlib import Path

import httpx
import pandas as pd

from taxi_demand_mlops.training import FEATURE_COLUMNS

DRIFT_DATA = Path("data/processed/drift.parquet")
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
) -> None:
    started = time.monotonic()
    async with httpx.AsyncClient(timeout=20) as client:
        for index in range(count):
            payload = _request_payload(frame.iloc[index], lag_scale)
            response = await client.post(url, json=payload)
            response.raise_for_status()
            if (index + 1) % 100 == 0:
                print(f"Sent {index + 1}/{count} drifted predictions")
            delay = started + (index + 1) / rate - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay April taxi features with intentionally shifted demand lags"
    )
    parser.add_argument("--url", default="http://127.0.0.1:8000/v1/predict")
    parser.add_argument("--data", type=Path, default=DRIFT_DATA)
    parser.add_argument("--count", type=int, default=600)
    parser.add_argument("--rate", type=float, default=30)
    parser.add_argument("--lag-scale", type=float, default=3)
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
    asyncio.run(
        _send_requests(
            frame,
            url=args.url,
            count=args.count,
            rate=args.rate,
            lag_scale=args.lag_scale,
        )
    )
    print(
        f"Completed {args.count} predictions at about {args.rate:g} requests/s; "
        f"scaled demand lags by {args.lag_scale:g}."
    )


if __name__ == "__main__":
    main()
