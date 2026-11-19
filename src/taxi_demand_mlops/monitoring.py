from __future__ import annotations

import json
import math
import threading
from collections import deque
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
from prometheus_client import Counter, Gauge, Histogram, make_asgi_app

DRIFT_FEATURES = ("lag_1", "lag_24", "lag_168")
DRIFT_WINDOW_SIZE = 300
DRIFT_RECOMPUTE_EVERY = 25
DRIFT_THRESHOLD = 0.2
DEFAULT_PROFILE_PATH = Path("monitoring/reference_profile.json")

API_REQUESTS = Counter(
    "taxi_api_requests_total",
    "HTTP requests served by the taxi demand API",
    ["method", "path", "status"],
)
API_REQUEST_DURATION = Histogram(
    "taxi_api_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "path"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2),
)
MODEL_PREDICTIONS = Counter(
    "taxi_model_predictions_total",
    "Successful demand forecasts",
    ["model_version"],
)
FORECAST_DEMAND = Histogram(
    "taxi_forecast_demand",
    "Distribution of forecast hourly pickups",
    ["model_version"],
    buckets=(0.1, 1, 2, 5, 10, 20, 50, 100, 250, 500, 1000, 2500, 5000),
)
INPUT_PSI = Gauge(
    "taxi_input_psi",
    "Population stability index for the latest input window",
    ["feature"],
)
INPUT_WINDOW_SAMPLES = Gauge(
    "taxi_input_drift_window_samples",
    "Number of requests in the current input drift window",
    ["feature"],
)
DRIFT_EVENTS = Counter(
    "taxi_input_drift_events_total",
    "Input PSI threshold crossings",
    ["feature"],
)
PROMETHEUS_APP = make_asgi_app()


def build_reference_profile(
    reference: pd.DataFrame,
    reference_period: str,
    source_data_md5: str | None = None,
) -> dict[str, object]:
    missing_features = set(DRIFT_FEATURES).difference(reference.columns)
    if missing_features:
        raise ValueError(f"Reference data is missing features: {sorted(missing_features)}")
    if reference.empty:
        raise ValueError("Reference data must not be empty")

    features: dict[str, dict[str, list[float]]] = {}
    for feature in DRIFT_FEATURES:
        values = pd.to_numeric(reference[feature], errors="coerce").to_numpy(
            dtype=np.float64
        )
        if values.size == 0:
            raise ValueError(f"Reference feature {feature!r} has no values")
        if not np.isfinite(values).all():
            raise ValueError(f"Reference feature {feature!r} contains non-finite values")

        edges = np.unique(np.quantile(values, np.linspace(0, 1, 11)))
        if edges.size < 2:
            spread = max(abs(float(edges[0])) * 1e-6, 0.5)
            edges = np.array([edges[0] - spread, edges[0] + spread])
        counts = np.histogram(values, bins=edges)[0]
        if counts.sum() == 0:
            raise ValueError(f"Reference feature {feature!r} has no histogram counts")
        shares = counts / counts.sum()
        features[feature] = {
            "edges": edges.astype(float).tolist(),
            "reference_shares": shares.astype(float).tolist(),
        }

    profile: dict[str, object] = {
        "reference_period": reference_period,
        "features": features,
    }
    if source_data_md5 is not None:
        profile["source_data_md5"] = source_data_md5
    return profile


def load_reference_profile(
    path: Path = DEFAULT_PROFILE_PATH,
) -> dict[str, object]:
    with path.open(encoding="utf-8") as file:
        profile = json.load(file)
    if not isinstance(profile, dict) or not isinstance(profile.get("features"), dict):
        raise TypeError(f"Invalid monitoring reference profile: {path}")

    features = profile["features"]
    if set(features) != set(DRIFT_FEATURES):
        raise ValueError(
            "Monitoring profile features must be exactly "
            + ", ".join(DRIFT_FEATURES)
        )
    for feature in DRIFT_FEATURES:
        _validated_feature_profile(feature, features[feature])
    return profile


def _validated_feature_profile(
    feature: str,
    entry: object,
) -> tuple[list[float], list[float]]:
    if not isinstance(entry, dict):
        raise TypeError(f"Invalid profile entry for {feature!r}")
    try:
        edges = np.asarray(entry["edges"], dtype=np.float64)
        shares = np.asarray(entry["reference_shares"], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid histogram profile for {feature!r}") from error
    if (
        edges.ndim != 1
        or shares.ndim != 1
        or edges.size != shares.size + 1
        or edges.size < 2
        or not np.isfinite(edges).all()
        or not np.isfinite(shares).all()
        or not np.all(np.diff(edges) > 0)
        or (shares < 0).any()
        or not np.isclose(shares.sum(), 1)
    ):
        raise ValueError(f"Invalid histogram profile for {feature!r}")
    return edges.tolist(), shares.tolist()


def population_stability_index(
    edges: list[float] | np.ndarray,
    reference_shares: list[float] | np.ndarray,
    sample: list[float] | np.ndarray,
) -> float:
    bin_edges = np.asarray(edges, dtype=np.float64)
    reference = np.asarray(reference_shares, dtype=np.float64)
    values = np.asarray(sample, dtype=np.float64)
    if (
        bin_edges.ndim != 1
        or reference.ndim != 1
        or bin_edges.size != reference.size + 1
        or bin_edges.size < 2
        or not np.isfinite(bin_edges).all()
        or not np.all(np.diff(bin_edges) > 0)
        or reference.size == 0
        or values.size == 0
        or not np.isfinite(values).all()
        or not np.isfinite(reference).all()
        or (reference < 0).any()
        or not np.isclose(reference.sum(), 1)
    ):
        raise ValueError("Invalid edges, reference shares, or sample for PSI")

    clipped = np.clip(values, bin_edges[0], bin_edges[-1])
    live_counts = np.histogram(clipped, bins=bin_edges)[0]
    live = live_counts / live_counts.sum()
    live = np.clip(live, 1e-4, None)
    expected = np.clip(reference, 1e-4, None)
    return float(np.sum((live - expected) * np.log(live / expected)))


class InputDriftMonitor:
    def __init__(
        self,
        profile: Mapping[str, object],
        *,
        window_size: int = DRIFT_WINDOW_SIZE,
        recompute_every: int = DRIFT_RECOMPUTE_EVERY,
        threshold: float = DRIFT_THRESHOLD,
    ) -> None:
        if window_size < 1 or recompute_every < 1:
            raise ValueError("window_size and recompute_every must be positive")
        if not math.isfinite(threshold) or threshold <= 0:
            raise ValueError("threshold must be a positive finite number")

        profiles = profile.get("features")
        if not isinstance(profiles, Mapping):
            raise TypeError("Monitoring profile has no feature distributions")
        if set(profiles) != set(DRIFT_FEATURES):
            raise ValueError(
                "Monitoring profile features must be exactly "
                + ", ".join(DRIFT_FEATURES)
            )
        self._profiles = {
            feature: _validated_feature_profile(feature, profiles[feature])
            for feature in DRIFT_FEATURES
        }
        self._windows = {
            feature: deque(maxlen=window_size) for feature in DRIFT_FEATURES
        }
        self._seen = dict.fromkeys(DRIFT_FEATURES, 0)
        self._last_scores = dict.fromkeys(DRIFT_FEATURES, 0.0)
        self._recompute_every = recompute_every
        self._window_size = window_size
        self._threshold = threshold
        self._lock = threading.Lock()

        for feature in DRIFT_FEATURES:
            INPUT_PSI.labels(feature).set(0)
            INPUT_WINDOW_SAMPLES.labels(feature).set(0)

    def observe(self, features: Mapping[str, float]) -> None:
        with self._lock:
            for feature in DRIFT_FEATURES:
                value = float(features[feature])
                if not math.isfinite(value):
                    raise ValueError(f"Input feature {feature!r} must be finite")
                window = self._windows[feature]
                window.append(value)
                self._seen[feature] += 1
                INPUT_WINDOW_SAMPLES.labels(feature).set(len(window))

                if len(window) < self._window_size:
                    continue
                if (self._seen[feature] - self._window_size) % self._recompute_every:
                    continue

                edges, reference_shares = self._profiles[feature]
                score = population_stability_index(
                    edges,
                    reference_shares,
                    list(window),
                )
                INPUT_PSI.labels(feature).set(score)
                if (
                    score >= self._threshold
                    and self._last_scores[feature] < self._threshold
                ):
                    DRIFT_EVENTS.labels(feature).inc()
                self._last_scores[feature] = score


def observe_prediction(model_version: str, forecast: float) -> None:
    MODEL_PREDICTIONS.labels(model_version).inc()
    FORECAST_DEMAND.labels(model_version).observe(forecast)
