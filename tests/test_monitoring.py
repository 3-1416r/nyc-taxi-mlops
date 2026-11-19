import numpy as np
import pandas as pd
import pytest
from prometheus_client import REGISTRY

from taxi_demand_mlops.monitoring import (
    DRIFT_FEATURES,
    InputDriftMonitor,
    build_reference_profile,
    population_stability_index,
)


def test_psi_is_low_for_reference_and_high_for_shifted_values() -> None:
    rng = np.random.default_rng(42)
    values = rng.normal(loc=10, scale=2, size=2_000)
    profile = build_reference_profile(
        pd.DataFrame({feature: values for feature in DRIFT_FEATURES}),
        reference_period="test",
    )
    feature_profile = profile["features"]["lag_1"]

    reference_psi = population_stability_index(
        feature_profile["edges"],
        feature_profile["reference_shares"],
        rng.normal(loc=10, scale=2, size=2_000),
    )
    shifted_psi = population_stability_index(
        feature_profile["edges"],
        feature_profile["reference_shares"],
        rng.normal(loc=30, scale=2, size=2_000),
    )

    assert reference_psi < 0.2
    assert shifted_psi > 0.2


def test_reference_profile_handles_constant_features() -> None:
    profile = build_reference_profile(
        pd.DataFrame({feature: [0.0] * 20 for feature in DRIFT_FEATURES}),
        reference_period="test",
    )
    feature_profile = profile["features"]["lag_24"]

    assert len(feature_profile["edges"]) == 2
    assert feature_profile["reference_shares"] == [1.0]
    assert population_stability_index(
        feature_profile["edges"],
        feature_profile["reference_shares"],
        [0.0] * 20,
    ) == 0


def test_psi_rejects_empty_live_window() -> None:
    with pytest.raises(ValueError, match="Invalid edges"):
        population_stability_index([0, 1], [1], [])


def test_monitor_exports_psi_after_rolling_window_fills() -> None:
    rng = np.random.default_rng(42)
    reference_values = rng.normal(loc=10, scale=2, size=1_000)
    profile = build_reference_profile(
        pd.DataFrame(
            {feature: reference_values for feature in DRIFT_FEATURES}
        ),
        reference_period="test",
    )
    monitor = InputDriftMonitor(profile, window_size=20, recompute_every=1)

    for _ in range(20):
        monitor.observe({feature: 30 for feature in DRIFT_FEATURES})

    psi = REGISTRY.get_sample_value("taxi_input_psi", {"feature": "lag_1"})
    sample_count = REGISTRY.get_sample_value(
        "taxi_input_drift_window_samples",
        {"feature": "lag_1"},
    )
    assert psi is not None and psi > 0.2
    assert sample_count == 20
