import io
import json

import pytest

from taxi_demand_mlops import alerting
from taxi_demand_mlops.alerting import drift_alert_fingerprint


def _alert(feature: str, active_at: str) -> dict[str, object]:
    return {
        "state": "firing",
        "activeAt": active_at,
        "labels": {
            "alertname": "TaxiInputDrift",
            "feature": feature,
        },
    }


def test_drift_alert_fingerprint_is_independent_of_alert_order() -> None:
    first = [_alert("lag_1", "2026-11-19T12:00:00Z")]
    second = [
        _alert("lag_168", "2026-11-19T12:00:05Z"),
        _alert("lag_1", "2026-11-19T12:00:00Z"),
    ]

    assert drift_alert_fingerprint(first) == drift_alert_fingerprint(second)
    assert len(drift_alert_fingerprint(first)) == 64


def test_drift_alert_fingerprint_rejects_empty_or_unrelated_alerts() -> None:
    with pytest.raises(ValueError, match="empty"):
        drift_alert_fingerprint([])
    with pytest.raises(ValueError, match="unexpected"):
        drift_alert_fingerprint(
            [
                {
                    "activeAt": "2026-11-19T12:00:00Z",
                    "labels": {"alertname": "TaxiApiDown"},
                }
            ]
        )


def test_fetch_alerts_selects_only_firing_input_drift(monkeypatch) -> None:
    payload = {
        "status": "success",
        "data": {
            "alerts": [
                _alert("lag_1", "2026-11-19T12:00:00Z"),
                {
                    **_alert("lag_24", "2026-11-19T12:00:00Z"),
                    "state": "pending",
                },
                {
                    **_alert("lag_168", "2026-11-19T12:00:00Z"),
                    "labels": {"alertname": "TaxiApiDown"},
                },
            ]
        },
    }

    def fake_urlopen(url, timeout):
        assert url == "http://prometheus:9090/api/v1/alerts"
        assert timeout == 10
        return io.BytesIO(json.dumps(payload).encode())

    monkeypatch.setattr(alerting, "urlopen", fake_urlopen)

    assert len(alerting.fetch_firing_drift_alerts("http://prometheus:9090")) == 1
