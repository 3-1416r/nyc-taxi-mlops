from __future__ import annotations

import hashlib
import json
from urllib.request import urlopen

ALERT_NAME = "TaxiInputDrift"


def fetch_firing_drift_alerts(prometheus_url: str) -> list[dict[str, object]]:
    url = f"{prometheus_url.rstrip('/')}/api/v1/alerts"
    try:
        with urlopen(url, timeout=10) as response:
            payload = json.load(response)
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Unable to read Prometheus alerts at {url}") from error
    if not isinstance(payload, dict) or payload.get("status") != "success":
        raise ValueError(f"Prometheus returned an invalid alerts response from {url}")
    data = payload.get("data")
    alerts = data.get("alerts") if isinstance(data, dict) else None
    if not isinstance(alerts, list):
        raise TypeError("Prometheus alerts response has no alert list")

    firing = []
    for alert in alerts:
        if not isinstance(alert, dict):
            raise TypeError("Prometheus returned a malformed alert entry")
        labels = alert.get("labels")
        if not isinstance(labels, dict):
            raise TypeError("Prometheus alert is missing labels")
        if labels.get("alertname") == ALERT_NAME and alert.get("state") == "firing":
            firing.append(alert)
    return firing


def drift_alert_fingerprint(alerts: list[dict[str, object]]) -> str:
    if not alerts:
        raise ValueError("Cannot fingerprint an empty drift-alert set")
    active_times = []
    for alert in alerts:
        active_at = alert.get("activeAt")
        labels = alert.get("labels")
        if not isinstance(active_at, str) or not active_at:
            raise ValueError("Firing drift alert is missing activeAt")
        if not isinstance(labels, dict) or labels.get("alertname") != ALERT_NAME:
            raise ValueError("Alert set contains an unexpected alert")
        active_times.append(active_at)
    event = json.dumps(
        {
            "alert": ALERT_NAME,
            "active_at": min(active_times),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(event.encode("utf-8")).hexdigest()
