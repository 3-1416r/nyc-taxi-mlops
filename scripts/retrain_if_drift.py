from __future__ import annotations

import argparse
import math
import time

from taxi_demand_mlops.alerting import (
    drift_alert_fingerprint,
    fetch_firing_drift_alerts,
)
from taxi_demand_mlops.retraining import run_retraining

DEFAULT_PROMETHEUS_URL = "http://127.0.0.1:9090"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Trigger one retraining run when Prometheus reports firing input drift"
    )
    parser.add_argument("--prometheus-url", default=DEFAULT_PROMETHEUS_URL)
    parser.add_argument("--lag-scale", type=float, default=1.0)
    parser.add_argument("--poll-interval", type=float, default=10)
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Keep polling and retrain for each new firing alert",
    )
    args = parser.parse_args()
    if not math.isfinite(args.lag_scale) or args.lag_scale <= 0:
        parser.error("--lag-scale must be a positive finite number")
    if not math.isfinite(args.poll_interval) or args.poll_interval <= 0:
        parser.error("--poll-interval must be a positive finite number")

    last_alert_id = None
    while True:
        alerts = fetch_firing_drift_alerts(args.prometheus_url)
        if alerts:
            alert_id = drift_alert_fingerprint(alerts)
            if alert_id != last_alert_id:
                result = run_retraining(
                    alert_id=alert_id,
                    lag_scale=args.lag_scale,
                )
                print(
                    f"Drift trigger {result.status}: run={result.run_id}, "
                    f"candidate={result.candidate_version}, "
                    f"champion={result.champion_version}. {result.reason}"
                )
                last_alert_id = alert_id
        else:
            last_alert_id = None
            print("No firing TaxiInputDrift alert.")
        if not args.watch:
            return
        time.sleep(args.poll_interval)


if __name__ == "__main__":
    main()
