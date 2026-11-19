from __future__ import annotations

import argparse

from taxi_demand_mlops.retraining import run_retraining


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retrain with chronological holdout evaluation and guarded promotion"
    )
    parser.add_argument("--alert-id")
    parser.add_argument("--lag-scale", type=float, default=1.0)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    args = parser.parse_args()

    result = run_retraining(
        alert_id=args.alert_id,
        lag_scale=args.lag_scale,
        validation_fraction=args.validation_fraction,
    )
    print(
        f"Retraining {result.status}: run={result.run_id}, "
        f"candidate={result.candidate_version}, champion={result.champion_version}. "
        f"{result.reason}"
    )


if __name__ == "__main__":
    main()
