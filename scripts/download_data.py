from __future__ import annotations

import argparse
from pathlib import Path

from taxi_demand_mlops.data import download_months


def main() -> None:
    parser = argparse.ArgumentParser(description="Download NYC TLC Yellow Taxi data")
    parser.add_argument("--start-month", default="2024-01")
    parser.add_argument("--end-month", default="2024-04")
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    download_months(
        args.start_month,
        args.end_month,
        args.output_dir,
        force=args.force,
    )


if __name__ == "__main__":
    main()
