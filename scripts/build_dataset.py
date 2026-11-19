from __future__ import annotations

import argparse
from pathlib import Path

from taxi_demand_mlops.data import build_hourly_dataset, month_range, split_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build hourly pickup-zone features and chronological splits"
    )
    parser.add_argument("--start-month", default="2024-01")
    parser.add_argument("--end-month", default="2024-04")
    parser.add_argument("--train-end-month", default="2024-02")
    parser.add_argument("--reference-end-month", default="2024-03")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()

    input_paths = [
        args.raw_dir / f"yellow_tripdata_{month}.parquet"
        for month in month_range(args.start_month, args.end_month)
    ]
    missing_paths = [path for path in input_paths if not path.is_file()]
    if missing_paths:
        raise FileNotFoundError(
            "Missing TLC files. Run scripts/download_data.py first: "
            + ", ".join(str(path) for path in missing_paths)
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = build_hourly_dataset(input_paths, args.start_month, args.end_month)
    full_path = args.output_dir / "hourly_demand.parquet"
    dataset.to_parquet(full_path, index=False)
    print(f"Wrote {len(dataset):,} hourly zone rows to {full_path}")

    splits = split_dataset(
        dataset,
        train_end_month=args.train_end_month,
        reference_end_month=args.reference_end_month,
    )
    for name, frame in splits.items():
        output_path = args.output_dir / f"{name}.parquet"
        frame.to_parquet(output_path, index=False)
        print(
            f"Wrote {len(frame):,} {name} rows "
            f"({frame['pickup_hour'].min()} to {frame['pickup_hour'].max()}) "
            f"to {output_path}"
        )


if __name__ == "__main__":
    main()
