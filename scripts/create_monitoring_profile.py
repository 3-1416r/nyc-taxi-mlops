from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from taxi_demand_mlops.monitoring import build_reference_profile

REFERENCE_DATA = Path("data/processed/reference.parquet")
DVC_POINTER = Path("data/processed.dvc")
DEFAULT_OUTPUT = Path("monitoring/reference_profile.json")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a PSI reference profile from the March reference period"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if not REFERENCE_DATA.is_file():
        raise FileNotFoundError(
            f"Missing {REFERENCE_DATA}; download and build the DVC datasets first"
        )
    with DVC_POINTER.open(encoding="utf-8") as file:
        pointer = yaml.safe_load(file)
    outputs = pointer.get("outs") if isinstance(pointer, dict) else None
    if not outputs or not isinstance(outputs[0], dict) or "md5" not in outputs[0]:
        raise ValueError(f"Missing DVC fingerprint in {DVC_POINTER}")

    reference = pd.read_parquet(REFERENCE_DATA)
    profile = build_reference_profile(
        reference,
        reference_period="2024-03",
        source_data_md5=str(outputs[0]["md5"]),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(profile, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"Wrote PSI reference profile from {len(reference):,} rows to {args.output}"
    )


if __name__ == "__main__":
    main()
