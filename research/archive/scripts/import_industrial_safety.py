from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


# Project root: C:\Projects\SIH26165
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Make project packages such as "ml" importable.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from ml.data.adapters.industrial_safety import load_and_convert


DEFAULT_INPUT = (
    PROJECT_ROOT
    / "data"
    / "public"
    / "industrial_safety"
    / "raw"
    / "_with_accidents_description.csv"
)

DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "industrial_safety_canonical.parquet"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert the Industrial Safety dataset "
            "to the RAKSHAK canonical format."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to the raw Industrial Safety CSV.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path for the canonical Parquet dataset.",
    )

    args = parser.parse_args()

    print(f"Loading: {args.input}")

    df = load_and_convert(args.input)

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_parquet(
        args.output,
        index=False,
    )

    print()
    print(f"Input records : {len(df)}")
    print(f"Output records: {len(df)}")
    print(f"Output file   : {args.output}")

    print()
    print("Canonical columns:")
    for column in df.columns:
        print(f"  - {column}")


if __name__ == "__main__":
    main()