from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path


DIAMETER_PATTERN = re.compile(r"(?<![A-Za-z0-9])(D[A-Za-z0-9]+)(?![A-Za-z0-9])")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Group CSV files by DXX in the filename and keep only a selected "
            "1-based row range in each output file."
        )
    )
    parser.add_argument("source", type=Path, help="Source folder containing CSV files.")
    parser.add_argument("output", type=Path, help="Output folder for grouped files.")
    parser.add_argument(
        "--start-row",
        type=int,
        default=79,
        help="1-based start row to keep (inclusive). Default: 79",
    )
    parser.add_argument(
        "--end-row",
        type=int,
        default=88,
        help="1-based end row to keep (inclusive). Default: 88",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove the output folder first if it already exists.",
    )
    parser.add_argument(
        "--segment-index",
        type=int,
        default=None,
        help=(
            "Optional 1-based filename segment index after splitting the base name "
            "on '-' to use as the grouping key."
        ),
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.source.is_dir():
        raise FileNotFoundError(f"Source folder does not exist: {args.source}")
    if args.start_row < 1 or args.end_row < args.start_row:
        raise ValueError("Row range must satisfy 1 <= start-row <= end-row.")


def prepare_output(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output folder already exists: {output_dir}. Use --overwrite to replace it."
            )
        try:
            shutil.rmtree(output_dir)
        except PermissionError:
            # Reuse the folder if Windows still holds a temporary handle on a file.
            pass
    output_dir.mkdir(parents=True, exist_ok=True)


def extract_diameter(name: str, segment_index: int | None) -> str:
    if segment_index is not None:
        parts = name.split("-")
        zero_based = segment_index - 1
        if zero_based < 0 or zero_based >= len(parts):
            raise ValueError(
                f"Filename {name} does not have segment {segment_index} after splitting on '-'."
            )
        return parts[zero_based]

    match = DIAMETER_PATTERN.search(name)
    if not match:
        raise ValueError(f"Could not find DXX in filename: {name}")
    return match.group(1)


def process_file(
    source_file: Path,
    output_root: Path,
    start_row: int,
    end_row: int,
    segment_index: int | None,
) -> str:
    diameter = extract_diameter(source_file.stem, segment_index)
    target_dir = output_root / diameter
    target_dir.mkdir(parents=True, exist_ok=True)
    target_file = target_dir / source_file.name

    with source_file.open("r", encoding="utf-8", newline="") as infile:
        lines = infile.readlines()

    if len(lines) < end_row:
        raise ValueError(
            f"File {source_file} has only {len(lines)} rows, cannot keep {start_row}-{end_row}."
        )

    selected = lines[start_row - 1 : end_row]

    with target_file.open("w", encoding="utf-8", newline="") as outfile:
        outfile.writelines(selected)

    return diameter


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()
    validate_args(args)
    prepare_output(args.output, args.overwrite)

    counts: dict[str, int] = {}
    processed = 0

    for source_file in sorted(args.source.glob("*.csv")):
        diameter = process_file(
            source_file,
            args.output,
            args.start_row,
            args.end_row,
            args.segment_index,
        )
        counts[diameter] = counts.get(diameter, 0) + 1
        processed += 1

    if processed == 0:
        raise ValueError(f"No CSV files found in {args.source}")

    print(f"processed={processed}")
    for diameter in sorted(counts):
        print(f"{diameter}={counts[diameter]}")


if __name__ == "__main__":
    main()
