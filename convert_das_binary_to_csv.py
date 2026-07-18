import argparse
import csv
import sys
from array import array
from pathlib import Path


TYPE_CODES = {
    "int16": "h",
    "float64": "d",
}


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def iter_source_files(input_dir: Path):
    for path in sorted(input_dir.iterdir()):
        if path.is_file():
            yield path


def load_values(path: Path, dtype: str, endian: str) -> array:
    type_code = TYPE_CODES[dtype]
    values = array(type_code)
    item_size = values.itemsize
    file_size = path.stat().st_size

    if file_size % item_size != 0:
        raise ValueError(
            f"{path} size {file_size} is not aligned to {dtype} item size {item_size}."
        )

    with path.open("rb") as f:
        values.frombytes(f.read())

    if endian in {"little", "big"} and endian != sys.byteorder:
        values.byteswap()

    return values


def resolve_shape(total_values: int, rows: int | None, cols: int | None) -> tuple[int, int]:
    if rows is not None and rows <= 0:
        raise ValueError("--rows must be greater than 0.")
    if cols is not None and cols <= 0:
        raise ValueError("--cols must be greater than 0.")

    if rows is not None and cols is not None:
        if total_values != rows * cols:
            raise ValueError(
                f"Expected {rows}x{cols}={rows * cols} values, got {total_values}."
            )
        return rows, cols

    if cols is not None:
        if total_values % cols != 0:
            raise ValueError(
                f"{total_values} values cannot be evenly reshaped with {cols} columns."
            )
        return total_values // cols, cols

    if rows is not None:
        if total_values % rows != 0:
            raise ValueError(
                f"{total_values} values cannot be evenly reshaped with {rows} rows."
            )
        return rows, total_values // rows

    raise ValueError("At least one of --rows or --cols must be provided.")


def format_value(value, dtype: str, float_decimals: int | None) -> str | int:
    if dtype == "float64":
        if float_decimals is None:
            return repr(float(value))
        return f"{float(value):.{float_decimals}f}"
    return int(value)


def write_csv(
    values: array,
    rows: int,
    cols: int,
    output_path: Path,
    dtype: str,
    float_decimals: int | None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        for row_idx in range(rows):
            start = row_idx * cols
            end = start + cols
            writer.writerow(
                format_value(value, dtype=dtype, float_decimals=float_decimals)
                for value in values[start:end]
            )


def convert_file(
    input_path: Path,
    output_path: Path,
    dtype: str,
    endian: str,
    rows: int | None,
    cols: int | None,
    float_decimals: int | None,
) -> tuple[int, int]:
    values = load_values(input_path, dtype=dtype, endian=endian)
    resolved_rows, resolved_cols = resolve_shape(len(values), rows=rows, cols=cols)
    write_csv(
        values,
        resolved_rows,
        resolved_cols,
        output_path,
        dtype=dtype,
        float_decimals=float_decimals,
    )
    return resolved_rows, resolved_cols


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch-convert DAS binary files to CSV."
    )
    parser.add_argument("input_dir", help="Input data directory")
    parser.add_argument("output_dir", help="Output CSV directory")
    parser.add_argument(
        "--dtype",
        choices=sorted(TYPE_CODES),
        default="int16",
        help="Binary element type. Default: int16",
    )
    parser.add_argument(
        "--endian",
        choices=["little", "big", "native"],
        default="little",
        help="Binary byte order. Default: little",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=None,
        help="Fixed row count",
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=2000,
        help="Fixed column count. Default: 2000",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Convert only the first N files",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing CSV files",
    )
    parser.add_argument(
        "--float-decimals",
        type=int,
        default=None,
        help="Format float64 output with a fixed number of decimal places",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    endian = sys.byteorder if args.endian == "native" else args.endian

    if not input_dir.exists():
        parser.error(f"Input directory does not exist: {input_dir}")
    if not input_dir.is_dir():
        parser.error(f"Input path is not a directory: {input_dir}")

    source_files = list(iter_source_files(input_dir))
    if args.limit is not None:
        source_files = source_files[: args.limit]

    if not source_files:
        print("No input files found.", file=sys.stderr)
        return 1

    total = len(source_files)
    converted = 0

    for index, input_path in enumerate(source_files, start=1):
        output_path = output_dir / f"{input_path.name}.csv"
        if output_path.exists() and not args.overwrite:
            print(f"[{index}/{total}] skip existing: {output_path}")
            continue

        resolved_rows, resolved_cols = convert_file(
            input_path=input_path,
            output_path=output_path,
            dtype=args.dtype,
            endian=endian,
            rows=args.rows,
            cols=args.cols,
            float_decimals=args.float_decimals,
        )
        converted += 1
        print(
            f"[{index}/{total}] converted: {input_path.name} -> "
            f"{output_path.name} ({resolved_rows}x{resolved_cols})"
        )

    print(f"done. total={total}, converted={converted}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
