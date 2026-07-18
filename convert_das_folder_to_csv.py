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


def write_csv(values: array, rows: int, cols: int, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        for row_idx in range(rows):
            start = row_idx * cols
            end = start + cols
            writer.writerow(values[start:end])


def convert_file(input_path: Path, output_path: Path, cols: int) -> tuple[int, int]:
    file_size = input_path.stat().st_size
    if file_size % 2 != 0:
        raise ValueError(f"{input_path} 不是 2 字节对齐，无法按 int16 读取。")

    values = load_int16_file(input_path)
    total_values = len(values)
    if total_values % cols != 0:
        raise ValueError(
            f"{input_path} 共 {total_values} 个 int16 数值，不能按 {cols} 列整齐展开。"
        )

    rows = total_values // cols
    write_csv(values, rows, cols, output_path)
    return rows, cols


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "将 DAS 原始二进制文件批量转换为 CSV。"
            " 默认按 little-endian int16 读取，并按指定列数展开。"
        )
    )
    parser.add_argument("input_dir", help="源数据目录")
    parser.add_argument("output_dir", help="CSV 输出目录")
    parser.add_argument(
        "--cols",
        type=int,
        default=150,
        help="每行列数，默认 2000",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="只转换前 N 个文件，便于抽样验证",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="如果目标 CSV 已存在则覆盖",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        parser.error(f"源目录不存在: {input_dir}")
    if not input_dir.is_dir():
        parser.error(f"源路径不是目录: {input_dir}")
    if args.cols <= 0:
        parser.error("--cols 必须大于 0")

    source_files = list(iter_source_files(input_dir))
    if args.limit is not None:
        source_files = source_files[: args.limit]

    if not source_files:
        print("没有发现可转换的文件。", file=sys.stderr)
        return 1

    total = len(source_files)
    converted = 0

    for index, input_path in enumerate(source_files, start=1):
        output_path = output_dir / f"{input_path.name}.csv"
        if output_path.exists() and not args.overwrite:
            print(f"[{index}/{total}] 跳过已存在文件: {output_path}")
            continue

        rows, cols = convert_file(input_path, output_path, args.cols)
        converted += 1
        print(f"[{index}/{total}] 已转换: {input_path.name} -> {output_path.name} ({rows}x{cols})")

    print(f"完成。共处理 {total} 个文件，新转换 {converted} 个文件。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
