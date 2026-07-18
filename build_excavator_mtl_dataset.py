from __future__ import annotations

import argparse
import csv
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path


IDLE = "\u6020\u901f"
CONSTRUCTION = "\u65bd\u5de5"
NORMAL_CONSTRUCTION = "\u6b63\u5e38\u65bd\u5de5"
EXCAVATION_CONSTRUCTION = "\u6316\u6398\u65bd\u5de5"
EXCAVATION_CUTTING = "\u6316\u6398\u5207\u524a"
OPEN_TRENCH_CONSTRUCTION = "\u5f00\u6316\u65bd\u5de5"
KNOCK_GROUND = "\u6572\u5730\u9762"
KNOCK_POSITION = "\u6572\u51fb\u5b9a\u4f4d"
KNOCK_GROUND_POSITION = "\u6572\u51fb\u5730\u9762\u5b9a\u4f4d"
KNOCK_GROUND_ALT = "\u6572\u51fb\u5730\u9762"
DRIVE_PARALLEL = "\u5e73\u884c\u884c\u9a76"
DRIVE_VERTICAL = "\u5782\u76f4\u7ba1\u9053\u884c\u9a76"

EVENT_MAP = {
    IDLE: IDLE,
    CONSTRUCTION: CONSTRUCTION,
    NORMAL_CONSTRUCTION: CONSTRUCTION,
    EXCAVATION_CONSTRUCTION: CONSTRUCTION,
    EXCAVATION_CUTTING: CONSTRUCTION,
    OPEN_TRENCH_CONSTRUCTION: CONSTRUCTION,
    KNOCK_GROUND: "\u6572\u51fb",
    KNOCK_GROUND_ALT: "\u6572\u51fb",
    KNOCK_POSITION: "\u6572\u51fb",
    KNOCK_GROUND_POSITION: "\u6572\u51fb",
    DRIVE_PARALLEL: "\u884c\u9a76",
    DRIVE_VERTICAL: "\u884c\u9a76",
}

DISTANCE_MAP = {
    "5m": "5m",
    "10m": "5m",
    "20m": "20m",
    "26m": "20m",
    "40m": "40m",
}

OLD_PATTERN = re.compile(
    r"^10km-\u7ba1\u9053(?P<distance>\d+)m-(?P<soil>[^-]+)-(?P<event>[^-]+)(?:-(?P<time>\d+))?-(?P<seq>\d+)$"
)
NEW_PATTERN = re.compile(
    r"^10km-(?P<soil>land|sand|shizi)-(?P<distance>\d+)m-(?P<event>[^-]+)-(?P<time>\d+)-(?P<seq>\d+)$"
)
VERTICAL_PATTERN = re.compile(
    r"^10km-(?P<soil>land|sand|shizi)-\u5782\u76f4\u7ba1\u9053\u884c\u9a76-(?P<from>\u8fd1|\u8fdc)-(?P<to>\u8fd1|\u8fdc)-(?P<time>\d+)-(?P<seq>\d+)$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a cropped multi-task CSV dataset from the excavator folder. "
            "Outputs grouped samples and train/val/test manifests."
        )
    )
    parser.add_argument("source", type=Path, help="Source folder containing CSV files.")
    parser.add_argument("output", type=Path, help="Output dataset root folder.")
    parser.add_argument("--start-row", type=int, default=79, help="1-based start row, default 79.")
    parser.add_argument("--end-row", type=int, default=88, help="1-based end row, default 88.")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Train split ratio.")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation split ratio.")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test split ratio.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for splitting.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove the output folder first if it already exists.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.source.is_dir():
        raise FileNotFoundError(f"Source folder does not exist: {args.source}")
    if args.start_row < 1 or args.end_row < args.start_row:
        raise ValueError("Row range must satisfy 1 <= start-row <= end-row.")
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError("train-ratio + val-ratio + test-ratio must equal 1.0.")


def prepare_output(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output folder already exists: {output_dir}. Use --overwrite to replace it."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def parse_sample(path: Path) -> tuple[str | None, str | None, str | None]:
    stem = path.stem

    old_match = OLD_PATTERN.match(stem)
    if old_match:
        raw_distance = f"{old_match.group('distance')}m"
        raw_event = old_match.group("event")
    else:
        new_match = NEW_PATTERN.match(stem)
        if new_match:
            raw_distance = f"{new_match.group('distance')}m"
            raw_event = new_match.group("event")
        else:
            vertical_match = VERTICAL_PATTERN.match(stem)
            if vertical_match:
                return None, None, "distance_ambiguous_vertical"
            return None, None, "unparsed_name"

    event_label = EVENT_MAP.get(raw_event)
    if event_label is None:
        return None, None, f"unknown_event:{raw_event}"

    distance_label = DISTANCE_MAP.get(raw_distance)
    if distance_label is None:
        return None, None, f"unknown_distance:{raw_distance}"

    return event_label, distance_label, None


def crop_csv(source_file: Path, target_file: Path, start_row: int, end_row: int) -> None:
    with source_file.open("r", encoding="utf-8", newline="") as infile:
        lines = infile.readlines()

    if len(lines) < end_row:
        raise ValueError(
            f"File {source_file} has only {len(lines)} rows, cannot keep {start_row}-{end_row}."
        )

    target_file.parent.mkdir(parents=True, exist_ok=True)
    with target_file.open("w", encoding="utf-8", newline="") as outfile:
        outfile.writelines(lines[start_row - 1 : end_row])


def split_bucket(
    files: list[Path],
    train_ratio: float,
    val_ratio: float,
    rng: random.Random,
) -> tuple[list[Path], list[Path], list[Path]]:
    shuffled = list(files)
    rng.shuffle(shuffled)

    train_count = int(len(shuffled) * train_ratio)
    val_count = int(len(shuffled) * val_ratio)
    train_files = shuffled[:train_count]
    val_files = shuffled[train_count : train_count + val_count]
    test_files = shuffled[train_count + val_count :]
    return train_files, val_files, test_files


def write_manifest(path: Path, rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "event_label", "distance_label"])
        writer.writerows(rows)


def write_skipped(path: Path, rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "reason"])
        writer.writerows(rows)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()
    validate_args(args)
    prepare_output(args.output, args.overwrite)

    rng = random.Random(args.seed)
    samples_root = args.output / "samples"

    grouped_files: dict[tuple[str, str], list[Path]] = defaultdict(list)
    skipped_rows: list[list[str]] = []
    combo_counter: Counter[tuple[str, str]] = Counter()

    source_files = sorted(args.source.glob("*.csv"))
    if not source_files:
        raise ValueError(f"No CSV files found in {args.source}")

    for source_file in source_files:
        event_label, distance_label, issue = parse_sample(source_file)
        if issue is not None:
            skipped_rows.append([str(source_file), issue])
            continue

        target_file = samples_root / event_label / distance_label / source_file.name
        crop_csv(source_file, target_file, args.start_row, args.end_row)
        grouped_files[(event_label, distance_label)].append(target_file)
        combo_counter[(event_label, distance_label)] += 1

    manifest_rows = {"train": [], "val": [], "test": []}
    split_counter: dict[str, Counter[tuple[str, str]]] = {
        "train": Counter(),
        "val": Counter(),
        "test": Counter(),
    }

    for combo_key, files in sorted(grouped_files.items()):
        train_files, val_files, test_files = split_bucket(
            files=files,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            rng=rng,
        )
        event_label, distance_label = combo_key
        for split_name, split_files in (
            ("train", train_files),
            ("val", val_files),
            ("test", test_files),
        ):
            for file_path in split_files:
                manifest_rows[split_name].append([str(file_path), event_label, distance_label])
                split_counter[split_name][combo_key] += 1

    for split_name in ("train", "val", "test"):
        write_manifest(args.output / f"{split_name}.csv", manifest_rows[split_name])
    write_skipped(args.output / "skipped_samples.csv", skipped_rows)

    print(f"source_files={len(source_files)}")
    print(f"kept_files={sum(combo_counter.values())}")
    print(f"skipped_files={len(skipped_rows)}")
    for (event_label, distance_label), count in sorted(combo_counter.items()):
        print(f"kept\t{event_label}\t{distance_label}\t{count}")
    for split_name in ("train", "val", "test"):
        print(f"{split_name}_samples={len(manifest_rows[split_name])}")
        for (event_label, distance_label), count in sorted(split_counter[split_name].items()):
            print(f"{split_name}\t{event_label}\t{distance_label}\t{count}")


if __name__ == "__main__":
    main()
