from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path


DATASET_ROOT = Path(r"D:\proj 1\converted_csv\MTL43")
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1
SEED = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create MTL43 train/val/test manifests with a configurable seed.")
    parser.add_argument("--dataset-root", default=str(DATASET_ROOT), type=str)
    parser.add_argument("--output-root", default=str(DATASET_ROOT), type=str)
    parser.add_argument("--seed", default=SEED, type=int)
    parser.add_argument("--train-ratio", default=TRAIN_RATIO, type=float)
    parser.add_argument("--val-ratio", default=VAL_RATIO, type=float)
    return parser.parse_args()


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "event_label", "distance_label", "sample_mode"])
        writer.writeheader()
        writer.writerows(rows)


def split_group(items: list[dict[str, str]], train_ratio: float, val_ratio: float) -> dict[str, list[dict[str, str]]]:
    total = len(items)
    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)
    if total >= 3:
        if train_count == 0:
            train_count = 1
        if val_count == 0:
            val_count = 1
        if train_count + val_count >= total:
            val_count = max(1, total - train_count - 1)
    return {
        "train": items[:train_count],
        "val": items[train_count: train_count + val_count],
        "test": items[train_count + val_count:],
    }


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    rng = random.Random(args.seed)
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)

    for path in sorted((dataset_root / "walking").glob("*.csv")):
        grouped[("walking", "_")].append(
            {"path": str(path), "event_label": "walking", "distance_label": "", "sample_mode": "row"}
        )
    for path in sorted((dataset_root / "driving").glob("*.csv")):
        grouped[("driving", "_")].append(
            {"path": str(path), "event_label": "driving", "distance_label": "", "sample_mode": "group3"}
        )
    for path in sorted((dataset_root / "background").glob("*.csv")):
        grouped[("background", "_")].append(
            {"path": str(path), "event_label": "background", "distance_label": "", "sample_mode": "file"}
        )
    for distance in ("5m", "20m", "40m"):
        for path in sorted((dataset_root / "excavator" / distance).glob("*.csv")):
            grouped[("excavator", distance)].append(
                {
                    "path": str(path),
                    "event_label": "excavator",
                    "distance_label": distance,
                    "sample_mode": "file",
                }
            )

    split_rows = {"train": [], "val": [], "test": []}
    for group_rows in grouped.values():
        rows = list(group_rows)
        rng.shuffle(rows)
        split = split_group(rows, args.train_ratio, args.val_ratio)
        for split_name, split_items in split.items():
            split_rows[split_name].extend(split_items)

    for split_name, rows in split_rows.items():
        rng.shuffle(rows)
        write_manifest(output_root / f"{split_name}.csv", rows)
        print(f"{split_name}: {len(rows)}")


if __name__ == "__main__":
    main()
