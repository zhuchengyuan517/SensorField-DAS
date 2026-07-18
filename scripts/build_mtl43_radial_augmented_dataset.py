from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path


DISTANCE_LABEL_ALIASES = {
    "5m": "Alarm area",
    "20m": "Tracking area",
    "40m": "No-threat area",
    "Alarm area": "Alarm area",
    "Tracking area": "Tracking area",
    "No-threat area": "No-threat area",
}

DEFAULT_DISTANCE_CLASSES = ("Alarm area", "Tracking area", "No-threat area")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build an MTL43 manifest set whose train split tops up Task 2 (Radial Threat) "
            "to a target count per class using duplicated rows intended for augmentation."
        )
    )
    parser.add_argument("--source-root", default=r"D:\proj 1\converted_csv\MTL43", type=str)
    parser.add_argument("--output-root", default=r"D:\proj 1\converted_csv\MTL43\_radial_aug1000", type=str)
    parser.add_argument("--target-per-class", default=1000, type=int)
    parser.add_argument(
        "--distance-classes",
        default=",".join(DEFAULT_DISTANCE_CLASSES),
        type=str,
        help="Comma-separated Radial Threat class order.",
    )
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_label_list(payload: str) -> list[str]:
    labels = [item.strip() for item in payload.split(",") if item.strip()]
    if not labels:
        raise ValueError("distance-classes cannot be empty")
    return labels


def canonicalize_distance_label(label_text: str) -> str:
    label = str(label_text).strip()
    if not label:
        return ""
    return DISTANCE_LABEL_ALIASES.get(label, label)


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    fieldnames = ["path", "event_label", "distance_label", "sample_mode", "augment_profile", "supervision_profile"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def prepare_output_root(output_root: Path, overwrite: bool) -> None:
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {output_root}. Use --overwrite to replace it.")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)


def normalize_row(row: dict[str, str]) -> dict[str, str]:
    return {
        "path": str(row["path"]).strip(),
        "event_label": str(row["event_label"]).strip(),
        "distance_label": canonicalize_distance_label(row.get("distance_label", "")),
        "sample_mode": str(row.get("sample_mode", "file")).strip() or "file",
        "augment_profile": str(row.get("augment_profile", "")).strip() or "base",
        "supervision_profile": str(row.get("supervision_profile", "")).strip() or "full",
    }


def top_up_train_rows(
    rows: list[dict[str, str]],
    distance_classes: list[str],
    target_per_class: int,
    seed: int,
) -> tuple[list[dict[str, str]], dict[str, object]]:
    rng = random.Random(seed)
    normalized_rows = [normalize_row(row) for row in rows]
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in normalized_rows:
        distance_label = row["distance_label"]
        if distance_label in distance_classes:
            grouped[distance_label].append(row)

    added_rows: list[dict[str, str]] = []
    source_counts = {label: len(grouped[label]) for label in distance_classes}
    duplicate_counts: dict[str, int] = {}
    for label in distance_classes:
        base_rows = grouped[label]
        if not base_rows:
            raise ValueError(f"Cannot augment Radial Threat class '{label}' because it has no source rows.")
        current_count = len(base_rows)
        if current_count >= target_per_class:
            duplicate_counts[label] = 0
            continue
        shortfall = target_per_class - current_count
        duplicate_counts[label] = shortfall
        for _ in range(shortfall):
            duplicate = dict(rng.choice(base_rows))
            duplicate["augment_profile"] = "location"
            duplicate["supervision_profile"] = "radial_only"
            added_rows.append(duplicate)

    output_rows = normalized_rows + added_rows
    output_counts = Counter(
        row["distance_label"]
        for row in output_rows
        if row["distance_label"] in distance_classes
    )
    summary = {
        "source_radial_counts": source_counts,
        "duplicate_radial_counts": duplicate_counts,
        "output_radial_counts": {label: int(output_counts[label]) for label in distance_classes},
        "source_train_rows": len(normalized_rows),
        "output_train_rows": len(output_rows),
        "added_train_rows": len(added_rows),
    }
    return output_rows, summary


def main() -> int:
    args = parse_args()
    source_root = Path(args.source_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    distance_classes = parse_label_list(args.distance_classes)
    prepare_output_root(output_root, args.overwrite)

    summary: dict[str, object] = {
        "source_root": str(source_root),
        "output_root": str(output_root),
        "target_per_class": int(args.target_per_class),
        "distance_classes": distance_classes,
        "seed": int(args.seed),
    }

    split_counts: dict[str, dict[str, int]] = {}
    for split in ("train", "val", "test"):
        source_path = source_root / f"{split}.csv"
        if not source_path.is_file():
            raise FileNotFoundError(f"Missing manifest: {source_path}")
        rows = read_manifest(source_path)
        if split == "train":
            output_rows, train_summary = top_up_train_rows(rows, distance_classes, int(args.target_per_class), int(args.seed))
            summary["train_topup"] = train_summary
        else:
            output_rows = [normalize_row(row) for row in rows]

        write_manifest(output_root / f"{split}.csv", output_rows)
        split_counts[split] = dict(
            Counter(
                row["distance_label"] or "unlabeled"
                for row in output_rows
            )
        )

    summary["distance_counts_by_split"] = split_counts
    with (output_root / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
