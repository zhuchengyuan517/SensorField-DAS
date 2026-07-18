from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_EVENT_CLASSES = ("walking", "excavator", "driving", "background")
DEFAULT_DISTANCE_CLASSES = ("Alarm area", "Tracking area", "No-threat area")
DISTANCE_LABEL_ALIASES = {
    "5m": "Alarm area",
    "20m": "Tracking area",
    "40m": "No-threat area",
    "Alarm area": "Alarm area",
    "Tracking area": "Tracking area",
    "No-threat area": "No-threat area",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build an MTL43 manifest set that independently tops up Task 1 (event_type) "
            "and Task 2 (radial_threat) to target effective sample counts via augmentation-ready duplicates."
        )
    )
    parser.add_argument("--source-root", default=r"D:\proj 1\converted_csv\MTL43", type=str)
    parser.add_argument(
        "--output-root",
        default=r"D:\proj 1\converted_csv\MTL43\_task12_event3000_radial2000",
        type=str,
    )
    parser.add_argument("--event-target-per-class", default=3000, type=int)
    parser.add_argument("--radial-target-per-class", default=2000, type=int)
    parser.add_argument("--event-classes", default=",".join(DEFAULT_EVENT_CLASSES), type=str)
    parser.add_argument("--distance-classes", default=",".join(DEFAULT_DISTANCE_CLASSES), type=str)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_label_list(payload: str) -> list[str]:
    labels = [item.strip() for item in payload.split(",") if item.strip()]
    if not labels:
        raise ValueError("label list cannot be empty")
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


def count_csv_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return sum(1 for line in handle if line.strip())


def compute_effective_units(sample_mode: str, row_count: int) -> int:
    if sample_mode == "file":
        return 1
    if sample_mode == "row":
        return row_count
    if sample_mode == "group3":
        return row_count // 3
    raise ValueError(f"Unsupported sample_mode '{sample_mode}'")


def normalize_row(row: dict[str, str], source_root: Path, row_count_cache: dict[Path, int]) -> dict[str, object]:
    sample_path = Path(str(row["path"]).strip())
    if not sample_path.is_absolute():
        sample_path = (source_root / sample_path).resolve()
    sample_mode = str(row.get("sample_mode", "file")).strip() or "file"
    if sample_path not in row_count_cache:
        row_count_cache[sample_path] = count_csv_rows(sample_path)
    row_count = row_count_cache[sample_path]
    effective_units = compute_effective_units(sample_mode, row_count)
    return {
        "path": str(sample_path),
        "event_label": str(row["event_label"]).strip(),
        "distance_label": canonicalize_distance_label(row.get("distance_label", "")),
        "sample_mode": sample_mode,
        "augment_profile": str(row.get("augment_profile", "")).strip() or "base",
        "supervision_profile": str(row.get("supervision_profile", "")).strip() or "full",
        "_effective_units": int(effective_units),
    }


def serialize_row(row: dict[str, object]) -> dict[str, str]:
    return {
        "path": str(row["path"]),
        "event_label": str(row["event_label"]),
        "distance_label": str(row.get("distance_label", "")),
        "sample_mode": str(row.get("sample_mode", "file")),
        "augment_profile": str(row.get("augment_profile", "base")),
        "supervision_profile": str(row.get("supervision_profile", "full")),
    }


def effective_counts_by_key(rows: list[dict[str, object]], key: str, allowed: set[str] | None = None) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        value = str(row.get(key, "")).strip()
        if not value:
            continue
        if allowed is not None and value not in allowed:
            continue
        counts[value] += int(row["_effective_units"])
    return {label: int(counts[label]) for label in sorted(counts.keys())}


def top_up_event_rows(
    rows: list[dict[str, object]],
    event_classes: list[str],
    target_per_class: int,
    seed: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    rng = random.Random(seed)
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        event_label = str(row["event_label"])
        if event_label in event_classes:
            grouped[event_label].append(row)

    added_rows: list[dict[str, object]] = []
    source_counts = {
        label: sum(int(row["_effective_units"]) for row in grouped[label])
        for label in event_classes
    }
    duplicate_counts: dict[str, int] = {}
    duplicate_row_counts: dict[str, int] = {}

    for label in event_classes:
        base_rows = grouped[label]
        if not base_rows:
            raise ValueError(f"Cannot augment Task 1 class '{label}' because it has no source rows.")
        current_count = source_counts[label]
        if current_count >= target_per_class:
            duplicate_counts[label] = 0
            duplicate_row_counts[label] = 0
            continue

        shortfall = target_per_class - current_count
        added_effective = 0
        added_manifest_rows = 0
        while added_effective < shortfall:
            remaining = shortfall - added_effective
            compatible = [row for row in base_rows if int(row["_effective_units"]) <= remaining]
            chosen = dict(rng.choice(compatible if compatible else base_rows))
            chosen["augment_profile"] = "event"
            chosen["supervision_profile"] = "event_only"
            chosen["distance_label"] = ""
            added_rows.append(chosen)
            added_effective += int(chosen["_effective_units"])
            added_manifest_rows += 1

        duplicate_counts[label] = int(added_effective)
        duplicate_row_counts[label] = int(added_manifest_rows)

    output_rows = rows + added_rows
    output_counts = effective_counts_by_key(output_rows, "event_label", set(event_classes))
    summary = {
        "source_event_effective_counts": {label: int(source_counts.get(label, 0)) for label in event_classes},
        "duplicate_event_effective_counts": {label: int(duplicate_counts.get(label, 0)) for label in event_classes},
        "duplicate_event_manifest_rows": {label: int(duplicate_row_counts.get(label, 0)) for label in event_classes},
        "output_event_effective_counts": {label: int(output_counts.get(label, 0)) for label in event_classes},
        "added_event_manifest_rows": int(len(added_rows)),
    }
    return added_rows, summary


def top_up_radial_rows(
    rows: list[dict[str, object]],
    distance_classes: list[str],
    target_per_class: int,
    seed: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    rng = random.Random(seed)
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        distance_label = str(row["distance_label"]).strip()
        if distance_label in distance_classes:
            grouped[distance_label].append(row)

    added_rows: list[dict[str, object]] = []
    source_counts = {
        label: sum(int(row["_effective_units"]) for row in grouped[label])
        for label in distance_classes
    }
    duplicate_counts: dict[str, int] = {}

    for label in distance_classes:
        base_rows = grouped[label]
        if not base_rows:
            raise ValueError(f"Cannot augment Task 2 class '{label}' because it has no source rows.")
        current_count = source_counts[label]
        if current_count >= target_per_class:
            duplicate_counts[label] = 0
            continue

        shortfall = target_per_class - current_count
        added_effective = 0
        while added_effective < shortfall:
            remaining = shortfall - added_effective
            compatible = [row for row in base_rows if int(row["_effective_units"]) <= remaining]
            chosen = dict(rng.choice(compatible if compatible else base_rows))
            chosen["augment_profile"] = "location"
            chosen["supervision_profile"] = "radial_only"
            added_rows.append(chosen)
            added_effective += int(chosen["_effective_units"])

        duplicate_counts[label] = int(added_effective)

    output_rows = rows + added_rows
    output_counts = effective_counts_by_key(output_rows, "distance_label", set(distance_classes))
    summary = {
        "source_radial_effective_counts": {label: int(source_counts.get(label, 0)) for label in distance_classes},
        "duplicate_radial_effective_counts": {label: int(duplicate_counts.get(label, 0)) for label in distance_classes},
        "output_radial_effective_counts": {label: int(output_counts.get(label, 0)) for label in distance_classes},
        "added_radial_manifest_rows": int(len(added_rows)),
    }
    return added_rows, summary


def main() -> int:
    args = parse_args()
    source_root = Path(args.source_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    event_classes = parse_label_list(args.event_classes)
    distance_classes = parse_label_list(args.distance_classes)
    prepare_output_root(output_root, args.overwrite)

    summary: dict[str, object] = {
        "source_root": str(source_root),
        "output_root": str(output_root),
        "event_target_per_class": int(args.event_target_per_class),
        "radial_target_per_class": int(args.radial_target_per_class),
        "event_classes": event_classes,
        "distance_classes": distance_classes,
        "seed": int(args.seed),
    }

    row_count_cache: dict[Path, int] = {}
    split_manifest_rows: dict[str, int] = {}
    for split in ("train", "val", "test"):
        source_path = source_root / f"{split}.csv"
        if not source_path.is_file():
            raise FileNotFoundError(f"Missing manifest: {source_path}")
        rows_raw = read_manifest(source_path)
        rows = [normalize_row(row, source_root, row_count_cache) for row in rows_raw]

        if split == "train":
            event_added, event_summary = top_up_event_rows(
                rows,
                event_classes=event_classes,
                target_per_class=int(args.event_target_per_class),
                seed=int(args.seed),
            )
            radial_added, radial_summary = top_up_radial_rows(
                rows,
                distance_classes=distance_classes,
                target_per_class=int(args.radial_target_per_class),
                seed=int(args.seed) + 1,
            )
            output_rows = rows + event_added + radial_added
            summary["train_event_topup"] = event_summary
            summary["train_radial_topup"] = radial_summary
            summary["train_added_manifest_rows"] = {
                "event_only": int(len(event_added)),
                "radial_only": int(len(radial_added)),
                "total": int(len(event_added) + len(radial_added)),
            }
        else:
            output_rows = rows

        serialized_rows = [serialize_row(row) for row in output_rows]
        write_manifest(output_root / f"{split}.csv", serialized_rows)
        split_manifest_rows[split] = int(len(serialized_rows))

    summary["manifest_rows_by_split"] = split_manifest_rows
    with (output_root / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
