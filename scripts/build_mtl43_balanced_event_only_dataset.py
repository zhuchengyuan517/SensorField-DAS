from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_CLASSES = ("walking", "excavator", "driving", "background")
DEFAULT_TARGET_OVERRIDES = {"driving": 1350}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a roughly balanced MTL43 event-only manifest set using effective sample counts. "
            "For example, sample_mode=group3 contributes three effective samples per CSV."
        )
    )
    parser.add_argument(
        "--source-root",
        default=r"D:\proj 1\converted_csv\MTL43",
        type=str,
        help="Directory containing source train/val/test manifests.",
    )
    parser.add_argument(
        "--output-root",
        default=r"D:\proj 1\converted_csv\MTL43\_single_task_manifests\event_only_balanced1500_drive1350",
        type=str,
        help="Directory to write balanced train/val/test manifests.",
    )
    parser.add_argument(
        "--target-per-class",
        default=1500,
        type=int,
        help="Default effective sample target for classes with enough samples.",
    )
    parser.add_argument(
        "--classes",
        default="walking,excavator,driving,background",
        type=str,
        help="Comma-separated class order.",
    )
    parser.add_argument(
        "--target-overrides",
        default="driving=1350",
        type=str,
        help="Optional per-class effective sample targets, e.g. driving=1350.",
    )
    parser.add_argument(
        "--upsample-shortfall-classes",
        default="driving",
        type=str,
        help="Classes allowed to duplicate training rows to fill small effective-count shortfalls.",
    )
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_label_list(payload: str) -> list[str]:
    labels = [item.strip() for item in payload.split(",") if item.strip()]
    if not labels:
        raise ValueError("classes cannot be empty")
    return labels


def parse_target_overrides(payload: str) -> dict[str, int]:
    overrides: dict[str, int] = {}
    if not payload.strip():
        return overrides
    for item in payload.split(","):
        chunk = item.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise ValueError(f"Invalid target override '{chunk}'. Expected label=count.")
        label, value = chunk.split("=", 1)
        label = label.strip()
        value = value.strip()
        if not label or not value:
            raise ValueError(f"Invalid target override '{chunk}'. Expected label=count.")
        overrides[label] = int(value)
    return overrides


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def prepare_output_dir(output_root: Path, overwrite: bool) -> None:
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {output_root}. Use --overwrite to replace it.")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)


def resolve_manifest_paths(source_root: Path) -> dict[str, Path]:
    manifest_paths = {
        split: source_root / f"{split}.csv"
        for split in ("train", "val", "test")
    }
    missing = [str(path) for path in manifest_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Could not find train/val/test manifests under source-root. Missing: "
            + ", ".join(missing)
        )
    return manifest_paths


def compute_split_targets(split_sizes: dict[str, int], target_total: int) -> dict[str, int]:
    total = sum(split_sizes.values())
    if total <= 0:
        return {split: 0 for split in split_sizes}

    raw_targets = {
        split: (target_total * split_sizes[split]) / total
        for split in split_sizes
    }
    floor_targets = {split: int(value) for split, value in raw_targets.items()}
    assigned = sum(floor_targets.values())
    remainder = max(target_total - assigned, 0)
    ranked_splits = sorted(
        split_sizes.keys(),
        key=lambda split: (raw_targets[split] - floor_targets[split]),
        reverse=True,
    )
    for split in ranked_splits[:remainder]:
        floor_targets[split] += 1
    return floor_targets


def count_csv_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return sum(1 for _ in csv.reader(handle))


def compute_effective_units(sample_mode: str, row_count: int) -> int:
    if sample_mode == "file":
        return 1
    if sample_mode == "row":
        return row_count
    if sample_mode == "group3":
        return row_count // 3
    raise ValueError(f"Unsupported sample_mode '{sample_mode}'")


def sanitize_row(row: dict[str, object]) -> dict[str, str]:
    return {
        "path": str(row["path"]),
        "event_label": str(row["event_label"]),
        "distance_label": str(row.get("distance_label", "")),
        "sample_mode": str(row.get("sample_mode", "file")),
    }


def duplicate_train_rows(
    selected_train_rows: list[dict[str, object]],
    shortfall: int,
    rng: random.Random,
) -> tuple[list[dict[str, object]], int]:
    duplicates: list[dict[str, object]] = []
    remaining = shortfall
    while remaining > 0:
        compatible = [
            row for row in selected_train_rows
            if int(row["_effective_units"]) <= remaining
        ]
        if not compatible:
            break
        duplicate = dict(rng.choice(compatible))
        duplicates.append(duplicate)
        remaining -= int(duplicate["_effective_units"])
    return duplicates, remaining


def build_balanced_manifests(
    classes: list[str],
    source_root: Path,
    output_root: Path,
    target_per_class: int,
    target_overrides: dict[str, int],
    upsample_shortfall_classes: set[str],
    seed: int,
) -> dict[str, object]:
    manifest_paths = resolve_manifest_paths(source_root)
    source_rows: dict[str, list[dict[str, str]]] = {
        split: read_manifest(path)
        for split, path in manifest_paths.items()
    }

    grouped: dict[str, dict[str, list[dict[str, object]]]] = {
        label: {split: [] for split in ("train", "val", "test")}
        for label in classes
    }
    row_count_cache: dict[Path, int] = {}
    for split, rows in source_rows.items():
        for row in rows:
            label = row["event_label"].strip()
            if label not in grouped:
                continue
            normalized_row: dict[str, object] = dict(row)
            sample_path = Path(str(normalized_row["path"]))
            if not sample_path.is_absolute():
                sample_path = (source_root / sample_path).resolve()
            normalized_row["path"] = sample_path
            # Event-only recognition ignores auxiliary distance/location text labels.
            normalized_row["distance_label"] = ""
            normalized_row["sample_mode"] = str(normalized_row.get("sample_mode", "file")).strip() or "file"
            grouped[label][split].append(normalized_row)

    rng = random.Random(seed)
    output_rows: dict[str, list[dict[str, str]]] = {split: [] for split in ("train", "val", "test")}
    summary: dict[str, object] = {
        "classes": classes,
        "target_per_class": int(target_per_class),
        "target_overrides": {label: int(value) for label, value in target_overrides.items()},
        "upsample_shortfall_classes": sorted(upsample_shortfall_classes),
        "seed": int(seed),
        "source_file_counts": {},
        "source_effective_counts": {},
        "missing_file_counts": {},
        "selected_file_counts": {},
        "selected_effective_counts": {},
        "duplicate_file_counts": {},
        "duplicate_effective_counts": {},
        "effective_target_by_class": {},
        "effective_shortfall_after_fill": {},
    }

    for label in classes:
        label_source = grouped[label]
        missing_counter: dict[str, int] = {}
        filtered_label_source: dict[str, list[dict[str, object]]] = {}
        effective_unit_sizes: set[int] = set()

        for split, rows in label_source.items():
            kept_rows: list[dict[str, object]] = []
            missing = 0
            for row in rows:
                sample_path = Path(str(row["path"]))
                if not sample_path.is_file():
                    missing += 1
                    continue
                if sample_path not in row_count_cache:
                    row_count_cache[sample_path] = count_csv_rows(sample_path)
                row_count = row_count_cache[sample_path]
                sample_mode = str(row.get("sample_mode", "file")).strip() or "file"
                effective_units = compute_effective_units(sample_mode=sample_mode, row_count=row_count)
                if effective_units <= 0:
                    continue
                normalized = dict(row)
                normalized["_row_count"] = row_count
                normalized["_effective_units"] = effective_units
                kept_rows.append(normalized)
                effective_unit_sizes.add(effective_units)
            filtered_label_source[split] = kept_rows
            missing_counter[split] = missing

        label_source = filtered_label_source
        source_file_counts = {split: len(rows) for split, rows in label_source.items()}
        source_effective_counts = {
            split: sum(int(row["_effective_units"]) for row in rows)
            for split, rows in label_source.items()
        }
        target_total = int(target_overrides.get(label, target_per_class))
        total_available_effective = sum(source_effective_counts.values())

        summary["source_file_counts"][label] = source_file_counts
        summary["source_effective_counts"][label] = source_effective_counts
        summary["missing_file_counts"][label] = missing_counter
        summary["selected_file_counts"][label] = {}
        summary["selected_effective_counts"][label] = {}
        summary["duplicate_file_counts"][label] = {}
        summary["duplicate_effective_counts"][label] = {}
        summary["effective_target_by_class"][label] = target_total

        selected_by_split: dict[str, list[dict[str, object]]] = {split: [] for split in ("train", "val", "test")}
        duplicate_by_split: dict[str, list[dict[str, object]]] = {split: [] for split in ("train", "val", "test")}

        if total_available_effective >= target_total:
            if len(effective_unit_sizes) != 1:
                raise ValueError(
                    f"Class '{label}' mixes effective unit sizes {sorted(effective_unit_sizes)}; "
                    "this builder expects one effective-unit size per class."
                )
            unit_size = next(iter(effective_unit_sizes))
            target_file_total = target_total // unit_size
            split_file_targets = compute_split_targets(
                split_sizes=source_file_counts,
                target_total=target_file_total,
            )
            for split, rows in label_source.items():
                shuffled = list(rows)
                rng.shuffle(shuffled)
                keep = min(split_file_targets[split], len(shuffled))
                selected_by_split[split] = shuffled[:keep]
            remaining_shortfall = max(
                target_total - sum(
                    int(row["_effective_units"])
                    for split_rows in selected_by_split.values()
                    for row in split_rows
                ),
                0,
            )
        else:
            for split, rows in label_source.items():
                selected_by_split[split] = list(rows)
            remaining_shortfall = target_total - total_available_effective
            if remaining_shortfall > 0 and label in upsample_shortfall_classes:
                duplicates, remaining_shortfall = duplicate_train_rows(
                    selected_train_rows=selected_by_split["train"],
                    shortfall=remaining_shortfall,
                    rng=rng,
                )
                duplicate_by_split["train"].extend(duplicates)

        summary["effective_shortfall_after_fill"][label] = int(max(remaining_shortfall, 0))

        for split in ("train", "val", "test"):
            selected_rows = selected_by_split[split] + duplicate_by_split[split]
            output_rows[split].extend(sanitize_row(row) for row in selected_rows)
            summary["selected_file_counts"][label][split] = len(selected_rows)
            summary["selected_effective_counts"][label][split] = sum(
                int(row["_effective_units"]) for row in selected_rows
            )
            summary["duplicate_file_counts"][label][split] = len(duplicate_by_split[split])
            summary["duplicate_effective_counts"][label][split] = sum(
                int(row["_effective_units"]) for row in duplicate_by_split[split]
            )

    for split in output_rows:
        rng.shuffle(output_rows[split])
        write_manifest(output_root / f"{split}.csv", output_rows[split])

    split_file_counts = {}
    split_effective_counts = {}
    for split, rows in output_rows.items():
        file_counts = Counter(row["event_label"] for row in rows)
        effective_counts = defaultdict(int)
        for row in rows:
            sample_path = Path(row["path"])
            row_count = row_count_cache[sample_path]
            effective_counts[row["event_label"]] += compute_effective_units(
                sample_mode=row.get("sample_mode", "file"),
                row_count=row_count,
            )
        split_file_counts[split] = dict(sorted(file_counts.items()))
        split_effective_counts[split] = dict(sorted(effective_counts.items()))

    summary["split_file_counts"] = split_file_counts
    summary["split_effective_counts"] = split_effective_counts
    summary["split_sizes"] = {split: len(rows) for split, rows in output_rows.items()}
    return summary


def main() -> None:
    args = parse_args()
    classes = parse_label_list(args.classes)
    target_overrides = DEFAULT_TARGET_OVERRIDES | parse_target_overrides(args.target_overrides)
    upsample_shortfall_classes = set(parse_label_list(args.upsample_shortfall_classes))
    source_root = Path(args.source_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"source-root does not exist: {source_root}")
    prepare_output_dir(output_root, overwrite=args.overwrite)

    summary = build_balanced_manifests(
        classes=classes,
        source_root=source_root,
        output_root=output_root,
        target_per_class=args.target_per_class,
        target_overrides=target_overrides,
        upsample_shortfall_classes=upsample_shortfall_classes,
        seed=args.seed,
    )
    summary_path = output_root / "balanced_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    with (output_root / "balanced_summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["split", "event_label", "file_count", "effective_count"])
        for split, counts in summary["split_file_counts"].items():
            for label in classes:
                writer.writerow(
                    [
                        split,
                        label,
                        counts.get(label, 0),
                        summary["split_effective_counts"][split].get(label, 0),
                    ]
                )

    print(f"Balanced manifests written to: {output_root}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
