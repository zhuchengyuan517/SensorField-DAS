from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
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
        description="Build a clean MTL43 location-only manifest from train/val/test.csv."
    )
    parser.add_argument("--source-root", default=r"D:\proj 1\converted_csv\MTL43", type=str)
    parser.add_argument(
        "--output-root",
        default=(
            r"D:\proj 1\converted_csv\MTL43\_single_task_manifests"
            r"\location_only_clean_seed123"
        ),
        type=str,
    )
    parser.add_argument(
        "--distance-classes",
        default=",".join(DEFAULT_DISTANCE_CLASSES),
        type=str,
        help="Comma-separated class order to keep.",
    )
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


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["path", "event_label", "distance_label", "sample_mode"],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    source_root = Path(args.source_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    distance_classes = set(parse_label_list(args.distance_classes))
    summary: dict[str, object] = {
        "source_root": str(source_root),
        "output_root": str(output_root),
        "distance_classes": sorted(distance_classes),
        "splits": {},
    }

    for split in ("train", "val", "test"):
        input_path = source_root / f"{split}.csv"
        if not input_path.is_file():
            raise FileNotFoundError(f"Missing source manifest: {input_path}")
        rows = []
        skipped_missing = 0
        for row in read_rows(input_path):
            distance_label = canonicalize_distance_label(row.get("distance_label", ""))
            if distance_label not in distance_classes:
                continue
            source_path = Path(str(row["path"]).strip())
            if not source_path.is_file():
                skipped_missing += 1
                continue
            rows.append(
                {
                    "path": str(source_path),
                    "event_label": str(row.get("event_label", "")).strip() or "excavator",
                    "distance_label": distance_label,
                    "sample_mode": str(row.get("sample_mode", "file")).strip() or "file",
                }
            )
        write_rows(output_root / f"{split}.csv", rows)
        summary["splits"][split] = {
            "rows": len(rows),
            "distance_counts": dict(Counter(row["distance_label"] for row in rows)),
            "event_counts": dict(Counter(row["event_label"] for row in rows)),
            "skipped_missing": skipped_missing,
        }

    with (output_root / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
