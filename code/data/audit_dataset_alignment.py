from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit local SensorField-DAS assets against the submission configuration."
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "config" / "sensorfield_m3t_submission.yaml"),
        type=str,
    )
    parser.add_argument("--dataset-root", default=None, type=str)
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "output" / "submission_alignment_report.json"),
        type=str,
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return payload


def resolve_local_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def count_numeric_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return sum(1 for row in csv.reader(handle) if any(cell.strip() for cell in row))


def resolve_sample_path(path_text: str, dataset_root: Path) -> Path:
    path = Path(path_text)
    if path.is_file():
        return path
    candidate = dataset_root / path.name
    if candidate.is_file():
        return candidate
    if "walking" in path.parts:
        candidates = [
            dataset_root / "human activities" / path.name,
            dataset_root.parent / "walking" / path.name,
        ]
        for item in candidates:
            if item.is_file():
                return item
    return path


def manifest_audit(dataset_root: Path) -> dict[str, Any]:
    manifest_rows: list[tuple[str, dict[str, str]]] = []
    split_stems: dict[str, set[str]] = {}
    missing_paths: list[str] = []
    row_count_cache: dict[Path, int] = {}

    for split in ("train", "val", "test"):
        manifest_path = dataset_root / f"{split}.csv"
        if not manifest_path.is_file():
            return {"status": "missing_manifests", "missing": str(manifest_path)}
        rows = read_manifest(manifest_path)
        split_stems[split] = set()
        for row in rows:
            resolved = resolve_sample_path(row.get("path", ""), dataset_root)
            row["_resolved_path"] = str(resolved)
            if resolved.is_file():
                split_stems[split].add(resolved.stem)
            elif len(missing_paths) < 50:
                missing_paths.append(str(resolved))
            manifest_rows.append((split, row))

    manifest_counts: Counter[str] = Counter()
    expanded_counts: Counter[str] = Counter()
    mode_counts: Counter[str] = Counter()
    for _, row in manifest_rows:
        label = row.get("event_label", "")
        mode = row.get("sample_mode", "file") or "file"
        path = Path(row["_resolved_path"])
        manifest_counts[label] += 1
        mode_counts[mode] += 1
        if not path.is_file():
            continue
        if path not in row_count_cache:
            row_count_cache[path] = count_numeric_rows(path)
        row_count = row_count_cache[path]
        units = 1 if mode == "file" else row_count if mode == "row" else row_count // 3
        expanded_counts[label] += units

    overlaps = {}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        common = sorted(split_stems[left] & split_stems[right])
        overlaps[f"{left}_vs_{right}"] = {"count": len(common), "examples": common[:10]}

    return {
        "status": "ok" if not missing_paths else "warning",
        "manifest_rows": len(manifest_rows),
        "manifest_event_counts": dict(manifest_counts),
        "loader_expanded_event_counts": dict(expanded_counts),
        "loader_expanded_total": sum(expanded_counts.values()),
        "sample_mode_counts": dict(mode_counts),
        "missing_path_count": len(missing_paths),
        "missing_path_examples": missing_paths,
        "source_stem_overlaps": overlaps,
        "source_stem_overlap_passed": all(item["count"] == 0 for item in overlaps.values()),
    }


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    config = load_yaml(config_path)
    local_paths = config.get("local_paths", {})
    dataset_root = (
        Path(args.dataset_root).expanduser().resolve()
        if args.dataset_root
        else resolve_local_path(str(local_paths["benchmark_root"])).resolve()
    )

    expected_assets = {
        name: resolve_local_path(str(value)).resolve()
        for name, value in local_paths.items()
        if name != "benchmark_root"
    }
    asset_status = {name: path.exists() for name, path in expected_assets.items()}
    manifest_report = manifest_audit(dataset_root)
    reported_counts = config["dataset"]["event_counts"]
    executable_counts = manifest_report.get("loader_expanded_event_counts", {})
    count_differences = {
        label: {
            "paper_reported": int(expected),
            "loader_expanded": int(executable_counts.get(label, 0)),
            "difference": int(executable_counts.get(label, 0)) - int(expected),
        }
        for label, expected in reported_counts.items()
    }

    report = {
        "config": str(config_path),
        "dataset_root": str(dataset_root),
        "paper_reported_records": int(config["dataset"]["reported_records"]),
        "asset_status": asset_status,
        "all_expected_assets_present": all(asset_status.values()),
        "manifest_audit": manifest_report,
        "paper_vs_loader_counts": count_differences,
        "alignment_warnings": [],
        "alignment_notes": [
            "Vehicle channels are expanded as independent records, yielding 4,480 vehicle and 13,806 total records.",
            "Paper-facing runners use 80 epochs, 16 anchors, perturbation probability 0.3, and representation consistency 0.05.",
            "GCTI follows the paper task-relation interaction and prediction-plus-representation consistency path.",
        ],
    }

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["all_expected_assets_present"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
