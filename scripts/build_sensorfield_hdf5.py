from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVENT_MAP = {"background": 0, "walking": 1, "excavator": 2, "driving": 3}
EVENT_PUBLIC_NAMES = {
    "background": "background_noise",
    "walking": "human_activity",
    "excavator": "mechanical_excavation",
    "driving": "vehicle_driving",
}
FINE_EVENT_MAP = {
    "N/A": 0,
    "walking": 1,
    "striking": 2,
    "hoeing": 3,
    "construction": 4,
    "excavation": 5,
    "cutting": 6,
    "unknown": 7,
}
DISTANCE_MAP = {"N/A": -1, "Alarm area": 0, "Tracking area": 1, "No-threat area": 2}
DISTANCE_VALUE = {"Alarm area": 5.0, "Tracking area": 20.0, "No-threat area": 40.0}
SOIL_MAP = {"land": 0, "sand": 1, "stone": 2, "unknown": 3}
BATCH_MAP = {"0715": "B001", "0716": "B002"}
SPLIT_NAMES = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the paper-aligned SensorField-DAS HDF5 release.")
    parser.add_argument(
        "--dataset-root",
        default=str(PROJECT_ROOT / "converted_csv" / "MTL43"),
        type=str,
    )
    parser.add_argument(
        "--condition-root",
        default=str(PROJECT_ROOT / "converted_csv" / "sensorfield_mtl43_condition_splits_strict"),
        type=str,
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "public_dataset_release" / "SensorField_DAS_v1.h5"),
        type=str,
    )
    parser.add_argument(
        "--private-map",
        default=str(PROJECT_ROOT / "public_dataset_release" / "private_mapping.csv"),
        type=str,
    )
    parser.add_argument("--compression-level", default=4, type=int)
    return parser.parse_args()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def resolve_sample_path(path_text: str, dataset_root: Path) -> Path:
    path = Path(path_text)
    if path.is_file():
        return path.resolve()
    candidates = []
    if path.parent.name == "walking" or "walking" in path.parts:
        candidates.extend(
            [
                dataset_root / "human activities" / path.name,
                dataset_root.parent / "walking" / path.name,
            ]
        )
    candidates.extend(
        [
            dataset_root / path.name,
            dataset_root / "background" / path.name,
            dataset_root / "driving" / path.name,
        ]
    )
    for distance in ("5m", "20m", "40m"):
        candidates.append(dataset_root / "excavator" / distance / path.name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Unable to resolve manifest sample: {path.name}")


def load_source_rows(dataset_root: Path) -> list[dict[str, Any]]:
    merged: dict[Path, dict[str, Any]] = {}
    for split in SPLIT_NAMES:
        manifest = dataset_root / f"{split}.csv"
        if not manifest.is_file():
            raise FileNotFoundError(f"Missing IID manifest: {manifest}")
        for row in read_manifest(manifest):
            path = resolve_sample_path(row["path"], dataset_root)
            if path in merged and merged[path]["iid_split"] != split:
                raise ValueError(f"Source file appears in multiple IID splits: {path.name}")
            merged[path] = {
                "path": path,
                "event_label": row["event_label"].strip(),
                "distance_label": row.get("distance_label", "").strip(),
                "iid_split": split,
            }
    event_order = {name: index for index, name in enumerate(EVENT_MAP)}
    return sorted(merged.values(), key=lambda row: (event_order[row["event_label"]], row["path"].name))


def matrix_shape(path: Path) -> tuple[int, int]:
    rows = 0
    columns = 0
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.reader(handle):
            if not any(cell.strip() for cell in row):
                continue
            rows += 1
            if columns == 0:
                columns = len(row)
            elif len(row) != columns:
                raise ValueError(f"Ragged CSV matrix: {path.name}")
    if rows == 0 or columns == 0:
        raise ValueError(f"Empty CSV matrix: {path.name}")
    return rows, columns


def load_numeric_matrix(
    path: Path, expected_shape: tuple[int, int]
) -> tuple[np.ndarray, float, list[str], str]:
    raw = path.read_bytes()
    file_hash = hashlib.sha256(raw).hexdigest()
    text = raw.decode("utf-8-sig")
    flattened_text = text.replace("\r", "").replace("\n", ",")
    numeric = np.fromstring(flattened_text, dtype=np.float32, sep=",")
    if numeric.size == expected_shape[0] * expected_shape[1]:
        numeric = numeric.reshape(expected_shape)
    else:
        # Curated release files are numeric-only; retain a tolerant fallback for anomalies.
        frame = pd.read_csv(path, header=None, encoding="utf-8-sig", low_memory=False)
        numeric = frame.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32, copy=True)
    numeric[np.isinf(numeric)] = np.nan
    invalid = ~np.isfinite(numeric)
    nan_ratio = float(invalid.mean()) if invalid.size else 0.0
    warnings = []
    if invalid.any():
        row_means = np.nanmean(numeric, axis=1)
        row_means = np.where(np.isfinite(row_means), row_means, 0.0).astype(np.float32)
        row_indices, column_indices = np.where(invalid)
        numeric[row_indices, column_indices] = row_means[row_indices]
        warnings.append("nan_or_inf_filled")
    return numeric, nan_ratio, warnings, file_hash


def parse_batch(filename: str) -> str:
    return next((public for private, public in BATCH_MAP.items() if private in filename), "B003")


def parse_soil(filename: str) -> str:
    lowered = filename.lower()
    matches = []
    if "land" in lowered:
        matches.append("land")
    if "sand" in lowered:
        matches.append("sand")
    if "shizi" in lowered or "stone" in lowered or "石子" in filename:
        matches.append("stone")
    return matches[0] if len(matches) == 1 else "unknown"


def parse_fine_event(event_label: str, filename: str) -> str:
    lowered = filename.lower()
    if event_label == "walking":
        if "敲" in filename or "strik" in lowered or "knock" in lowered:
            return "striking"
        if "锄" in filename or "hoe" in lowered or "chutou" in lowered:
            return "hoeing"
        if "行走" in filename or "walk" in lowered:
            return "walking"
        return "unknown"
    if event_label == "excavator":
        if "切削" in filename or "cut" in lowered:
            return "cutting"
        if "挖掘" in filename or "开挖" in filename or "excavat" in lowered or "dig" in lowered:
            return "excavation"
        if "施工" in filename or "construct" in lowered:
            return "construction"
        return "unknown"
    return "N/A"


def create_compressed_dataset(
    group: h5py.Group,
    name: str,
    values: Any,
    compression: int,
    dtype: Any | None = None,
) -> h5py.Dataset:
    array = np.asarray(values, dtype=dtype)
    return group.create_dataset(
        name,
        data=array,
        compression="gzip",
        compression_opts=compression,
        shuffle=True,
    )


def load_condition_assignments(condition_root: Path, dataset_root: Path) -> dict[str, dict[Path, str]]:
    assignments: dict[str, dict[Path, str]] = {}
    for protocol in ("region", "soil", "acquisition"):
        protocol_map: dict[Path, str] = {}
        protocol_dir = condition_root / f"{protocol}_level"
        for split in SPLIT_NAMES:
            manifest = protocol_dir / f"{split}.csv"
            if not manifest.is_file():
                raise FileNotFoundError(f"Missing condition manifest: {manifest}")
            for row in read_manifest(manifest):
                path = resolve_sample_path(row["path"], dataset_root)
                if path in protocol_map and protocol_map[path] != split:
                    raise ValueError(f"Condition split overlap for {path.name}")
                protocol_map[path] = split
        assignments[protocol] = protocol_map
    return assignments


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    condition_root = Path(args.condition_root).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    private_map_path = Path(args.private_map).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    private_map_path.parent.mkdir(parents=True, exist_ok=True)

    source_rows = load_source_rows(dataset_root)
    conditions = load_condition_assignments(condition_root, dataset_root)
    source_shapes = {row["path"]: matrix_shape(row["path"]) for row in source_rows}
    sample_count = sum(source_shapes[row["path"]][0] if row["event_label"] == "driving" else 1 for row in source_rows)
    flat_length = sum(rows * columns for rows, columns in source_shapes.values())
    logging.info("Preparing %s records and %s float32 values", sample_count, flat_length)
    if sample_count != 13806:
        raise ValueError(f"Expected 13,806 records, found {sample_count}")

    string_dtype = h5py.string_dtype(encoding="utf-8")
    root_attrs = {
        "dataset_name": "SensorField-DAS",
        "version": "v1.0",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "dataset_unit": "one localized event field, except vehicle channels are independent records",
        "observation_window_seconds": 5,
        "standardized_sampling_rate_hz": 2000,
        "anonymization_level": "public_release",
        "source_description": "Cross-regional distributed acoustic sensing records under multiple acquisition and soil conditions.",
        "public_release_note": (
            "Sensitive source paths, dates, locations, project identifiers, and physical zone identifiers "
            "are removed or anonymized."
        ),
    }

    split_indices = {"iid": {name: [] for name in SPLIT_NAMES}}
    split_indices.update({protocol: {name: [] for name in SPLIT_NAMES} for protocol in conditions})
    event_counts: Counter[str] = Counter()
    fine_counts: Counter[str] = Counter()
    soil_counts: Counter[str] = Counter()
    distance_counts: Counter[str] = Counter()
    warnings_count = 0
    private_rows = []
    segment_registry: dict[tuple[str, str, int], int] = {}

    signal_index_values = np.empty((sample_count, 2), dtype=np.int64)
    signal_shape_values = np.empty((sample_count, 2), dtype=np.int32)
    vector_values = {
        "event_type": np.empty(sample_count, dtype=np.int16),
        "fine_event": np.empty(sample_count, dtype=np.int16),
        "distance_label": np.empty(sample_count, dtype=np.int16),
        "distance_value_m": np.empty(sample_count, dtype=np.float32),
        "soil_condition": np.empty(sample_count, dtype=np.int16),
        "segment_id": np.empty(sample_count, dtype=np.int32),
        "sampling_rate_hz": np.full(sample_count, 2000.0, dtype=np.float32),
        "is_background": np.empty(sample_count, dtype=np.bool_),
        "has_distance_label": np.empty(sample_count, dtype=np.bool_),
    }
    meta_values = {
        name: [""] * sample_count
        for name in (
            "sample_id",
            "public_file_id",
            "file_sha256",
            "source_batch_id",
            "original_filename_hash",
            "source_group_hash",
            "parse_status",
            "parse_warning",
        )
    }
    quality_values = {
        "signal_length": np.empty(sample_count, dtype=np.int32),
        "nan_ratio": np.empty(sample_count, dtype=np.float32),
        "mean": np.empty(sample_count, dtype=np.float32),
        "std": np.empty(sample_count, dtype=np.float32),
        "rms": np.empty(sample_count, dtype=np.float32),
        "max_abs": np.empty(sample_count, dtype=np.float32),
        "is_valid": np.empty(sample_count, dtype=np.bool_),
    }

    with h5py.File(output_path, "w") as handle:
        for key, value in root_attrs.items():
            handle.attrs[key] = value
        data = handle.create_group("data")
        labels = handle.create_group("labels")
        meta = handle.create_group("meta")
        label_maps = meta.create_group("label_maps")
        quality = handle.create_group("quality")
        splits = handle.create_group("splits")

        signals_flat = data.create_dataset(
            "signals_flat",
            shape=(flat_length,),
            dtype=np.float32,
            compression="gzip",
            compression_opts=args.compression_level,
            shuffle=True,
            chunks=(min(flat_length, 1_000_000),),
        )
        sample_index = 0
        flat_offset = 0
        signal_buffer: list[np.ndarray] = []
        signal_buffer_size = 0
        signal_buffer_start = 0

        def flush_signal_buffer() -> None:
            nonlocal signal_buffer, signal_buffer_size, signal_buffer_start
            if not signal_buffer:
                return
            values = np.concatenate(signal_buffer)
            signals_flat[signal_buffer_start : signal_buffer_start + values.size] = values
            signal_buffer_start += values.size
            signal_buffer = []
            signal_buffer_size = 0

        for source_index, source in enumerate(source_rows, start=1):
            if source_index == 1 or source_index % 250 == 0:
                logging.info("Reading source file %s / %s", source_index, len(source_rows))
            path = source["path"]
            matrix, nan_ratio, parse_warnings, file_hash = load_numeric_matrix(path, source_shapes[path])
            event_label = source["event_label"]
            filename_hash = sha256_text(path.name)
            source_group_hash = sha256_text(path.stem)
            batch_id = parse_batch(path.name)
            soil = parse_soil(path.name)
            fine_event = parse_fine_event(event_label, path.name)
            distance_label = source["distance_label"] or "N/A"
            distance_code = DISTANCE_MAP.get(distance_label, -1)
            distance_value = DISTANCE_VALUE.get(distance_label, math.nan)
            segment_key = (batch_id, soil, 2000)
            if segment_key not in segment_registry:
                segment_registry[segment_key] = len(segment_registry)
            channel_signals = [matrix[row_index][:, None] for row_index in range(matrix.shape[0])]
            if event_label != "driving":
                channel_signals = [matrix.T]

            for channel_index, signal in enumerate(channel_signals):
                flat = np.asarray(signal, dtype=np.float32).reshape(-1)
                end = flat_offset + flat.size
                signal_buffer.append(flat)
                signal_buffer_size += flat.size
                if signal_buffer_size >= 5_000_000:
                    flush_signal_buffer()
                signal_index_values[sample_index] = (flat_offset, flat.size)
                signal_shape_values[sample_index] = signal.shape

                vector_values["event_type"][sample_index] = EVENT_MAP[event_label]
                vector_values["fine_event"][sample_index] = FINE_EVENT_MAP[fine_event]
                vector_values["distance_label"][sample_index] = distance_code
                vector_values["distance_value_m"][sample_index] = distance_value
                vector_values["soil_condition"][sample_index] = SOIL_MAP[soil]
                vector_values["segment_id"][sample_index] = segment_registry[segment_key]
                vector_values["is_background"][sample_index] = event_label == "background"
                vector_values["has_distance_label"][sample_index] = distance_code >= 0

                sample_id = f"SAMPLE_{sample_index + 1:08d}"
                meta_values["sample_id"][sample_index] = sample_id
                meta_values["public_file_id"][sample_index] = f"PUB_{sample_index + 1:08d}"
                meta_values["file_sha256"][sample_index] = file_hash
                meta_values["source_batch_id"][sample_index] = batch_id
                meta_values["original_filename_hash"][sample_index] = filename_hash
                meta_values["source_group_hash"][sample_index] = source_group_hash
                meta_values["parse_status"][sample_index] = "parsed"
                meta_values["parse_warning"][sample_index] = ";".join(parse_warnings)

                quality_values["signal_length"][sample_index] = flat.size
                quality_values["nan_ratio"][sample_index] = nan_ratio
                quality_values["mean"][sample_index] = float(np.mean(flat))
                quality_values["std"][sample_index] = float(np.std(flat))
                quality_values["rms"][sample_index] = float(np.sqrt(np.mean(np.square(flat))))
                quality_values["max_abs"][sample_index] = float(np.max(np.abs(flat)))
                quality_values["is_valid"][sample_index] = nan_ratio <= 0.2

                split_indices["iid"][source["iid_split"]].append(sample_index)
                for protocol, protocol_map in conditions.items():
                    split = protocol_map.get(path)
                    if split is None:
                        raise ValueError(f"Missing {protocol} assignment for {path.name}")
                    split_indices[protocol][split].append(sample_index)

                event_counts[EVENT_PUBLIC_NAMES[event_label]] += 1
                fine_counts[fine_event] += 1
                soil_counts[soil] += 1
                distance_counts[distance_label] += 1
                warnings_count += bool(parse_warnings)
                private_rows.append(
                    {
                        "original_path": str(path),
                        "original_filename": path.name,
                        "channel_index": channel_index if event_label == "driving" else "",
                        "sample_id": sample_id,
                        "source_group_hash": source_group_hash,
                        "file_sha256": file_hash,
                    }
                )
                sample_index += 1
                flat_offset = end

        flush_signal_buffer()

        if sample_index != sample_count or flat_offset != flat_length:
            raise RuntimeError(
                f"Write count mismatch: samples={sample_index}/{sample_count}, values={flat_offset}/{flat_length}"
            )

        create_compressed_dataset(data, "signal_index", signal_index_values, args.compression_level)
        create_compressed_dataset(data, "signal_shape", signal_shape_values, args.compression_level)
        for name, values in vector_values.items():
            create_compressed_dataset(labels, name, values, args.compression_level)
        for name, values in meta_values.items():
            create_compressed_dataset(meta, name, values, args.compression_level, string_dtype)
        for name, values in quality_values.items():
            create_compressed_dataset(quality, name, values, args.compression_level)

        maps = {
            "event_type_json": {EVENT_PUBLIC_NAMES[key]: value for key, value in EVENT_MAP.items()},
            "fine_event_json": FINE_EVENT_MAP,
            "distance_label_json": DISTANCE_MAP,
            "soil_condition_json": SOIL_MAP,
            "segment_map_json": {
                f"SEG_{value + 1:04d}": value for _, value in sorted(segment_registry.items(), key=lambda item: item[1])
            },
        }
        for name, payload in maps.items():
            label_maps.create_dataset(name, data=json.dumps(payload, ensure_ascii=False, sort_keys=True), dtype=string_dtype)
        for protocol, protocol_splits in split_indices.items():
            group = splits.create_group(protocol)
            for split, indices in protocol_splits.items():
                values = np.asarray(indices, dtype=np.int64)
                group.create_dataset(
                    split,
                    data=values,
                    compression="gzip" if values.size else None,
                    compression_opts=args.compression_level if values.size else None,
                    shuffle=bool(values.size),
                )

    with private_map_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(private_rows[0]))
        writer.writeheader()
        writer.writerows(private_rows)

    statistics = {
        "dataset_name": "SensorField-DAS",
        "curated_source_files": len(source_rows),
        "total_records": sample_count,
        "event_type_counts": dict(event_counts),
        "fine_event_counts": dict(fine_counts),
        "soil_condition_counts": dict(soil_counts),
        "distance_label_counts": dict(distance_counts),
        "sampling_rate_hz_counts": {"2000": sample_count},
        "segment_id_counts": {
            f"SEG_{segment_id + 1:04d}": int(count)
            for segment_id, count in zip(*np.unique(vector_values["segment_id"], return_counts=True))
        },
        "invalid_record_count": int((~quality_values["is_valid"]).sum()),
        "parsing_warning_count": warnings_count,
        "iid_split_counts": {name: len(values) for name, values in split_indices["iid"].items()},
        "condition_split_counts": {
            protocol: {name: len(values) for name, values in protocol_splits.items()}
            for protocol, protocol_splits in split_indices.items()
            if protocol != "iid"
        },
    }
    stats_path = output_path.parent / "dataset_statistics.json"
    stats_path.write_text(json.dumps(statistics, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path = output_path.parent / "build_report.md"
    report_path.write_text(
        "# SensorField-DAS Build Report\n\n"
        f"- Output: `{output_path.name}`\n"
        f"- Curated source files: {len(source_rows):,}\n"
        f"- Total records: {sample_count:,}\n"
        f"- Flattened float32 values: {flat_length:,}\n"
        f"- Event counts: `{json.dumps(dict(event_counts), ensure_ascii=False)}`\n"
        f"- IID split counts: `{json.dumps(statistics['iid_split_counts'])}`\n"
        f"- Condition split counts: `{json.dumps(statistics['condition_split_counts'])}`\n"
        f"- Invalid records: {statistics['invalid_record_count']}\n"
        f"- Parsing warnings: {warnings_count}\n"
        "- Vehicle unit: one channel per record.\n"
        "- Selection policy: only source files listed by the canonical MTL43 manifests were packaged; "
        "unlisted or nonconforming files were excluded.\n"
        "- Original signal files were read only and were not modified.\n"
        "- Public HDF5 groups: `/data`, `/labels`, `/meta`, `/quality`, and `/splits`.\n"
        "- The local `private_mapping.csv` is excluded from version control and must not be placed in a public archive.\n",
        encoding="utf-8",
    )
    logging.info("Created %s", output_path)
    logging.info("Created %s", stats_path)
    logging.info("Created private mapping %s", private_map_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
