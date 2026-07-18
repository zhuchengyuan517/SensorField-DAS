from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import h5py
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the PipeDAS public HDF5 file.")
    parser.add_argument("--h5", required=True, help="Path to the HDF5 file to validate.")
    return parser.parse_args()


def read_json_string(dataset: h5py.Dataset) -> dict:
    raw = dataset[()]
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return json.loads(raw)


def collect_string_values(handle: h5py.File) -> list[str]:
    values: list[str] = []
    for key, value in handle.attrs.items():
        if isinstance(value, str):
            values.append(value)
    meta_group = handle["/meta"]
    for dataset_name in meta_group.keys():
        dataset = meta_group[dataset_name]
        if isinstance(dataset, h5py.Group):
            continue
        if dataset.dtype.kind in {"O", "S", "U"}:
            data = dataset[()]
            if dataset.ndim == 0:
                values.append(data.decode("utf-8") if isinstance(data, bytes) else str(data))
            else:
                for item in data[: min(len(data), 200)]:
                    values.append(item.decode("utf-8") if isinstance(item, bytes) else str(item))
    return values


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    h5_path = Path(args.h5)
    errors: list[str] = []
    warnings: list[str] = []

    with h5py.File(h5_path, "r") as handle:
        sample_count = len(handle["/meta/sample_id"])
        logging.info("Validating %s with %s samples", h5_path, sample_count)

        expected_one_dim = [
            "/labels/event_type",
            "/labels/fine_event",
            "/labels/distance_label",
            "/labels/distance_value_m",
            "/labels/soil_condition",
            "/labels/segment_id",
            "/labels/sampling_rate_hz",
            "/labels/is_background",
            "/labels/has_distance_label",
            "/meta/public_file_id",
            "/meta/file_sha256",
            "/meta/source_batch_id",
            "/meta/original_filename_hash",
            "/meta/parse_status",
            "/meta/parse_warning",
            "/quality/signal_length",
            "/quality/nan_ratio",
            "/quality/mean",
            "/quality/std",
            "/quality/rms",
            "/quality/max_abs",
            "/quality/is_valid",
        ]
        for dataset_path in expected_one_dim:
            length = len(handle[dataset_path])
            if length != sample_count:
                errors.append(f"{dataset_path} length mismatch: {length} != {sample_count}")

        signal_index = handle["/data/signal_index"][:]
        signals_flat_length = len(handle["/data/signals_flat"])
        for row_index, (start, length) in enumerate(signal_index):
            if start < 0 or length < 0 or start + length > signals_flat_length:
                errors.append(
                    f"signal_index out of bounds at row {row_index}: start={start} length={length} flat={signals_flat_length}"
                )
                break

        label_maps = {
            "event_type": read_json_string(handle["/meta/label_maps/event_type_json"]),
            "fine_event": read_json_string(handle["/meta/label_maps/fine_event_json"]),
            "distance_label": read_json_string(handle["/meta/label_maps/distance_label_json"]),
            "soil_condition": read_json_string(handle["/meta/label_maps/soil_condition_json"]),
            "segment_map": read_json_string(handle["/meta/label_maps/segment_map_json"]),
        }
        for map_name, payload in label_maps.items():
            if not payload:
                errors.append(f"label map {map_name} is empty")

        splits_group = handle["/splits"]

        def validate_split_node(node: h5py.Group | h5py.Dataset, prefix: str) -> None:
            if isinstance(node, h5py.Dataset):
                indices = node[:]
                if indices.size == 0:
                    return
                if np.min(indices) < 0 or np.max(indices) >= sample_count:
                    errors.append(f"split index out of range in {prefix}")
                return
            for child_name in node.keys():
                validate_split_node(node[child_name], f"{prefix}/{child_name}")

        validate_split_node(splits_group, "/splits")

        sensitive_patterns = [
            re.compile(r"0715|0716"),
            re.compile(r"converted_csv", re.IGNORECASE),
            re.compile(r"proj 1", re.IGNORECASE),
            re.compile(r"[A-Z]:\\"),
            re.compile(r"(?i)dn\d+"),
            re.compile(r"(?i)\bd\d{2,4}\b"),
        ]
        for value in collect_string_values(handle):
            for pattern in sensitive_patterns:
                if pattern.search(value):
                    errors.append(f"sensitive string leakage detected: {pattern.pattern}")
                    break
            if errors:
                break

    if errors:
        logging.error("Validation failed with %s errors", len(errors))
        for error in errors:
            logging.error(error)
        return 1

    if warnings:
        for warning in warnings:
            logging.warning(warning)
    logging.info("Validation succeeded for %s", h5_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
