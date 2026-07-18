from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.anonymizer import (
    SegmentIdRegistry,
    build_private_mapping_row,
    build_public_file_id,
    sha256_file,
    sha256_text,
)
from src.hdf5_writer import HDF5DatasetWriter
from src.label_parser import LabelParser
from src.split_builder import (
    build_cross_segment_folds,
    build_random_split,
    build_segment_holdout_split,
)
from src.stats_report import (
    build_statistics_payload,
    write_build_report,
    write_dataset_card,
    write_dataset_statistics,
)
from src.zone_extractor import (
    extract_window_time_major,
    generate_background_windows,
    resolve_window_rows,
    select_event_window,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the PipeDAS public HDF5 release.")
    parser.add_argument("--input", required=True, help="Input root that contains converted CSV batches.")
    parser.add_argument("--output", required=True, help="Output HDF5 path.")
    parser.add_argument("--config", required=True, help="YAML label config path.")
    parser.add_argument("--private-map", required=True, help="Private local mapping CSV path.")
    return parser.parse_args()


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def ensure_numeric_label(label: str, registry: Dict[str, int]) -> int:
    if label not in registry:
        registry[label] = len(registry)
    return registry[label]


def load_numeric_csv(path: Path) -> tuple[np.ndarray, np.ndarray, List[str]]:
    warnings: List[str] = []
    raw_df = pd.read_csv(path, header=None, dtype=str, encoding="utf-8", low_memory=False)
    numeric_df = raw_df.apply(pd.to_numeric, errors="coerce")
    numeric_df = numeric_df.loc[:, numeric_df.notna().any(axis=0)]
    if numeric_df.empty:
        raise ValueError("no_numeric_columns")

    raw_signal = numeric_df.to_numpy(dtype=np.float32)
    if raw_signal.ndim == 1:
        raw_signal = raw_signal.reshape(-1, 1)
    if raw_signal.shape[1] == 0:
        raise ValueError("empty_numeric_matrix")

    raw_signal = np.where(np.isinf(raw_signal), np.nan, raw_signal).astype(np.float32)
    invalid_mask = ~np.isfinite(raw_signal)
    if invalid_mask.any():
        col_means = np.nanmean(raw_signal, axis=0)
        col_means = np.where(np.isfinite(col_means), col_means, 0.0).astype(np.float32)
        row_idx, col_idx = np.where(~np.isfinite(raw_signal))
        raw_signal[row_idx, col_idx] = col_means[col_idx]
        warnings.append("nan_or_inf_filled")
    return raw_signal, invalid_mask.astype(bool), warnings


def compute_quality(signal: np.ndarray, invalid_mask: np.ndarray, nan_ratio_threshold: float) -> Dict[str, float]:
    flat = signal.reshape(-1).astype(np.float32)
    nan_ratio = float(invalid_mask.mean()) if invalid_mask.size else 0.0
    return {
        "signal_length": int(flat.size),
        "nan_ratio": nan_ratio,
        "mean": float(np.mean(flat)),
        "std": float(np.std(flat)),
        "rms": float(np.sqrt(np.mean(np.square(flat)))),
        "max_abs": float(np.max(np.abs(flat))),
        "is_valid": bool(nan_ratio <= nan_ratio_threshold),
    }


def sampling_rate_to_label(value: float) -> str:
    if value != value:
        return "NaN"
    if float(value).is_integer():
        return str(int(value))
    return f"{value:g}"


def derive_skip_reason(parse_result, config: dict) -> str | None:
    extraction_config = config["signal_extraction"]
    if extraction_config["skip_explicit_background_files"] and parse_result.event_type == "background_noise":
        return "explicit_background_skipped"
    if extraction_config["skip_unresolved_events"] and "event_type_unresolved" in parse_result.parse_warning:
        return "event_type_unresolved"
    return None


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    input_root = Path(args.input)
    output_h5 = Path(args.output)
    private_map_path = Path(args.private_map)
    config = load_config(Path(args.config))

    output_h5.parent.mkdir(parents=True, exist_ok=True)
    private_map_path.parent.mkdir(parents=True, exist_ok=True)

    parser = LabelParser(config)
    segment_registry = SegmentIdRegistry(prefix=config["anonymization"]["segment_prefix"])
    root_attrs = {
        "dataset_name": config["dataset"]["name"],
        "version": config["dataset"]["version"],
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "anonymization_level": config["dataset"]["anonymization_level"],
        "source_description": config["dataset"]["source_description"],
        "public_release_note": config["dataset"]["public_release_note"],
    }
    writer = HDF5DatasetWriter(output_h5, root_attrs)

    input_root_resolved = input_root.resolve()
    excluded_roots = {
        root
        for root in {private_map_path.parent.resolve(), output_h5.parent.resolve()}
        if root != input_root_resolved
    }
    csv_paths = [
        path
        for path in sorted(input_root.rglob("*.csv"))
        if path.resolve() != private_map_path.resolve()
        and not any(root == path.resolve().parent or root in path.resolve().parents for root in excluded_roots)
    ]
    logging.info("Found %s CSV files to process", len(csv_paths))

    event_map = dict(config["label_maps"]["event_type"])
    fine_event_map = dict(config["label_maps"]["fine_event"])
    soil_map = dict(config["label_maps"]["soil_condition"])
    distance_map = {config["distance"]["default_label"]: 0}

    sample_summaries: List[Dict[str, object]] = []
    private_rows: List[Dict[str, str]] = []
    failed_examples: List[Dict[str, str]] = []
    warning_counter: Counter[str] = Counter()
    skipped_files = 0
    positive_samples = 0
    generated_background_samples = 0
    total_flat_length = 0

    nan_ratio_threshold = float(config["quality"]["nan_ratio_threshold"])
    public_file_prefix = config["anonymization"]["public_file_prefix"]
    split_warnings: List[str] = []

    try:
        for source_index, csv_path in enumerate(csv_paths):
            if source_index % 250 == 0:
                logging.info("Processing file %s / %s", source_index + 1, len(csv_paths))

            parse_result = parser.parse_path(csv_path)
            original_path_hash = sha256_text(str(csv_path.resolve()))
            original_filename_hash = sha256_text(csv_path.name)
            file_sha256 = sha256_file(csv_path)

            skip_reason = derive_skip_reason(parse_result, config)
            if skip_reason is not None:
                skipped_files += 1
                private_rows.append(
                    build_private_mapping_row(
                        original_path=csv_path,
                        sample_id="",
                        segment_public_id="",
                        parse_result=skip_reason,
                        path_hash=original_path_hash,
                    )
                )
                if len(failed_examples) < 200:
                    failed_examples.append(
                        {
                            "path_hash": original_path_hash,
                            "status": skip_reason,
                            "warning": ";".join(parse_result.parse_warning),
                        }
                    )
                continue

            try:
                raw_signal, raw_invalid_mask, load_warnings = load_numeric_csv(csv_path)
            except Exception as exc:  # noqa: BLE001
                skipped_files += 1
                status = f"read_error:{type(exc).__name__}"
                private_rows.append(
                    build_private_mapping_row(
                        original_path=csv_path,
                        sample_id="",
                        segment_public_id="",
                        parse_result=status,
                        path_hash=original_path_hash,
                    )
                )
                if len(failed_examples) < 200:
                    failed_examples.append(
                        {
                            "path_hash": original_path_hash,
                            "status": status,
                            "warning": str(exc),
                        }
                    )
                logging.warning("Skipping %s because %s", csv_path, exc)
                continue

            warnings = list(parse_result.parse_warning)
            warnings.extend(load_warnings)
            for warning in warnings:
                warning_counter[warning] += 1

            try:
                window_rows = resolve_window_rows(config, parse_result.event_type, parse_result.fine_event)
                event_window = select_event_window(raw_signal, window_rows)
            except Exception as exc:  # noqa: BLE001
                skipped_files += 1
                status = f"crop_error:{type(exc).__name__}"
                private_rows.append(
                    build_private_mapping_row(
                        original_path=csv_path,
                        sample_id="",
                        segment_public_id="",
                        parse_result=status,
                        path_hash=original_path_hash,
                    )
                )
                if len(failed_examples) < 200:
                    failed_examples.append(
                        {
                            "path_hash": original_path_hash,
                            "status": status,
                            "warning": str(exc),
                        }
                    )
                continue

            segment_record = segment_registry.get_or_create(
                parse_result.source_batch_id,
                parse_result.sampling_rate_hz,
                parse_result.soil_condition,
            )

            def append_public_sample(
                sample_signal: np.ndarray,
                sample_invalid_mask: np.ndarray,
                event_type_name: str,
                fine_event_name: str,
                distance_label_name: str,
                distance_value_m: float,
                has_distance_label: bool,
                is_background: bool,
                sample_mode: str,
            ) -> None:
                nonlocal total_flat_length, positive_samples, generated_background_samples

                quality = compute_quality(sample_signal, sample_invalid_mask, nan_ratio_threshold)
                if not quality["is_valid"]:
                    warning_counter[f"high_nan_ratio>{nan_ratio_threshold:g}"] += 1

                distance_label_id = ensure_numeric_label(distance_label_name, distance_map)
                event_type_id = int(event_map[event_type_name])
                fine_event_id = int(fine_event_map.get(fine_event_name, fine_event_map["unknown"]))
                soil_condition_id = int(soil_map[parse_result.soil_condition])

                written_index = len(sample_summaries)
                sample_id = f"SAMPLE_{written_index + 1:08d}"
                public_file_id = build_public_file_id(
                    public_file_prefix,
                    parse_result.source_batch_id,
                    written_index,
                )
                parse_warning_text = ";".join(sorted(set(warnings + [sample_mode])))

                writer.append_sample(
                    signal=sample_signal,
                    record={
                        "event_type": event_type_id,
                        "fine_event": fine_event_id,
                        "distance_label": distance_label_id,
                        "distance_value_m": np.float32(distance_value_m),
                        "soil_condition": soil_condition_id,
                        "segment_id": np.int32(segment_record.numeric_id),
                        "sampling_rate_hz": np.float32(parse_result.sampling_rate_hz),
                        "is_background": bool(is_background),
                        "has_distance_label": bool(has_distance_label),
                        "sample_id": sample_id,
                        "public_file_id": public_file_id,
                        "file_sha256": file_sha256,
                        "source_batch_id": parse_result.source_batch_id,
                        "original_filename_hash": original_filename_hash,
                        "parse_status": "warning" if warnings else parse_result.parse_status,
                        "parse_warning": parse_warning_text,
                        "signal_length": np.int32(quality["signal_length"]),
                        "nan_ratio": np.float32(quality["nan_ratio"]),
                        "mean": np.float32(quality["mean"]),
                        "std": np.float32(quality["std"]),
                        "rms": np.float32(quality["rms"]),
                        "max_abs": np.float32(quality["max_abs"]),
                        "is_valid": bool(quality["is_valid"]),
                    },
                )
                total_flat_length += int(quality["signal_length"])
                sample_summaries.append(
                    {
                        "sample_id": sample_id,
                        "public_file_id": public_file_id,
                        "source_batch_id": parse_result.source_batch_id,
                        "event_type": event_type_name,
                        "fine_event": fine_event_name if fine_event_name in fine_event_map else "unknown",
                        "distance_label": distance_label_name,
                        "soil_condition": parse_result.soil_condition,
                        "segment_id": segment_record.numeric_id,
                        "segment_public_id": segment_record.public_id,
                        "sampling_rate_hz": parse_result.sampling_rate_hz,
                        "sampling_rate_label": sampling_rate_to_label(parse_result.sampling_rate_hz),
                        "is_valid": bool(quality["is_valid"]),
                    }
                )
                private_rows.append(
                    build_private_mapping_row(
                        original_path=csv_path,
                        sample_id=sample_id,
                        segment_public_id=segment_record.public_id,
                        parse_result=sample_mode,
                        path_hash=original_path_hash,
                    )
                )
                if is_background:
                    generated_background_samples += 1
                else:
                    positive_samples += 1

            event_signal = extract_window_time_major(raw_signal, event_window)
            event_invalid_mask = raw_invalid_mask[event_window.start_row : event_window.end_row, :].T
            append_public_sample(
                sample_signal=event_signal,
                sample_invalid_mask=event_invalid_mask,
                event_type_name=parse_result.event_type,
                fine_event_name=parse_result.fine_event if parse_result.fine_event in fine_event_map else "unknown",
                distance_label_name=parse_result.distance_label,
                distance_value_m=parse_result.distance_value_m,
                has_distance_label=parse_result.has_distance_label,
                is_background=False,
                sample_mode="event_crop",
            )

            background_windows = generate_background_windows(
                raw_signal=raw_signal,
                event_window=event_window,
                background_per_event=int(config["signal_extraction"]["background_per_event"]),
                guard_rows=int(config["signal_extraction"]["background_guard_rows"]),
                stride_mode=str(config["signal_extraction"]["candidate_stride_mode"]),
            )
            for background_index, background_window in enumerate(background_windows, start=1):
                background_signal = extract_window_time_major(raw_signal, background_window)
                background_invalid_mask = raw_invalid_mask[
                    background_window.start_row : background_window.end_row,
                    :,
                ].T
                append_public_sample(
                    sample_signal=background_signal,
                    sample_invalid_mask=background_invalid_mask,
                    event_type_name="background_noise",
                    fine_event_name="N/A",
                    distance_label_name=config["distance"]["default_label"],
                    distance_value_m=math.nan,
                    has_distance_label=False,
                    is_background=True,
                    sample_mode=f"background_crop_{background_index}",
                )

        event_type_ids = [event_map[summary["event_type"]] for summary in sample_summaries]
        segment_ids = [summary["segment_id"] for summary in sample_summaries]
        splits = {
            "random": build_random_split(event_type_ids, config["random_seed"], split_warnings),
            "segment_holdout": build_segment_holdout_split(segment_ids, config["random_seed"], split_warnings),
            "cross_segment": build_cross_segment_folds(segment_ids, 5, config["random_seed"], split_warnings),
        }
        writer.write_splits(splits)

        writer.write_label_maps(
            {
                "event_type_json": event_map,
                "fine_event_json": fine_event_map,
                "distance_label_json": distance_map,
                "soil_condition_json": soil_map,
                "segment_map_json": json.loads(segment_registry.to_public_json()),
            }
        )
    finally:
        writer.close()

    private_df = pd.DataFrame(private_rows)
    private_df.to_csv(private_map_path, index=False, encoding="utf-8")

    stats_payload = build_statistics_payload(sample_summaries, warning_counter)
    write_dataset_statistics(output_h5.parent / "dataset_statistics.json", stats_payload)
    write_dataset_card(
        output_h5.parent / "dataset_card.md",
        {
            "dataset_name": config["dataset"]["name"],
            "version": config["dataset"]["version"],
            "total_samples": len(sample_summaries),
            "source_batches": sorted({summary["source_batch_id"] for summary in sample_summaries})
            or sorted(config["batch_map"].values()),
        },
    )
    write_build_report(
        output_h5.parent / "build_report.md",
        {
            "output_h5": str(output_h5),
            "total_files_scanned": len(csv_paths),
            "samples_written": len(sample_summaries),
            "files_skipped": skipped_files,
            "invalid_samples": int(stats_payload["invalid_samples"]),
            "positive_samples": positive_samples,
            "generated_background_samples": generated_background_samples,
            "warning_counter": dict(warning_counter),
            "split_warnings": split_warnings,
            "failed_examples": failed_examples,
            "hdf5_summary": {
                "signals_flat_length": total_flat_length,
                "sample_count": len(sample_summaries),
                "segment_count": len({summary["segment_public_id"] for summary in sample_summaries}),
                "distance_label_count": len(distance_map),
            },
        },
    )
    logging.info("Build completed. Public HDF5 written to %s", output_h5)
    logging.info("Private mapping written to %s", private_map_path)
    logging.info("Total written samples: %s", len(sample_summaries))
    logging.info("Positive samples: %s", positive_samples)
    logging.info("Generated background samples: %s", generated_background_samples)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
