from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Mapping

import h5py
import numpy as np


class HDF5DatasetWriter:
    def __init__(self, output_path: Path, root_attrs: Mapping[str, str]) -> None:
        self.output_path = output_path
        self.h5 = h5py.File(output_path, "w")
        self._closed = False
        self.signal_offset = 0
        self.sample_count = 0
        for key, value in root_attrs.items():
            self.h5.attrs[key] = value

        string_dtype = h5py.string_dtype(encoding="utf-8")
        self.datasets: Dict[str, h5py.Dataset] = {}

        self._ensure_group("/data")
        self._ensure_group("/labels")
        self._ensure_group("/meta")
        self._ensure_group("/meta/label_maps")
        self._ensure_group("/quality")
        self._ensure_group("/splits")

        self.datasets["/data/signals_flat"] = self._create_resizable_dataset(
            "/data/signals_flat",
            shape=(0,),
            maxshape=(None,),
            dtype=np.float32,
        )
        self.datasets["/data/signal_index"] = self._create_resizable_dataset(
            "/data/signal_index",
            shape=(0, 2),
            maxshape=(None, 2),
            dtype=np.int64,
        )
        self.datasets["/data/signal_shape"] = self._create_resizable_dataset(
            "/data/signal_shape",
            shape=(0, 2),
            maxshape=(None, 2),
            dtype=np.int32,
        )

        label_specs = {
            "/labels/event_type": np.int16,
            "/labels/fine_event": np.int16,
            "/labels/distance_label": np.int16,
            "/labels/distance_value_m": np.float32,
            "/labels/soil_condition": np.int16,
            "/labels/segment_id": np.int32,
            "/labels/sampling_rate_hz": np.float32,
            "/labels/is_background": np.bool_,
            "/labels/has_distance_label": np.bool_,
        }
        for dataset_path, dtype in label_specs.items():
            self.datasets[dataset_path] = self._create_resizable_dataset(
                dataset_path,
                shape=(0,),
                maxshape=(None,),
                dtype=dtype,
            )

        meta_specs = {
            "/meta/sample_id": string_dtype,
            "/meta/public_file_id": string_dtype,
            "/meta/file_sha256": string_dtype,
            "/meta/source_batch_id": string_dtype,
            "/meta/original_filename_hash": string_dtype,
            "/meta/parse_status": string_dtype,
            "/meta/parse_warning": string_dtype,
        }
        for dataset_path, dtype in meta_specs.items():
            self.datasets[dataset_path] = self._create_resizable_dataset(
                dataset_path,
                shape=(0,),
                maxshape=(None,),
                dtype=dtype,
            )

        quality_specs = {
            "/quality/signal_length": np.int32,
            "/quality/nan_ratio": np.float32,
            "/quality/mean": np.float32,
            "/quality/std": np.float32,
            "/quality/rms": np.float32,
            "/quality/max_abs": np.float32,
            "/quality/is_valid": np.bool_,
        }
        for dataset_path, dtype in quality_specs.items():
            self.datasets[dataset_path] = self._create_resizable_dataset(
                dataset_path,
                shape=(0,),
                maxshape=(None,),
                dtype=dtype,
            )

    def append_sample(self, signal: np.ndarray, record: Mapping[str, object]) -> None:
        signal = np.asarray(signal, dtype=np.float32)
        if signal.ndim == 1:
            signal = signal.reshape(-1, 1)
        flat = signal.reshape(-1).astype(np.float32)
        start = self.signal_offset
        length = int(flat.size)

        self._append_1d("/data/signals_flat", flat)
        self._append_row("/data/signal_index", np.array([start, length], dtype=np.int64))
        self._append_row("/data/signal_shape", np.array(signal.shape[:2], dtype=np.int32))

        for key in (
            "event_type",
            "fine_event",
            "distance_label",
            "distance_value_m",
            "soil_condition",
            "segment_id",
            "sampling_rate_hz",
            "is_background",
            "has_distance_label",
        ):
            self._append_scalar(f"/labels/{key}", record[key])

        for key in (
            "sample_id",
            "public_file_id",
            "file_sha256",
            "source_batch_id",
            "original_filename_hash",
            "parse_status",
            "parse_warning",
        ):
            self._append_scalar(f"/meta/{key}", record[key])

        for key in (
            "signal_length",
            "nan_ratio",
            "mean",
            "std",
            "rms",
            "max_abs",
            "is_valid",
        ):
            self._append_scalar(f"/quality/{key}", record[key])

        self.signal_offset += length
        self.sample_count += 1

    def write_label_maps(self, label_maps: Mapping[str, object]) -> None:
        string_dtype = h5py.string_dtype(encoding="utf-8")
        for key, value in label_maps.items():
            dataset_path = f"/meta/label_maps/{key}"
            if dataset_path in self.h5:
                del self.h5[dataset_path]
            payload = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True)
            self.h5.create_dataset(dataset_path, data=payload, dtype=string_dtype)

    def write_splits(self, splits: Mapping[str, object]) -> None:
        splits_group = self.h5["/splits"]
        for key in list(splits_group.keys()):
            del splits_group[key]
        for split_name, split_value in splits.items():
            self._write_split_object(f"/splits/{split_name}", split_value)

    def flush(self) -> None:
        if not self._closed:
            self.h5.flush()

    def close(self) -> None:
        if self._closed:
            return
        self.h5.flush()
        self.h5.close()
        self._closed = True

    def _write_split_object(self, base_path: str, value: object) -> None:
        if isinstance(value, dict):
            self._ensure_group(base_path)
            for child_key, child_value in value.items():
                self._write_split_object(f"{base_path}/{child_key}", child_value)
            return
        array = np.asarray(value, dtype=np.int64)
        if base_path in self.h5:
            del self.h5[base_path]
        self.h5.create_dataset(
            base_path,
            data=array,
            compression="gzip",
            shuffle=True,
        )

    def _ensure_group(self, path: str) -> h5py.Group:
        return self.h5.require_group(path)

    def _create_resizable_dataset(
        self,
        path: str,
        shape: Iterable[int],
        maxshape: Iterable[int],
        dtype: object,
    ) -> h5py.Dataset:
        return self.h5.create_dataset(
            path,
            shape=tuple(shape),
            maxshape=tuple(maxshape),
            dtype=dtype,
            compression="gzip",
            shuffle=True,
        )

    def _append_1d(self, dataset_path: str, values: np.ndarray) -> None:
        dataset = self.datasets[dataset_path]
        old_size = dataset.shape[0]
        new_size = old_size + values.shape[0]
        dataset.resize((new_size,))
        dataset[old_size:new_size] = values

    def _append_row(self, dataset_path: str, row: np.ndarray) -> None:
        dataset = self.datasets[dataset_path]
        old_size = dataset.shape[0]
        new_size = old_size + 1
        dataset.resize((new_size, dataset.shape[1]))
        dataset[old_size] = row

    def _append_scalar(self, dataset_path: str, value: object) -> None:
        dataset = self.datasets[dataset_path]
        old_size = dataset.shape[0]
        new_size = old_size + 1
        dataset.resize((new_size,))
        dataset[old_size] = value
