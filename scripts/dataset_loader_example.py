from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load a sample from SensorField-DAS HDF5.")
    parser.add_argument(
        "--h5",
        default=str(Path(__file__).resolve().parents[1] / "public_dataset_release" / "SensorField_DAS_v1.h5"),
        help="Path to the SensorField-DAS HDF5 file.",
    )
    parser.add_argument("--index", type=int, default=0, help="Sample index to load.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    with h5py.File(args.h5, "r") as handle:
        start, length = handle["/data/signal_index"][args.index]
        time_steps, channels = handle["/data/signal_shape"][args.index]
        flat = handle["/data/signals_flat"][start:start + length]
        signal = np.asarray(flat, dtype=np.float32).reshape((time_steps, channels))

        sample_id = handle["/meta/sample_id"][args.index]
        if isinstance(sample_id, bytes):
            sample_id = sample_id.decode("utf-8")
        event_type_id = int(handle["/labels/event_type"][args.index])
        raw_map = handle["/meta/label_maps/event_type_json"][()]
        event_map = json.loads(raw_map.decode("utf-8") if isinstance(raw_map, bytes) else raw_map)
        event_name = next(name for name, value in event_map.items() if value == event_type_id)
        print(f"sample_id={sample_id}")
        print(f"signal_shape={signal.shape}")
        print(f"event_type_id={event_type_id}")
        print(f"event_type={event_name}")
        print(f"signal_mean={float(signal.mean()):.6f}")
        print(f"signal_std={float(signal.std()):.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
