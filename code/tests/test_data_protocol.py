import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = PROJECT_ROOT / "examples" / "das_csv"
for path in (PROJECT_ROOT, EXAMPLES_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from sensorfield_dataset import (  # noqa: E402
    DISTANCE_IGNORE_INDEX,
    MultiTaskCSVDataset,
    canonicalize_dataset_manifests,
)
from LibMTL.model import SensorFieldM3T  # noqa: E402
from sensorfield_metrics import classification_metrics  # noqa: E402


def _write_signal(path: Path, rows: int, cols: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = np.linspace(-1.0, 1.0, rows * cols, dtype=np.float32).reshape(rows, cols)
    np.savetxt(path, values, delimiter=",", fmt="%.6f")


class SensorFieldTPAMIProtocolTest(unittest.TestCase):
    def test_vehicle_channels_are_independent_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            signal_path = root / "vehicle.csv"
            _write_signal(signal_path, rows=10, cols=64)
            manifest_path = root / "train.csv"
            with manifest_path.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["path", "event_label", "distance_label", "sample_mode"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "path": str(signal_path),
                        "event_label": "driving",
                        "distance_label": "",
                        "sample_mode": "row",
                    }
                )

            dataset = MultiTaskCSVDataset(
                manifest_path=manifest_path,
                event_to_idx={"driving": 0},
                distance_to_idx={"Alarm area": 0},
                input_height=1,
                input_width=64,
            )
            self.assertEqual(len(dataset), 10)
            signal, labels = dataset[9]
            self.assertEqual(tuple(signal.shape), (1, 1, 64))
            self.assertEqual(tuple(labels["task_mask"].tolist()), (1.0, 0.0))

    def test_stf_uses_joint_three_channel_spatial_groups(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            signal_path = root / "six_channels.csv"
            _write_signal(signal_path, rows=6, cols=64)
            manifest_path = root / "train.csv"
            with manifest_path.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["path", "event_label", "distance_label", "sample_mode"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "path": str(signal_path),
                        "event_label": "excavator",
                        "distance_label": "Alarm area",
                        "sample_mode": "file",
                    }
                )

            dataset = MultiTaskCSVDataset(
                manifest_path=manifest_path,
                event_to_idx={"excavator": 0},
                distance_to_idx={"Alarm area": 0},
                input_height=1,
                input_width=64,
                return_multiview=True,
                stf_spatial_fusion="group3_mean",
                stft_n_fft=16,
                stft_hop_length=8,
                stft_win_length=16,
            )
            with signal_path.open("r", encoding="utf-8-sig", newline="") as handle:
                signal = np.loadtxt(handle, delimiter=",", dtype=np.float32)
            actual = dataset._compute_stft_feature(signal).squeeze(0)

            row_spectra = []
            for row in signal:
                spectrum = torch.stft(
                    torch.from_numpy(row),
                    n_fft=16,
                    hop_length=8,
                    win_length=16,
                    window=dataset._stft_window,
                    return_complex=True,
                    center=True,
                )
                row_spectra.append(torch.log1p(torch.abs(spectrum)))
            expected = torch.cat(
                [
                    torch.stack(row_spectra[:3], dim=0).mean(dim=0),
                    torch.stack(row_spectra[3:], dim=0).mean(dim=0),
                ],
                dim=0,
            )

            self.assertEqual(actual.shape[0], 2 * (16 // 2 + 1))
            torch.testing.assert_close(actual, expected)

    def test_canonicalized_multiview_shapes_and_task_mask(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "MTL43"
            actual_walking = root / "human activities" / "walk.csv"
            stale_walking = root / "walking" / "walk.csv"
            excavator = root / "excavator" / "5m" / "exc.csv"
            _write_signal(actual_walking, rows=1, cols=64)
            _write_signal(excavator, rows=6, cols=64)
            for split in ("train", "val", "test"):
                with (root / f"{split}.csv").open("w", encoding="utf-8-sig", newline="") as handle:
                    writer = csv.DictWriter(
                        handle,
                        fieldnames=["path", "event_label", "distance_label", "sample_mode"],
                    )
                    writer.writeheader()
                    writer.writerow(
                        {
                            "path": str(stale_walking),
                            "event_label": "walking",
                            "distance_label": "",
                            "sample_mode": "file",
                        }
                    )
                    writer.writerow(
                        {
                            "path": str(excavator),
                            "event_label": "excavator",
                            "distance_label": "Alarm area",
                            "sample_mode": "file",
                        }
                    )

            out_root = Path(tmp) / "canonical"
            summary = canonicalize_dataset_manifests(root, out_root)
            self.assertEqual(len(summary["canonicalized_paths"]), 3)

            dataset = MultiTaskCSVDataset(
                manifest_path=out_root / "train.csv",
                event_to_idx={"walking": 0, "excavator": 1},
                distance_to_idx={"Alarm area": 0},
                input_height=1,
                input_width=64,
                return_multiview=True,
                spatial_adapter="center",
                stf_size=32,
                gaf_size=32,
                normalize="sample",
                stft_n_fft=16,
                stft_hop_length=8,
                stft_win_length=16,
            )
            inputs, labels = dataset[1]
            self.assertEqual(tuple(inputs["raw"].shape), (1, 64))
            self.assertEqual(tuple(inputs["stf"].shape), (1, 32, 32))
            self.assertEqual(tuple(inputs["gaf"].shape), (1, 32, 32))
            self.assertEqual(int(labels["distance_cls"].item()), 0)
            self.assertEqual(tuple(labels["task_mask"].tolist()), (1.0, 1.0))
            _, walking_labels = dataset[0]
            self.assertEqual(int(walking_labels["distance_cls"].item()), DISTANCE_IGNORE_INDEX)
            self.assertEqual(tuple(walking_labels["task_mask"].tolist()), (1.0, 0.0))

    def test_metrics_and_missing_view_forward(self) -> None:
        bundle = classification_metrics(
            targets=[0, 1, 1, 2],
            predictions=[0, 1, 2, 2],
            probabilities=[
                [0.9, 0.05, 0.05],
                [0.1, 0.8, 0.1],
                [0.2, 0.3, 0.5],
                [0.1, 0.1, 0.8],
            ],
            class_names=["a", "b", "c"],
        )
        self.assertGreater(bundle.task_score, 0.7)
        self.assertEqual(bundle.confusion.sum(), 4)

        model = SensorFieldM3T(
            task_output_dims={"event_type": 4, "distance_cls": 3},
            hidden_dim=32,
            num_anchors=4,
            num_heads=4,
            stf_size=32,
            gaf_size=32,
            return_auxiliary=True,
        )
        batch = {
            "raw": torch.randn(2, 1, 128),
            "stf": torch.randn(2, 1, 32, 32),
            "gaf": torch.randn(2, 1, 32, 32),
            "modality_mask": torch.tensor([[1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]),
        }
        outputs = model(batch)
        self.assertEqual(tuple(outputs["event_type"].shape), (2, 4))
        self.assertEqual(tuple(outputs["distance_cls"].shape), (2, 3))
        self.assertTrue(torch.isfinite(outputs["taef_outputs"]["alpha"]).all())


if __name__ == "__main__":
    unittest.main()
