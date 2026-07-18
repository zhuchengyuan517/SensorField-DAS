from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LIB_ROOT = PROJECT_ROOT / "libmtl_das_patch"
if str(LIB_ROOT) not in sys.path:
    sys.path.insert(0, str(LIB_ROOT))

from LibMTL.model import SensorFieldMEDHTT, ordinal_predictions  # noqa: E402


IGNORE_INDEX = -1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train SensorField-MEDHTT on the public PipeDAS HDF5 release."
    )
    parser.add_argument(
        "--h5",
        default=str(PROJECT_ROOT / "public_dataset_release" / "PipeDAS_Multi_v1.h5"),
        type=str,
    )
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config" / "label_config.yaml"), type=str)
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "output" / "sensorfield_medhtt_hdf5"), type=str)
    parser.add_argument("--condition-label", default="fine_event", choices=["fine_event", "soil_condition"])
    parser.add_argument(
        "--radial-order",
        default="distance_asc",
        choices=["distance_asc", "distance_desc", "raw"],
        help="Ordinal order for radial labels. distance_asc preserves physical distance order.",
    )
    parser.add_argument("--epochs", default=5, type=int)
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--num-workers", default=0, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--max-samples", default=0, type=int, help="0 uses the full HDF5 dataset.")
    parser.add_argument("--input-height", default=6, type=int)
    parser.add_argument("--input-width", default=2048, type=int)
    parser.add_argument("--normalize", default="sample", choices=["sample", "none"])
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--weight-decay", default=5e-4, type=float)
    parser.add_argument("--max-grad-norm", default=1.0, type=float)
    parser.add_argument("--event-loss-weight", default=1.0, type=float)
    parser.add_argument("--radial-loss-weight", default=1.2, type=float)
    parser.add_argument("--condition-loss-weight", default=1.0, type=float)
    parser.add_argument("--hidden-dim", default=96, type=int)
    parser.add_argument("--num-heads", default=4, type=int)
    parser.add_argument("--shared-tokens", default=6, type=int)
    parser.add_argument("--raw-tokens", default=8, type=int)
    parser.add_argument("--stf-tokens", default=8, type=int)
    parser.add_argument("--gaf-tokens", default=8, type=int)
    parser.add_argument("--stf-size", default=64, type=int)
    parser.add_argument("--gaf-size", default=48, type=int)
    parser.add_argument("--stft-n-fft", default=128, type=int)
    parser.add_argument("--stft-hop-length", default=64, type=int)
    parser.add_argument("--stft-win-length", default=128, type=int)
    parser.add_argument("--propagation-steps", default=1, type=int)
    parser.add_argument("--enabled-views", default="raw,stf,gaf", type=str)
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--dec-loss-weight", default=0.05, type=float)
    parser.add_argument("--cep-loss-weight", default=0.02, type=float)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but is not available.")
        return torch.device("cuda")
    if name == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def invert_label_map(label_map: dict[str, int]) -> dict[int, str]:
    return {int(value): str(key) for key, value in label_map.items()}


def compact_map(values: np.ndarray, valid_mask: np.ndarray) -> dict[int, int]:
    raw_values = sorted(int(value) for value in np.unique(values[valid_mask]))
    return {raw_value: index for index, raw_value in enumerate(raw_values)}


def radial_compact_map(
    distance_raw: np.ndarray,
    distance_value_m: np.ndarray,
    valid_mask: np.ndarray,
    radial_order: str,
) -> dict[int, int]:
    raw_values = [int(value) for value in np.unique(distance_raw[valid_mask])]
    if radial_order == "raw":
        raw_values = sorted(raw_values)
    else:
        def sort_key(raw_value: int) -> tuple[float, int]:
            values = distance_value_m[(distance_raw == raw_value) & valid_mask]
            finite_values = values[np.isfinite(values)]
            if finite_values.size:
                return float(np.median(finite_values)), raw_value
            return float(raw_value), raw_value

        raw_values = sorted(raw_values, key=sort_key, reverse=(radial_order == "distance_desc"))
    return {raw_value: index for index, raw_value in enumerate(raw_values)}


def names_from_compact_map(raw_to_compact: dict[int, int], raw_to_name: dict[int, str]) -> list[str]:
    names = [""] * len(raw_to_compact)
    for raw_value, compact_value in raw_to_compact.items():
        names[compact_value] = raw_to_name.get(raw_value, str(raw_value))
    return names


def build_labels_and_maps(
    h5_path: Path,
    config: dict[str, Any],
    condition_label: str,
    radial_order: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with h5py.File(h5_path, "r") as handle:
        event_raw = handle["/labels/event_type"][:].astype(np.int64)
        distance_raw = handle["/labels/distance_label"][:].astype(np.int64)
        distance_value_m = handle["/labels/distance_value_m"][:].astype(np.float32)
        has_distance = handle["/labels/has_distance_label"][:].astype(bool)
        is_background = handle["/labels/is_background"][:].astype(bool)
        condition_raw = handle[f"/labels/{condition_label}"][:].astype(np.int64)
        quality_valid = handle["/quality/is_valid"][:].astype(bool) if "/quality/is_valid" in handle else np.ones_like(is_background)

    event_map = compact_map(event_raw, quality_valid)
    radial_map = radial_compact_map(distance_raw, distance_value_m, quality_valid & has_distance, radial_order)
    if condition_label == "fine_event":
        configured = config["label_maps"]["fine_event"]
        invalid_condition_ids = {
            int(configured.get("N/A", -999)),
            int(configured.get("unknown", -998)),
        }
        condition_valid = quality_valid & (~is_background) & (~np.isin(condition_raw, list(invalid_condition_ids)))
    else:
        condition_valid = quality_valid & (~is_background)
    condition_map = compact_map(condition_raw, condition_valid)

    label_arrays = {
        "event_type": np.full(event_raw.shape, IGNORE_INDEX, dtype=np.int64),
        "radial_threat": np.full(event_raw.shape, IGNORE_INDEX, dtype=np.int64),
        "threat_condition": np.full(event_raw.shape, IGNORE_INDEX, dtype=np.int64),
    }
    for raw_value, compact_value in event_map.items():
        label_arrays["event_type"][event_raw == raw_value] = compact_value
    for raw_value, compact_value in radial_map.items():
        label_arrays["radial_threat"][(distance_raw == raw_value) & has_distance] = compact_value
    for raw_value, compact_value in condition_map.items():
        label_arrays["threat_condition"][(condition_raw == raw_value) & condition_valid] = compact_value

    event_names = names_from_compact_map(event_map, invert_label_map(config["label_maps"]["event_type"]))
    if "distance_label" in config["label_maps"]:
        distance_names_raw = invert_label_map(config["label_maps"]["distance_label"])
    else:
        distance_names_raw = {}
        for raw_value in radial_map:
            values = distance_value_m[(distance_raw == raw_value) & has_distance]
            finite_values = values[np.isfinite(values)]
            if finite_values.size:
                median_value = float(np.median(finite_values))
                distance_names_raw[raw_value] = f"{median_value:g}m"
            else:
                distance_names_raw[raw_value] = str(raw_value)
    radial_names = names_from_compact_map(radial_map, distance_names_raw)
    condition_names = names_from_compact_map(
        condition_map,
        invert_label_map(config["label_maps"][condition_label]),
    )
    metadata = {
        "event_raw_to_compact": event_map,
        "radial_raw_to_compact": radial_map,
        "condition_raw_to_compact": condition_map,
        "event_names": event_names,
        "radial_names": radial_names,
        "condition_names": condition_names,
        "radial_order": radial_order,
        "quality_valid": quality_valid,
        "stratify_labels": label_arrays["event_type"],
    }
    return label_arrays, metadata


def can_stratify(labels: np.ndarray) -> bool:
    values, counts = np.unique(labels, return_counts=True)
    return len(values) > 1 and int(counts.min()) >= 2


def split_optional_stratify(
    indices: np.ndarray,
    labels: np.ndarray,
    test_size: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    stratify = labels if can_stratify(labels) else None
    first, second = train_test_split(
        indices,
        test_size=test_size,
        random_state=seed,
        shuffle=True,
        stratify=stratify,
    )
    return np.asarray(first, dtype=np.int64), np.asarray(second, dtype=np.int64)


def build_splits(
    valid_indices: np.ndarray,
    stratify_labels: np.ndarray,
    seed: int,
    max_samples: int,
) -> dict[str, np.ndarray]:
    candidate_indices = np.asarray(valid_indices, dtype=np.int64)
    if max_samples > 0 and max_samples < len(candidate_indices):
        labels = stratify_labels[candidate_indices]
        stratify = labels if can_stratify(labels) else None
        candidate_indices, _unused = train_test_split(
            candidate_indices,
            train_size=max_samples,
            random_state=seed,
            shuffle=True,
            stratify=stratify,
        )
        candidate_indices = np.asarray(candidate_indices, dtype=np.int64)

    train_indices, temp_indices = split_optional_stratify(
        candidate_indices,
        stratify_labels[candidate_indices],
        test_size=0.30,
        seed=seed,
    )
    val_indices, test_indices = split_optional_stratify(
        temp_indices,
        stratify_labels[temp_indices],
        test_size=0.50,
        seed=seed + 1,
    )
    return {
        "train": np.asarray(train_indices, dtype=np.int64),
        "val": np.asarray(val_indices, dtype=np.int64),
        "test": np.asarray(test_indices, dtype=np.int64),
    }


class PipeDASHDF5Dataset(Dataset):
    def __init__(
        self,
        h5_path: Path,
        indices: np.ndarray,
        label_arrays: dict[str, np.ndarray],
        input_height: int,
        input_width: int,
        normalize: str = "sample",
    ) -> None:
        self.h5_path = Path(h5_path)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.label_arrays = label_arrays
        self.input_height = int(input_height)
        self.input_width = int(input_width)
        self.normalize = normalize
        self._handle: h5py.File | None = None

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_handle"] = None
        return state

    @property
    def handle(self) -> h5py.File:
        if self._handle is None:
            self._handle = h5py.File(self.h5_path, "r")
        return self._handle

    def _load_signal(self, sample_index: int) -> torch.Tensor:
        start, length = self.handle["/data/signal_index"][sample_index]
        time_steps, channels = self.handle["/data/signal_shape"][sample_index]
        flat = self.handle["/data/signals_flat"][start : start + length]
        signal = np.asarray(flat, dtype=np.float32).reshape((int(time_steps), int(channels))).T
        tensor = torch.from_numpy(signal).unsqueeze(0).unsqueeze(0)
        if tensor.shape[-2:] != (self.input_height, self.input_width):
            tensor = F.interpolate(
                tensor,
                size=(self.input_height, self.input_width),
                mode="bilinear",
                align_corners=False,
            )
        tensor = tensor.squeeze(0)
        if self.normalize == "sample":
            mean = tensor.mean()
            std = tensor.std(unbiased=False)
            tensor = (tensor - mean) / std.clamp_min(1e-6)
        return tensor.float()

    def __getitem__(self, item: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        sample_index = int(self.indices[item])
        signal = self._load_signal(sample_index)
        labels = {
            name: torch.tensor(int(values[sample_index]), dtype=torch.long)
            for name, values in self.label_arrays.items()
        }
        return signal, labels


def to_device(
    inputs: torch.Tensor,
    labels: dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    return inputs.to(device), {key: value.to(device) for key, value in labels.items()}


def masked_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    valid = targets != IGNORE_INDEX
    if not valid.any():
        return logits.sum() * 0.0
    return F.cross_entropy(logits[valid], targets[valid], weight=weight)


def ordinal_bce_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    valid = targets != IGNORE_INDEX
    if not valid.any():
        return logits.sum() * 0.0
    valid_targets = targets[valid]
    thresholds = torch.arange(logits.size(1), device=logits.device).view(1, -1)
    cumulative_targets = (valid_targets.view(-1, 1) > thresholds).float()
    return F.binary_cross_entropy_with_logits(logits[valid], cumulative_targets)


def class_weights(labels: np.ndarray, indices: np.ndarray, num_classes: int, device: torch.device) -> torch.Tensor:
    selected = labels[indices]
    selected = selected[selected != IGNORE_INDEX]
    counts = np.bincount(selected, minlength=num_classes).astype(np.float64)
    weights = np.zeros(num_classes, dtype=np.float32)
    nonzero = counts > 0
    if nonzero.any():
        weights[nonzero] = float(counts[nonzero].sum()) / (float(nonzero.sum()) * counts[nonzero])
        weights[nonzero] = weights[nonzero] / max(float(weights[nonzero].mean()), 1e-6)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def metric_block(targets: list[int], predictions: list[int], label_names: list[str]) -> dict[str, Any]:
    if not targets:
        return {"acc": None, "macro_f1": None, "support": 0}
    labels = list(range(len(label_names)))
    return {
        "acc": float(accuracy_score(targets, predictions)),
        "macro_f1": float(f1_score(targets, predictions, labels=labels, average="macro", zero_division=0)),
        "support": int(len(targets)),
    }


def run_epoch(
    model: SensorFieldMEDHTT,
    loader: DataLoader,
    device: torch.device,
    split: str,
    event_weight: torch.Tensor,
    condition_weight: torch.Tensor,
    optimizer: torch.optim.Optimizer | None,
    args: argparse.Namespace,
    label_names: dict[str, list[str]],
) -> dict[str, Any]:
    is_train = optimizer is not None
    model.train(is_train)

    total_samples = 0
    total_loss = 0.0
    total_event_loss = 0.0
    total_radial_loss = 0.0
    total_condition_loss = 0.0
    total_aux_loss = 0.0
    targets = {"event_type": [], "radial_threat": [], "threat_condition": []}
    predictions = {"event_type": [], "radial_threat": [], "threat_condition": []}

    for batch_index, (inputs, labels) in enumerate(loader, start=1):
        inputs, labels = to_device(inputs, labels, device)
        task_validity = {name: value != IGNORE_INDEX for name, value in labels.items()}

        with torch.set_grad_enabled(is_train):
            outputs = model(inputs, task_validity=task_validity)
            event_loss = masked_cross_entropy(outputs["event_type"], labels["event_type"], event_weight)
            radial_loss = ordinal_bce_loss(outputs["radial_threat"], labels["radial_threat"])
            condition_loss = masked_cross_entropy(
                outputs["threat_condition"],
                labels["threat_condition"],
                condition_weight,
            )
            aux_loss = outputs["event_type"].sum() * 0.0
            for value in outputs.get("aux_losses", {}).values():
                if value is not None:
                    aux_loss = aux_loss + value
            loss = (
                args.event_loss_weight * event_loss
                + args.radial_loss_weight * radial_loss
                + args.condition_loss_weight * condition_loss
                + aux_loss
            )

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()

        batch_size = int(inputs.size(0))
        total_samples += batch_size
        total_loss += float(loss.detach().item()) * batch_size
        total_event_loss += float(event_loss.detach().item()) * batch_size
        total_radial_loss += float(radial_loss.detach().item()) * batch_size
        total_condition_loss += float(condition_loss.detach().item()) * batch_size
        total_aux_loss += float(aux_loss.detach().item()) * batch_size

        event_pred = outputs["event_type"].detach().argmax(dim=-1).cpu().tolist()
        event_target = labels["event_type"].detach().cpu().tolist()
        targets["event_type"].extend(event_target)
        predictions["event_type"].extend(event_pred)

        radial_valid = labels["radial_threat"] != IGNORE_INDEX
        if radial_valid.any():
            radial_pred = ordinal_predictions(outputs["radial_threat"].detach())[radial_valid].cpu().tolist()
            radial_target = labels["radial_threat"][radial_valid].detach().cpu().tolist()
            targets["radial_threat"].extend(radial_target)
            predictions["radial_threat"].extend(radial_pred)

        condition_valid = labels["threat_condition"] != IGNORE_INDEX
        if condition_valid.any():
            condition_pred = outputs["threat_condition"].detach().argmax(dim=-1)[condition_valid].cpu().tolist()
            condition_target = labels["threat_condition"][condition_valid].detach().cpu().tolist()
            targets["threat_condition"].extend(condition_target)
            predictions["threat_condition"].extend(condition_pred)

        if is_train and (batch_index == 1 or batch_index % 25 == 0):
            print(
                f"{split} batch {batch_index}/{len(loader)} "
                f"loss={total_loss / max(total_samples, 1):.4f}",
                flush=True,
            )

    metrics = {
        "split": split,
        "loss": total_loss / max(total_samples, 1),
        "event_loss": total_event_loss / max(total_samples, 1),
        "radial_loss": total_radial_loss / max(total_samples, 1),
        "condition_loss": total_condition_loss / max(total_samples, 1),
        "aux_loss": total_aux_loss / max(total_samples, 1),
        "num_samples": total_samples,
    }
    metrics["event"] = metric_block(targets["event_type"], predictions["event_type"], label_names["event_type"])
    metrics["radial"] = metric_block(targets["radial_threat"], predictions["radial_threat"], label_names["radial_threat"])
    metrics["condition"] = metric_block(
        targets["threat_condition"],
        predictions["threat_condition"],
        label_names["threat_condition"],
    )
    score_terms = [
        block["acc"]
        for block in (metrics["event"], metrics["radial"], metrics["condition"])
        if block["acc"] is not None
    ]
    metrics["score"] = float(np.mean(score_terms)) if score_terms else 0.0
    metrics["_targets"] = targets
    metrics["_predictions"] = predictions
    return metrics


def strip_payload(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metrics.items() if not key.startswith("_")}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def write_history(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "epoch",
        "split",
        "loss",
        "event_loss",
        "radial_loss",
        "condition_loss",
        "aux_loss",
        "event_acc",
        "radial_acc",
        "condition_acc",
        "score",
        "num_samples",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def flatten_history_row(epoch: int, metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "epoch": epoch,
        "split": metrics["split"],
        "loss": metrics["loss"],
        "event_loss": metrics["event_loss"],
        "radial_loss": metrics["radial_loss"],
        "condition_loss": metrics["condition_loss"],
        "aux_loss": metrics["aux_loss"],
        "event_acc": metrics["event"]["acc"],
        "radial_acc": metrics["radial"]["acc"],
        "condition_acc": metrics["condition"]["acc"],
        "score": metrics["score"],
        "num_samples": metrics["num_samples"],
    }


def write_report_artifacts(
    run_dir: Path,
    split: str,
    metrics: dict[str, Any],
    label_names: dict[str, list[str]],
) -> None:
    for task_name, names in label_names.items():
        targets = metrics["_targets"][task_name]
        predictions = metrics["_predictions"][task_name]
        if not targets:
            continue
        labels = list(range(len(names)))
        report = classification_report(
            targets,
            predictions,
            labels=labels,
            target_names=names,
            zero_division=0,
            output_dict=True,
        )
        matrix = confusion_matrix(targets, predictions, labels=labels)
        write_json(run_dir / f"{split}_{task_name}_classification_report.json", report)
        with (run_dir / f"{split}_{task_name}_confusion.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow([""] + names)
            for name, row in zip(names, matrix.tolist()):
                writer.writerow([name] + row)


def label_counts(label_arrays: dict[str, np.ndarray], splits: dict[str, np.ndarray]) -> dict[str, dict[str, dict[str, int]]]:
    payload: dict[str, dict[str, dict[str, int]]] = {}
    for split_name, indices in splits.items():
        payload[split_name] = {}
        for task_name, labels in label_arrays.items():
            selected = labels[indices]
            selected = selected[selected != IGNORE_INDEX]
            values, counts = np.unique(selected, return_counts=True)
            payload[split_name][task_name] = {str(int(value)): int(count) for value, count in zip(values, counts)}
    return payload


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    h5_path = Path(args.h5).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve()
    output_root = Path(args.output_dir).expanduser().resolve()
    run_dir = output_root / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(config_path)
    label_arrays, metadata = build_labels_and_maps(h5_path, config, args.condition_label, args.radial_order)
    valid_indices = np.flatnonzero(metadata["quality_valid"])
    splits = build_splits(valid_indices, metadata["stratify_labels"], args.seed, args.max_samples)
    label_names = {
        "event_type": metadata["event_names"],
        "radial_threat": metadata["radial_names"],
        "threat_condition": metadata["condition_names"],
    }
    task_output_dims = {
        "event_type": len(label_names["event_type"]),
        "radial_threat": len(label_names["radial_threat"]),
        "threat_condition": len(label_names["threat_condition"]),
    }
    if any(value < 1 for value in task_output_dims.values()) or task_output_dims["radial_threat"] < 2:
        raise ValueError(f"Invalid task output dimensions inferred from HDF5 labels: {task_output_dims}")

    device = resolve_device(args.device)
    print(f"Run directory: {run_dir}")
    print(f"Device: {device}")
    print(f"Task output dims: {task_output_dims}")
    print(f"Split sizes: { {name: len(idx) for name, idx in splits.items()} }")
    print(f"Label names: {label_names}")

    datasets = {
        split: PipeDASHDF5Dataset(
            h5_path=h5_path,
            indices=indices,
            label_arrays=label_arrays,
            input_height=args.input_height,
            input_width=args.input_width,
            normalize=args.normalize,
        )
        for split, indices in splits.items()
    }
    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        ),
    }

    model = SensorFieldMEDHTT(
        task_output_dims=task_output_dims,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        shared_tokens=args.shared_tokens,
        raw_tokens=args.raw_tokens,
        stf_tokens=args.stf_tokens,
        gaf_tokens=args.gaf_tokens,
        stf_size=args.stf_size,
        gaf_size=args.gaf_size,
        stft_n_fft=args.stft_n_fft,
        stft_hop_length=args.stft_hop_length,
        stft_win_length=args.stft_win_length,
        propagation_steps=args.propagation_steps,
        enabled_views=args.enabled_views,
        dropout=args.dropout,
        dec_loss_weight=args.dec_loss_weight,
        cep_loss_weight=args.cep_loss_weight,
    ).to(device)

    event_weight = class_weights(
        label_arrays["event_type"],
        splits["train"],
        task_output_dims["event_type"],
        device,
    )
    condition_weight = class_weights(
        label_arrays["threat_condition"],
        splits["train"],
        task_output_dims["threat_condition"],
        device,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    run_config = {
        "args": vars(args),
        "task_output_dims": task_output_dims,
        "label_names": label_names,
        "label_maps": {
            key: {str(raw): int(compact) for raw, compact in value.items()}
            for key, value in {
                "event_raw_to_compact": metadata["event_raw_to_compact"],
                "radial_raw_to_compact": metadata["radial_raw_to_compact"],
                "condition_raw_to_compact": metadata["condition_raw_to_compact"],
            }.items()
        },
        "split_sizes": {name: int(len(indices)) for name, indices in splits.items()},
        "label_counts": label_counts(label_arrays, splits),
    }
    write_json(run_dir / "run_config.json", run_config)

    best_score = -1.0
    best_epoch = 0
    history_rows: list[dict[str, Any]] = []
    best_path = run_dir / "best.pt"
    last_path = run_dir / "last.pt"

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_metrics = run_epoch(
            model,
            loaders["train"],
            device,
            "train",
            event_weight,
            condition_weight,
            optimizer,
            args,
            label_names,
        )
        val_metrics = run_epoch(
            model,
            loaders["val"],
            device,
            "val",
            event_weight,
            condition_weight,
            None,
            args,
            label_names,
        )
        history_rows.extend([flatten_history_row(epoch, train_metrics), flatten_history_row(epoch, val_metrics)])
        write_history(run_dir / "history.csv", history_rows)

        print(
            "train "
            f"loss={train_metrics['loss']:.4f} score={train_metrics['score']:.4f} "
            f"event={train_metrics['event']['acc']:.4f} radial={train_metrics['radial']['acc']} "
            f"condition={train_metrics['condition']['acc']}"
        )
        print(
            "val   "
            f"loss={val_metrics['loss']:.4f} score={val_metrics['score']:.4f} "
            f"event={val_metrics['event']['acc']:.4f} radial={val_metrics['radial']['acc']} "
            f"condition={val_metrics['condition']['acc']}"
        )

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_metrics": strip_payload(val_metrics),
            "run_config": run_config,
        }
        torch.save(checkpoint, last_path)
        if val_metrics["score"] > best_score:
            best_score = float(val_metrics["score"])
            best_epoch = epoch
            torch.save(checkpoint, best_path)

    if best_path.is_file():
        checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics = run_epoch(
        model,
        loaders["test"],
        device,
        "test",
        event_weight,
        condition_weight,
        None,
        args,
        label_names,
    )
    write_report_artifacts(run_dir, "test", test_metrics, label_names)
    summary = {
        "best_epoch": best_epoch,
        "best_val_score": best_score,
        "test_metrics": strip_payload(test_metrics),
        "run_dir": str(run_dir),
    }
    write_json(run_dir / "summary.json", summary)
    print("\nTraining complete.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
