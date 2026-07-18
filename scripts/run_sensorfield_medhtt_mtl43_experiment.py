from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset, Subset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LIB_ROOT = PROJECT_ROOT / "libmtl_das_patch"
EXAMPLES_ROOT = LIB_ROOT / "examples" / "das_csv"
for path in (LIB_ROOT, EXAMPLES_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from create_dataset import DISTANCE_IGNORE_INDEX, MultiTaskCSVDataset, parse_label_list  # noqa: E402
from LibMTL.model import SensorFieldMEDHTT, ordinal_predictions  # noqa: E402


IGNORE_INDEX = -1
DEFAULT_EVENT_CLASSES = "walking,excavator,driving,background"
DEFAULT_DISTANCE_CLASSES = "Alarm area,Tracking area,No-threat area"
DEFAULT_CONDITION_CLASSES = "digging,walking,parallel_driving,vehicle_driving"

CONDITION_TOKENS = (
    ("knocking", ("\u6572\u51fb\u5730\u9762\u5b9a\u4f4d", "\u6572\u51fb\u5730\u9762", "\u6572\u51fb")),
    ("idle", ("\u6020\u901f", "idle")),
    ("parallel_driving", ("\u5e73\u884c\u884c\u9a76", "parallel")),
    ("crossing", ("\u5782\u76f4\u7ba1\u9053\u884c\u9a76", "\u5782\u76f4", "crossing")),
    (
        "digging",
        (
            "\u5f00\u6316\u65bd\u5de5",
            "\u6316\u6398\u5207\u524a",
            "\u6b63\u5e38\u65bd\u5de5",
            "\u65bd\u5de5",
            "\u5f00\u6316",
            "\u6316\u6398",
            "\u5207\u524a",
        ),
    ),
    ("vehicle_driving", ("\u6c7d\u8f66", "vehicle", "drive", "\u884c\u9a76")),
    ("walking", ("\u884c\u8d70", "walking", "walk", "\u4eba\u884c\u8d70")),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train SensorField-MEDHTT on converted_csv/MTL43 manifests."
    )
    parser.add_argument("--dataset-path", default=str(PROJECT_ROOT / "converted_csv" / "MTL43"), type=str)
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "output" / "sensorfield_medhtt_mtl43"), type=str)
    parser.add_argument("--event-classes", default=DEFAULT_EVENT_CLASSES, type=str)
    parser.add_argument("--distance-classes", default=DEFAULT_DISTANCE_CLASSES, type=str)
    parser.add_argument("--condition-classes", default=DEFAULT_CONDITION_CLASSES, type=str)
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--num-workers", default=0, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--max-train-samples", default=0, type=int)
    parser.add_argument("--max-val-samples", default=0, type=int)
    parser.add_argument("--max-test-samples", default=0, type=int)
    parser.add_argument("--input-height", default=6, type=int)
    parser.add_argument("--input-width", default=2048, type=int)
    parser.add_argument("--normalize", default="sample", choices=["sample", "none"])
    parser.add_argument("--train-augment", action="store_true", default=False)
    parser.add_argument("--augment-noise-std", default=0.005, type=float)
    parser.add_argument("--augment-gain-std", default=0.02, type=float)
    parser.add_argument("--augment-shift", default=50, type=int)
    parser.add_argument("--augment-mask-width", default=100, type=int)
    parser.add_argument("--augment-drop-rows", default=0, type=int)
    parser.add_argument("--location-aug-repeats", default=0, type=int)
    parser.add_argument("--location-aug-noise-std", default=0.01, type=float)
    parser.add_argument("--location-aug-gain-std", default=0.05, type=float)
    parser.add_argument("--location-aug-shift", default=120, type=int)
    parser.add_argument("--location-aug-mask-width", default=240, type=int)
    parser.add_argument("--location-aug-drop-rows", default=1, type=int)
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
    parser.add_argument("--disable-med", action="store_true", default=False)
    parser.add_argument("--disable-htt", action="store_true", default=False)
    parser.add_argument("--disable-bti", action="store_true", default=False)
    parser.add_argument("--disable-cep", action="store_true", default=False)
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
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    if name == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def infer_condition_from_path(path: Path | str) -> str:
    text = str(path).lower()
    for condition_name, tokens in CONDITION_TOKENS:
        if any(token.lower() in text for token in tokens):
            return condition_name
    return "unknown"


def ordinal_bce_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    valid = targets != IGNORE_INDEX
    if not valid.any():
        return logits.sum() * 0.0
    thresholds = torch.arange(logits.size(1), device=logits.device).view(1, -1)
    cumulative_targets = (targets[valid].view(-1, 1) > thresholds).float()
    return F.binary_cross_entropy_with_logits(logits[valid], cumulative_targets)


def masked_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    valid = targets != IGNORE_INDEX
    if not valid.any():
        return logits.sum() * 0.0
    return F.cross_entropy(logits[valid], targets[valid], weight=weight)


class MTL43MEDHTTDataset(MultiTaskCSVDataset):
    def __init__(
        self,
        manifest_path: Path,
        event_to_idx: dict[str, int],
        distance_to_idx: dict[str, int],
        condition_to_idx: dict[str, int],
        event_idx_to_name: dict[int, str],
        input_height: int,
        input_width: int,
        normalize: str,
        augment: bool,
        augment_noise_std: float,
        augment_gain_std: float,
        augment_shift: int,
        augment_mask_width: int,
        augment_drop_rows: int,
        location_aug_repeats: int,
        location_aug_noise_std: float,
        location_aug_gain_std: float,
        location_aug_shift: int,
        location_aug_mask_width: int,
        location_aug_drop_rows: int,
    ) -> None:
        super().__init__(
            manifest_path=manifest_path,
            event_to_idx=event_to_idx,
            distance_to_idx=distance_to_idx,
            input_height=input_height,
            input_width=input_width,
            sample_level="manifest",
            normalize=normalize,
            augment=augment,
            augment_noise_std=augment_noise_std,
            augment_gain_std=augment_gain_std,
            augment_shift=augment_shift,
            augment_mask_width=augment_mask_width,
            augment_drop_rows=augment_drop_rows,
            location_aug_repeats=location_aug_repeats,
            location_aug_noise_std=location_aug_noise_std,
            location_aug_gain_std=location_aug_gain_std,
            location_aug_shift=location_aug_shift,
            location_aug_mask_width=location_aug_mask_width,
            location_aug_drop_rows=location_aug_drop_rows,
            use_stft_aux=False,
        )
        self.condition_to_idx = condition_to_idx
        self.event_idx_to_name = event_idx_to_name
        for sample in self.samples:
            event_name = self.event_idx_to_name[int(sample["event_label"])]
            condition_name = infer_condition_from_path(sample["path"])
            if event_name == "background" or condition_name not in self.condition_to_idx:
                condition_label = IGNORE_INDEX
            else:
                condition_label = self.condition_to_idx[condition_name]
            sample["condition_name"] = condition_name
            sample["condition_label"] = condition_label

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        signal_tensor, labels = super().__getitem__(index)
        sample = self.samples[index]
        supervision_profile = str(sample.get("supervision_profile", "full")).strip() or "full"
        event_target = labels["event_type"]
        radial_target = labels["distance_cls"]
        condition_target = torch.tensor(int(sample["condition_label"]), dtype=torch.long)
        if supervision_profile == "radial_only":
            event_target = torch.tensor(IGNORE_INDEX, dtype=torch.long)
            condition_target = torch.tensor(IGNORE_INDEX, dtype=torch.long)
        elif supervision_profile == "event_only":
            radial_target = torch.tensor(IGNORE_INDEX, dtype=torch.long)
            condition_target = torch.tensor(IGNORE_INDEX, dtype=torch.long)
        return signal_tensor, {
            "event_type": event_target,
            "radial_threat": radial_target,
            "threat_condition": condition_target,
        }


def maybe_subset(dataset: Dataset, max_samples: int, seed: int) -> Dataset:
    if max_samples <= 0 or max_samples >= len(dataset):
        return dataset
    rng = np.random.default_rng(seed)
    indices = np.arange(len(dataset))
    rng.shuffle(indices)
    return Subset(dataset, indices[:max_samples].tolist())


def build_datasets(args: argparse.Namespace) -> tuple[dict[str, Dataset], dict[str, list[str]], dict[str, Any]]:
    dataset_path = Path(args.dataset_path).expanduser().resolve()
    event_classes = parse_label_list(args.event_classes)
    distance_classes = parse_label_list(args.distance_classes)
    condition_classes = parse_label_list(args.condition_classes)
    event_to_idx = {name: index for index, name in enumerate(event_classes)}
    distance_to_idx = {name: index for index, name in enumerate(distance_classes)}
    condition_to_idx = {name: index for index, name in enumerate(condition_classes)}
    event_idx_to_name = {index: name for name, index in event_to_idx.items()}

    datasets: dict[str, Dataset] = {}
    base_datasets: dict[str, MTL43MEDHTTDataset] = {}
    for split in ("train", "val", "test"):
        manifest_path = dataset_path / f"{split}.csv"
        dataset = MTL43MEDHTTDataset(
            manifest_path=manifest_path,
            event_to_idx=event_to_idx,
            distance_to_idx=distance_to_idx,
            condition_to_idx=condition_to_idx,
            event_idx_to_name=event_idx_to_name,
            input_height=args.input_height,
            input_width=args.input_width,
            normalize=args.normalize,
            augment=args.train_augment and split == "train",
            augment_noise_std=args.augment_noise_std,
            augment_gain_std=args.augment_gain_std,
            augment_shift=args.augment_shift,
            augment_mask_width=args.augment_mask_width,
            augment_drop_rows=args.augment_drop_rows,
            location_aug_repeats=args.location_aug_repeats,
            location_aug_noise_std=args.location_aug_noise_std,
            location_aug_gain_std=args.location_aug_gain_std,
            location_aug_shift=args.location_aug_shift,
            location_aug_mask_width=args.location_aug_mask_width,
            location_aug_drop_rows=args.location_aug_drop_rows,
        )
        base_datasets[split] = dataset
        limit = getattr(args, f"max_{split}_samples")
        datasets[split] = maybe_subset(dataset, int(limit), args.seed + len(datasets))

    label_names = {
        "event_type": event_classes,
        "radial_threat": distance_classes,
        "threat_condition": condition_classes,
    }
    metadata = {
        "dataset_path": str(dataset_path),
        "base_split_sizes": {split: len(dataset) for split, dataset in base_datasets.items()},
        "active_split_sizes": {split: len(dataset) for split, dataset in datasets.items()},
        "label_counts": summarize_label_counts(base_datasets),
    }
    return datasets, label_names, metadata


def summarize_label_counts(datasets: dict[str, MTL43MEDHTTDataset]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for split, dataset in datasets.items():
        counters = {
            "event_type": Counter(),
            "radial_threat": Counter(),
            "threat_condition": Counter(),
            "condition_name_all": Counter(),
        }
        for sample in dataset.samples:
            supervision_profile = str(sample.get("supervision_profile", "full")).strip() or "full"
            if supervision_profile in {"full", "event_only"}:
                counters["event_type"][int(sample["event_label"])] += 1
            if supervision_profile in {"full", "radial_only"} and int(sample["distance_label"]) != DISTANCE_IGNORE_INDEX:
                counters["radial_threat"][int(sample["distance_label"])] += 1
            if supervision_profile == "full" and int(sample["condition_label"]) != IGNORE_INDEX:
                counters["threat_condition"][int(sample["condition_label"])] += 1
            counters["condition_name_all"][str(sample["condition_name"])] += 1
        payload[split] = {
            name: {str(key): int(value) for key, value in counter.items()}
            for name, counter in counters.items()
        }
    return payload


def class_weights(
    samples: list[dict[str, Any]],
    key: str,
    num_classes: int,
    device: torch.device,
    predicate: Any | None = None,
) -> torch.Tensor:
    labels = []
    for sample in samples:
        if predicate is not None and not predicate(sample):
            continue
        value = int(sample[key])
        if value != IGNORE_INDEX:
            labels.append(value)
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64) if labels else np.ones(num_classes)
    weights = np.zeros(num_classes, dtype=np.float32)
    nonzero = counts > 0
    if nonzero.any():
        weights[nonzero] = float(counts[nonzero].sum()) / (float(nonzero.sum()) * counts[nonzero])
        weights[nonzero] = weights[nonzero] / max(float(weights[nonzero].mean()), 1e-6)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def get_base_dataset(dataset: Dataset) -> MTL43MEDHTTDataset:
    if isinstance(dataset, Subset):
        return dataset.dataset  # type: ignore[return-value]
    return dataset  # type: ignore[return-value]


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
        inputs = inputs.to(device)
        labels = {key: value.to(device) for key, value in labels.items()}
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

        event_valid = labels["event_type"] != IGNORE_INDEX
        if event_valid.any():
            event_pred = outputs["event_type"].detach().argmax(dim=-1)[event_valid].cpu().tolist()
            event_target = labels["event_type"][event_valid].detach().cpu().tolist()
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

        if is_train and (batch_index == 1 or batch_index % 50 == 0):
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


def write_report_artifacts(
    run_dir: Path,
    split: str,
    metrics: dict[str, Any],
    label_names: dict[str, list[str]],
) -> None:
    for task_name, names in label_names.items():
        task_targets = metrics["_targets"][task_name]
        task_predictions = metrics["_predictions"][task_name]
        if not task_targets:
            continue
        labels = list(range(len(names)))
        report = classification_report(
            task_targets,
            task_predictions,
            labels=labels,
            target_names=names,
            zero_division=0,
            output_dict=True,
        )
        matrix = confusion_matrix(task_targets, task_predictions, labels=labels)
        write_json(run_dir / f"{split}_{task_name}_classification_report.json", report)
        with (run_dir / f"{split}_{task_name}_confusion.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow([""] + names)
            for name, row in zip(names, matrix.tolist()):
                writer.writerow([name] + row)


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    output_root = Path(args.output_dir).expanduser().resolve()
    run_dir = output_root / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    datasets, label_names, metadata = build_datasets(args)
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
    task_output_dims = {
        "event_type": len(label_names["event_type"]),
        "radial_threat": len(label_names["radial_threat"]),
        "threat_condition": len(label_names["threat_condition"]),
    }
    print(f"Run directory: {run_dir}")
    print(f"Device: {device}")
    print(f"Task output dims: {task_output_dims}")
    print(f"Split sizes: {metadata['active_split_sizes']}")
    print(f"Label names: {label_names}")

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
        disable_med=args.disable_med,
        disable_htt=args.disable_htt,
        disable_bti=args.disable_bti,
        disable_cep=args.disable_cep,
    ).to(device)

    train_base = get_base_dataset(datasets["train"])
    event_weight = class_weights(
        train_base.samples,
        "event_label",
        task_output_dims["event_type"],
        device,
        predicate=lambda sample: str(sample.get("supervision_profile", "full")).strip() in {"full", "event_only"},
    )
    condition_weight = class_weights(
        train_base.samples,
        "condition_label",
        task_output_dims["threat_condition"],
        device,
        predicate=lambda sample: str(sample.get("supervision_profile", "full")).strip() == "full",
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    run_config = {
        "args": vars(args),
        "task_output_dims": task_output_dims,
        "label_names": label_names,
        "metadata": metadata,
    }
    write_json(run_dir / "run_config.json", run_config)

    history_rows: list[dict[str, Any]] = []
    best_score = -1.0
    best_epoch = 0
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
