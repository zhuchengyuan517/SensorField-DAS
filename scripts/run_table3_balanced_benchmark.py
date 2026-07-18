from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LIB_ROOT = PROJECT_ROOT / "libmtl_das_patch"
EXAMPLE_ROOT = LIB_ROOT / "examples" / "das_csv"
for path in (LIB_ROOT, EXAMPLE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from create_dataset import canonicalize_distance_label  # noqa: E402
from create_dataset_mmit import MMITEventDataset  # noqa: E402
from single_task_signal import save_histories, save_json, save_confusion_artifacts  # noqa: E402
from LibMTL.model import DASMultiModalNet, FocalLoss, SensorFieldM3T, compute_total_loss  # noqa: E402
from LibMTL.model.das_multimodal_net import GAFBranch, ResidualBlock2D  # noqa: E402


DEFAULT_DATASET_ROOT = (
    PROJECT_ROOT
    / "converted_csv"
    / "MTL43"
    / "_single_task_manifests"
    / "event_only_balanced1500_drive1350"
)
DEFAULT_LOCATION_DATASET_ROOT = (
    PROJECT_ROOT / "converted_csv" / "MTL43" / "_single_task_manifests" / "location_only"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "output" / "table3_balanced_benchmark"
EVENT_CLASSES = ("walking", "excavator", "driving", "background")
DISTANCE_CLASSES = ("Alarm area", "Tracking area", "No-threat area")


@dataclass(frozen=True)
class SingleTaskSpec:
    name: str
    label_column: str
    target_key: str
    output_key: str
    class_names: tuple[str, ...]
    display_name: str


def get_task_spec(task_name: str) -> SingleTaskSpec:
    if task_name == "event":
        return SingleTaskSpec(
            name="event",
            label_column="event_label",
            target_key="event_type",
            output_key="event_type",
            class_names=EVENT_CLASSES,
            display_name="Event type",
        )
    if task_name == "location":
        return SingleTaskSpec(
            name="location",
            label_column="distance_label",
            target_key="distance_cls",
            output_key="distance_cls",
            class_names=DISTANCE_CLASSES,
            display_name="Location",
        )
    raise ValueError(f"Unsupported task: {task_name}")


def canonicalize_task_label(task_name: str, label_text: str) -> str:
    if task_name == "location":
        return canonicalize_distance_label(label_text)
    return str(label_text).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rerun single-task DAS baselines on the balanced DAS benchmark."
    )
    parser.add_argument("--task", default="event", choices=["event", "location"])
    parser.add_argument("--dataset_root", default=str(DEFAULT_DATASET_ROOT), type=str)
    parser.add_argument("--output_root", default=str(DEFAULT_OUTPUT_ROOT), type=str)
    parser.add_argument("--models", default="resnet,vgg,vit,proposed", type=str)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--batch_size", default=24, type=int)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--sample_level", default="manifest", choices=["manifest", "file", "row", "group3"])
    parser.add_argument("--normalize", default="sample", choices=["sample", "none"])
    parser.add_argument("--raw_length", default=4096, type=int)
    parser.add_argument("--stft_size", default=128, type=int)
    parser.add_argument("--gaf_size", default=128, type=int)
    parser.add_argument("--stft_n_fft", default=256, type=int)
    parser.add_argument("--stft_hop_length", default=128, type=int)
    parser.add_argument("--stft_win_length", default=256, type=int)
    parser.add_argument("--lr", default=3e-4, type=float)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--step_size", default=10, type=int)
    parser.add_argument("--gamma", default=0.5, type=float)
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--d_model", default=128, type=int)
    parser.add_argument("--num_heads", default=4, type=int)
    parser.add_argument("--sensorfield_num_anchors", default=8, type=int)
    parser.add_argument("--sensorfield_raw_tokens", default=6, type=int)
    parser.add_argument("--sensorfield_stf_tokens", default=48, type=int)
    parser.add_argument("--sensorfield_gaf_tokens", default=48, type=int)
    parser.add_argument("--sensorfield_fac_loss_weight", default=0.0, type=float)
    parser.add_argument("--sensorfield_taef_loss_weight", default=0.0, type=float)
    parser.add_argument("--sensorfield_gcti_loss_weight", default=0.0, type=float)
    parser.add_argument("--sensorfield_enabled_views", default="raw,stf,gaf", type=str)
    parser.add_argument("--gaf_patch_size", default=8, type=int)
    parser.add_argument("--gaf_depth", default=4, type=int)
    parser.add_argument("--lambda_value", default=0.1, type=float)
    parser.add_argument("--focal_gamma", default=2.0, type=float)
    parser.add_argument(
        "--train_sampler",
        default="event_balanced",
        choices=["none", "event_balanced", "class_balanced", "distance_balanced"],
    )
    parser.add_argument("--train_augment", action="store_true", default=True)
    parser.add_argument("--augment_noise_std", default=0.01, type=float)
    parser.add_argument("--augment_gain_std", default=0.05, type=float)
    parser.add_argument("--augment_shift", default=128, type=int)
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


def build_run_root(output_root: Path) -> Path:
    return output_root / datetime.now().strftime("%Y%m%d_%H%M%S")


class MMITSingleTaskDataset(MMITEventDataset):
    """MMIT view dataset with a configurable single-task target.

    Returned views keep the existing shapes:
    raw [1, L], stft [1, H, W], gaf [1, H, W].
    Labels use task-specific keys such as event_type or distance_cls.
    """

    def __init__(
        self,
        manifest_path: Path,
        task_spec: SingleTaskSpec,
        raw_length: int,
        stft_size: int,
        gaf_size: int,
        sample_level: str,
        normalize: str,
        augment: bool,
        augment_noise_std: float,
        augment_gain_std: float,
        augment_shift: int,
        stft_n_fft: int,
        stft_hop_length: int,
        stft_win_length: int,
    ) -> None:
        self.task_spec = task_spec
        self.label_to_idx = {label: index for index, label in enumerate(task_spec.class_names)}
        super().__init__(
            manifest_path=manifest_path,
            event_to_idx={},
            raw_length=raw_length,
            stft_size=stft_size,
            gaf_size=gaf_size,
            sample_level=sample_level,
            normalize=normalize,
            augment=augment,
            augment_noise_std=augment_noise_std,
            augment_gain_std=augment_gain_std,
            augment_shift=augment_shift,
            stft_n_fft=stft_n_fft,
            stft_hop_length=stft_hop_length,
            stft_win_length=stft_win_length,
        )

    def _read_manifest(self):
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        samples = []
        with self.manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            required_columns = {"path", self.task_spec.label_column}
            if reader.fieldnames is None or not required_columns.issubset(set(reader.fieldnames)):
                raise ValueError(
                    f"{self.manifest_path} must contain columns: "
                    f"{','.join(sorted(required_columns))}"
                )
            for row in reader:
                path = Path(row["path"])
                if not path.is_absolute():
                    path = (self.manifest_path.parent / path).resolve()
                if not path.is_file():
                    raise FileNotFoundError(f"CSV sample does not exist: {path}")

                label_text = canonicalize_task_label(
                    self.task_spec.name,
                    row.get(self.task_spec.label_column, ""),
                )
                if label_text not in self.label_to_idx:
                    raise KeyError(
                        f"Unknown {self.task_spec.label_column} label '{label_text}' "
                        f"in {self.manifest_path}"
                    )

                sample_mode = row.get("sample_mode", "").strip() or "file"
                if self.sample_level != "manifest":
                    sample_mode = self.sample_level
                row_count = self._count_rows(path)
                target_label = self.label_to_idx[label_text]

                if sample_mode == "file":
                    samples.append(
                        {
                            "path": path,
                            "sample_mode": "file",
                            "target_label": target_label,
                            "label_text": label_text,
                        }
                    )
                elif sample_mode == "group3":
                    group_count = row_count // 3
                    for group_index in range(group_count):
                        samples.append(
                            {
                                "path": path,
                                "sample_mode": "group3",
                                "row_start": group_index * 3,
                                "row_end": group_index * 3 + 3,
                                "target_label": target_label,
                                "label_text": label_text,
                            }
                        )
                elif sample_mode == "row":
                    for row_index in range(row_count):
                        samples.append(
                            {
                                "path": path,
                                "sample_mode": "row",
                                "row_index": row_index,
                                "target_label": target_label,
                                "label_text": label_text,
                            }
                        )
                else:
                    raise ValueError(f"Unsupported sample mode '{sample_mode}' in {self.manifest_path}")

        if not samples:
            raise ValueError(f"No usable samples found in {self.manifest_path}")
        return samples

    def __getitem__(self, index):
        sample = self.samples[index]
        signal_2d = self._augment_signal(self._extract_signal(sample))
        raw_signal = self._normalize_signal(self._flatten_signal(signal_2d))  # [1, L]
        stft = self._compute_stft(raw_signal)  # [1, H, W]
        gaf = self._compute_gaf(raw_signal)  # [1, H, W]
        target = torch.tensor(sample["target_label"], dtype=torch.long)
        return {"raw": raw_signal, "stft": stft, "gaf": gaf}, {self.task_spec.target_key: target}


class GAFOnlySingleTaskDataset(Dataset):
    def __init__(
        self,
        manifest_path: Path,
        task_spec: SingleTaskSpec,
        raw_length: int,
        stft_size: int,
        gaf_size: int,
        sample_level: str,
        normalize: str,
        augment: bool,
        augment_noise_std: float,
        augment_gain_std: float,
        augment_shift: int,
        stft_n_fft: int,
        stft_hop_length: int,
        stft_win_length: int,
    ) -> None:
        self.task_spec = task_spec
        self.base_dataset = MMITSingleTaskDataset(
            manifest_path=manifest_path,
            task_spec=task_spec,
            raw_length=raw_length,
            stft_size=stft_size,
            gaf_size=gaf_size,
            sample_level=sample_level,
            normalize=normalize,
            augment=augment,
            augment_noise_std=augment_noise_std,
            augment_gain_std=augment_gain_std,
            augment_shift=augment_shift,
            stft_n_fft=stft_n_fft,
            stft_hop_length=stft_hop_length,
            stft_win_length=stft_win_length,
        )
        self.samples = self.base_dataset.samples

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample, labels = self.base_dataset[index]
        return sample["gaf"], labels[self.task_spec.target_key]


def gaf_collate(batch):
    images = torch.stack([inputs for inputs, _ in batch], dim=0)
    targets = torch.stack([target for _, target in batch], dim=0)
    return images, targets


def build_gaf_dataloaders(args: argparse.Namespace) -> dict[str, DataLoader]:
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    task_spec = get_task_spec(args.task)
    loaders: dict[str, DataLoader] = {}
    for split in ("train", "val", "test"):
        dataset = GAFOnlySingleTaskDataset(
            manifest_path=dataset_root / f"{split}.csv",
            task_spec=task_spec,
            raw_length=args.raw_length,
            stft_size=args.stft_size,
            gaf_size=args.gaf_size,
            sample_level=args.sample_level,
            normalize=args.normalize,
            augment=(args.train_augment and split == "train"),
            augment_noise_std=args.augment_noise_std,
            augment_gain_std=args.augment_gain_std,
            augment_shift=args.augment_shift,
            stft_n_fft=args.stft_n_fft,
            stft_hop_length=args.stft_hop_length,
            stft_win_length=args.stft_win_length,
        )
        sampler = None
        shuffle = split == "train"
        if split == "train" and args.train_sampler != "none":
            counts = np.zeros(len(task_spec.class_names), dtype=np.float64)
            for sample in dataset.samples:
                counts[int(sample["target_label"])] += 1.0
            counts = np.maximum(counts, 1.0)
            weights = torch.as_tensor(
                [1.0 / counts[int(sample["target_label"])] for sample in dataset.samples],
                dtype=torch.double,
            )
            sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
            shuffle = False
        loaders[split] = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=gaf_collate,
        )
    return loaders


def build_multimodal_dataloaders(args: argparse.Namespace) -> dict[str, DataLoader]:
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    task_spec = get_task_spec(args.task)
    loaders: dict[str, DataLoader] = {}
    for split in ("train", "val", "test"):
        dataset = MMITSingleTaskDataset(
            manifest_path=dataset_root / f"{split}.csv",
            task_spec=task_spec,
            raw_length=args.raw_length,
            stft_size=args.stft_size,
            gaf_size=args.gaf_size,
            sample_level=args.sample_level,
            normalize=args.normalize,
            augment=(args.train_augment and split == "train"),
            augment_noise_std=args.augment_noise_std,
            augment_gain_std=args.augment_gain_std,
            augment_shift=args.augment_shift,
            stft_n_fft=args.stft_n_fft,
            stft_hop_length=args.stft_hop_length,
            stft_win_length=args.stft_win_length,
        )
        sampler = None
        shuffle = split == "train"
        if split == "train" and args.train_sampler != "none":
            counts = np.zeros(len(task_spec.class_names), dtype=np.float64)
            for sample in dataset.samples:
                counts[int(sample["target_label"])] += 1.0
            counts = np.maximum(counts, 1.0)
            weights = torch.as_tensor(
                [1.0 / counts[int(sample["target_label"])] for sample in dataset.samples],
                dtype=torch.double,
            )
            sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
            shuffle = False
        loaders[split] = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=lambda batch: (
                {
                    "raw": torch.stack([sample["raw"] for sample, _ in batch], dim=0),
                    "stft": torch.stack([sample["stft"] for sample, _ in batch], dim=0),
                    "gaf": torch.stack([sample["gaf"] for sample, _ in batch], dim=0),
                },
                {task_spec.target_key: torch.stack([labels[task_spec.target_key] for _, labels in batch], dim=0)},
            ),
        )
    return loaders


class GAFResNetClassifier(nn.Module):
    def __init__(self, num_classes: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
        )
        self.layer1 = nn.Sequential(
            ResidualBlock2D(32, 64, stride=2),
            ResidualBlock2D(64, 64, stride=1),
        )
        self.layer2 = nn.Sequential(
            ResidualBlock2D(64, 128, stride=2),
            ResidualBlock2D(128, 128, stride=1),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.stem(inputs)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.pool(x)
        return self.head(x)


class GAFVGGClassifier(nn.Module):
    def __init__(self, num_classes: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.features(inputs)
        x = self.pool(x)
        return self.head(x)


class GAFViTClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        d_model: int,
        num_heads: int,
        patch_size: int,
        depth: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.backbone = GAFBranch(
            in_channels=1,
            embed_dim=d_model,
            patch_size=patch_size,
            num_heads=num_heads,
            depth=depth,
            dropout=dropout,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.backbone(inputs)
        return self.head(features)


def build_baseline_model(
    model_name: str,
    args: argparse.Namespace,
    task_spec: SingleTaskSpec,
) -> nn.Module:
    num_classes = len(task_spec.class_names)
    if model_name == "resnet":
        return GAFResNetClassifier(num_classes=num_classes, dropout=args.dropout)
    if model_name == "vgg":
        return GAFVGGClassifier(num_classes=num_classes, dropout=args.dropout)
    if model_name == "vit":
        return GAFViTClassifier(
            num_classes=num_classes,
            d_model=args.d_model,
            num_heads=args.num_heads,
            patch_size=args.gaf_patch_size,
            depth=args.gaf_depth,
            dropout=args.dropout,
        )
    if model_name == "proposed":
        return DASMultiModalNet(
            num_classes=num_classes,
            d_model=args.d_model,
            tf_branch_dim=args.d_model,
            temporal_branch_dim=args.d_model,
            gaf_patch_size=args.gaf_patch_size,
            gaf_depth=max(2, args.gaf_depth // 2),
            num_heads=args.num_heads,
            dropout=args.dropout,
        )
    if model_name == "sensorfield_m3t":
        return SensorFieldM3T(
            task_output_dims={task_spec.output_key: num_classes},
            hidden_dim=args.d_model,
            num_anchors=args.sensorfield_num_anchors,
            num_heads=args.num_heads,
            raw_tokens=args.sensorfield_raw_tokens,
            stf_tokens=args.sensorfield_stf_tokens,
            gaf_tokens=args.sensorfield_gaf_tokens,
            stf_size=args.stft_size,
            gaf_size=args.gaf_size,
            stft_n_fft=args.stft_n_fft,
            stft_hop_length=args.stft_hop_length,
            stft_win_length=args.stft_win_length,
            fac_loss_weight=args.sensorfield_fac_loss_weight,
            taef_loss_weight=args.sensorfield_taef_loss_weight,
            gcti_loss_weight=args.sensorfield_gcti_loss_weight,
            view_drop_prob=0.0,
            enable_view_consistency=False,
            enabled_views=args.sensorfield_enabled_views,
            dropout=args.dropout,
        )
    raise ValueError(f"Unsupported model: {model_name}")


def build_confusion_matrix(targets: list[int], predictions: list[int], num_classes: int) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for target, prediction in zip(targets, predictions):
        matrix[int(target), int(prediction)] += 1
    return matrix


def compute_macro_metrics(matrix: np.ndarray) -> dict[str, float]:
    precisions = []
    recalls = []
    f1_scores = []
    total = int(matrix.sum())
    correct = int(np.trace(matrix))
    for idx in range(matrix.shape[0]):
        tp = float(matrix[idx, idx])
        fp = float(matrix[:, idx].sum() - tp)
        fn = float(matrix[idx, :].sum() - tp)
        precision = tp / max(tp + fp, 1.0)
        recall = tp / max(tp + fn, 1.0)
        f1 = 0.0 if precision + recall == 0 else (2.0 * precision * recall) / (precision + recall)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
    return {
        "acc": correct / max(total, 1),
        "macro_precision": float(np.mean(precisions)) if precisions else 0.0,
        "macro_recall": float(np.mean(recalls)) if recalls else 0.0,
        "macro_f1": float(np.mean(f1_scores)) if f1_scores else 0.0,
        "num_samples": total,
    }


def run_baseline_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: AdamW | None,
    collect_predictions: bool,
    num_classes: int,
) -> dict[str, float | list[int]]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_samples = 0
    total_correct = 0
    targets_all: list[int] = []
    predictions_all: list[int] = []

    progress = tqdm(loader, leave=False)
    for images, targets in progress:
        images = images.to(device)
        targets = targets.to(device)
        with torch.set_grad_enabled(is_train):
            logits = model(images)
            loss = criterion(logits, targets)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        predictions = logits.argmax(dim=1)
        batch_size = images.size(0)
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size
        total_correct += int((predictions == targets).sum().item())
        if collect_predictions:
            targets_all.extend(targets.detach().cpu().tolist())
            predictions_all.extend(predictions.detach().cpu().tolist())
        progress.set_postfix(loss=f"{total_loss / max(total_samples, 1):.4f}")

    metrics: dict[str, float | list[int]] = {
        "loss": total_loss / max(total_samples, 1),
        "acc": total_correct / max(total_samples, 1),
        "num_samples": total_samples,
    }
    if collect_predictions:
        matrix = build_confusion_matrix(targets_all, predictions_all, num_classes)
        metrics.update(compute_macro_metrics(matrix))
        metrics["targets"] = targets_all
        metrics["predictions"] = predictions_all
    return metrics


def run_proposed_epoch(
    model: DASMultiModalNet,
    loader: DataLoader,
    device: torch.device,
    criterion: FocalLoss,
    lambda_value: float,
    optimizer: AdamW | None,
    collect_predictions: bool,
    target_key: str,
    num_classes: int,
) -> dict[str, float | list[int]]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_cls = 0.0
    total_dec = 0.0
    total_cons = 0.0
    total_samples = 0
    total_correct = 0
    targets_all: list[int] = []
    predictions_all: list[int] = []

    progress = tqdm(loader, leave=False)
    for batch_inputs, batch_labels in progress:
        raw = batch_inputs["raw"].to(device)
        stft = batch_inputs["stft"].to(device)
        gaf = batch_inputs["gaf"].to(device)
        targets = batch_labels[target_key].to(device)

        with torch.set_grad_enabled(is_train):
            outputs = model(raw, stft, gaf)
            loss_out = compute_total_loss(outputs, targets, criterion, lambda1=lambda_value, lambda2=lambda_value)
            loss = loss_out.total_loss
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        predictions = outputs["logits"].argmax(dim=1)
        batch_size = raw.size(0)
        total_samples += batch_size
        total_loss += float(loss.item()) * batch_size
        total_cls += float(loss_out.cls_loss.item()) * batch_size
        total_dec += float(loss_out.disentanglement_loss.item()) * batch_size
        total_cons += float(loss_out.consistency_loss.item()) * batch_size
        total_correct += int((predictions == targets).sum().item())
        if collect_predictions:
            targets_all.extend(targets.detach().cpu().tolist())
            predictions_all.extend(predictions.detach().cpu().tolist())
        progress.set_postfix(loss=f"{total_loss / max(total_samples, 1):.4f}")

    metrics: dict[str, float | list[int]] = {
        "loss": total_loss / max(total_samples, 1),
        "cls_loss": total_cls / max(total_samples, 1),
        "dec_loss": total_dec / max(total_samples, 1),
        "cons_loss": total_cons / max(total_samples, 1),
        "acc": total_correct / max(total_samples, 1),
        "num_samples": total_samples,
    }
    if collect_predictions:
        matrix = build_confusion_matrix(targets_all, predictions_all, num_classes)
        metrics.update(compute_macro_metrics(matrix))
        metrics["targets"] = targets_all
        metrics["predictions"] = predictions_all
    return metrics


def run_sensorfield_epoch(
    model: SensorFieldM3T,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: AdamW | None,
    collect_predictions: bool,
    target_key: str,
    output_key: str,
    num_classes: int,
) -> dict[str, float | list[int]]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_cls = 0.0
    total_aux = 0.0
    total_samples = 0
    total_correct = 0
    targets_all: list[int] = []
    predictions_all: list[int] = []

    progress = tqdm(loader, leave=False)
    for batch_inputs, batch_labels in progress:
        inputs = {
            "raw": batch_inputs["raw"].to(device),
            "stf": batch_inputs["stft"].to(device),
            "gaf": batch_inputs["gaf"].to(device),
        }
        targets = batch_labels[target_key].to(device)

        with torch.set_grad_enabled(is_train):
            outputs = model(inputs)
            logits = outputs[output_key]
            cls_loss = criterion(logits, targets)
            aux_losses = outputs.get("aux_losses", {})
            aux_loss = logits.sum() * 0.0
            for value in aux_losses.values():
                if torch.is_tensor(value):
                    aux_loss = aux_loss + value
            loss = cls_loss + aux_loss
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        predictions = logits.argmax(dim=1)
        batch_size = targets.size(0)
        total_samples += batch_size
        total_loss += float(loss.item()) * batch_size
        total_cls += float(cls_loss.item()) * batch_size
        total_aux += float(aux_loss.detach().item()) * batch_size
        total_correct += int((predictions == targets).sum().item())
        if collect_predictions:
            targets_all.extend(targets.detach().cpu().tolist())
            predictions_all.extend(predictions.detach().cpu().tolist())
        progress.set_postfix(loss=f"{total_loss / max(total_samples, 1):.4f}")

    metrics: dict[str, float | list[int]] = {
        "loss": total_loss / max(total_samples, 1),
        "cls_loss": total_cls / max(total_samples, 1),
        "aux_loss": total_aux / max(total_samples, 1),
        "acc": total_correct / max(total_samples, 1),
        "num_samples": total_samples,
    }
    if collect_predictions:
        matrix = build_confusion_matrix(targets_all, predictions_all, num_classes)
        metrics.update(compute_macro_metrics(matrix))
        metrics["targets"] = targets_all
        metrics["predictions"] = predictions_all
    return metrics


def train_one_model(
    model_name: str,
    args: argparse.Namespace,
    run_root: Path,
    device: torch.device,
) -> dict[str, object]:
    task_spec = get_task_spec(args.task)
    num_classes = len(task_spec.class_names)
    model_dir = run_root / model_name
    history_dir = model_dir / "history"
    model_dir.mkdir(parents=True, exist_ok=True)
    history_dir.mkdir(parents=True, exist_ok=True)
    save_json(model_dir / "run_config.json", vars(args) | {"model_name": model_name})

    if model_name in {"proposed", "sensorfield_m3t"}:
        loaders = build_multimodal_dataloaders(args)
    else:
        loaders = build_gaf_dataloaders(args)

    model = build_baseline_model(model_name, args, task_spec).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    criterion = FocalLoss(gamma=args.focal_gamma) if model_name == "proposed" else nn.CrossEntropyLoss()

    train_history = []
    val_history = []
    test_history = []
    best_val_f1 = float("-inf")
    best_epoch = 0
    best_val_metrics: dict[str, object] | None = None
    best_test_metrics: dict[str, object] | None = None

    for epoch in range(1, args.epochs + 1):
        print(f"\n[{model_name}] Epoch {epoch}/{args.epochs}")
        if model_name == "proposed":
            train_metrics = run_proposed_epoch(
                model=model,
                loader=loaders["train"],
                device=device,
                criterion=criterion,
                lambda_value=args.lambda_value,
                optimizer=optimizer,
                collect_predictions=False,
                target_key=task_spec.target_key,
                num_classes=num_classes,
            )
            val_metrics = run_proposed_epoch(
                model=model,
                loader=loaders["val"],
                device=device,
                criterion=criterion,
                lambda_value=args.lambda_value,
                optimizer=None,
                collect_predictions=True,
                target_key=task_spec.target_key,
                num_classes=num_classes,
            )
        elif model_name == "sensorfield_m3t":
            train_metrics = run_sensorfield_epoch(
                model=model,
                loader=loaders["train"],
                device=device,
                criterion=criterion,
                optimizer=optimizer,
                collect_predictions=False,
                target_key=task_spec.target_key,
                output_key=task_spec.output_key,
                num_classes=num_classes,
            )
            val_metrics = run_sensorfield_epoch(
                model=model,
                loader=loaders["val"],
                device=device,
                criterion=criterion,
                optimizer=None,
                collect_predictions=True,
                target_key=task_spec.target_key,
                output_key=task_spec.output_key,
                num_classes=num_classes,
            )
        else:
            train_metrics = run_baseline_epoch(
                model=model,
                loader=loaders["train"],
                device=device,
                criterion=criterion,
                optimizer=optimizer,
                collect_predictions=False,
                num_classes=num_classes,
            )
            val_metrics = run_baseline_epoch(
                model=model,
                loader=loaders["val"],
                device=device,
                criterion=criterion,
                optimizer=None,
                collect_predictions=True,
                num_classes=num_classes,
            )

        scheduler.step()
        current_lr = float(optimizer.param_groups[0]["lr"])
        train_metrics["epoch"] = epoch
        train_metrics["lr"] = current_lr
        val_metrics["epoch"] = epoch
        val_metrics["lr"] = current_lr
        train_history.append({k: v for k, v in train_metrics.items() if k not in {"targets", "predictions"}})
        val_history.append({k: v for k, v in val_metrics.items() if k not in {"targets", "predictions"}})
        save_histories(history_dir, train_history, val_history, test_history)

        current_val_f1 = float(val_metrics["macro_f1"])
        print(
            f"[{model_name}] val precision={float(val_metrics['macro_precision']):.4f} "
            f"recall={float(val_metrics['macro_recall']):.4f} "
            f"f1={current_val_f1:.4f}"
        )

        if current_val_f1 >= best_val_f1:
            best_val_f1 = current_val_f1
            best_epoch = epoch
            best_val_metrics = dict(val_metrics)
            torch.save(model.state_dict(), model_dir / "best.pt")
            save_confusion_artifacts(history_dir, "best_val", val_metrics, list(task_spec.class_names))

            if model_name == "proposed":
                test_metrics = run_proposed_epoch(
                    model=model,
                    loader=loaders["test"],
                    device=device,
                    criterion=criterion,
                    lambda_value=args.lambda_value,
                    optimizer=None,
                    collect_predictions=True,
                    target_key=task_spec.target_key,
                    num_classes=num_classes,
                )
            elif model_name == "sensorfield_m3t":
                test_metrics = run_sensorfield_epoch(
                    model=model,
                    loader=loaders["test"],
                    device=device,
                    criterion=criterion,
                    optimizer=None,
                    collect_predictions=True,
                    target_key=task_spec.target_key,
                    output_key=task_spec.output_key,
                    num_classes=num_classes,
                )
            else:
                test_metrics = run_baseline_epoch(
                    model=model,
                    loader=loaders["test"],
                    device=device,
                    criterion=criterion,
                    optimizer=None,
                    collect_predictions=True,
                    num_classes=num_classes,
                )
            test_metrics["epoch"] = epoch
            best_test_metrics = dict(test_metrics)
            test_history.append({k: v for k, v in test_metrics.items() if k not in {"targets", "predictions"}})
            save_histories(history_dir, train_history, val_history, test_history)
            save_confusion_artifacts(history_dir, "best_test", test_metrics, list(task_spec.class_names))
            print(
                f"[{model_name}] test precision={float(test_metrics['macro_precision']):.4f} "
                f"recall={float(test_metrics['macro_recall']):.4f} "
                f"f1={float(test_metrics['macro_f1']):.4f}"
            )

    torch.save(model.state_dict(), model_dir / "last.pt")
    summary = {
        "model_name": model_name,
        "task": task_spec.name,
        "class_names": list(task_spec.class_names),
        "best_epoch": best_epoch,
        "best_val_metrics": best_val_metrics,
        "best_test_metrics": best_test_metrics,
        "device": str(device),
    }
    save_json(model_dir / "summary.json", summary)
    return summary


def format_float(value: float) -> str:
    return f"{float(value):.4f}"


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    models = [item.strip().lower() for item in args.models.split(",") if item.strip()]
    default_event_root = DEFAULT_DATASET_ROOT.expanduser().resolve()
    requested_root = Path(args.dataset_root).expanduser().resolve()
    if args.task == "location" and requested_root == default_event_root:
        args.dataset_root = str(DEFAULT_LOCATION_DATASET_ROOT)
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")
    task_spec = get_task_spec(args.task)

    run_root = build_run_root(Path(args.output_root).expanduser().resolve())
    run_root.mkdir(parents=True, exist_ok=True)
    print(f"Dataset Root: {dataset_root}")
    print(f"Run Root: {run_root}")
    print(f"Device: {device}")
    print(f"Task: {task_spec.display_name} ({task_spec.name})")
    print(f"Classes: {list(task_spec.class_names)}")
    print(f"Models: {models}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")

    summaries = []
    for model_name in models:
        summary = train_one_model(model_name=model_name, args=args, run_root=run_root, device=device)
        val_metrics = summary["best_val_metrics"] or {}
        test_metrics = summary["best_test_metrics"] or {}
        summaries.append(
            {
                "Task": args.task,
                "Model": model_name,
                "Best Epoch": summary["best_epoch"],
                "Val Precision": format_float(val_metrics.get("macro_precision", 0.0)),
                "Val Recall": format_float(val_metrics.get("macro_recall", 0.0)),
                "Val F1": format_float(val_metrics.get("macro_f1", 0.0)),
                "Test Precision": format_float(test_metrics.get("macro_precision", 0.0)),
                "Test Recall": format_float(test_metrics.get("macro_recall", 0.0)),
                "Test F1": format_float(test_metrics.get("macro_f1", 0.0)),
                "Source": str(run_root / model_name),
            }
        )

    summary_path = run_root / "table3_summary.csv"
    with summary_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)
    save_json(run_root / "table3_summary.json", summaries)
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
