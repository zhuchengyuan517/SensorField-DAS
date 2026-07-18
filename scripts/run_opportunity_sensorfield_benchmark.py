from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
LIBMTL_ROOT = ROOT / "libmtl_das_patch"
if str(LIBMTL_ROOT) not in sys.path:
    sys.path.insert(0, str(LIBMTL_ROOT))

from LibMTL.model.sensorfield_m3t import SensorFieldM3T  # noqa: E402


IGNORE_INDEX = -100
DATA_URL = "https://archive.ics.uci.edu/static/public/226/opportunity+activity+recognition.zip"

TASK_SPECS = {
    "locomotion": {
        "label_col": 243,
        "label_names": {
            1: "Stand",
            2: "Walk",
            4: "Sit",
            5: "Lie",
        },
    },
    "gesture": {
        "label_col": 249,
        "label_names": {
            406516: "Open Door 1",
            406517: "Open Door 2",
            404516: "Close Door 1",
            404517: "Close Door 2",
            406520: "Open Fridge",
            404520: "Close Fridge",
            406505: "Open Dishwasher",
            404505: "Close Dishwasher",
            406519: "Open Drawer 1",
            404519: "Close Drawer 1",
            406511: "Open Drawer 2",
            404511: "Close Drawer 2",
            406508: "Open Drawer 3",
            404508: "Close Drawer 3",
            408512: "Clean Table",
            407521: "Drink from Cup",
            405506: "Toggle Switch",
        },
    },
    "activity": {
        "label_col": 244,
        "label_names": {
            101: "Relaxing",
            102: "Coffee time",
            103: "Early morning",
            104: "Cleanup",
            105: "Sandwich time",
        },
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SensorField-M3T and HAR baselines on OPPORTUNITY cross-subject multitask recognition."
    )
    parser.add_argument("--dataset_root", default=str(ROOT / "_datasets" / "opportunity" / "OpportunityUCIDataset"), type=str)
    parser.add_argument("--output_root", default=str(ROOT / "_tmp_opportunity_sensorfield"), type=str)
    parser.add_argument("--feature_group", default="body", choices=["body", "all", "object", "ambient"], type=str)
    parser.add_argument("--window_size", default=90, type=int)
    parser.add_argument("--stride", default=45, type=int)
    parser.add_argument("--heldout_subject", default=4, type=int)
    parser.add_argument("--val_subject", default=3, type=int)
    parser.add_argument(
        "--val_fraction",
        default=0.20,
        type=float,
        help="If val_subject <= 0, sample this fraction from non-test source subjects for validation.",
    )
    parser.add_argument("--seeds", default="42", type=str)
    parser.add_argument(
        "--models",
        default="sensorfield_m3t,deepconvlstm,tinyhar,temporal_transformer,attend_discriminate",
        type=str,
    )
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--lr", default=3e-4, type=float)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--label_smoothing", default=0.0, type=float)
    parser.add_argument("--train_noise_std", default=0.0, type=float)
    parser.add_argument("--channel_dropout_prob", default=0.0, type=float)
    parser.add_argument("--hidden_dim", default=96, type=int)
    parser.add_argument("--num_anchors", default=8, type=int)
    parser.add_argument("--num_heads", default=4, type=int)
    parser.add_argument("--dropout", default=0.15, type=float)
    parser.add_argument("--fac_loss_weight", default=0.05, type=float)
    parser.add_argument("--taef_loss_weight", default=0.01, type=float)
    parser.add_argument("--gcti_loss_weight", default=0.01, type=float)
    parser.add_argument("--view_drop_prob", default=0.10, type=float)
    parser.add_argument("--early_stop_min_epochs", default=0, type=int)
    parser.add_argument("--early_stop_patience", default=0, type=int)
    parser.add_argument("--early_stop_min_delta", default=0.0, type=float)
    parser.add_argument("--max_windows_per_subject", default=0, type=int)
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Run a very small deterministic smoke test.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_int_list(payload: str) -> list[int]:
    return [int(item.strip()) for item in payload.split(",") if item.strip()]


def parse_model_list(payload: str) -> list[str]:
    return [item.strip().lower() for item in payload.split(",") if item.strip()]


def feature_indices(group: str) -> np.ndarray:
    if group == "all":
        return np.arange(1, 243, dtype=np.int64)
    if group == "object":
        return np.arange(134, 194, dtype=np.int64)
    if group == "ambient":
        return np.arange(194, 231, dtype=np.int64)
    # OPPORTUNITY body-worn sensors: inertial/body accelerometers plus UWB location tags.
    return np.concatenate([np.arange(1, 134, dtype=np.int64), np.arange(231, 243, dtype=np.int64)])


def subject_from_path(path: Path) -> int:
    match = re.match(r"S(\d+)-", path.name)
    if not match:
        raise ValueError(f"Cannot parse subject from OPPORTUNITY file name: {path.name}")
    return int(match.group(1))


def run_from_path(path: Path) -> str:
    return path.stem.split("-", 1)[1]


def task_label_maps() -> tuple[dict[str, dict[int, int]], dict[str, list[str]]]:
    id_to_index: dict[str, dict[int, int]] = {}
    names: dict[str, list[str]] = {}
    for task_name, spec in TASK_SPECS.items():
        ordered_ids = list(spec["label_names"].keys())
        id_to_index[task_name] = {label_id: idx for idx, label_id in enumerate(ordered_ids)}
        names[task_name] = [spec["label_names"][label_id] for label_id in ordered_ids]
    return id_to_index, names


def majority_label(values: np.ndarray, label_map: dict[int, int]) -> int:
    valid = values[np.isfinite(values)].astype(np.int64)
    valid = valid[valid != 0]
    if valid.size == 0:
        return IGNORE_INDEX
    counts = Counter(int(item) for item in valid if int(item) in label_map)
    if not counts:
        return IGNORE_INDEX
    label_id, _ = counts.most_common(1)[0]
    return label_map[label_id]


def fill_missing(features: np.ndarray) -> np.ndarray:
    features = features.astype(np.float32, copy=False)
    invalid = ~np.isfinite(features)
    if not invalid.any():
        return features
    col_means = np.nanmean(np.where(invalid, np.nan, features), axis=0)
    col_means = np.where(np.isfinite(col_means), col_means, 0.0).astype(np.float32)
    row_idx, col_idx = np.where(invalid)
    features[row_idx, col_idx] = col_means[col_idx]
    return features


@dataclass
class WindowData:
    x: np.ndarray
    labels: dict[str, np.ndarray]
    subjects: np.ndarray
    runs: np.ndarray
    task_label_names: dict[str, list[str]]
    feature_group: str


def cache_path(args: argparse.Namespace) -> Path:
    root = Path(args.output_root).expanduser().resolve() / "cache"
    root.mkdir(parents=True, exist_ok=True)
    limit = int(args.max_windows_per_subject)
    suffix = "all" if limit <= 0 else f"max{limit}"
    # Keep this version tag in the filename so reservoir-sampled capped windows
    # cannot accidentally reuse the earlier sequentially capped cache.
    cache_version = "reservoir_v2"
    return root / f"opportunity_{args.feature_group}_w{args.window_size}_s{args.stride}_{suffix}_{cache_version}.npz"


def save_cache(path: Path, data: WindowData) -> None:
    payload: dict[str, Any] = {
        "x": data.x,
        "subjects": data.subjects,
        "runs": data.runs,
        "feature_group": data.feature_group,
        "task_label_names": json.dumps(data.task_label_names),
    }
    for task_name, values in data.labels.items():
        payload[f"label_{task_name}"] = values
    np.savez_compressed(path, **payload)


def load_cache(path: Path) -> WindowData:
    cached = np.load(path, allow_pickle=False)
    labels = {task: cached[f"label_{task}"] for task in TASK_SPECS}
    return WindowData(
        x=cached["x"],
        labels=labels,
        subjects=cached["subjects"],
        runs=cached["runs"],
        task_label_names=json.loads(str(cached["task_label_names"])),
        feature_group=str(cached["feature_group"]),
    )


def build_windows(args: argparse.Namespace) -> WindowData:
    path = cache_path(args)
    if args.cache and path.is_file():
        return load_cache(path)

    dataset_dir = Path(args.dataset_root).expanduser().resolve() / "dataset"
    dat_files = sorted(dataset_dir.glob("S*-*.dat"))
    if not dat_files:
        raise FileNotFoundError(f"No OPPORTUNITY .dat files found under {dataset_dir}")

    selected_features = feature_indices(args.feature_group)
    id_to_index, task_names = task_label_maps()
    windows: list[np.ndarray] = []
    labels: dict[str, list[int]] = {task: [] for task in TASK_SPECS}
    subjects: list[int] = []
    runs: list[str] = []
    seen_per_subject: dict[int, int] = defaultdict(int)
    slots_per_subject: dict[int, list[int]] = defaultdict(list)
    rng = random.Random(20260626)
    limit = int(args.max_windows_per_subject)

    for dat_path in dat_files:
        subject_id = subject_from_path(dat_path)
        matrix = np.loadtxt(dat_path, dtype=np.float32)
        features = fill_missing(matrix[:, selected_features])
        label_columns = {task: matrix[:, spec["label_col"]] for task, spec in TASK_SPECS.items()}
        for start in range(0, max(features.shape[0] - args.window_size + 1, 0), args.stride):
            end = start + args.window_size
            current_labels = {
                task: majority_label(label_columns[task][start:end], id_to_index[task])
                for task in TASK_SPECS
            }
            if all(value == IGNORE_INDEX for value in current_labels.values()):
                continue
            seen_per_subject[subject_id] += 1
            current_window = features[start:end].T.copy()
            current_run = run_from_path(dat_path)
            if limit <= 0 or len(slots_per_subject[subject_id]) < limit:
                slot = len(windows)
                slots_per_subject[subject_id].append(slot)
                windows.append(current_window)
                for task, value in current_labels.items():
                    labels[task].append(value)
                subjects.append(subject_id)
                runs.append(current_run)
            else:
                replace_at = rng.randrange(seen_per_subject[subject_id])
                if replace_at < limit:
                    slot = slots_per_subject[subject_id][replace_at]
                    windows[slot] = current_window
                    for task, value in current_labels.items():
                        labels[task][slot] = value
                    subjects[slot] = subject_id
                    runs[slot] = current_run

    if not windows:
        raise RuntimeError("Window extraction produced no samples.")

    data = WindowData(
        x=np.stack(windows).astype(np.float32),
        labels={task: np.asarray(values, dtype=np.int64) for task, values in labels.items()},
        subjects=np.asarray(subjects, dtype=np.int64),
        runs=np.asarray(runs),
        task_label_names=task_names,
        feature_group=args.feature_group,
    )
    if args.cache:
        save_cache(path, data)
    return data


class OpportunityDataset(Dataset):
    def __init__(
        self,
        data: WindowData,
        indices: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
        is_train: bool = False,
        train_noise_std: float = 0.0,
        channel_dropout_prob: float = 0.0,
    ) -> None:
        self.x = data.x[indices]
        self.labels = {task: values[indices] for task, values in data.labels.items()}
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)
        self.is_train = bool(is_train)
        self.train_noise_std = float(train_noise_std)
        self.channel_dropout_prob = float(channel_dropout_prob)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        signal = (self.x[index] - self.mean) / self.std
        signal_tensor = torch.from_numpy(signal.astype(np.float32))
        if self.is_train:
            if self.train_noise_std > 0:
                signal_tensor = signal_tensor + torch.randn_like(signal_tensor) * self.train_noise_std
            if self.channel_dropout_prob > 0:
                keep_mask = (torch.rand(signal_tensor.size(0)) >= self.channel_dropout_prob).to(signal_tensor.dtype)
                if keep_mask.sum() > 0:
                    signal_tensor = signal_tensor * keep_mask.unsqueeze(-1)
        labels = {task: torch.tensor(values[index], dtype=torch.long) for task, values in self.labels.items()}
        return signal_tensor, labels


def split_indices(
    data: WindowData,
    heldout_subject: int,
    val_subject: int,
    val_fraction: float,
    smoke: bool,
) -> dict[str, np.ndarray]:
    subjects = data.subjects
    if val_subject > 0:
        train = np.where((subjects != heldout_subject) & (subjects != val_subject))[0]
        val = np.where(subjects == val_subject)[0]
    else:
        source = np.where(subjects != heldout_subject)[0]
        rng = np.random.default_rng(20260626)
        shuffled = source.copy()
        rng.shuffle(shuffled)
        val_count = max(1, int(round(len(shuffled) * float(val_fraction))))
        val = shuffled[:val_count]
        train = shuffled[val_count:]
    test = np.where(subjects == heldout_subject)[0]
    rng = np.random.default_rng(12345)
    if smoke:
        train = rng.choice(train, size=min(len(train), 128), replace=False)
        val = rng.choice(val, size=min(len(val), 64), replace=False)
        test = rng.choice(test, size=min(len(test), 64), replace=False)
    return {"train": train, "val": val, "test": test}


def make_loaders(data: WindowData, splits: dict[str, np.ndarray], args: argparse.Namespace) -> dict[str, DataLoader]:
    train_x = data.x[splits["train"]]
    mean = train_x.mean(axis=(0, 2), keepdims=False)[:, None]
    std = train_x.std(axis=(0, 2), keepdims=False)[:, None]
    std = np.maximum(std, 1e-5)
    loaders = {}
    for split, indices in splits.items():
        dataset = OpportunityDataset(
            data,
            indices=indices,
            mean=mean,
            std=std,
            is_train=split == "train",
            train_noise_std=args.train_noise_std if split == "train" else 0.0,
            channel_dropout_prob=args.channel_dropout_prob if split == "train" else 0.0,
        )
        loaders[split] = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=split == "train",
            num_workers=args.num_workers,
            pin_memory=False,
        )
    return loaders


class MultiTaskHeadMixin:
    task_names: tuple[str, ...]

    def _build_heads(self, hidden_dim: int, task_output_dims: dict[str, int]) -> nn.ModuleDict:
        self.task_names = tuple(task_output_dims.keys())
        return nn.ModuleDict({task: nn.Linear(hidden_dim, dim) for task, dim in task_output_dims.items()})

    def _head_outputs(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        return {task: self.heads[task](features) for task in self.task_names}


class DeepConvLSTM(nn.Module, MultiTaskHeadMixin):
    def __init__(self, input_channels: int, task_output_dims: dict[str, int], hidden_dim: int, dropout: float) -> None:
        super().__init__()
        conv_dim = hidden_dim
        self.conv = nn.Sequential(
            nn.Conv1d(input_channels, conv_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(conv_dim),
            nn.ReLU(),
            nn.Conv1d(conv_dim, conv_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(conv_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(conv_dim, hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.proj = nn.Sequential(nn.LayerNorm(hidden_dim * 2), nn.Dropout(dropout), nn.Linear(hidden_dim * 2, hidden_dim))
        self.heads = self._build_heads(hidden_dim, task_output_dims)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.conv(x).transpose(1, 2)
        output, _ = self.lstm(features)
        pooled = output.mean(dim=1)
        return self._head_outputs(self.proj(pooled))


class TinyHAR(nn.Module, MultiTaskHeadMixin):
    def __init__(self, input_channels: int, task_output_dims: dict[str, int], hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.stem = nn.Conv1d(input_channels, hidden_dim, kernel_size=1)
        self.blocks = nn.Sequential(
            DepthwiseBlock(hidden_dim, dropout),
            DepthwiseBlock(hidden_dim, dropout),
            DepthwiseBlock(hidden_dim, dropout),
        )
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attn = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))
        self.out = nn.Sequential(nn.LayerNorm(hidden_dim * 2), nn.Dropout(dropout), nn.Linear(hidden_dim * 2, hidden_dim))
        self.heads = self._build_heads(hidden_dim, task_output_dims)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.blocks(self.stem(x)).transpose(1, 2)
        output, _ = self.gru(features)
        weights = torch.softmax(self.attn(output), dim=1)
        pooled = (weights * output).sum(dim=1)
        return self._head_outputs(self.out(pooled))


class DepthwiseBlock(nn.Module):
    def __init__(self, channels: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=5, padding=2, groups=channels),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class TemporalTransformer(nn.Module, MultiTaskHeadMixin):
    def __init__(
        self,
        input_channels: int,
        task_output_dims: dict[str, int],
        hidden_dim: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.proj = nn.Conv1d(input_channels, hidden_dim, kernel_size=3, padding=1)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 3,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.cls = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.norm = nn.LayerNorm(hidden_dim)
        self.heads = self._build_heads(hidden_dim, task_output_dims)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        tokens = self.proj(x).transpose(1, 2)
        cls = self.cls.expand(tokens.size(0), -1, -1)
        encoded = self.encoder(torch.cat([cls, tokens], dim=1))
        return self._head_outputs(self.norm(encoded[:, 0]))


class AttendDiscriminate(nn.Module, MultiTaskHeadMixin):
    def __init__(self, input_channels: int, task_output_dims: dict[str, int], hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(input_channels, max(input_channels // 4, 8)),
            nn.ReLU(),
            nn.Linear(max(input_channels // 4, 8), input_channels),
            nn.Sigmoid(),
        )
        self.conv = nn.Sequential(
            nn.Conv1d(input_channels, hidden_dim, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attn = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))
        self.proj = nn.Sequential(nn.LayerNorm(hidden_dim * 2), nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU())
        self.heads = self._build_heads(hidden_dim, task_output_dims)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = x * self.channel_gate(x).unsqueeze(-1)
        features = self.conv(x).transpose(1, 2)
        output, _ = self.rnn(features)
        weights = torch.softmax(self.attn(output), dim=1)
        pooled = (weights * output).sum(dim=1)
        return self._head_outputs(self.proj(pooled))


class ConvTranClassifier(nn.Module, MultiTaskHeadMixin):
    """ConvTran-style convolutional tokenization plus Transformer encoder."""

    def __init__(
        self,
        input_channels: int,
        task_output_dims: dict[str, int],
        hidden_dim: int,
        num_heads: int,
        window_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.embed = nn.Sequential(
            nn.Conv1d(input_channels, hidden_dim, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )
        self.cls = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.pos = nn.Parameter(torch.zeros(1, window_size + 1, hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.norm = nn.LayerNorm(hidden_dim)
        self.heads = self._build_heads(hidden_dim, task_output_dims)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        tokens = self.embed(x).transpose(1, 2)
        cls = self.cls.expand(tokens.size(0), -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = tokens + self.pos[:, : tokens.size(1)]
        encoded = self.encoder(tokens)
        return self._head_outputs(self.norm(encoded[:, 0]))


class PatchTSTClassifier(nn.Module, MultiTaskHeadMixin):
    """PatchTST-style channel-independent patch encoding for classification."""

    def __init__(
        self,
        input_channels: int,
        task_output_dims: dict[str, int],
        hidden_dim: int,
        num_heads: int,
        dropout: float,
        patch_len: int = 15,
        patch_stride: int = 5,
    ) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.patch_len = patch_len
        self.patch_stride = patch_stride
        self.patch_embed = nn.Linear(patch_len, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 3,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.channel_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.out = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Dropout(dropout))
        self.heads = self._build_heads(hidden_dim, task_output_dims)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        patches = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_stride)
        batch_size, channels, patch_count, _ = patches.shape
        patch_tokens = self.patch_embed(patches.reshape(batch_size * channels, patch_count, self.patch_len))
        encoded = self.encoder(patch_tokens).mean(dim=1).reshape(batch_size, channels, -1)
        weights = torch.softmax(self.channel_gate(encoded), dim=1)
        pooled = (weights * encoded).sum(dim=1)
        return self._head_outputs(self.out(pooled))


class TSMixerBlock(nn.Module):
    def __init__(self, seq_len: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.time_norm = nn.LayerNorm(hidden_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(seq_len, seq_len * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(seq_len * 2, seq_len),
            nn.Dropout(dropout),
        )
        self.channel_norm = nn.LayerNorm(hidden_dim)
        self.channel_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        mixed_time = self.time_mlp(self.time_norm(tokens).transpose(1, 2)).transpose(1, 2)
        tokens = tokens + mixed_time
        tokens = tokens + self.channel_mlp(self.channel_norm(tokens))
        return tokens


class TSMixerClassifier(nn.Module, MultiTaskHeadMixin):
    """TSMixer-style all-MLP multivariate time-series classifier."""

    def __init__(
        self,
        input_channels: int,
        task_output_dims: dict[str, int],
        hidden_dim: int,
        window_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_channels, hidden_dim)
        self.blocks = nn.Sequential(
            TSMixerBlock(window_size, hidden_dim, dropout),
            TSMixerBlock(window_size, hidden_dim, dropout),
            TSMixerBlock(window_size, hidden_dim, dropout),
        )
        self.out = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Dropout(dropout))
        self.heads = self._build_heads(hidden_dim, task_output_dims)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        tokens = self.input_proj(x.transpose(1, 2))
        tokens = self.blocks(tokens)
        pooled = tokens.mean(dim=1)
        return self._head_outputs(self.out(pooled))


class TimesBlock(nn.Module):
    """Small TimesNet-style 2D temporal-variation block with fixed candidate periods."""

    def __init__(self, hidden_dim: int, dropout: float, periods: tuple[int, ...] = (2, 3, 5)) -> None:
        super().__init__()
        self.periods = periods
        self.period_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim),
                    nn.GELU(),
                    nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
                    nn.Dropout(dropout),
                )
                for _ in periods
            ]
        )
        self.norm = nn.BatchNorm1d(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, hidden_dim, seq_len = x.shape
        candidates = []
        for period, conv in zip(self.periods, self.period_convs):
            pad_len = (period - seq_len % period) % period
            current = F.pad(x, (0, pad_len)) if pad_len else x
            folded = current.reshape(batch_size, hidden_dim, -1, period)
            restored = conv(folded).reshape(batch_size, hidden_dim, -1)[..., :seq_len]
            candidates.append(restored)
        mixed = torch.stack(candidates, dim=0).mean(dim=0)
        return x + self.norm(mixed)


class TimesNetClassifier(nn.Module, MultiTaskHeadMixin):
    def __init__(self, input_channels: int, task_output_dims: dict[str, int], hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.proj = nn.Conv1d(input_channels, hidden_dim, kernel_size=1)
        self.blocks = nn.Sequential(TimesBlock(hidden_dim, dropout), TimesBlock(hidden_dim, dropout))
        self.out = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Dropout(dropout))
        self.heads = self._build_heads(hidden_dim, task_output_dims)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.blocks(self.proj(x))
        pooled = features.mean(dim=-1)
        return self._head_outputs(self.out(pooled))


class ModernTCNBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float, kernel_size: int = 31) -> None:
        super().__init__()
        self.dwconv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=hidden_dim)
        self.norm = nn.BatchNorm1d(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim * 4, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim * 4, hidden_dim, kernel_size=1),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(self.dwconv(x))
        return residual + self.ffn(x)


class ModernTCNClassifier(nn.Module, MultiTaskHeadMixin):
    def __init__(self, input_channels: int, task_output_dims: dict[str, int], hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.stem = nn.Conv1d(input_channels, hidden_dim, kernel_size=1)
        self.blocks = nn.Sequential(
            ModernTCNBlock(hidden_dim, dropout, kernel_size=31),
            ModernTCNBlock(hidden_dim, dropout, kernel_size=31),
            ModernTCNBlock(hidden_dim, dropout, kernel_size=15),
        )
        self.out = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Dropout(dropout))
        self.heads = self._build_heads(hidden_dim, task_output_dims)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.blocks(self.stem(x))
        pooled = features.mean(dim=-1)
        return self._head_outputs(self.out(pooled))


class ITransformerClassifier(nn.Module, MultiTaskHeadMixin):
    """iTransformer-style variate-token classifier."""

    def __init__(
        self,
        input_channels: int,
        task_output_dims: dict[str, int],
        hidden_dim: int,
        num_heads: int,
        window_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.value_embed = nn.Linear(window_size, hidden_dim)
        self.channel_pos = nn.Parameter(torch.zeros(1, input_channels, hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 3,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.channel_gate = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))
        self.out = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Dropout(dropout))
        self.heads = self._build_heads(hidden_dim, task_output_dims)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        tokens = self.value_embed(x) + self.channel_pos[:, : x.size(1)]
        encoded = self.encoder(tokens)
        weights = torch.softmax(self.channel_gate(encoded), dim=1)
        pooled = (weights * encoded).sum(dim=1)
        return self._head_outputs(self.out(pooled))


class TinierHARClassifier(nn.Module, MultiTaskHeadMixin):
    """TinierHAR-style compact residual depthwise CNN + GRU classifier."""

    def __init__(self, input_channels: int, task_output_dims: dict[str, int], hidden_dim: int, dropout: float) -> None:
        super().__init__()
        compact_dim = max(hidden_dim // 2, 32)
        self.stem = nn.Conv1d(input_channels, compact_dim, kernel_size=1)
        self.blocks = nn.Sequential(
            DepthwiseBlock(compact_dim, dropout),
            DepthwiseBlock(compact_dim, dropout),
        )
        self.gru = nn.GRU(compact_dim, compact_dim, batch_first=True, bidirectional=True)
        self.attn = nn.Sequential(nn.Linear(compact_dim * 2, compact_dim), nn.Tanh(), nn.Linear(compact_dim, 1))
        self.out = nn.Sequential(nn.LayerNorm(compact_dim * 2), nn.Dropout(dropout), nn.Linear(compact_dim * 2, hidden_dim))
        self.heads = self._build_heads(hidden_dim, task_output_dims)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.blocks(self.stem(x)).transpose(1, 2)
        output, _ = self.gru(features)
        weights = torch.softmax(self.attn(output), dim=1)
        pooled = (weights * output).sum(dim=1)
        return self._head_outputs(self.out(pooled))


class SensorFieldWrapper(nn.Module):
    def __init__(
        self,
        input_channels: int,
        task_output_dims: dict[str, int],
        hidden_dim: int,
        num_anchors: int,
        num_heads: int,
        dropout: float,
        args: argparse.Namespace,
    ) -> None:
        super().__init__()
        self.model = SensorFieldM3T(
            task_output_dims=task_output_dims,
            hidden_dim=hidden_dim,
            num_anchors=num_anchors,
            num_heads=num_heads,
            raw_tokens=8,
            raw_in_channels=input_channels,
            stf_tokens=24,
            gaf_tokens=24,
            stf_size=64,
            gaf_size=48,
            stft_n_fft=64,
            stft_hop_length=16,
            stft_win_length=64,
            fac_loss_weight=args.fac_loss_weight,
            taef_loss_weight=args.taef_loss_weight,
            gcti_loss_weight=args.gcti_loss_weight,
            view_drop_prob=args.view_drop_prob,
            enable_view_consistency=args.view_drop_prob > 0,
            view_consistency_weight=0.02 if args.view_drop_prob > 0 else 0.0,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> dict[str, Any]:
        # Explicit three-view OPPORTUNITY input for SensorField-M3T:
        # raw: [B, C, T], stf: [B, 1, H, W], gaf: [B, 1, H, W].
        # The model can also build these internally, but passing them here makes
        # the multimodal experimental protocol unambiguous.
        stf = self.model._build_stf_map_from_raw(x)
        gaf = self.model._build_gaf_from_raw(x)
        return self.model({"raw": x, "stf": stf, "gaf": gaf})


def build_model(model_name: str, input_channels: int, task_output_dims: dict[str, int], args: argparse.Namespace) -> nn.Module:
    if model_name == "sensorfield_m3t":
        return SensorFieldWrapper(
            input_channels,
            task_output_dims,
            args.hidden_dim,
            args.num_anchors,
            args.num_heads,
            args.dropout,
            args,
        )
    if model_name == "deepconvlstm":
        return DeepConvLSTM(input_channels, task_output_dims, args.hidden_dim, args.dropout)
    if model_name == "tinyhar":
        return TinyHAR(input_channels, task_output_dims, args.hidden_dim, args.dropout)
    if model_name == "temporal_transformer":
        return TemporalTransformer(input_channels, task_output_dims, args.hidden_dim, args.num_heads, args.dropout)
    if model_name == "attend_discriminate":
        return AttendDiscriminate(input_channels, task_output_dims, args.hidden_dim, args.dropout)
    if model_name == "convtran":
        return ConvTranClassifier(input_channels, task_output_dims, args.hidden_dim, args.num_heads, args.window_size, args.dropout)
    if model_name == "patchtst":
        return PatchTSTClassifier(input_channels, task_output_dims, args.hidden_dim, args.num_heads, args.dropout)
    if model_name == "tsmixer":
        return TSMixerClassifier(input_channels, task_output_dims, args.hidden_dim, args.window_size, args.dropout)
    if model_name == "timesnet":
        return TimesNetClassifier(input_channels, task_output_dims, args.hidden_dim, args.dropout)
    if model_name == "moderntcn":
        return ModernTCNClassifier(input_channels, task_output_dims, args.hidden_dim, args.dropout)
    if model_name == "itransformer":
        return ITransformerClassifier(input_channels, task_output_dims, args.hidden_dim, args.num_heads, args.window_size, args.dropout)
    if model_name == "tinierhar":
        return TinierHARClassifier(input_channels, task_output_dims, args.hidden_dim, args.dropout)
    raise ValueError(f"Unknown model: {model_name}")


def class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    valid = labels[labels != IGNORE_INDEX]
    if valid.size == 0:
        return torch.ones(num_classes, dtype=torch.float32)
    counts = np.bincount(valid, minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (num_classes * counts)
    weights = weights / max(weights.mean(), 1e-12)
    return torch.tensor(weights, dtype=torch.float32)


def make_criteria(
    data: WindowData,
    train_indices: np.ndarray,
    task_output_dims: dict[str, int],
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, nn.Module]:
    criteria = {}
    label_smoothing = float(max(0.0, min(1.0, args.label_smoothing)))
    for task, dim in task_output_dims.items():
        weights = class_weights(data.labels[task][train_indices], dim).to(device)
        criteria[task] = nn.CrossEntropyLoss(
            weight=weights,
            ignore_index=IGNORE_INDEX,
            label_smoothing=label_smoothing,
        )
    return criteria


def batch_to_device(batch: tuple[torch.Tensor, dict[str, torch.Tensor]], device: torch.device) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    x, labels = batch
    return x.to(device), {task: value.to(device) for task, value in labels.items()}


def compute_loss(outputs: dict[str, Any], labels: dict[str, torch.Tensor], criteria: dict[str, nn.Module]) -> torch.Tensor:
    losses = []
    for task, criterion in criteria.items():
        target = labels[task]
        if (target != IGNORE_INDEX).any():
            losses.append(criterion(outputs[task], target))
    if "aux_losses" in outputs:
        losses.extend(value for value in outputs["aux_losses"].values() if torch.is_tensor(value))
    if not losses:
        return torch.zeros((), device=next(iter(outputs.values())).device, requires_grad=True)
    return torch.stack([loss if loss.ndim == 0 else loss.mean() for loss in losses]).mean()


def train_epoch(model: nn.Module, loader: DataLoader, criteria: dict[str, nn.Module], optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    total_count = 0
    for batch in loader:
        x, labels = batch_to_device(batch, device)
        outputs = model(x)
        loss = compute_loss(outputs, labels, criteria)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += float(loss.item()) * x.size(0)
        total_count += x.size(0)
    return total_loss / max(total_count, 1)


def macro_far(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    total = cm.sum()
    fars = []
    for idx in range(num_classes):
        tp = cm[idx, idx]
        fp = cm[:, idx].sum() - tp
        fn = cm[idx, :].sum() - tp
        tn = total - tp - fp - fn
        fars.append(float(fp) / max(float(fp + tn), 1.0))
    return float(np.mean(fars))


def task_metrics(y_true: np.ndarray, logits: np.ndarray, num_classes: int) -> dict[str, float]:
    if y_true.size == 0:
        return {"acc": math.nan, "f1": math.nan, "auc": math.nan, "far": math.nan, "task_score": math.nan}
    y_pred = logits.argmax(axis=1)
    acc = float((y_pred == y_true).mean())
    f1 = float(f1_score(y_true, y_pred, average="macro", labels=list(range(num_classes)), zero_division=0))
    probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    try:
        auc = float(roc_auc_score(y_true, probs, multi_class="ovr", average="macro", labels=list(range(num_classes))))
    except Exception:
        auc = math.nan
    far = macro_far(y_true, y_pred, num_classes)
    auc_for_score = auc if math.isfinite(auc) else 0.0
    task_score = (acc + f1 + auc_for_score + (1.0 - far)) / 4.0
    return {"acc": acc, "f1": f1, "auc": auc, "far": far, "task_score": task_score}


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, task_output_dims: dict[str, int], device: torch.device) -> dict[str, float]:
    model.eval()
    logits_by_task: dict[str, list[np.ndarray]] = {task: [] for task in task_output_dims}
    labels_by_task: dict[str, list[np.ndarray]] = {task: [] for task in task_output_dims}
    for batch in loader:
        x, labels = batch_to_device(batch, device)
        outputs = model(x)
        for task in task_output_dims:
            target = labels[task]
            mask = target != IGNORE_INDEX
            if mask.any():
                logits_by_task[task].append(outputs[task][mask].detach().cpu().numpy())
                labels_by_task[task].append(target[mask].detach().cpu().numpy())

    result: dict[str, float] = {}
    task_scores = []
    for task, dim in task_output_dims.items():
        logits = np.concatenate(logits_by_task[task], axis=0) if logits_by_task[task] else np.zeros((0, dim), dtype=np.float32)
        target = np.concatenate(labels_by_task[task], axis=0) if labels_by_task[task] else np.zeros((0,), dtype=np.int64)
        metrics = task_metrics(target, logits, dim)
        for key, value in metrics.items():
            result[f"{task}_{key}"] = value
        if math.isfinite(metrics["task_score"]):
            task_scores.append(metrics["task_score"])
    result["mtl_score"] = float(np.mean(task_scores)) if task_scores else math.nan
    return result


def count_params(model: nn.Module) -> float:
    return sum(param.numel() for param in model.parameters()) / 1e6


def hook_estimate_flops_g(model: nn.Module, dummy: torch.Tensor) -> float | None:
    flops = 0.0
    handles = []

    def conv_hook(module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        nonlocal flops
        if not torch.is_tensor(output):
            return
        if isinstance(module, nn.Conv1d):
            kernel_ops = module.kernel_size[0] * (module.in_channels // module.groups)
        elif isinstance(module, nn.Conv2d):
            kernel_ops = module.kernel_size[0] * module.kernel_size[1] * (module.in_channels // module.groups)
        else:
            return
        flops += float(output.numel() * kernel_ops * 2)

    def linear_hook(module: nn.Linear, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        nonlocal flops
        if torch.is_tensor(output):
            flops += float(output.numel() * module.in_features * 2)

    def rnn_hook(module: nn.Module, inputs: tuple[torch.Tensor, ...], output: Any) -> None:
        nonlocal flops
        x = inputs[0]
        if not torch.is_tensor(x) or x.ndim < 3:
            return
        if getattr(module, "batch_first", False):
            batch_size, seq_len = int(x.shape[0]), int(x.shape[1])
        else:
            seq_len, batch_size = int(x.shape[0]), int(x.shape[1])
        hidden_size = int(module.hidden_size)
        num_layers = int(module.num_layers)
        directions = 2 if bool(module.bidirectional) else 1
        gates = 4 if isinstance(module, nn.LSTM) else 3
        for layer_idx in range(num_layers):
            layer_input_size = int(module.input_size) if layer_idx == 0 else hidden_size * directions
            per_direction = gates * hidden_size * (layer_input_size + hidden_size)
            flops += float(batch_size * seq_len * directions * per_direction * 2)

    def mha_hook(module: nn.MultiheadAttention, inputs: tuple[torch.Tensor, ...], output: Any) -> None:
        nonlocal flops
        if len(inputs) < 3:
            return
        query, key, value = inputs[:3]
        if not (torch.is_tensor(query) and torch.is_tensor(key) and torch.is_tensor(value)):
            return
        if module.batch_first:
            batch_size, q_len, embed_dim = int(query.shape[0]), int(query.shape[1]), int(query.shape[2])
            k_len = int(key.shape[1])
            v_len = int(value.shape[1])
        else:
            q_len, batch_size, embed_dim = int(query.shape[0]), int(query.shape[1]), int(query.shape[2])
            k_len = int(key.shape[0])
            v_len = int(value.shape[0])
        num_heads = int(module.num_heads)
        head_dim = embed_dim // max(num_heads, 1)
        # q/k/v projections, attention score/value products, and output projection.
        flops += float(batch_size * (q_len + k_len + v_len) * embed_dim * embed_dim * 2)
        flops += float(batch_size * num_heads * q_len * k_len * head_dim * 2)
        flops += float(batch_size * num_heads * q_len * k_len * head_dim * 2)
        flops += float(batch_size * q_len * embed_dim * embed_dim * 2)

    for module in model.modules():
        if isinstance(module, (nn.Conv1d, nn.Conv2d)):
            handles.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(linear_hook))
        elif isinstance(module, (nn.LSTM, nn.GRU)):
            handles.append(module.register_forward_hook(rnn_hook))
        elif isinstance(module, nn.MultiheadAttention):
            handles.append(module.register_forward_hook(mha_hook))
    try:
        with torch.no_grad():
            _ = model(dummy)
    except Exception:
        return None
    finally:
        for handle in handles:
            handle.remove()
    return flops / 1e9 if flops > 0 else None


def estimate_flops_g(model: nn.Module, input_channels: int, window_size: int, device: torch.device) -> float | None:
    try:
        from thop import profile
    except Exception:
        profile = None
    model.eval()
    dummy = torch.randn(1, input_channels, window_size, device=device)
    if profile is not None:
        try:
            macs, _ = profile(model, inputs=(dummy,), verbose=False)
            return float(macs) * 2.0 / 1e9
        except Exception:
            pass
    hook_flops = hook_estimate_flops_g(model, dummy)
    if hook_flops is not None:
        return hook_flops
    try:
        from torch.profiler import ProfilerActivity, profile as torch_profile

        with torch_profile(activities=[ProfilerActivity.CPU], with_flops=True) as prof:
            with torch.no_grad():
                _ = model(dummy)
        flops = sum(float(getattr(item, "flops", 0.0) or 0.0) for item in prof.key_averages())
        return flops / 1e9 if flops > 0 else None
    except Exception:
        return None


def train_one_model(
    model_name: str,
    data: WindowData,
    loaders: dict[str, DataLoader],
    splits: dict[str, np.ndarray],
    task_output_dims: dict[str, int],
    args: argparse.Namespace,
    seed: int,
    output_dir: Path,
) -> dict[str, Any]:
    set_seed(seed)
    device = torch.device("cpu")
    model = build_model(model_name, input_channels=data.x.shape[1], task_output_dims=task_output_dims, args=args).to(device)
    criteria = make_criteria(data, splits["train"], task_output_dims, device, args)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    model_dir = output_dir / model_name / f"seed_{seed:03d}"
    model_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_state = None
    best_val = -float("inf")
    best_epoch = 0
    stop_epoch = args.epochs
    epochs_without_improvement = 0
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, loaders["train"], criteria, optimizer, device)
        val_metrics = evaluate(model, loaders["val"], task_output_dims, device)
        scheduler.step()
        history.append({"epoch": epoch, "train_loss": train_loss, **{f"val_{k}": v for k, v in val_metrics.items()}})
        if val_metrics["mtl_score"] > best_val + args.early_stop_min_delta:
            best_val = val_metrics["mtl_score"]
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if (
            args.early_stop_patience > 0
            and epoch >= max(args.early_stop_min_epochs, 1)
            and epochs_without_improvement >= args.early_stop_patience
        ):
            stop_epoch = epoch
            break
    else:
        stop_epoch = args.epochs

    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics = evaluate(model, loaders["test"], task_output_dims, device)
    params_m = count_params(model)
    flops_g = estimate_flops_g(model, data.x.shape[1], args.window_size, device)

    with (model_dir / "history.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        if history:
            writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)
    summary = {
        "model": model_name,
        "seed": seed,
        "best_epoch": best_epoch,
        "stop_epoch": stop_epoch,
        "early_stopped": float(stop_epoch < args.epochs),
        "best_val_mtl_score": best_val,
        "params_m": params_m,
        "flops_g": flops_g,
        "seconds": time.time() - started,
        **test_metrics,
    }
    with (model_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    return summary


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["model"])].append(row)
    summary_rows = []
    metric_keys = [key for key in rows[0].keys() if key not in {"model", "seed"} and isinstance(rows[0].get(key), (int, float, type(None)))]
    for model, model_rows in grouped.items():
        out: dict[str, Any] = {"model": model, "runs": len(model_rows)}
        for key in metric_keys:
            values = np.asarray([row.get(key, math.nan) for row in model_rows], dtype=np.float64)
            values = values[np.isfinite(values)]
            out[f"{key}_mean"] = float(values.mean()) if values.size else math.nan
            out[f"{key}_std"] = float(values.std(ddof=1)) if values.size > 1 else 0.0
        summary_rows.append(out)
    return sorted(summary_rows, key=lambda item: item.get("mtl_score_mean", -1), reverse=True)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt_pm(mean: float, std: float, digits: int = 4) -> str:
    if not math.isfinite(mean):
        return "--"
    return f"{mean:.{digits}f} $\\pm$ {std:.{digits}f}"


def write_latex(path: Path, summary_rows: list[dict[str, Any]]) -> None:
    model_names = {
        "sensorfield_m3t": "SensorField-M3T",
        "deepconvlstm": "DeepConvLSTM",
        "tinyhar": "TinyHAR",
        "temporal_transformer": "Temporal Transformer",
        "attend_discriminate": "Attend-Discriminate",
        "convtran": "ConvTran",
        "patchtst": "PatchTST",
        "tsmixer": "TSMixer",
        "timesnet": "TimesNet",
        "moderntcn": "ModernTCN",
        "itransformer": "iTransformer",
        "tinierhar": "TinierHAR",
    }
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Cross-subject OPPORTUNITY generalization comparison. Metrics are reported as mean $\\pm$ std across seeds.}",
        "\\label{tab:opportunity_generalization}",
        "\\setlength{\\tabcolsep}{3.0pt}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{lccccccccc}",
        "\\toprule",
        "Model & Locomotion F1 & Gesture F1 & Activity F1 & Locomotion TaskScore & Gesture TaskScore & Activity TaskScore & MTLScore & Params (M) & FLOPs (G) \\\\",
        "\\midrule",
    ]
    best_mtl = max((row.get("mtl_score_mean", -float("inf")) for row in summary_rows), default=-float("inf"))
    for row in summary_rows:
        model = model_names.get(str(row["model"]), str(row["model"]))
        mtl_text = fmt_pm(row.get("mtl_score_mean", math.nan), row.get("mtl_score_std", 0.0))
        if abs(row.get("mtl_score_mean", math.nan) - best_mtl) < 1e-12:
            mtl_text = f"\\textbf{{{mtl_text}}}"
        lines.append(
            " & ".join(
                [
                    model,
                    fmt_pm(row.get("locomotion_f1_mean", math.nan), row.get("locomotion_f1_std", 0.0)),
                    fmt_pm(row.get("gesture_f1_mean", math.nan), row.get("gesture_f1_std", 0.0)),
                    fmt_pm(row.get("activity_f1_mean", math.nan), row.get("activity_f1_std", 0.0)),
                    fmt_pm(row.get("locomotion_task_score_mean", math.nan), row.get("locomotion_task_score_std", 0.0)),
                    fmt_pm(row.get("gesture_task_score_mean", math.nan), row.get("gesture_task_score_std", 0.0)),
                    fmt_pm(row.get("activity_task_score_mean", math.nan), row.get("activity_task_score_std", 0.0)),
                    mtl_text,
                    f"{row.get('params_m_mean', math.nan):.2f}",
                    f"{row.get('flops_g_mean', math.nan):.4f}",
                ]
            )
            + " \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}%", "}", "\\end{table*}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_protocol(path: Path, args: argparse.Namespace, data: WindowData, splits: dict[str, np.ndarray]) -> None:
    task_info = "\n".join(
        f"- {task}: {len(names)} classes, labels = {', '.join(names)}"
        for task, names in data.task_label_names.items()
    )
    if int(args.val_subject) > 0:
        split_text = f"validation subject S{args.val_subject}"
    else:
        split_text = f"validation fraction {args.val_fraction:.2f} sampled from source subjects"
    text = f"""# OPPORTUNITY SensorField-M3T Protocol

Dataset: OPPORTUNITY Activity Recognition, UCI id 226.
Source URL: {DATA_URL}

Protocol:
- Feature group: {args.feature_group}
- Window size / stride: {args.window_size} / {args.stride}
- Cross-subject split: test subject S{args.heldout_subject}; {split_text}
- Train/val/test windows: {len(splits['train'])}/{len(splits['val'])}/{len(splits['test'])}
- Null labels encoded as 0 in the official files are ignored per task.
- SensorField-M3T input views: raw multichannel signal, STF map generated by STFT, and GAF image generated from the raw temporal signal.
- Regularization: dropout={args.dropout}, weight_decay={args.weight_decay}, label_smoothing={args.label_smoothing}, train_noise_std={args.train_noise_std}, channel_dropout_prob={args.channel_dropout_prob}, view_drop_prob={args.view_drop_prob}.
- Early stopping: min_epochs={args.early_stop_min_epochs}, patience={args.early_stop_patience}, min_delta={args.early_stop_min_delta}.

Tasks:
{task_info}

Compared models:
- SensorField-M3T
- DeepConvLSTM / TinyHAR / Attend-Discriminate legacy references when selected
- ConvTran / PatchTST / TSMixer / TimesNet / ModernTCN / iTransformer / TinierHAR recent backbone adaptations when selected
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.epochs = min(args.epochs, 1)
        args.max_windows_per_subject = 160 if args.max_windows_per_subject <= 0 else min(args.max_windows_per_subject, 160)
        args.models = "sensorfield_m3t,convtran"
        args.cache = False
    seeds = parse_int_list(args.seeds)
    models = parse_model_list(args.models)
    run_stamp = f"{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
    output_dir = Path(args.output_root).expanduser().resolve() / run_stamp
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, ensure_ascii=False, indent=2)

    data = build_windows(args)
    splits = split_indices(
        data,
        heldout_subject=args.heldout_subject,
        val_subject=args.val_subject,
        val_fraction=args.val_fraction,
        smoke=args.smoke,
    )
    loaders = make_loaders(data, splits, args)
    task_output_dims = {task: len(names) for task, names in data.task_label_names.items()}
    write_protocol(output_dir / "protocol.md", args, data, splits)

    rows = []
    for seed in seeds:
        for model_name in models:
            print(f"[{time.strftime('%H:%M:%S')}] Training {model_name} seed={seed}")
            rows.append(train_one_model(model_name, data, loaders, splits, task_output_dims, args, seed, output_dir))
            write_csv(output_dir / "metrics_per_seed.csv", rows)
    summary_rows = summarize(rows)
    write_csv(output_dir / "summary.csv", summary_rows)
    write_latex(output_dir / "opportunity_benchmark.tex", summary_rows)
    print(f"Saved OPPORTUNITY benchmark outputs to: {output_dir}")


if __name__ == "__main__":
    main()
