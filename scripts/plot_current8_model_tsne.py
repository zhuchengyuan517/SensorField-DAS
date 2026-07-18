"""Export Event/Location t-SNE plots for the current eight-model benchmark.

The script loads the actual checkpoints referenced by the SensorField-M3T
benchmark table and extracts pre-head representations for both tasks.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import types
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import font_manager
from torch.utils.data import DataLoader

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score


ROOT = Path(__file__).resolve().parents[1]
PATCH_ROOT = ROOT / "libmtl_das_patch"
PATCH_EXAMPLE_DIR = PATCH_ROOT / "examples" / "das_csv"
GITHUB_ROOT = Path(r"D:\github\LibMTL-main\LibMTL-main")
GITHUB_EXAMPLE_DIR = GITHUB_ROOT / "examples" / "das_csv"

for path in (str(PATCH_ROOT), str(PATCH_EXAMPLE_DIR), str(GITHUB_ROOT), str(GITHUB_EXAMPLE_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

# Avoid executing the original LibMTL.__init__, which imports optional training
# dependencies. The visualization only needs model files from the patch plus the
# original ResNet implementation.
if "LibMTL" not in sys.modules:
    libmtl_pkg = types.ModuleType("LibMTL")
    libmtl_pkg.__path__ = [str(PATCH_ROOT / "LibMTL"), str(GITHUB_ROOT / "LibMTL")]
    sys.modules["LibMTL"] = libmtl_pkg
if "LibMTL.model" not in sys.modules:
    model_pkg = types.ModuleType("LibMTL.model")
    model_pkg.__path__ = [str(PATCH_ROOT / "LibMTL" / "model"), str(GITHUB_ROOT / "LibMTL" / "model")]
    sys.modules["LibMTL.model"] = model_pkg

from create_dataset import DISTANCE_IGNORE_INDEX  # noqa: E402
from create_dataset_imagefork import HybridMTL43Dataset, parse_label_list  # noqa: E402
from LibMTL.model.adapted_benchmark_imagefork import (  # noqa: E402
    DASMAEImageFork,
    M4oEImageFork,
    MultiModNImageFork,
    PipelineADWinTImageFork,
)
from LibMTL.model.sensorfield_m3t_imagefork import SensorFieldM3TImageFork  # noqa: E402
from LibMTL.model.resnet import resnet18  # noqa: E402
from sota_multitask_imagefork_benchmark import (  # noqa: E402
    ImageForkMultiModalDataset,
    build_model as build_sota_image_model,
    collate_multitask,
    resolve_device,
)


DEFAULT_DATASET_PATH = ROOT / "converted_csv" / "MTL43_imagefork_dedup_clean"
DEFAULT_MTL43_ROOT = ROOT / "converted_csv" / "MTL43"
DEFAULT_IMAGE_ROOT = ROOT / "converted_csv" / "flower_data_rl_dedup_clean"
DEFAULT_OUTPUT_ROOT = ROOT / "paper_assets" / "sensorfield_m3t_experiments" / "tsne_current8_models"

DEFAULT_RUNS: OrderedDict[str, Path] = OrderedDict(
    [
        ("ConvNeXt-Small", ROOT / "_tmp_compare_imagefork_sota_clean" / "20260518_133929" / "convnext_small"),
        ("MultiModN", ROOT / "_tmp_added4_bench" / "multimodn" / "20260518_215955"),
        ("M4oE", ROOT / "_tmp_added4_bench" / "m4oe" / "20260518_220341"),
        (
            "DAS-MAE + downstream fine-tuning head",
            ROOT / "_tmp_added4_bench" / "dasmae" / "20260518_220731",
        ),
        ("PipelineADWinT", ROOT / "_tmp_added4_bench" / "pipelineadwint" / "20260518_221136"),
        ("Aligned-MTL", ROOT / "_tmp_compare_aligned_clean" / "20260518_160714"),
        ("MoCo-weighting", ROOT / "_tmp_compare_moco_weighting_clean" / "20260518_160414"),
        (
            "SensorField-M3T",
            ROOT / "_tmp_sensorfield_convnext_expert_hybrid" / "evt1_loc1_freeze6_resume" / "20260518_164757",
        ),
    ]
)

EVENT_COLOR_MAP = {
    "walking": "#31688E",
    "excavator": "#C44536",
    "driving": "#2E7D32",
    "background": "#6D597A",
}
LOCATION_COLOR_MAP = {
    "Alarm area": "#E07A2D",
    "Tracking area": "#2A9D8F",
    "No-threat area": "#8D6E63",
}
EVENT_MARKERS = {"walking": "o", "excavator": "s", "driving": "^", "background": "D"}
LOCATION_MARKERS = {"Alarm area": "o", "Tracking area": "s", "No-threat area": "^"}
DISPLAY_SHORT_NAMES = {
    "DAS-MAE + downstream fine-tuning head": "DAS-MAE + FT",
    "MoCo-weighting": "MoCo",
    "SensorField-M3T": "SensorField-M3T",
}


class HybridInferenceDataset(HybridMTL43Dataset):
    """Hybrid dataset variant that keeps sample paths for traceability."""

    def __getitem__(self, index):
        sample = self.samples[index]
        payload, labels = super().__getitem__(index)
        payload["path"] = str(sample["path"])
        return payload, labels


def hybrid_collate_with_meta(batch):
    batch_size = len(batch)
    event_targets = torch.empty(batch_size, dtype=torch.long)
    distance_targets = torch.empty(batch_size, dtype=torch.long)
    csv_inputs, csv_indices = [], []
    image_inputs, image_indices = [], []
    sample_paths = []
    input_types = []

    for idx, (sample, labels) in enumerate(batch):
        event_targets[idx] = labels["event_type"]
        distance_targets[idx] = labels["distance_cls"]
        sample_paths.append(sample.get("path", ""))
        input_types.append(sample["input_type"])
        if sample["input_type"] == "csv":
            csv_inputs.append(sample["input_data"])
            csv_indices.append(idx)
        else:
            image_inputs.append(sample["input_data"])
            image_indices.append(idx)

    collated = {
        "batch_size": batch_size,
        "csv_inputs": torch.stack(csv_inputs, dim=0) if csv_inputs else None,
        "csv_indices": torch.tensor(csv_indices, dtype=torch.long),
        "image_inputs": torch.stack(image_inputs, dim=0) if image_inputs else None,
        "image_indices": torch.tensor(image_indices, dtype=torch.long),
        "sample_paths": sample_paths,
        "input_types": input_types,
    }
    labels = {
        "event_type": event_targets,
        "distance_cls": distance_targets,
    }
    return collated, labels


class GlobalPoolDecoder(nn.Module):
    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        self.decoder = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_channels, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.decoder(inputs)


class _TransformResNetLTB(nn.Module):
    """Minimal LTB ResNet transform used only for inference-time features."""

    def __init__(self, encoder_list: list[nn.Module], task_name: list[str], device: torch.device) -> None:
        super().__init__()
        self.task_name = task_name
        self.task_num = len(task_name)
        self.device = device
        self.resnet_conv = nn.ModuleDict(
            {
                task: nn.Sequential(
                    encoder_list[tn].conv1,
                    encoder_list[tn].bn1,
                    encoder_list[tn].relu,
                    encoder_list[tn].maxpool,
                )
                for tn, task in enumerate(self.task_name)
            }
        )
        self.resnet_layer = nn.ModuleDict({})
        for layer_idx in range(4):
            self.resnet_layer[str(layer_idx)] = nn.ModuleList([])
            for task_idx in range(self.task_num):
                encoder = encoder_list[task_idx]
                self.resnet_layer[str(layer_idx)].append(getattr(encoder, f"layer{layer_idx + 1}"))
        self.alpha = nn.Parameter(torch.ones(6, self.task_num, self.task_num))
        self.use_deterministic_routing = True

    def _resolve_alpha(self, epoch: int, epochs: int) -> torch.Tensor:
        """Return task-routing weights with deterministic inference by default."""
        if self.use_deterministic_routing:
            route_idx = self.alpha.argmax(dim=-1)
            return F.one_hot(route_idx, num_classes=self.task_num).to(device=self.device, dtype=self.alpha.dtype)
        if epoch < epochs / 100:
            return torch.ones(6, self.task_num, self.task_num, device=self.device, dtype=self.alpha.dtype)
        tau = epochs / 20 / np.sqrt(epoch + 1)
        return F.gumbel_softmax(self.alpha, dim=-1, tau=tau, hard=True)

    def forward(self, inputs: torch.Tensor, epoch: int, epochs: int) -> list[torch.Tensor]:
        alpha = self._resolve_alpha(epoch, epochs)
        ss_rep = {idx: [None] * self.task_num for idx in range(5)}
        for layer_idx in range(5):
            for task_idx, task in enumerate(self.task_name):
                if layer_idx == 0:
                    ss_rep[layer_idx][task_idx] = self.resnet_conv[task](inputs)
                else:
                    child_rep = sum(
                        alpha[layer_idx, task_idx, other_idx] * ss_rep[layer_idx - 1][other_idx]
                        for other_idx in range(self.task_num)
                    )
                    ss_rep[layer_idx][task_idx] = self.resnet_layer[str(layer_idx - 1)][task_idx](child_rep)
        return ss_rep[4]


class MiniLTB(nn.Module):
    """Inference-only LTB wrapper matching the saved LibMTL state_dict keys."""

    def __init__(self, task_name: list[str], encoder_class, decoders: nn.ModuleDict, device: torch.device) -> None:
        super().__init__()
        self.task_name = task_name
        self.task_num = len(task_name)
        self.decoders = decoders
        encoders = [encoder_class() for _ in range(self.task_num)]
        self.encoder = _TransformResNetLTB(encoders, task_name, device)
        self.epoch = 100
        self.epochs = 100

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        reps = self.encoder(inputs, self.epoch, self.epochs)
        return {task: self.decoders[task](reps[idx]) for idx, task in enumerate(self.task_name)}


def build_resnet18_encoder(in_channels: int = 3, pretrained: bool = False):
    def encoder_class():
        model = resnet18(pretrained=pretrained)
        old_conv = model.conv1
        if old_conv.in_channels != in_channels:
            new_conv = nn.Conv2d(
                in_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=False,
            )
            with torch.no_grad():
                base_weight = old_conv.weight.mean(dim=1, keepdim=True)
                new_conv.weight.copy_(base_weight.repeat(1, in_channels, 1, 1) / max(in_channels, 1))
            model.conv1 = new_conv
        return model

    return encoder_class


class UnifiedHybridTensorDataset(torch.utils.data.Dataset):
    """Convert hybrid samples to one 3-channel tensor for LibMTL LTB baselines."""

    def __init__(
        self,
        manifest_path: Path,
        event_classes: list[str],
        distance_classes: list[str],
        csv_input_height: int,
        csv_input_width: int,
        image_size: int,
        normalize: str = "none",
    ) -> None:
        event_to_idx = {label: idx for idx, label in enumerate(event_classes)}
        distance_to_idx = {label: idx for idx, label in enumerate(distance_classes)}
        self.base_dataset = HybridInferenceDataset(
            manifest_path=manifest_path,
            event_to_idx=event_to_idx,
            distance_to_idx=distance_to_idx,
            csv_input_height=csv_input_height,
            csv_input_width=csv_input_width,
            image_size=image_size,
            normalize=normalize,
        )
        self.image_size = int(image_size)
        self.samples = self.base_dataset.samples

    def __len__(self) -> int:
        return len(self.base_dataset)

    def _to_unified_tensor(self, sample: dict[str, Any]) -> torch.Tensor:
        input_data = sample["input_data"]
        if sample["input_type"] == "image":
            if input_data.shape[-2:] != (self.image_size, self.image_size):
                input_data = F.interpolate(
                    input_data.unsqueeze(0),
                    size=(self.image_size, self.image_size),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
            return input_data.float()

        tensor = F.interpolate(
            input_data.unsqueeze(0),
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        if tensor.size(0) == 1:
            tensor = tensor.repeat(3, 1, 1)
        elif tensor.size(0) != 3:
            tensor = tensor.mean(dim=0, keepdim=True).repeat(3, 1, 1)
        return tensor.float()

    def __getitem__(self, index: int):
        sample, labels = self.base_dataset[index]
        return self._to_unified_tensor(sample), labels


def build_run_save_path(base_path: Path) -> Path:
    return base_path / datetime.now().strftime("%Y%m%d_%H%M%S")


def setup_publication_style() -> None:
    candidates = ["Times New Roman", "Cambria", "Georgia", "STIXGeneral", "DejaVu Serif"]
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.family"] = name
            break
    plt.rcParams.update(
        {
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.labelsize": 12.0,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 13.0,
            "figure.dpi": 220,
            "savefig.dpi": 350,
            "savefig.bbox": "tight",
        }
    )
    plt.rcParams["axes.unicode_minus"] = False


def extract_state_dict(checkpoint_obj: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint_obj, dict):
        if "model_state_dict" in checkpoint_obj:
            return checkpoint_obj["model_state_dict"]
        if "state_dict" in checkpoint_obj:
            return checkpoint_obj["state_dict"]
    return checkpoint_obj


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def run_config(run_dir: Path) -> dict[str, Any]:
    return load_json(run_dir / "run_config.json")


def to_device_hybrid(batch_inputs: dict[str, Any], batch_labels: dict[str, torch.Tensor], device: torch.device):
    moved = {
        "batch_size": batch_inputs["batch_size"],
        "sample_paths": batch_inputs.get("sample_paths", []),
        "input_types": batch_inputs.get("input_types", []),
        "csv_indices": batch_inputs["csv_indices"].to(device, non_blocking=True),
        "image_indices": batch_inputs["image_indices"].to(device, non_blocking=True),
        "csv_inputs": (
            batch_inputs["csv_inputs"].to(device, non_blocking=True)
            if batch_inputs["csv_inputs"] is not None
            else None
        ),
        "image_inputs": (
            batch_inputs["image_inputs"].to(device, non_blocking=True)
            if batch_inputs["image_inputs"] is not None
            else None
        ),
    }
    moved_labels = {key: value.to(device, non_blocking=True) for key, value in batch_labels.items()}
    return moved, moved_labels


def parse_enabled_views(value: str | list[str] | tuple[str, ...]) -> str:
    if isinstance(value, str):
        return value
    return ",".join(value)


def build_sensorfield_model(config: dict[str, Any], event_classes: list[str], distance_classes: list[str]) -> SensorFieldM3TImageFork:
    model = SensorFieldM3TImageFork(
        num_event_classes=len(event_classes),
        num_location_classes=len(distance_classes),
        hidden_dim=int(config.get("hidden_dim", 128)),
        num_anchors=int(config.get("num_anchors", 8)),
        num_heads=int(config.get("num_heads", 4)),
        fusion_dim=int(config.get("fusion_dim", 256)),
        raw_tokens=int(config.get("time_tokens", 6)),
        stf_tokens=int(config.get("freq_tokens", 48)),
        gaf_tokens=int(config.get("gaf_tokens", 48)),
        gaf_size=int(config.get("gaf_size", 48)),
        stft_n_fft=int(config.get("stft_n_fft", 256)),
        stft_hop_length=int(config.get("stft_hop_length", 128)),
        stft_win_length=int(config.get("stft_win_length", 256)),
        image_size=int(config.get("image_size", 224)),
        fac_loss_weight=float(config.get("fac_loss_weight", 0.0)),
        taef_loss_weight=float(config.get("taef_loss_weight", 0.0)),
        gcti_loss_weight=float(config.get("gcti_loss_weight", 0.0)),
        view_drop_prob=float(config.get("view_drop_prob", 0.0)),
        enable_view_consistency=bool(config.get("enable_view_consistency", False)),
        disable_fac=bool(config.get("disable_fac", False)),
        disable_complement=bool(config.get("disable_complement", False)),
        disable_taef=bool(config.get("disable_taef", False)),
        disable_gcti=bool(config.get("disable_gcti", False)),
        disable_view_consistency=bool(config.get("disable_view_consistency", False)),
        enabled_views=parse_enabled_views(config.get("enabled_views", "raw,stf,gaf")),
        view_consistency_weight=float(config.get("view_consistency_weight", 0.0)),
        view_noise_std=float(config.get("view_noise_std", 0.01)),
        location_image_backbone=str(config.get("location_image_backbone", "legacy_cnn")),
        location_image_backbone_pretrained=bool(config.get("location_image_backbone_pretrained", False)),
        image_location_ensemble_weight=float(config.get("image_location_ensemble_weight", 0.0)),
        image_location_specialist_blend=float(config.get("image_location_specialist_blend", 1.0)),
        image_event_expert_weight=float(config.get("image_event_expert_weight", 0.0)),
        image_location_expert_weight=float(config.get("image_location_expert_weight", 0.0)),
        dropout=float(config.get("dropout", 0.2)),
        return_auxiliary=True,
    )
    return model


def build_adapted_model(model_key: str, config: dict[str, Any], event_classes: list[str], distance_classes: list[str]) -> nn.Module:
    kwargs = {
        "num_event_classes": len(event_classes),
        "num_location_classes": len(distance_classes),
        "input_rows": int(config.get("csv_input_height", 6)),
        "hidden_dim": int(config.get("hidden_dim", 128)),
        "dropout": float(config.get("dropout", 0.2)),
        "stft_n_fft": int(config.get("stft_n_fft", 256)),
        "stft_hop_length": int(config.get("stft_hop_length", 128)),
        "stft_win_length": int(config.get("stft_win_length", 256)),
        "gaf_size": int(config.get("gaf_size", 48)),
        "image_size": int(config.get("image_size", 224)),
    }
    image_backbone = str(config.get("location_image_backbone", "resnet18"))
    image_pretrained = bool(config.get("location_image_backbone_pretrained", True))
    if model_key == "multimodn":
        return MultiModNImageFork(**kwargs, image_backbone=image_backbone, image_pretrained=image_pretrained)
    if model_key == "m4oe":
        return M4oEImageFork(**kwargs, image_backbone=image_backbone, image_pretrained=image_pretrained)
    if model_key == "das_mae":
        return DASMAEImageFork(
            num_event_classes=len(event_classes),
            num_location_classes=len(distance_classes),
            input_rows=int(config.get("csv_input_height", 6)),
            hidden_dim=int(config.get("hidden_dim", 128)),
            dropout=float(config.get("dropout", 0.2)),
            image_backbone=image_backbone,
            image_pretrained=image_pretrained,
        )
    if model_key == "pipelineadwint":
        return PipelineADWinTImageFork(**kwargs)
    raise ValueError(f"Unsupported adapted model key: {model_key}")


def build_ltb_model(event_classes: list[str], distance_classes: list[str], device: torch.device) -> MiniLTB:
    decoders = nn.ModuleDict(
        {
            "event_type": GlobalPoolDecoder(512, len(event_classes)),
            "distance_cls": GlobalPoolDecoder(512, len(distance_classes)),
        }
    )
    model = MiniLTB(
        task_name=["event_type", "distance_cls"],
        encoder_class=build_resnet18_encoder(in_channels=3, pretrained=False),
        decoders=decoders,
        device=device,
    )
    model.epoch = 100
    model.epochs = 100
    return model


def load_checkpoint(model: nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    state = extract_state_dict(torch.load(checkpoint_path, map_location=device))
    model.load_state_dict(state, strict=True)


def _zeros(batch_size: int, width: int, device: torch.device) -> torch.Tensor:
    return torch.zeros(batch_size, width, device=device)


def _pad_feature_dim(features: torch.Tensor, target_dim: int) -> torch.Tensor:
    """Right-pad feature vectors for hybrid branches with different hidden sizes."""
    if features.size(1) == target_dim:
        return features
    if features.size(1) > target_dim:
        return features[:, :target_dim]
    return F.pad(features, (0, target_dim - features.size(1)))


def extract_sensorfield_features(model: SensorFieldM3TImageFork, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    model.eval()
    event_features, event_labels = [], []
    location_features, location_labels = [], []
    paths = []
    with torch.no_grad():
        for batch_inputs, batch_labels in loader:
            batch_inputs, batch_labels = to_device_hybrid(batch_inputs, batch_labels, device)
            batch_size = int(batch_inputs["batch_size"])
            image_dim = int(model.location_head.in_features)
            csv_dim = int(getattr(model.csv_backbone, "hidden_dim", image_dim))
            feature_dim = max(image_dim, csv_dim)
            event_batch = _zeros(batch_size, feature_dim, device)
            location_batch = _zeros(batch_size, feature_dim, device)
            valid_location = torch.zeros(batch_size, dtype=torch.bool, device=device)

            csv_inputs = batch_inputs.get("csv_inputs")
            csv_indices = batch_inputs.get("csv_indices")
            if csv_inputs is not None and csv_indices.numel() > 0:
                csv_outputs = model.csv_backbone(csv_inputs)
                tokens = csv_outputs.get("gcti_outputs", {}).get("updated_task_tokens")
                if tokens is not None:
                    csv_event = _pad_feature_dim(tokens[:, 0], feature_dim)
                    event_batch.index_copy_(0, csv_indices, csv_event.to(event_batch.dtype))
                    if tokens.size(1) > 1:
                        csv_location = _pad_feature_dim(tokens[:, 1], feature_dim)
                        location_batch.index_copy_(0, csv_indices, csv_location.to(location_batch.dtype))

            image_inputs = batch_inputs.get("image_inputs")
            image_indices = batch_inputs.get("image_indices")
            if image_inputs is not None and image_indices.numel() > 0:
                image_feature, image_feature_map, _ = model._encode_location_image(image_inputs.float())
                image_event_hidden = model.image_event_tower(image_feature)
                image_location_hidden = model.image_location_tower(image_feature)
                if image_feature_map is not None:
                    image_location_logits, _ = model._forward_image_location(image_feature, image_feature_map)
                    classifier = model.image_location_classifier
                    if isinstance(classifier, nn.Sequential) and len(classifier) >= 2:
                        image_location_hidden = classifier[:-1](image_location_hidden)
                    _ = image_location_logits
                image_event_hidden = _pad_feature_dim(image_event_hidden, feature_dim)
                image_location_hidden = _pad_feature_dim(image_location_hidden, feature_dim)
                event_batch.index_copy_(0, image_indices, image_event_hidden.to(event_batch.dtype))
                location_batch.index_copy_(0, image_indices, image_location_hidden.to(location_batch.dtype))
                valid_location[image_indices] = batch_labels["distance_cls"][image_indices] != DISTANCE_IGNORE_INDEX

            event_features.append(event_batch.detach().cpu())
            event_labels.extend(batch_labels["event_type"].detach().cpu().tolist())
            if valid_location.any():
                location_features.append(location_batch[valid_location].detach().cpu())
                location_labels.extend(batch_labels["distance_cls"][valid_location].detach().cpu().tolist())
            paths.extend(batch_inputs.get("sample_paths", []))
    return _feature_payload(event_features, event_labels, location_features, location_labels, paths)


def extract_adapted_features(model: nn.Module, model_key: str, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    model.eval()
    event_features, event_labels = [], []
    location_features, location_labels = [], []
    paths = []
    with torch.no_grad():
        for batch_inputs, batch_labels in loader:
            batch_inputs, batch_labels = to_device_hybrid(batch_inputs, batch_labels, device)
            batch_size = int(batch_inputs["batch_size"])
            hidden_dim = int(getattr(model, "hidden_dim", 128))
            event_batch = _zeros(batch_size, hidden_dim, device)
            location_batch = _zeros(batch_size, hidden_dim, device)
            valid_location = torch.zeros(batch_size, dtype=torch.bool, device=device)

            csv_inputs = batch_inputs.get("csv_inputs")
            csv_indices = batch_inputs.get("csv_indices")
            if csv_inputs is not None and csv_indices.numel() > 0:
                signal = model.signal(_extract_csv_signal(csv_inputs)) if hasattr(model, "signal") else None
                if model_key == "multimodn":
                    state = torch.zeros(csv_inputs.size(0), hidden_dim, device=device, dtype=csv_inputs.dtype)
                    for name in ("raw", "stf", "gaf"):
                        update = model.modality_updates[name](torch.cat([state, signal[name]], dim=-1))
                        state = model.final_norm(state + update)
                    event_batch.index_copy_(0, csv_indices, state.to(event_batch.dtype))
                elif model_key == "m4oe":
                    features = {name: signal[name] for name in ("raw", "stf", "gaf")}
                    event_hidden = model._fuse_task(features, model.event_router)
                    event_batch.index_copy_(0, csv_indices, event_hidden.to(event_batch.dtype))
                elif model_key == "das_mae":
                    patches = model._patchify(_extract_csv_signal(csv_inputs))
                    tokens = model.patch_embed(patches)
                    encoded = model.encoder(tokens)
                    pooled = encoded.mean(dim=1)
                    event_hidden = model.event_tower(pooled)
                    event_batch.index_copy_(0, csv_indices, event_hidden.to(event_batch.dtype))
                elif model_key == "pipelineadwint":
                    pseudo_images = signal["pseudo_image"]
                    visual_hidden = model.visual_backbone(pseudo_images)
                    event_hidden = model.event_tower(visual_hidden)
                    event_batch.index_copy_(0, csv_indices, event_hidden.to(event_batch.dtype))

            image_inputs = batch_inputs.get("image_inputs")
            image_indices = batch_inputs.get("image_indices")
            if image_inputs is not None and image_indices.numel() > 0:
                if model_key in {"multimodn", "das_mae"}:
                    image_feature = model.image_head.encoder(image_inputs.float())
                    event_hidden = model.image_head.event_tower(image_feature)
                    location_hidden = model.image_head.location_tower(image_feature)
                elif model_key == "m4oe":
                    image_feature = model.image_encoder(image_inputs.float())
                    features = {"image": image_feature}
                    event_hidden = model._fuse_task(features, model.event_router)
                    location_hidden = model._fuse_task(features, model.location_router)
                else:
                    image_feature = model.visual_backbone(image_inputs.float())
                    event_hidden = model.event_tower(image_feature)
                    location_hidden = model.location_tower(image_feature)
                event_batch.index_copy_(0, image_indices, event_hidden.to(event_batch.dtype))
                location_batch.index_copy_(0, image_indices, location_hidden.to(location_batch.dtype))
                valid_location[image_indices] = batch_labels["distance_cls"][image_indices] != DISTANCE_IGNORE_INDEX

            event_features.append(event_batch.detach().cpu())
            event_labels.extend(batch_labels["event_type"].detach().cpu().tolist())
            if valid_location.any():
                location_features.append(location_batch[valid_location].detach().cpu())
                location_labels.extend(batch_labels["distance_cls"][valid_location].detach().cpu().tolist())
            paths.extend(batch_inputs.get("sample_paths", []))
    return _feature_payload(event_features, event_labels, location_features, location_labels, paths)


def _extract_csv_signal(csv_inputs: torch.Tensor) -> torch.Tensor:
    if csv_inputs.ndim == 4:
        return csv_inputs[:, 0].float()
    if csv_inputs.ndim == 3:
        return csv_inputs.float()
    raise ValueError(f"Unsupported CSV input shape: {tuple(csv_inputs.shape)}")


def extract_sota_features(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    model.eval()
    event_features, event_labels = [], []
    location_features, location_labels = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = {key: value.to(device, non_blocking=True) for key, value in labels.items()}
            outputs = model(inputs)
            feats = outputs["features"].detach().cpu()
            event_features.append(feats)
            event_labels.extend(labels["event_type"].detach().cpu().tolist())
            valid_mask = labels["distance_cls"] != DISTANCE_IGNORE_INDEX
            if valid_mask.any():
                location_features.append(feats[valid_mask.detach().cpu()])
                location_labels.extend(labels["distance_cls"][valid_mask].detach().cpu().tolist())
    return _feature_payload(event_features, event_labels, location_features, location_labels, [])


def extract_ltb_features(model: MiniLTB, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    model.eval()
    event_features, event_labels = [], []
    location_features, location_labels = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = {key: value.to(device, non_blocking=True) for key, value in labels.items()}
            reps = model.encoder(inputs, model.epoch, model.epochs)
            event_rep = F.adaptive_avg_pool2d(reps[0], (1, 1)).flatten(1).detach().cpu()
            location_rep = F.adaptive_avg_pool2d(reps[1], (1, 1)).flatten(1)
            event_features.append(event_rep)
            event_labels.extend(labels["event_type"].detach().cpu().tolist())
            valid_mask = labels["distance_cls"] != DISTANCE_IGNORE_INDEX
            if valid_mask.any():
                location_features.append(location_rep[valid_mask].detach().cpu())
                location_labels.extend(labels["distance_cls"][valid_mask].detach().cpu().tolist())
    return _feature_payload(event_features, event_labels, location_features, location_labels, [])


def _feature_payload(
    event_features: list[torch.Tensor],
    event_labels: list[int],
    location_features: list[torch.Tensor],
    location_labels: list[int],
    sample_paths: list[str],
) -> dict[str, Any]:
    return {
        "event_features": torch.cat(event_features, dim=0).numpy(),
        "event_labels": np.asarray(event_labels, dtype=np.int64),
        "location_features": (
            torch.cat(location_features, dim=0).numpy()
            if location_features
            else np.zeros((0, 1), dtype=np.float32)
        ),
        "location_labels": np.asarray(location_labels, dtype=np.int64),
        "sample_paths": sample_paths,
    }


def compute_tsne(features: np.ndarray, random_state: int = 42) -> np.ndarray:
    if features.ndim != 2:
        raise ValueError(f"Expected 2D features, got {features.shape}")
    if features.shape[0] < 3:
        return np.zeros((features.shape[0], 2), dtype=np.float32)
    features = np.asarray(features, dtype=np.float32)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    n_components = min(50, features.shape[1], features.shape[0] - 1)
    reduced = PCA(n_components=n_components, random_state=random_state).fit_transform(features) if n_components >= 2 else features
    perplexity = max(5, min(30, features.shape[0] // 8))
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate="auto",
        init="pca",
        random_state=random_state,
        max_iter=1500,
    )
    return tsne.fit_transform(reduced).astype(np.float32)


def compute_centroid_stats(embeddings: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    unique_labels = sorted(set(labels.tolist()))
    centroids = {}
    for label in unique_labels:
        points = embeddings[labels == label]
        centroids[int(label)] = points.mean(axis=0)
    pairwise = []
    for i, label_i in enumerate(unique_labels):
        for label_j in unique_labels[i + 1 :]:
            pairwise.append(float(np.linalg.norm(centroids[int(label_i)] - centroids[int(label_j)])))
    if not pairwise:
        return 0.0, 0.0
    return float(min(pairwise)), float(np.mean(pairwise))


def save_embedding_csv(path: Path, embeddings: np.ndarray, labels: np.ndarray, label_names: list[str], task_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["x", "y", "label_idx", "label_name", "task"])
        for point, label in zip(embeddings, labels):
            writer.writerow([f"{float(point[0]):.8f}", f"{float(point[1]):.8f}", int(label), label_names[int(label)], task_name])


def _style_tsne_axis(ax, title: str) -> None:
    ax.set_title(title, pad=7, fontweight="semibold")
    ax.set_xlabel("t-SNE dimension 1")
    ax.set_ylabel("t-SNE dimension 2")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


def _scatter_tsne(ax, embeddings: np.ndarray, labels: np.ndarray, label_names: list[str], color_map: dict[str, str], marker_map: dict[str, str]):
    handles = []
    for label_idx, label_name in enumerate(label_names):
        mask = labels == label_idx
        if not np.any(mask):
            continue
        points = embeddings[mask]
        handle = ax.scatter(
            points[:, 0],
            points[:, 1],
            s=18,
            alpha=0.80,
            c=color_map.get(label_name, "#444444"),
            marker=marker_map.get(label_name, "o"),
            label=label_name,
            edgecolors="white",
            linewidths=0.28,
        )
        centroid = points.mean(axis=0)
        ax.scatter(
            centroid[0],
            centroid[1],
            s=76,
            c=color_map.get(label_name, "#444444"),
            marker="X",
            edgecolors="black",
            linewidths=0.5,
            zorder=5,
        )
        handles.append(handle)
    return handles


def plot_tsne(path: Path, embeddings: np.ndarray, labels: np.ndarray, label_names: list[str], color_map: dict[str, str], marker_map: dict[str, str], title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.7, 4.7))
    handles = _scatter_tsne(ax, embeddings, labels, label_names, color_map, marker_map)
    _style_tsne_axis(ax, title)
    if handles:
        ax.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.12),
            ncol=min(4, len(handles)),
            frameon=False,
            handletextpad=0.4,
            columnspacing=0.8,
            markerscale=1.1,
        )
    fig.savefig(path)
    if path.suffix.lower() == ".png":
        fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def plot_tsne_grid(path: Path, panels: list[tuple[str, np.ndarray, np.ndarray]], label_names: list[str], color_map: dict[str, str], marker_map: dict[str, str], figure_title: str) -> None:
    if not panels:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = 4
    rows = math.ceil(len(panels) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.6, rows * 3.55))
    axes = np.atleast_1d(axes).reshape(rows, cols)
    legend_handles = None
    panel_labels = [chr(ord("a") + idx) for idx in range(len(panels))]

    for idx, (model_name, embeddings, labels) in enumerate(panels):
        ax = axes[idx // cols, idx % cols]
        handles = _scatter_tsne(ax, embeddings, labels, label_names, color_map, marker_map)
        if legend_handles is None and handles:
            legend_handles = handles
        title_name = DISPLAY_SHORT_NAMES.get(model_name, model_name)
        _style_tsne_axis(ax, f"({panel_labels[idx]}) {title_name}")

    for idx in range(len(panels), rows * cols):
        axes[idx // cols, idx % cols].axis("off")

    fig.suptitle(figure_title, y=0.998, fontsize=12, fontweight="semibold")
    if legend_handles:
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.022),
            ncol=min(4, len(legend_handles)),
            frameon=False,
            handletextpad=0.5,
            columnspacing=1.0,
            markerscale=1.1,
        )
    fig.tight_layout(rect=[0.015, 0.08, 0.985, 0.955])
    fig.savefig(path)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def build_loaders(args: argparse.Namespace, event_classes: list[str], distance_classes: list[str]):
    event_to_idx = {label: idx for idx, label in enumerate(event_classes)}
    distance_to_idx = {label: idx for idx, label in enumerate(distance_classes)}
    dataset_path = Path(args.dataset_path).expanduser().resolve()
    hybrid_dataset = HybridInferenceDataset(
        manifest_path=dataset_path / f"{args.split}.csv",
        event_to_idx=event_to_idx,
        distance_to_idx=distance_to_idx,
        csv_input_height=args.csv_input_height,
        csv_input_width=args.csv_input_width,
        image_size=args.image_size,
        normalize="none",
    )
    hybrid_loader = DataLoader(
        hybrid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=hybrid_collate_with_meta,
    )
    sota_dataset = ImageForkMultiModalDataset(
        manifest_path=dataset_path / f"{args.split}.csv",
        event_to_idx=event_to_idx,
        distance_to_idx=distance_to_idx,
        raw_length=args.raw_length,
        stft_size=args.stft_size,
        gaf_size=args.sota_gaf_size,
        image_size=args.image_size,
        normalize="sample",
        augment=False,
        stft_n_fft=args.stft_n_fft,
        stft_hop_length=args.stft_hop_length,
        stft_win_length=args.stft_win_length,
    )
    sota_loader = DataLoader(
        sota_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_multitask,
    )
    ltb_dataset = UnifiedHybridTensorDataset(
        manifest_path=dataset_path / f"{args.split}.csv",
        event_classes=event_classes,
        distance_classes=distance_classes,
        csv_input_height=args.csv_input_height,
        csv_input_width=args.csv_input_width,
        image_size=args.image_size,
        normalize="none",
    )
    ltb_loader = DataLoader(
        ltb_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return hybrid_loader, sota_loader, ltb_loader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot t-SNE distributions for the current eight comparison models.")
    parser.add_argument("--dataset_path", default=str(DEFAULT_DATASET_PATH), type=str)
    parser.add_argument("--save_path", default=str(DEFAULT_OUTPUT_ROOT), type=str)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--gpu_id", default=0, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--batch_size", default=24, type=int)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--csv_input_height", default=6, type=int)
    parser.add_argument("--csv_input_width", default=10000, type=int)
    parser.add_argument("--image_size", default=224, type=int)
    parser.add_argument("--raw_length", default=4096, type=int)
    parser.add_argument("--stft_size", default=128, type=int)
    parser.add_argument("--sota_gaf_size", default=128, type=int)
    parser.add_argument("--stft_n_fft", default=256, type=int)
    parser.add_argument("--stft_hop_length", default=128, type=int)
    parser.add_argument("--stft_win_length", default=256, type=int)
    parser.add_argument("--event_classes", default="walking,excavator,driving,background", type=str)
    parser.add_argument("--distance_classes", default="Alarm area,Tracking area,No-threat area", type=str)
    parser.add_argument("--models", default=",".join(DEFAULT_RUNS.keys()), type=str)
    return parser.parse_args()


def model_kind(model_name: str, config: dict[str, Any]) -> str:
    config_model = str(config.get("model", "")).strip().lower()
    if model_name == "ConvNeXt-Small":
        return "convnext"
    if model_name in {"Aligned-MTL", "MoCo-weighting"}:
        return "ltb"
    if config_model:
        return config_model
    lookup = {
        "MultiModN": "multimodn",
        "M4oE": "m4oe",
        "DAS-MAE + downstream fine-tuning head": "das_mae",
        "PipelineADWinT": "pipelineadwint",
        "SensorField-M3T": "sensorfield_m3t",
    }
    return lookup[model_name]


def main() -> None:
    setup_publication_style()
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = resolve_device(args.gpu_id)
    event_classes = parse_label_list(args.event_classes)
    distance_classes = parse_label_list(args.distance_classes)
    selected_models = [item.strip() for item in args.models.split(",") if item.strip()]
    unknown = [name for name in selected_models if name not in DEFAULT_RUNS]
    if unknown:
        raise ValueError(f"Unknown models: {unknown}. Supported: {list(DEFAULT_RUNS)}")

    save_root = build_run_save_path(Path(args.save_path).expanduser().resolve())
    save_root.mkdir(parents=True, exist_ok=True)
    hybrid_loader, sota_loader, ltb_loader = build_loaders(args, event_classes, distance_classes)

    summary_rows = []
    event_panels = []
    location_panels = []
    failures = []

    for model_name in selected_models:
        run_dir = DEFAULT_RUNS[model_name].resolve()
        checkpoint = run_dir / "best.pt"
        config = run_config(run_dir)
        kind = model_kind(model_name, config)
        model_dir = save_root / model_name.replace(" ", "_").replace("+", "plus").replace("/", "_")
        model_dir.mkdir(parents=True, exist_ok=True)

        try:
            if not checkpoint.is_file():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
            print(f"Extracting {model_name} from {checkpoint}")
            if kind == "convnext":
                model = build_sota_image_model("convnext_small", len(event_classes), len(distance_classes), pretrained=False, dropout=0.2).to(device)
                load_checkpoint(model, checkpoint, device)
                payload = extract_sota_features(model, sota_loader, device)
            elif kind == "ltb":
                model = build_ltb_model(event_classes, distance_classes, device).to(device)
                load_checkpoint(model, checkpoint, device)
                payload = extract_ltb_features(model, ltb_loader, device)
            elif kind == "sensorfield_m3t":
                model = build_sensorfield_model(config, event_classes, distance_classes).to(device)
                load_checkpoint(model, checkpoint, device)
                payload = extract_sensorfield_features(model, hybrid_loader, device)
            else:
                model = build_adapted_model(kind, config, event_classes, distance_classes).to(device)
                load_checkpoint(model, checkpoint, device)
                payload = extract_adapted_features(model, kind, hybrid_loader, device)

            event_embedding = compute_tsne(payload["event_features"], random_state=args.seed)
            save_embedding_csv(model_dir / "event_tsne_points.csv", event_embedding, payload["event_labels"], event_classes, "event")
            plot_tsne(
                model_dir / "event_tsne.png",
                event_embedding,
                payload["event_labels"],
                event_classes,
                EVENT_COLOR_MAP,
                EVENT_MARKERS,
                f"{model_name} event-feature t-SNE",
            )
            event_panels.append((model_name, event_embedding, payload["event_labels"]))
            event_silhouette = (
                float(silhouette_score(event_embedding, payload["event_labels"]))
                if len(set(payload["event_labels"].tolist())) > 1
                else 0.0
            )
            event_min_dist, event_mean_dist = compute_centroid_stats(event_embedding, payload["event_labels"])

            location_features = payload["location_features"]
            location_labels = payload["location_labels"]
            if location_features.shape[0] >= 3:
                location_embedding = compute_tsne(location_features, random_state=args.seed)
                save_embedding_csv(model_dir / "location_tsne_points.csv", location_embedding, location_labels, distance_classes, "location")
                plot_tsne(
                    model_dir / "location_tsne.png",
                    location_embedding,
                    location_labels,
                    distance_classes,
                    LOCATION_COLOR_MAP,
                    LOCATION_MARKERS,
                    f"{model_name} location-feature t-SNE",
                )
                location_panels.append((model_name, location_embedding, location_labels))
                location_silhouette = (
                    float(silhouette_score(location_embedding, location_labels))
                    if len(set(location_labels.tolist())) > 1
                    else 0.0
                )
                location_min_dist, location_mean_dist = compute_centroid_stats(location_embedding, location_labels)
            else:
                location_silhouette = 0.0
                location_min_dist = 0.0
                location_mean_dist = 0.0

            summary = {
                "model_name": model_name,
                "kind": kind,
                "run_dir": str(run_dir),
                "checkpoint": str(checkpoint),
                "event_samples": int(payload["event_features"].shape[0]),
                "event_feature_dim": int(payload["event_features"].shape[1]),
                "event_silhouette": event_silhouette,
                "event_min_centroid_dist": event_min_dist,
                "event_mean_centroid_dist": event_mean_dist,
                "location_samples": int(location_features.shape[0]),
                "location_feature_dim": int(location_features.shape[1]) if location_features.ndim == 2 else 0,
                "location_silhouette": location_silhouette,
                "location_min_centroid_dist": location_min_dist,
                "location_mean_centroid_dist": location_mean_dist,
                "status": "completed",
            }
            with (model_dir / "summary.json").open("w", encoding="utf-8") as handle:
                json.dump(summary, handle, ensure_ascii=False, indent=2)
            summary_rows.append(summary)
        except Exception as exc:
            failure = {
                "model_name": model_name,
                "kind": kind,
                "run_dir": str(run_dir),
                "checkpoint": str(checkpoint),
                "status": "failed",
                "error": repr(exc),
            }
            failures.append(failure)
            with (model_dir / "failure.json").open("w", encoding="utf-8") as handle:
                json.dump(failure, handle, ensure_ascii=False, indent=2)
            print(f"[FAILED] {model_name}: {exc}")

    if summary_rows:
        summary_path = save_root / "tsne_summary.csv"
        with summary_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
        plot_tsne_grid(
            save_root / "event_tsne_grid.png",
            event_panels,
            event_classes,
            EVENT_COLOR_MAP,
            EVENT_MARKERS,
            "Event-task t-SNE comparison across models",
        )
        plot_tsne_grid(
            save_root / "location_tsne_grid.png",
            location_panels,
            distance_classes,
            LOCATION_COLOR_MAP,
            LOCATION_MARKERS,
            "Location-task t-SNE comparison across models",
        )
        print(f"Saved t-SNE outputs to: {save_root}")
        print(f"Summary table: {summary_path}")
    if failures:
        failure_path = save_root / "tsne_failures.json"
        with failure_path.open("w", encoding="utf-8") as handle:
            json.dump(failures, handle, ensure_ascii=False, indent=2)
        print(f"Failures recorded at: {failure_path}")


if __name__ == "__main__":
    main()
