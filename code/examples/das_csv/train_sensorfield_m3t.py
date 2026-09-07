import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from PIL import Image

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - optional at runtime
    plt = None

try:
    from openpyxl import Workbook
except Exception:  # pragma: no cover - optional at runtime
    Workbook = None

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[1]
WORKSPACE_ROOT = CURRENT_DIR.parents[2]
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(1, str(PROJECT_ROOT))

from sensorfield_dataset import (
    DISTANCE_IGNORE_INDEX,
    audit_dataset_manifests,
    canonicalize_dataset_manifests,
    csv_dataloader,
    ensure_mtl43_manifests,
    parse_label_list,
)
from LibMTL.model.pipemmtl import PipeMMTL
from LibMTL.model.sensorfield_m3t import SensorFieldM3T
from LibMTL.model.condition_baselines import build_condition_baseline
from sensorfield_metrics import attach_task_metrics, classification_metrics, metric_rows

DEFAULT_DATASET_PATH = WORKSPACE_ROOT / "converted_csv" / "MTL43"
DEFAULT_SAVE_PATH = PROJECT_ROOT / "examples" / "das_csv" / "runs" / "sensorfield_m3t"
DEFAULT_LOCATION_IMAGE_ROOT = WORKSPACE_ROOT / "_datasets" / "location_images"


class GHMCrossEntropyLoss(nn.Module):
    """A simple multi-class GHM-C variant using 1-pt as gradient proxy."""

    def __init__(self, bins=10, momentum=0.75, ignore_index=None):
        super().__init__()
        if bins <= 1:
            raise ValueError("bins must be greater than 1 for GHM.")
        self.bins = bins
        self.momentum = momentum
        self.ignore_index = ignore_index
        edges = torch.linspace(0, 1, steps=bins + 1)
        edges[-1] += 1e-6
        self.register_buffer("edges", edges)
        self.register_buffer("acc_sum", torch.zeros(bins))

    def forward(self, logits, targets):
        if logits.ndim != 2:
            raise ValueError(f"Expected [B, C] logits for GHM loss, got {tuple(logits.shape)}")
        if targets.ndim != 1:
            raise ValueError(f"Expected [B] targets for GHM loss, got {tuple(targets.shape)}")
        if self.ignore_index is not None:
            valid_mask = targets != self.ignore_index
            if not valid_mask.any():
                return logits.sum() * 0.0
            logits = logits[valid_mask]
            targets = targets[valid_mask]
        if logits.size(0) == 0:
            return logits.sum() * 0.0

        with torch.no_grad():
            probs = torch.softmax(logits, dim=1)
            pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
            grad_proxy = (1.0 - pt).clamp_(0.0, 1.0)
            weights = torch.zeros_like(grad_proxy)
            total = grad_proxy.numel()
            non_empty_bins = 0

            for idx in range(self.bins):
                left = self.edges[idx]
                right = self.edges[idx + 1]
                in_bin = (grad_proxy >= left) & (grad_proxy < right)
                count = int(in_bin.sum().item())
                if count == 0:
                    continue
                non_empty_bins += 1
                if self.training and self.momentum > 0:
                    self.acc_sum[idx] = self.momentum * self.acc_sum[idx] + (1.0 - self.momentum) * count
                    effective = self.acc_sum[idx]
                else:
                    effective = torch.tensor(float(count), device=grad_proxy.device)
                weights[in_bin] = total / effective.clamp_min(1.0)

            if non_empty_bins > 0:
                weights = weights / non_empty_bins
            else:
                weights.fill_(1.0)

        ce = F.cross_entropy(logits, targets, reduction="none")
        return (ce * weights).sum() / max(logits.size(0), 1)


class SafeCrossEntropyLoss(nn.Module):
    """Cross-entropy that returns zero when a whole batch is ignored."""

    def __init__(self, ignore_index=None):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        if logits.ndim != 2:
            raise ValueError(f"Expected [B, C] logits for CE loss, got {tuple(logits.shape)}")
        if targets.ndim != 1:
            raise ValueError(f"Expected [B] targets for CE loss, got {tuple(targets.shape)}")
        if self.ignore_index is not None:
            valid_mask = targets != self.ignore_index
            if not valid_mask.any():
                return logits.sum() * 0.0
            logits = logits[valid_mask]
            targets = targets[valid_mask]
        if logits.size(0) == 0:
            return logits.sum() * 0.0
        return F.cross_entropy(logits, targets)


def build_run_save_path(base_path):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return base_path / timestamp


def set_trainable(module, trainable):
    for parameter in module.parameters():
        parameter.requires_grad = bool(trainable)


def load_location_image_pretrained(model, checkpoint_path, device):
    if not checkpoint_path:
        return False
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        print(f"Skip loading location-image pretrain because file was not found: {path}")
        return False

    checkpoint = torch.load(path, map_location=device)
    state = checkpoint.get("location_image_state_dict", checkpoint)
    if "model_state_dict" in checkpoint and isinstance(checkpoint["model_state_dict"], dict):
        model_state = checkpoint["model_state_dict"]
    else:
        model_state = None
    module_names = (
        "location_image_encoder",
        "location_image_proj",
        "location_image_aux_head",
        "image_location_tower",
        "location_head",
        "image_event_tower",
        "image_event_head",
        "image_event_expert_head",
        "image_location_expert_head",
    )

    def _load_generic_backbone(source_state):
        if not isinstance(source_state, dict):
            return False
        if not hasattr(model, "location_image_encoder"):
            return False
        encoder = getattr(model, "location_image_encoder")
        if not hasattr(encoder, "backbone"):
            return False
        generic_backbone = {
            key[len("backbone.") :]: value
            for key, value in source_state.items()
            if key.startswith("backbone.")
        }
        if not generic_backbone:
            return False
        encoder.backbone.load_state_dict(generic_backbone, strict=True)
        generic_event_head = {
            key[len("event_head.") :]: value
            for key, value in source_state.items()
            if key.startswith("event_head.")
        }
        if generic_event_head and hasattr(model, "image_event_expert_head"):
            getattr(model, "image_event_expert_head").load_state_dict(generic_event_head, strict=True)
        generic_location_head = {
            key[len("location_head.") :]: value
            for key, value in source_state.items()
            if key.startswith("location_head.")
        }
        if generic_location_head and hasattr(model, "image_location_expert_head"):
            getattr(model, "image_location_expert_head").load_state_dict(generic_location_head, strict=True)
        return True

    loaded_parts = 0
    for module_name in module_names:
        if not hasattr(model, module_name):
            continue
        if module_name in state:
            try:
                getattr(model, module_name).load_state_dict(state[module_name], strict=True)
            except RuntimeError as exc:
                raise RuntimeError(
                    f"Failed to load {module_name} from {path}. "
                    "Make sure the pretraining branch dimension matches PipeMMTL fusion_dim/location image size."
                ) from exc
            loaded_parts += 1
    if loaded_parts == 0 and _load_generic_backbone(state):
        loaded_parts = 1
    if loaded_parts == 0 and isinstance(state, dict):
        fallback_state = {}
        for key, value in state.items():
            if any(key.startswith(f"{module_name}.") for module_name in module_names):
                fallback_state[key] = value
        if fallback_state:
            model.load_state_dict(fallback_state, strict=False)
            loaded_parts = 1
    if loaded_parts == 0 and model_state is not None and _load_generic_backbone(model_state):
        loaded_parts = 1
    if loaded_parts == 0 and model_state is not None:
        fallback_state = {}
        for key, value in model_state.items():
            if any(key.startswith(f"{module_name}.") for module_name in module_names):
                fallback_state[key] = value
        if fallback_state:
            model.load_state_dict(fallback_state, strict=False)
            loaded_parts = 1
    if loaded_parts == 0:
        raise ValueError(f"No location-image weights found in checkpoint: {path}")
    print(f"Loaded location-image pretrained weights from {path}")
    return True


def parse_args():
    parser = argparse.ArgumentParser(description="Train PipeMMTL on CSV-based DAS signals.")
    parser.add_argument(
        "--model",
        default="pipemmtl",
        choices=[
            "pipemmtl",
            "sensorfield_m3t",
            "convnext_small",
            "multimodn",
            "m4oe",
            "das_mae",
            "pipelineadwint",
            "aligned_mtl",
            "moco_mtl",
        ],
    )
    parser.add_argument("--dataset_path", default=str(DEFAULT_DATASET_PATH), type=str)
    parser.add_argument("--save_path", default=str(DEFAULT_SAVE_PATH), type=str)
    parser.add_argument("--bs", default=8, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=80, type=int, help="Maximum training epochs.")
    parser.add_argument("--num_workers", default=0, type=int, help="DataLoader workers.")
    parser.add_argument("--gpu_id", default=0, type=int, help="CUDA device id when CUDA is available.")
    parser.add_argument("--seed", default=42, type=int, help="Random seed.")
    parser.add_argument(
        "--input_height",
        default=6,
        type=int,
        help="Target sample height after resizing.",
    )
    parser.add_argument("--input_width", default=10000, type=int, help="Target signal width after resizing.")
    parser.add_argument(
        "--return_multiview",
        action="store_true",
        default=False,
        help="Return dict inputs with raw/stf/gaf views instead of the legacy tensor payload.",
    )
    parser.add_argument(
        "--spatial_adapter",
        default="center",
        choices=["center", "mean", "learned"],
        help="How retained 1/6/10 spatial channels are reduced to the Raw [1, 10000] view.",
    )
    parser.add_argument(
        "--stf_spatial_fusion",
        default="group3_mean",
        choices=["group3_mean", "channel_stack"],
        help="Fuse each adjacent three-channel sensing group before composing the STF map.",
    )
    parser.add_argument("--stf_size", default=224, type=int, help="Resolution for the STF map.")
    parser.add_argument("--normalize", default="none", choices=["sample", "none"])
    parser.add_argument(
        "--event_classes",
        default="walking,excavator,driving,background",
        type=str,
        help="Comma-separated event labels in index order.",
    )
    parser.add_argument(
        "--distance_classes",
        default="Alarm area,Tracking area,No-threat area",
        type=str,
        help="Comma-separated location labels in index order.",
    )
    parser.add_argument("--lr", default=3e-5, type=float, help="AdamW learning rate.")
    parser.add_argument("--weight_decay", default=5e-4, type=float, help="AdamW weight decay.")
    parser.add_argument("--step_size", default=10, type=int, help="StepLR step size.")
    parser.add_argument("--gamma", default=0.7, type=float, help="StepLR gamma.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Gradient clipping value. <=0 disables.")
    parser.add_argument("--event_loss_weight", default=1.0, type=float, help="Loss weight for event classification.")
    parser.add_argument(
        "--location_loss_weight",
        default=1.2,
        type=float,
        help="Loss weight for location classification.",
    )
    parser.add_argument(
        "--early_stop_patience",
        default=8,
        type=int,
        help="Stop if validation score does not improve for this many epochs. <=0 disables early stop.",
    )
    parser.add_argument(
        "--early_stop_min_delta",
        default=5e-4,
        type=float,
        help="Minimum validation score improvement required to reset early stopping patience.",
    )
    parser.add_argument("--embed_dim", default=128, type=int, help="Shared token embedding dimension.")
    parser.add_argument("--fusion_dim", default=256, type=int, help="Hidden dimension after multimodal fusion.")
    parser.add_argument("--time_tokens", default=6, type=int, help="Token count for time-domain features.")
    parser.add_argument("--freq_tokens", default=48, type=int, help="Token count for frequency-domain features.")
    parser.add_argument("--gaf_tokens", default=48, type=int, help="Token count for GAF image features.")
    parser.add_argument("--gaf_size", default=224, type=int, help="Resolution used for Gramian Angular Field images.")
    parser.add_argument("--prior_tokens", default=8, type=int, help="Learnable prior token count.")
    parser.add_argument("--num_heads", default=4, type=int, help="Attention head count.")
    parser.add_argument("--dropout", default=0.2, type=float, help="Dropout in PipeMMTL.")
    parser.add_argument(
        "--baseline_pretrained",
        action="store_true",
        default=False,
        help="Use cached ImageNet initialization for visual condition baselines when available.",
    )
    parser.add_argument(
        "--task_balance",
        default="equal",
        choices=["equal", "aligned_mtl", "moco"],
        help="Task-gradient combination used during training.",
    )
    parser.add_argument("--hidden_dim", default=128, type=int, help="Hidden dimension for SensorField-M3T.")
    parser.add_argument("--num_anchors", default=16, type=int, help="Anchor count used by SensorField-M3T FAC.")
    parser.add_argument("--fac_loss_weight", default=0.1, type=float, help="FAC auxiliary loss weight.")
    parser.add_argument("--taef_loss_weight", default=0.0, type=float, help="TAEF diversity loss weight.")
    parser.add_argument("--gcti_loss_weight", default=0.01, type=float, help="GCTI consistency loss weight.")
    parser.add_argument("--view_drop_prob", default=0.3, type=float, help="Probability of applying one-view perturbation.")
    parser.add_argument(
        "--enable_view_consistency",
        action="store_true",
        default=True,
        help="Enable full-view versus perturbed-view consistency regularization.",
    )
    parser.add_argument("--disable_fac", action="store_true", default=False)
    parser.add_argument("--disable_complement", action="store_true", default=False)
    parser.add_argument("--disable_taef", action="store_true", default=False)
    parser.add_argument("--disable_gcti", action="store_true", default=False)
    parser.add_argument("--disable_view_consistency", action="store_true", default=False)
    parser.add_argument("--enabled_views", default="raw,stf,gaf", type=str)
    parser.add_argument(
        "--view_consistency_weight",
        default=0.05,
        type=float,
        help="Representation consistency coefficient used by GCTI.",
    )
    parser.add_argument("--view_noise_std", default=0.01, type=float, help="Noise std for the perturbed view.")
    parser.add_argument("--location_time_weight", default=0.35, type=float, help="Task-2 weight for time-domain features.")
    parser.add_argument("--location_stft_weight", default=2.0, type=float, help="Task-2 weight for STFT/frequency features.")
    parser.add_argument("--location_gaf_weight", default=1.0, type=float, help="Task-2 weight for GAF/spatial features.")
    parser.add_argument("--location_image_weight", default=1.0, type=float, help="Task-2 weight for location-image features.")
    parser.add_argument("--stft_n_fft", default=256, type=int)
    parser.add_argument("--stft_hop_length", default=128, type=int)
    parser.add_argument("--stft_win_length", default=256, type=int)
    parser.add_argument(
        "--train_sampler",
        default="event_distance_balanced",
        choices=["none", "event_balanced", "distance_balanced", "joint_balanced", "event_distance_balanced"],
    )
    parser.add_argument("--train_augment", dest="train_augment", action="store_true")
    parser.add_argument("--disable_train_augment", dest="train_augment", action="store_false")
    parser.set_defaults(train_augment=True)
    parser.add_argument("--location_aug_repeats", default=1, type=int)
    parser.add_argument("--location_aug_noise_std", default=0.002, type=float)
    parser.add_argument("--location_aug_gain_std", default=0.01, type=float)
    parser.add_argument("--location_aug_shift", default=24, type=int)
    parser.add_argument("--location_aug_mask_width", default=0, type=int)
    parser.add_argument("--location_aug_drop_rows", default=0, type=int)
    parser.add_argument(
        "--location_image_root",
        default=str(DEFAULT_LOCATION_IMAGE_ROOT),
        type=str,
        help="Optional auxiliary STFT image dataset root with train/val/test subfolders.",
    )
    parser.add_argument("--location_image_size", default=224, type=int, help="Image size for auxiliary STFT images.")
    parser.add_argument(
        "--location_image_aux_weight",
        default=0.0,
        type=float,
        help="Weight for auxiliary location-image supervision. <=0 disables it.",
    )
    parser.add_argument(
        "--location_image_aux_bs",
        default=16,
        type=int,
        help="Batch size for auxiliary location-image supervision.",
    )
    parser.add_argument(
        "--location_image_pretrained_path",
        default="",
        type=str,
        help="Optional pretrained checkpoint for the location-image branch.",
    )
    parser.add_argument(
        "--freeze_location_image_epochs",
        default=5,
        type=int,
        help="Freeze the pretrained location-image encoder/projection for the first N epochs. <=0 disables freezing.",
    )
    parser.add_argument("--loss_type", default="ce", choices=["ghm", "ce"], help="Classification loss type.")
    parser.add_argument("--ghm_bins", default=10, type=int, help="Number of bins used by GHM.")
    parser.add_argument("--ghm_momentum", default=0.75, type=float, help="Moving average momentum for GHM bins.")
    parser.add_argument(
        "--eval_level",
        default="file",
        choices=["row", "file"],
        help="Use row-level or file-level voting metrics on validation/test.",
    )
    parser.add_argument(
        "--vote_method",
        default="mean_logits",
        choices=["mean_logits", "majority_vote"],
        help="How to aggregate row predictions into file predictions.",
    )
    parser.add_argument(
        "--test_every_epoch",
        action="store_true",
        default=False,
        help="Evaluate the test split every epoch. Off by default to avoid test-set overuse.",
    )
    parser.add_argument(
        "--save_every_epoch",
        action="store_true",
        default=False,
        help="Save one checkpoint per epoch under epoch_checkpoints for later audit.",
    )
    parser.add_argument(
        "--selection_metric",
        default="mtl_score",
        choices=["mtl_score", "mean_task_score", "pareto_balanced", "old_score", "val_loss", "last"],
        help="Validation-only checkpoint selection rule.",
    )
    parser.add_argument("--max_train_batches", default=0, type=int, help="Debug only: limit train batches per epoch.")
    parser.add_argument("--max_eval_batches", default=0, type=int, help="Debug only: limit val/test batches per epoch.")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(gpu_id):
    if torch.cuda.is_available():
        return torch.device(f"cuda:{gpu_id}")
    return torch.device("cpu")


def to_device(batch_inputs, batch_labels, device):
    if torch.is_tensor(batch_inputs):
        batch_inputs = batch_inputs.to(device, non_blocking=True)
    elif isinstance(batch_inputs, dict):
        batch_inputs = {
            key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in batch_inputs.items()
        }
    else:
        raise TypeError(f"Unsupported batch input type: {type(batch_inputs)!r}")
    batch_labels = {key: value.to(device, non_blocking=True) for key, value in batch_labels.items()}
    return batch_inputs, batch_labels


def infer_batch_size(batch_inputs):
    if torch.is_tensor(batch_inputs):
        return int(batch_inputs.size(0))
    if isinstance(batch_inputs, dict):
        for value in batch_inputs.values():
            if torch.is_tensor(value):
                return int(value.size(0))
    raise ValueError("Unable to infer batch size from batch inputs.")


def accuracy_from_logits(logits, labels, ignore_index=None):
    if ignore_index is not None:
        valid_mask = labels != ignore_index
        if not valid_mask.any():
            return 0, 0
        logits = logits[valid_mask]
        labels = labels[valid_mask]
    predictions = torch.argmax(logits, dim=1)
    return (predictions == labels).sum().item(), labels.size(0)


def _task_balanced_backward(model, task_losses, aux_loss, method):
    """Backpropagate two task losses with optional Aligned-MTL or MoCo gradients."""
    if method == "equal":
        (task_losses[0] + task_losses[1] + aux_loss).backward()
        return

    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    task_grads = [
        torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
        for loss in task_losses
    ]
    shared_indices = [
        index
        for index, (event_grad, location_grad) in enumerate(zip(*task_grads))
        if event_grad is not None and location_grad is not None
    ]
    if not shared_indices:
        (task_losses[0] + task_losses[1] + aux_loss).backward()
        return

    event_vector = torch.cat([task_grads[0][index].reshape(-1) for index in shared_indices])
    location_vector = torch.cat([task_grads[1][index].reshape(-1) for index in shared_indices])

    if method == "aligned_mtl":
        gradients = torch.stack([event_vector, location_vector])
        gram = gradients @ gradients.t()
        eigenvalues, eigenvectors = torch.linalg.eigh(gram)
        tolerance = eigenvalues.max().clamp_min(1e-12) * max(gram.shape) * torch.finfo(gram.dtype).eps
        keep = eigenvalues > tolerance
        if keep.any():
            eigenvalues = eigenvalues[keep]
            eigenvectors = eigenvectors[:, keep]
            transform = eigenvalues.min().sqrt() * (
                eigenvectors @ torch.diag(eigenvalues.rsqrt()) @ eigenvectors.t()
            )
            weights = transform.sum(dim=0)
        else:
            weights = gradients.new_ones(2)
    elif method == "moco":
        state = getattr(model, "_moco_weighting_state", None)
        if state is None or state["y"].shape != (2, event_vector.numel()):
            state = {
                "step": 0,
                "y": event_vector.new_zeros((2, event_vector.numel())),
                "lambda": event_vector.new_full((2,), 0.5),
            }
            model._moco_weighting_state = state
        state["step"] += 1
        normalized = torch.stack(
            [
                event_vector / event_vector.norm().clamp_min(1e-8) * task_losses[0].detach(),
                location_vector / location_vector.norm().clamp_min(1e-8) * task_losses[1].detach(),
            ]
        )
        step = float(state["step"])
        state["y"].add_(-(0.5 / step**0.5) * (state["y"] - normalized))
        gram = state["y"] @ state["y"].t()
        state["lambda"] = torch.softmax(
            state["lambda"] - (0.1 / step**0.5) * gram @ state["lambda"], dim=0
        )
        weights = state["lambda"]
    else:
        raise KeyError(f"Unsupported task balancing method: {method}")

    for index, parameter in enumerate(parameters):
        event_grad, location_grad = task_grads[0][index], task_grads[1][index]
        if event_grad is None and location_grad is None:
            continue
        if event_grad is None:
            parameter.grad = weights[1] * location_grad
        elif location_grad is None:
            parameter.grad = weights[0] * event_grad
        else:
            parameter.grad = weights[0] * event_grad + weights[1] * location_grad
    if aux_loss.requires_grad:
        aux_loss.backward()


def build_loss(name, bins, momentum, ignore_index=None):
    if name == "ghm":
        return GHMCrossEntropyLoss(bins=bins, momentum=momentum, ignore_index=ignore_index)
    if name == "ce":
        return SafeCrossEntropyLoss(ignore_index=ignore_index)
    raise ValueError(f"Unsupported loss type: {name}")


class LocationImageAuxDataset(Dataset):
    def __init__(self, dataset_root, split, distance_to_idx, image_size):
        self.dataset_root = Path(dataset_root)
        self.split = split
        self.distance_to_idx = distance_to_idx
        self.image_size = int(image_size)
        self.samples = self._collect_samples()

    def _collect_samples(self):
        split_root = self.dataset_root / self.split
        samples = []
        if not split_root.is_dir():
            return samples
        for label_name, label_idx in self.distance_to_idx.items():
            class_dir = split_root / label_name
            if not class_dir.is_dir():
                continue
            for path in sorted(class_dir.glob("*.png")):
                samples.append((path, label_idx))
            for path in sorted(class_dir.glob("*.jpg")):
                samples.append((path, label_idx))
            for path in sorted(class_dir.glob("*.jpeg")):
                samples.append((path, label_idx))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
            image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
            image_np = np.asarray(image, dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)
        return image_tensor, torch.tensor(label, dtype=torch.long)


def build_location_image_loader(dataset_root, split, distance_classes, batch_size, num_workers, image_size):
    if not dataset_root:
        return None
    root = Path(dataset_root).expanduser().resolve()
    if not root.is_dir():
        return None
    dataset = LocationImageAuxDataset(
        dataset_root=root,
        split=split,
        distance_to_idx={label: idx for idx, label in enumerate(distance_classes)},
        image_size=image_size,
    )
    if len(dataset) == 0:
        return None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        drop_last=(split == "train" and len(dataset) > 1),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def _finalize_file_metrics(file_buffers, vote_method):
    if not file_buffers:
        return {
            "file_count": 0,
            "location_file_count": 0,
            "event_targets": [],
            "event_predictions": [],
            "event_probabilities": [],
            "location_targets": [],
            "location_predictions": [],
            "location_probabilities": [],
        }

    event_targets = []
    event_predictions = []
    event_probabilities = []
    location_targets = []
    location_predictions = []
    location_probabilities = []
    for bucket in file_buffers.values():
        event_logits = torch.stack(bucket["event_logits"])

        if vote_method == "mean_logits":
            event_vector = event_logits.mean(dim=0)
            event_pred = event_vector.argmax().item()
        else:
            event_votes = event_logits.argmax(dim=1)
            event_pred = torch.bincount(event_votes, minlength=event_logits.size(1)).argmax().item()
            event_vector = event_logits.mean(dim=0)

        event_targets.append(int(bucket["event_label"]))
        event_predictions.append(int(event_pred))
        event_probabilities.append(torch.softmax(event_vector, dim=0).cpu().tolist())

        if bucket["location_label"] != DISTANCE_IGNORE_INDEX:
            location_logits = torch.stack(bucket["location_logits"])
            if vote_method == "mean_logits":
                location_vector = location_logits.mean(dim=0)
                location_pred = location_vector.argmax().item()
            else:
                location_votes = location_logits.argmax(dim=1)
                location_pred = torch.bincount(location_votes, minlength=location_logits.size(1)).argmax().item()
                location_vector = location_logits.mean(dim=0)
            location_targets.append(int(bucket["location_label"]))
            location_predictions.append(int(location_pred))
            location_probabilities.append(torch.softmax(location_vector, dim=0).cpu().tolist())

    return {
        "file_count": len(file_buffers),
        "location_file_count": len(location_targets),
        "event_targets": event_targets,
        "event_predictions": event_predictions,
        "event_probabilities": event_probabilities,
        "location_targets": location_targets,
        "location_predictions": location_predictions,
        "location_probabilities": location_probabilities,
    }


def run_epoch(
    model,
    loader,
    device,
    split,
    event_criterion,
    location_criterion,
    event_classes=None,
    distance_classes=None,
    optimizer=None,
    event_loss_weight=1.0,
    location_loss_weight=1.0,
    max_grad_norm=0.0,
    aggregate_by_file=False,
    vote_method="mean_logits",
    collect_predictions=False,
    max_batches=0,
    task_balance="equal",
):
    is_train = optimizer is not None
    model.train(is_train)
    event_criterion.train(is_train)
    location_criterion.train(is_train)

    total_samples = 0
    total_loss = 0.0
    total_event_loss = 0.0
    total_location_loss = 0.0
    total_location_loss_samples = 0
    total_aux_loss = 0.0
    aux_loss_totals = defaultdict(float)
    row_event_correct = 0
    row_location_correct = 0
    row_location_total = 0
    file_buffers = defaultdict(lambda: {"event_logits": [], "location_logits": []})
    sample_offset = 0
    dataset_samples = getattr(loader.dataset, "samples", None)
    event_targets_all, event_predictions_all, event_probabilities_all = [], [], []
    location_targets_all, location_predictions_all, location_probabilities_all = [], [], []

    progress = tqdm(loader, desc=split, leave=False)
    for batch_index, (batch_inputs, batch_labels) in enumerate(progress, start=1):
        if max_batches and batch_index > max_batches:
            break
        batch_inputs, batch_labels = to_device(batch_inputs, batch_labels, device)
        event_targets = batch_labels["event_type"]
        location_targets = batch_labels["distance_cls"]

        with torch.set_grad_enabled(is_train):
            if isinstance(model, SensorFieldM3T):
                outputs = model(batch_inputs, task_mask=batch_labels.get("task_mask"))
            else:
                outputs = model(batch_inputs)
            event_logits = outputs["event_type"]
            location_logits = outputs.get("location", outputs["distance_cls"])

            event_loss = event_criterion(event_logits, event_targets)
            location_loss = location_criterion(location_logits, location_targets)
            aux_losses = outputs.get("aux_losses", {})
            aux_loss = event_logits.new_zeros(())
            for name, value in aux_losses.items():
                if value is None:
                    continue
                aux_loss = aux_loss + value
                aux_loss_totals[name] += float(value.detach().item()) * infer_batch_size(batch_inputs)
            loss = event_loss_weight * event_loss + location_loss_weight * location_loss + aux_loss

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                _task_balanced_backward(
                    model,
                    (event_loss_weight * event_loss, location_loss_weight * location_loss),
                    aux_loss,
                    task_balance,
                )
                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

        batch_size = infer_batch_size(batch_inputs)
        total_samples += batch_size
        total_loss += loss.item() * batch_size
        total_event_loss += event_loss.item() * batch_size
        total_aux_loss += float(aux_loss.detach().item()) * batch_size
        batch_event_correct, _ = accuracy_from_logits(event_logits, event_targets)
        batch_location_correct, batch_location_total = accuracy_from_logits(
            location_logits,
            location_targets,
            ignore_index=DISTANCE_IGNORE_INDEX,
        )
        total_location_loss += location_loss.item() * batch_location_total
        total_location_loss_samples += batch_location_total
        row_event_correct += batch_event_correct
        row_location_correct += batch_location_correct
        row_location_total += batch_location_total

        if collect_predictions:
            batch_event_pred = torch.argmax(event_logits, dim=1)
            batch_event_prob = torch.softmax(event_logits, dim=1)
            event_targets_all.extend(event_targets.detach().cpu().tolist())
            event_predictions_all.extend(batch_event_pred.detach().cpu().tolist())
            event_probabilities_all.extend(batch_event_prob.detach().cpu().tolist())
            valid_location = location_targets != DISTANCE_IGNORE_INDEX
            if valid_location.any():
                batch_location_pred = torch.argmax(location_logits, dim=1)
                batch_location_prob = torch.softmax(location_logits, dim=1)
                location_targets_all.extend(location_targets[valid_location].detach().cpu().tolist())
                location_predictions_all.extend(batch_location_pred[valid_location].detach().cpu().tolist())
                location_probabilities_all.extend(batch_location_prob[valid_location].detach().cpu().tolist())

        if aggregate_by_file:
            if dataset_samples is None:
                raise ValueError("Dataset does not expose samples metadata for file-level voting.")
            batch_meta = dataset_samples[sample_offset: sample_offset + batch_size]
            for idx, meta in enumerate(batch_meta):
                key = str(meta["path"])
                bucket = file_buffers[key]
                bucket["event_logits"].append(event_logits[idx].detach().cpu())
                bucket["event_label"] = int(meta["event_label"])
                bucket["location_label"] = int(meta["distance_label"])
                if int(meta["distance_label"]) != DISTANCE_IGNORE_INDEX:
                    bucket["location_logits"].append(location_logits[idx].detach().cpu())
            sample_offset += batch_size

        avg_loss = total_loss / max(total_samples, 1)
        avg_event_acc = row_event_correct / max(total_samples, 1)
        avg_location_acc = row_location_correct / max(row_location_total, 1)
        progress.set_postfix(
            loss=f"{avg_loss:.4f}",
            event_acc=f"{avg_event_acc:.4f}",
            location_acc=f"{avg_location_acc:.4f}",
            aux=f"{total_aux_loss / max(total_samples, 1):.4f}",
        )

    row_event_acc = row_event_correct / max(total_samples, 1)
    row_location_acc = row_location_correct / max(row_location_total, 1)
    row_score = (row_event_acc + row_location_acc) / 2.0

    selected_payload = {
        "event_targets": event_targets_all,
        "event_predictions": event_predictions_all,
        "event_probabilities": event_probabilities_all,
        "location_targets": location_targets_all,
        "location_predictions": location_predictions_all,
        "location_probabilities": location_probabilities_all,
    }

    if aggregate_by_file:
        file_metrics = _finalize_file_metrics(file_buffers, vote_method=vote_method)
        selected_payload = {
            "event_targets": file_metrics["event_targets"],
            "event_predictions": file_metrics["event_predictions"],
            "event_probabilities": file_metrics["event_probabilities"],
            "location_targets": file_metrics["location_targets"],
            "location_predictions": file_metrics["location_predictions"],
            "location_probabilities": file_metrics["location_probabilities"],
        }
        file_count = file_metrics["file_count"]
    else:
        file_count = 0

    metrics = {
        "split": split,
        "loss": total_loss / max(total_samples, 1),
        "event_loss": total_event_loss / max(total_samples, 1),
        "location_loss": total_location_loss / max(total_location_loss_samples, 1),
        "aux_loss": total_aux_loss / max(total_samples, 1),
        "row_event_acc": row_event_acc,
        "row_location_acc": row_location_acc,
        "row_score": row_score,
        "num_samples": total_samples,
        "file_count": file_count,
        "location_file_count": file_metrics["location_file_count"] if aggregate_by_file else 0,
        "metric_level": "file" if aggregate_by_file else "row",
    }
    for name, total_value in aux_loss_totals.items():
        metrics[name] = total_value / max(total_samples, 1)
    if collect_predictions:
        metrics.update(selected_payload)
        metrics = attach_task_metrics(
            metrics,
            event_classes=list(event_classes or [str(idx) for idx in range(max(event_logits.size(1), 1))]),
            distance_classes=list(distance_classes or [str(idx) for idx in range(max(location_logits.size(1), 1))]),
        )
    else:
        metrics.update(
            {
                "event_acc": row_event_acc,
                "location_acc": row_location_acc,
                "event_macro_f1": row_event_acc,
                "location_macro_f1": row_location_acc,
                "event_auc": 0.5,
                "location_auc": 0.5,
                "event_far": 0.0,
                "location_far": 0.0,
                "event_task_score": (row_event_acc + row_event_acc + 0.5 + 1.0) / 4.0,
                "location_task_score": (row_location_acc + row_location_acc + 0.5 + 1.0) / 4.0,
            }
        )
        metrics["mtl_score"] = (metrics["event_task_score"] + metrics["location_task_score"]) / 2.0
        metrics["score"] = (row_event_acc + row_location_acc) / 2.0
    return metrics


def run_location_image_aux_epoch(
    model,
    loader,
    device,
    criterion,
    optimizer,
    aux_weight=1.0,
    max_grad_norm=0.0,
):
    if loader is None or aux_weight <= 0:
        return {
            "location_image_aux_loss": 0.0,
            "location_image_aux_acc": 0.0,
            "location_image_aux_samples": 0,
        }

    model.train(True)
    criterion.train(True)
    total_loss = 0.0
    total_samples = 0
    total_correct = 0

    progress = tqdm(loader, desc="train_location_image_aux", leave=False)
    for images, labels in progress:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model.forward_location_image(images)
        base_loss = criterion(logits, labels)
        loss = aux_weight * base_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        batch_size = images.size(0)
        total_samples += batch_size
        total_loss += base_loss.item() * batch_size
        predictions = torch.argmax(logits, dim=1)
        total_correct += (predictions == labels).sum().item()
        progress.set_postfix(
            loss=f"{(total_loss / max(total_samples, 1)):.4f}",
            acc=f"{(total_correct / max(total_samples, 1)):.4f}",
        )

    return {
        "location_image_aux_loss": total_loss / max(total_samples, 1),
        "location_image_aux_acc": total_correct / max(total_samples, 1),
        "location_image_aux_samples": total_samples,
    }


def write_history_csv(path, rows):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            formatted = {}
            for key, value in row.items():
                if isinstance(value, float):
                    formatted[key] = f"{value:.10f}"
                else:
                    formatted[key] = value
            writer.writerow(formatted)


def write_history_xlsx(path, sheets):
    if Workbook is None:
        print("Skip writing XLSX history because openpyxl is not installed.")
        return
    workbook = Workbook()
    first = True
    for sheet_name, rows in sheets.items():
        worksheet = workbook.active if first else workbook.create_sheet(title=sheet_name)
        worksheet.title = sheet_name
        first = False
        if not rows:
            continue
        headers = list(rows[0].keys())
        worksheet.append(headers)
        for row in rows:
            worksheet.append([row.get(header) for header in headers])
    workbook.save(path)


def build_confusion_matrix(targets, predictions, num_classes):
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for target, prediction in zip(targets, predictions):
        matrix[int(target), int(prediction)] += 1
    return matrix


def build_classification_report(matrix, label_names):
    rows = []
    total = int(matrix.sum())
    correct = int(np.trace(matrix))
    for idx, label_name in enumerate(label_names):
        tp = float(matrix[idx, idx])
        fp = float(matrix[:, idx].sum() - tp)
        fn = float(matrix[idx, :].sum() - tp)
        support = int(matrix[idx, :].sum())
        precision = tp / max(tp + fp, 1.0)
        recall = tp / max(tp + fn, 1.0)
        f1 = 0.0 if precision + recall == 0 else (2 * precision * recall) / (precision + recall)
        rows.append(
            {
                "label": label_name,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
            }
        )
    accuracy = correct / max(total, 1)
    rows.extend(
        [
            {"label": "accuracy", "precision": accuracy, "recall": accuracy, "f1": accuracy, "support": total},
            {
                "label": "macro_avg",
                "precision": float(np.mean([row["precision"] for row in rows])) if rows else 0.0,
                "recall": float(np.mean([row["recall"] for row in rows])) if rows else 0.0,
                "f1": float(np.mean([row["f1"] for row in rows])) if rows else 0.0,
                "support": total,
            },
        ]
    )
    return rows


def save_confusion_artifacts(history_dir, prefix, metrics, event_classes, distance_classes):
    if "event_targets" not in metrics:
        return
    if metrics["event_targets"]:
        event_bundle = classification_metrics(
            metrics["event_targets"],
            metrics["event_predictions"],
            metrics.get("event_probabilities"),
            event_classes,
        )
        event_matrix = event_bundle.confusion
        write_history_csv(
            history_dir / f"{prefix}_event_confusion.csv",
            [{"label": event_classes[idx], **{event_classes[j]: int(event_matrix[idx, j]) for j in range(len(event_classes))}} for idx in range(len(event_classes))],
        )
        write_history_csv(history_dir / f"{prefix}_event_report.csv", metric_rows("event_type", event_bundle))
    if metrics["location_targets"]:
        location_bundle = classification_metrics(
            metrics["location_targets"],
            metrics["location_predictions"],
            metrics.get("location_probabilities"),
            distance_classes,
        )
        location_matrix = location_bundle.confusion
        write_history_csv(
            history_dir / f"{prefix}_location_confusion.csv",
            [{"label": distance_classes[idx], **{distance_classes[j]: int(location_matrix[idx, j]) for j in range(len(distance_classes))}} for idx in range(len(distance_classes))],
        )
        write_history_csv(
            history_dir / f"{prefix}_location_report.csv",
            metric_rows("distance_cls", location_bundle),
        )


def strip_prediction_payload(metrics):
    return {
        key: value
        for key, value in metrics.items()
        if key
        not in {
            "event_targets",
            "event_predictions",
            "event_probabilities",
            "location_targets",
            "location_predictions",
            "location_probabilities",
            "_event_metric_bundle",
            "_location_metric_bundle",
        }
    }


def plot_curves(history_dir, train_history, val_history, test_history):
    if plt is None:
        print("Skip plotting because matplotlib is not installed.")
        return
    if not train_history:
        return

    def plot_metric_group(filename, metrics):
        plt.figure(figsize=(12, 10))
        for idx, metric in enumerate(metrics, start=1):
            plt.subplot(len(metrics), 1, idx)
            for label, rows in [("train", train_history), ("val", val_history), ("test", test_history)]:
                if not rows:
                    continue
                filtered = [
                    (row["epoch"], row[metric])
                    for row in rows
                    if metric in row and isinstance(row[metric], (int, float)) and np.isfinite(row[metric])
                ]
                if not filtered:
                    continue
                epochs = [epoch for epoch, _ in filtered]
                values = [value for _, value in filtered]
                plt.plot(epochs, values, label=f"{label}_{metric}")
            plt.xlabel("Epoch")
            plt.ylabel(metric)
            plt.legend()
            plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(history_dir / filename, dpi=200)
        plt.close()

    plot_metric_group("loss_curves.png", ["loss", "event_loss", "location_loss"])
    plot_metric_group("acc_curves.png", ["event_acc", "location_acc", "score"])
    plot_metric_group("taskscore_curves.png", ["event_task_score", "location_task_score", "mtl_score"])


def save_histories(history_dir, train_history, val_history, test_history):
    history_dir.mkdir(parents=True, exist_ok=True)
    write_history_csv(history_dir / "train_history.csv", train_history)
    write_history_csv(history_dir / "val_history.csv", val_history)
    write_history_csv(history_dir / "test_history.csv", test_history)
    write_history_xlsx(
        history_dir / "training_history.xlsx",
        {
            "train": train_history,
            "val": val_history,
            "test": test_history,
        },
    )
    plot_curves(history_dir, train_history, val_history, test_history)


def save_json(path, payload):
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def checkpoint_selection_value(metrics, selection_metric):
    if selection_metric in {"mtl_score", "mean_task_score"}:
        return float(metrics.get("mtl_score", metrics.get("score", 0.0)))
    if selection_metric == "pareto_balanced":
        return min(float(metrics.get("event_task_score", 0.0)), float(metrics.get("location_task_score", 0.0)))
    if selection_metric == "old_score":
        return (
            float(metrics.get("row_event_acc", metrics.get("event_acc", 0.0)))
            + float(metrics.get("row_location_acc", metrics.get("location_acc", 0.0)))
        ) / 2.0
    if selection_metric == "val_loss":
        return -float(metrics.get("loss", 0.0))
    if selection_metric == "last":
        return float(metrics.get("epoch", 0))
    raise KeyError(f"Unsupported selection metric: {selection_metric}")


def main():
    args = parse_args()
    set_seed(args.seed)

    dataset_path = Path(args.dataset_path).expanduser().resolve()
    if not dataset_path.is_dir():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
    manifest_files = [dataset_path / name for name in ("train.csv", "val.csv", "test.csv")]
    if not all(path.is_file() for path in manifest_files):
        ensure_mtl43_manifests(dataset_path=dataset_path, seed=args.seed, overwrite=True)

    save_root = Path(args.save_path).expanduser().resolve()
    save_path = build_run_save_path(save_root)
    save_path.mkdir(parents=True, exist_ok=True)
    history_dir = save_path / "history"
    history_dir.mkdir(parents=True, exist_ok=True)
    args.save_path = str(save_path)
    canonical_dataset_path = save_path / "canonicalized_manifests"
    manifest_summary = canonicalize_dataset_manifests(dataset_path, canonical_dataset_path)
    audit_summary = audit_dataset_manifests(canonical_dataset_path, save_path / "manifest_audit.json")
    save_json(save_path / "canonicalization_summary.json", manifest_summary)
    dataset_path = canonical_dataset_path

    device = resolve_device(args.gpu_id)
    event_classes = parse_label_list(args.event_classes)
    distance_classes = parse_label_list(args.distance_classes)
    eval_by_file = args.eval_level == "file"

    print(f"Dataset Path: {dataset_path}")
    print(f"Model: {args.model}")
    print(f"Save Path: {save_path}")
    print(f"Device: {device}")
    print(f"Batch Size: {args.bs}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning Rate: {args.lr}")
    print(f"Weight Decay: {args.weight_decay}")
    print(f"Step Size: {args.step_size}")
    print(f"Gamma: {args.gamma}")
    print(f"Loss Type: {args.loss_type}")
    print(f"Train Sampler: {args.train_sampler}")
    print(f"Train Augment: {args.train_augment}")
    print(
        "Location Augmentation: "
        f"repeats={args.location_aug_repeats}, "
        f"noise_std={args.location_aug_noise_std}, "
        f"gain_std={args.location_aug_gain_std}, "
        f"shift={args.location_aug_shift}, "
        f"mask_width={args.location_aug_mask_width}, "
        f"drop_rows={args.location_aug_drop_rows}"
    )
    print(f"Location Image Root: {args.location_image_root}")
    print(f"Location Image Aux Weight: {args.location_image_aux_weight}")
    print(f"Location Image Size: {args.location_image_size}")
    print(f"Location Image Pretrained Path: {args.location_image_pretrained_path or 'None'}")
    print(f"Freeze Location Image Epochs: {args.freeze_location_image_epochs}")
    print(f"Eval Level: {args.eval_level}")
    print(f"Vote Method: {args.vote_method}")
    print(f"Selection Metric: {args.selection_metric}")
    print(f"Canonicalized Dataset Path: {dataset_path}")
    print(f"Leakage Audit Passed: {audit_summary['leakage_passed']}")
    print(f"Input Shape: (1, {args.input_height}, {args.input_width})")
    if args.model == "pipemmtl":
        print(
            "Input Interpretation: PipeMMTL recomputes time statistics, STFT, and GAF from the raw signal channel, "
            "and the location branch also uses a merged STFT image representation with optional auxiliary image supervision."
        )
    elif args.model == "sensorfield_m3t":
        print(
            "Input Interpretation: SensorField-M3T accepts existing raw/stft/gaf keys when available, "
            "and otherwise derives STF and GAF views from the existing raw signal tensor without changing dataset format."
        )
    else:
        print("Input Interpretation: condition baseline consumes Raw, STF, and GAF views from the shared MTL43 loader.")
    print(
        f"TPAMI Multiview: return_multiview={args.return_multiview or args.model != 'pipemmtl'}, "
        f"spatial_adapter={args.spatial_adapter}, stf_size={args.stf_size}, gaf_size={args.gaf_size}"
    )
    if args.model == "pipemmtl":
        print(
            f"PipeMMTL Config: embed_dim={args.embed_dim}, fusion_dim={args.fusion_dim}, "
            f"time_tokens={args.time_tokens}, freq_tokens={args.freq_tokens}, gaf_tokens={args.gaf_tokens}, gaf_size={args.gaf_size}"
        )
    elif args.model == "sensorfield_m3t":
        print(
            f"SensorField-M3T Config: hidden_dim={args.hidden_dim}, num_anchors={args.num_anchors}, "
            f"enabled_views={args.enabled_views}, fac_loss_weight={args.fac_loss_weight}, "
            f"taef_loss_weight={args.taef_loss_weight}, gcti_loss_weight={args.gcti_loss_weight}, "
            f"view_drop_prob={args.view_drop_prob}, disable_fac={args.disable_fac}, "
            f"disable_complement={args.disable_complement}, disable_taef={args.disable_taef}, "
            f"disable_gcti={args.disable_gcti}, disable_view_consistency={args.disable_view_consistency}"
        )
    else:
        print(
            f"Condition Baseline Config: hidden_dim={args.hidden_dim}, pretrained={args.baseline_pretrained}, "
            f"task_balance={args.task_balance}"
        )
    print(
        "Location Modality Weights: "
        f"time={args.location_time_weight}, "
        f"stft={args.location_stft_weight}, "
        f"gaf={args.location_gaf_weight}, "
        f"image={args.location_image_weight}"
    )
    print(
        f"STFT Config: n_fft={args.stft_n_fft}, hop_length={args.stft_hop_length}, "
        f"win_length={args.stft_win_length}, spatial_fusion={args.stf_spatial_fusion}"
    )
    print(f"Event Classes: {event_classes}")
    print(f"Location Classes: {distance_classes}")

    data_loader, _ = csv_dataloader(
        dataset_path=dataset_path,
        batch_size=args.bs,
        event_classes=event_classes,
        distance_classes=distance_classes,
        input_height=args.input_height,
        input_width=args.input_width,
        sample_level="manifest",
        normalize=args.normalize,
        num_workers=args.num_workers,
        augment=args.train_augment,
        location_aug_repeats=args.location_aug_repeats,
        location_aug_noise_std=args.location_aug_noise_std,
        location_aug_gain_std=args.location_aug_gain_std,
        location_aug_shift=args.location_aug_shift,
        location_aug_mask_width=args.location_aug_mask_width,
        location_aug_drop_rows=args.location_aug_drop_rows,
        return_multiview=args.return_multiview or args.model != "pipemmtl",
        spatial_adapter=args.spatial_adapter,
        stf_spatial_fusion=args.stf_spatial_fusion,
        stf_size=args.stf_size,
        gaf_size=args.gaf_size,
        train_sampler=args.train_sampler,
        use_stft_aux=False,
    )

    if args.model == "pipemmtl":
        model = PipeMMTL(
            num_event_classes=len(event_classes),
            num_location_classes=len(distance_classes),
            input_length=args.input_height * args.input_width,
            input_rows=args.input_height,
            embed_dim=args.embed_dim,
            fusion_dim=args.fusion_dim,
            time_tokens=args.time_tokens,
            freq_tokens=args.freq_tokens,
            gaf_tokens=args.gaf_tokens,
            gaf_size=args.gaf_size,
            prior_tokens=args.prior_tokens,
            num_heads=args.num_heads,
            dropout=args.dropout,
            stft_n_fft=args.stft_n_fft,
            stft_hop_length=args.stft_hop_length,
            stft_win_length=args.stft_win_length,
            location_image_size=args.location_image_size,
            location_time_weight=args.location_time_weight,
            location_stft_weight=args.location_stft_weight,
            location_gaf_weight=args.location_gaf_weight,
            location_image_weight=args.location_image_weight,
        ).to(device)
        load_location_image_pretrained(model, args.location_image_pretrained_path, device)
    elif args.model == "sensorfield_m3t":
        model = SensorFieldM3T(
            task_output_dims={
                "event_type": len(event_classes),
                "distance_cls": len(distance_classes),
            },
            hidden_dim=args.hidden_dim,
            num_anchors=args.num_anchors,
            num_heads=args.num_heads,
            stf_size=args.stf_size,
            gaf_size=args.gaf_size,
            stft_n_fft=args.stft_n_fft,
            stft_hop_length=args.stft_hop_length,
            stft_win_length=args.stft_win_length,
            fac_loss_weight=args.fac_loss_weight,
            taef_loss_weight=args.taef_loss_weight,
            gcti_loss_weight=args.gcti_loss_weight,
            view_drop_prob=args.view_drop_prob,
            enable_view_consistency=args.enable_view_consistency,
            disable_fac=args.disable_fac,
            disable_complement=args.disable_complement,
            disable_taef=args.disable_taef,
            disable_gcti=args.disable_gcti,
            disable_view_consistency=args.disable_view_consistency,
            enabled_views=args.enabled_views,
            view_consistency_weight=args.view_consistency_weight,
            view_noise_std=args.view_noise_std,
            dropout=args.dropout,
        ).to(device)
    else:
        model = build_condition_baseline(
            args.model,
            num_event_classes=len(event_classes),
            num_location_classes=len(distance_classes),
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            image_size=args.gaf_size,
            pretrained=args.baseline_pretrained,
        ).to(device)

    event_criterion = build_loss(args.loss_type, args.ghm_bins, args.ghm_momentum, ignore_index=None).to(device)
    location_criterion = build_loss(
        args.loss_type,
        args.ghm_bins,
        args.ghm_momentum,
        ignore_index=DISTANCE_IGNORE_INDEX,
    ).to(device)
    location_image_aux_criterion = build_loss(args.loss_type, args.ghm_bins, args.ghm_momentum, ignore_index=None).to(device)

    if args.model == "pipemmtl":
        location_image_loader = build_location_image_loader(
            dataset_root=args.location_image_root,
            split="train",
            distance_classes=distance_classes,
            batch_size=args.location_image_aux_bs,
            num_workers=args.num_workers,
            image_size=args.location_image_size,
        )
        if location_image_loader is not None:
            print(f"Auxiliary location-image samples: {len(location_image_loader.dataset)}")
        else:
            print("Auxiliary location-image supervision: disabled")
    else:
        location_image_loader = None
        print(f"Auxiliary location-image supervision: disabled for {args.model}")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    best_score = float("-inf")
    best_epoch = 0
    best_val_metrics = None
    patience_counter = 0
    train_history, val_history, test_history = [], [], []

    save_json(save_path / "run_config.json", vars(args))

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        freeze_image_branch = (
            args.model == "pipemmtl"
            and args.freeze_location_image_epochs > 0
            and epoch <= args.freeze_location_image_epochs
        )
        if args.model == "pipemmtl":
            set_trainable(model.location_image_encoder, not freeze_image_branch)
            set_trainable(model.location_image_proj, not freeze_image_branch)
        if freeze_image_branch:
            print("Location image encoder/projection frozen for warm-up.")

        train_metrics = run_epoch(
            model=model,
            loader=data_loader["train"],
            device=device,
            split="train",
            event_criterion=event_criterion,
            location_criterion=location_criterion,
            event_classes=event_classes,
            distance_classes=distance_classes,
            optimizer=optimizer,
            event_loss_weight=args.event_loss_weight,
            location_loss_weight=args.location_loss_weight,
            max_grad_norm=args.max_grad_norm,
            aggregate_by_file=False,
            vote_method=args.vote_method,
            max_batches=args.max_train_batches,
            task_balance=args.task_balance,
        )
        aux_metrics = run_location_image_aux_epoch(
            model=model,
            loader=location_image_loader,
            device=device,
            criterion=location_image_aux_criterion,
            optimizer=optimizer,
            aux_weight=args.location_image_aux_weight,
            max_grad_norm=args.max_grad_norm,
        )
        train_metrics.update(aux_metrics)
        val_metrics = run_epoch(
            model=model,
            loader=data_loader["val"],
            device=device,
            split="val",
            event_criterion=event_criterion,
            location_criterion=location_criterion,
            event_classes=event_classes,
            distance_classes=distance_classes,
            optimizer=None,
            event_loss_weight=args.event_loss_weight,
            location_loss_weight=args.location_loss_weight,
            aggregate_by_file=eval_by_file,
            vote_method=args.vote_method,
            collect_predictions=True,
            max_batches=args.max_eval_batches,
        )

        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        for metric_block in (train_metrics, val_metrics):
            metric_block["epoch"] = epoch
            metric_block["lr"] = current_lr

        train_history.append(strip_prediction_payload(train_metrics))
        val_history.append(strip_prediction_payload(val_metrics))
        save_histories(history_dir, train_history, val_history, test_history)

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "train_metrics": strip_prediction_payload(train_metrics.copy()),
                "val_metrics": strip_prediction_payload(val_metrics.copy()),
                "args": vars(args),
            },
            save_path / "last.pt",
        )
        if args.save_every_epoch:
            epoch_ckpt_dir = save_path / "epoch_checkpoints"
            epoch_ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "train_metrics": strip_prediction_payload(train_metrics.copy()),
                    "val_metrics": strip_prediction_payload(val_metrics.copy()),
                    "args": vars(args),
                },
                epoch_ckpt_dir / f"epoch_{epoch:03d}.pt",
            )

        print(
            "train loss={train_loss:.4f} row_event_acc={train_event_acc:.4f} row_location_acc={train_location_acc:.4f} | "
            "aux_image_loss={aux_image_loss:.4f} aux_image_acc={aux_image_acc:.4f} | "
            "val loss={val_loss:.4f} {val_level}_event_acc={val_event_acc:.4f} {val_level}_location_acc={val_location_acc:.4f} "
            "event_score={event_score:.4f} location_score={location_score:.4f} mtl_score={val_score:.4f}".format(
                train_loss=train_metrics["loss"],
                train_event_acc=train_metrics["row_event_acc"],
                train_location_acc=train_metrics["row_location_acc"],
                aux_image_loss=train_metrics["location_image_aux_loss"],
                aux_image_acc=train_metrics["location_image_aux_acc"],
                val_loss=val_metrics["loss"],
                val_level=val_metrics["metric_level"],
                val_event_acc=val_metrics["event_acc"],
                val_location_acc=val_metrics["location_acc"],
                event_score=val_metrics["event_task_score"],
                location_score=val_metrics["location_task_score"],
                val_score=val_metrics["mtl_score"],
            )
        )

        current_selection_score = checkpoint_selection_value(val_metrics, args.selection_metric)
        if args.selection_metric == "last" or current_selection_score > best_score + args.early_stop_min_delta:
            best_score = current_selection_score
            best_epoch = epoch
            best_val_metrics = strip_prediction_payload(val_metrics.copy())
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "train_metrics": strip_prediction_payload(train_metrics.copy()),
                    "val_metrics": strip_prediction_payload(val_metrics.copy()),
                    "args": vars(args),
                },
                save_path / "best.pt",
            )
            save_confusion_artifacts(history_dir, "best_val", val_metrics, event_classes, distance_classes)
            print(
                f"Best checkpoint updated at epoch {epoch} with "
                f"validation {args.selection_metric}={best_score:.4f}."
            )
        else:
            patience_counter += 1
            print(f"No validation improvement. Early-stop counter: {patience_counter}")

        if args.test_every_epoch:
            interim_test = run_epoch(
                model=model,
                loader=data_loader["test"],
                device=device,
                split="test",
                event_criterion=event_criterion,
                location_criterion=location_criterion,
                event_classes=event_classes,
                distance_classes=distance_classes,
                optimizer=None,
                event_loss_weight=args.event_loss_weight,
                location_loss_weight=args.location_loss_weight,
                aggregate_by_file=eval_by_file,
                vote_method=args.vote_method,
                collect_predictions=True,
                max_batches=args.max_eval_batches,
            )
            interim_test["epoch"] = epoch
            interim_test["lr"] = current_lr
            test_history.append(strip_prediction_payload(interim_test))
            save_histories(history_dir, train_history, val_history, test_history)
            print(
                "test {level}_event_acc={event_acc:.4f} {level}_location_acc={location_acc:.4f} score={score:.4f}".format(
                    level=interim_test["metric_level"],
                    event_acc=interim_test["event_acc"],
                    location_acc=interim_test["location_acc"],
                    score=interim_test["score"],
                )
            )

        if args.early_stop_patience > 0 and patience_counter >= args.early_stop_patience:
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    best_test_metrics = None
    best_ckpt_path = save_path / "best.pt"
    if best_ckpt_path.is_file():
        checkpoint = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        best_test_metrics = run_epoch(
            model=model,
            loader=data_loader["test"],
            device=device,
            split="best_test",
            event_criterion=event_criterion,
            location_criterion=location_criterion,
            event_classes=event_classes,
            distance_classes=distance_classes,
            optimizer=None,
            event_loss_weight=args.event_loss_weight,
            location_loss_weight=args.location_loss_weight,
            aggregate_by_file=eval_by_file,
            vote_method=args.vote_method,
            collect_predictions=True,
            max_batches=args.max_eval_batches,
        )
        best_test_metrics["epoch"] = best_epoch
        best_test_metrics["lr"] = 0.0
        save_confusion_artifacts(history_dir, "best_test", best_test_metrics, event_classes, distance_classes)
        test_history = [strip_prediction_payload(best_test_metrics)]
        save_histories(history_dir, train_history, val_history, test_history)

    summary = {
        "best_epoch": best_epoch,
        "best_val_score": best_score,
        "selection_metric": args.selection_metric,
        "best_val_metrics": best_val_metrics,
        "best_test_metrics": strip_prediction_payload(best_test_metrics) if best_test_metrics is not None else None,
        "device": str(device),
    }
    save_json(save_path / "summary.json", summary)

    print("\nTraining complete.")
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation {args.selection_metric}: {best_score:.4f}")
    if best_test_metrics is not None:
        print(
            "Best test metrics: "
            f"loss={best_test_metrics['loss']:.4f}, "
            f"{best_test_metrics['metric_level']}_event_acc={best_test_metrics['event_acc']:.4f}, "
            f"{best_test_metrics['metric_level']}_location_acc={best_test_metrics['location_acc']:.4f}, "
            f"mtl_score={best_test_metrics['mtl_score']:.4f}"
        )


if __name__ == "__main__":
    main()
