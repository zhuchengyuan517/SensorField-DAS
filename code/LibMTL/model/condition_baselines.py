from __future__ import annotations

from typing import Any
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
except Exception:  # pragma: no cover - optional dependency
    timm = None


VIEW_NAMES = ("raw", "stf", "gaf")


def _validate_views(inputs: dict[str, torch.Tensor]) -> None:
    missing = [name for name in VIEW_NAMES if name not in inputs]
    if missing:
        raise KeyError(f"Missing required views: {missing}")
    if inputs["raw"].ndim != 3:
        raise ValueError(f"raw must have shape [B, 1, T], got {tuple(inputs['raw'].shape)}")
    for name in ("stf", "gaf"):
        if inputs[name].ndim != 4:
            raise ValueError(f"{name} must have shape [B, 1, H, W], got {tuple(inputs[name].shape)}")


class RawViewEncoder(nn.Module):
    """Lightweight temporal encoder: [B, 1, 10000] -> [B, D]."""

    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        widths = (32, 64, 96, hidden_dim)
        layers: list[nn.Module] = []
        in_channels = 1
        for width in widths:
            layers.extend(
                [
                    nn.Conv1d(in_channels, width, kernel_size=9, stride=3, padding=4, bias=False),
                    nn.BatchNorm1d(width),
                    nn.GELU(),
                ]
            )
            in_channels = width
        self.net = nn.Sequential(*layers)
        self.proj = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(dropout))

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        return self.proj(self.net(raw.float()))


class MapViewEncoder(nn.Module):
    """Lightweight image encoder: [B, 1, 224, 224] -> [B, D]."""

    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        widths = (32, 64, 96, hidden_dim)
        layers: list[nn.Module] = []
        in_channels = 1
        for width in widths:
            layers.extend(
                [
                    nn.Conv2d(in_channels, width, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(width),
                    nn.GELU(),
                ]
            )
            in_channels = width
        self.net = nn.Sequential(*layers)
        self.proj = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout))

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        return self.proj(self.net(feature_map.float()))


class ThreeViewEncoder(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.raw = RawViewEncoder(hidden_dim, dropout)
        self.stf = MapViewEncoder(hidden_dim, dropout)
        self.gaf = MapViewEncoder(hidden_dim, dropout)

    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        _validate_views(inputs)
        return {name: getattr(self, name)(inputs[name]) for name in VIEW_NAMES}


class TaskHeads(nn.Module):
    def __init__(self, hidden_dim: int, num_event_classes: int, num_location_classes: int, dropout: float) -> None:
        super().__init__()
        self.event_tower = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)
        )
        self.location_tower = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)
        )
        self.event_head = nn.Linear(hidden_dim, num_event_classes)
        self.location_head = nn.Linear(hidden_dim, num_location_classes)

    def forward(self, event_feature: torch.Tensor, location_feature: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        location_feature = event_feature if location_feature is None else location_feature
        event_logits = self.event_head(self.event_tower(event_feature))
        location_logits = self.location_head(self.location_tower(location_feature))
        return {"event_type": event_logits, "location": location_logits, "distance_cls": location_logits}


class SharedThreeViewMTL(nn.Module):
    """Shared multi-view backbone used with EW, Aligned-MTL, or MoCo task weighting."""

    def __init__(self, num_event_classes: int, num_location_classes: int, hidden_dim: int = 128, dropout: float = 0.2, **_: Any) -> None:
        super().__init__()
        self.views = ThreeViewEncoder(hidden_dim, dropout)
        self.fusion = nn.Sequential(
            nn.LayerNorm(hidden_dim * 3),
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.heads = TaskHeads(hidden_dim, num_event_classes, num_location_classes, dropout)

    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        features = self.views(inputs)
        fused = self.fusion(torch.cat([features[name] for name in VIEW_NAMES], dim=-1))
        return self.heads(fused)


class MultiModNConditionBaseline(nn.Module):
    """Sequential modality-conditioned state updates over Raw, STF, and GAF."""

    def __init__(self, num_event_classes: int, num_location_classes: int, hidden_dim: int = 128, dropout: float = 0.2, **_: Any) -> None:
        super().__init__()
        self.views = ThreeViewEncoder(hidden_dim, dropout)
        self.updates = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.LayerNorm(hidden_dim * 2),
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for name in VIEW_NAMES
            }
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.heads = TaskHeads(hidden_dim, num_event_classes, num_location_classes, dropout)

    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        features = self.views(inputs)
        state = torch.zeros_like(features["raw"])
        for name in VIEW_NAMES:
            state = self.norm(state + self.updates[name](torch.cat([state, features[name]], dim=-1)))
        return self.heads(state)


class M4oEConditionBaseline(nn.Module):
    """Task-specific mixture-of-experts routing over three view experts."""

    def __init__(self, num_event_classes: int, num_location_classes: int, hidden_dim: int = 128, dropout: float = 0.2, **_: Any) -> None:
        super().__init__()
        self.views = ThreeViewEncoder(hidden_dim, dropout)
        self.experts = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.LayerNorm(hidden_dim),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for name in VIEW_NAMES
            }
        )
        self.event_router = nn.Sequential(nn.LayerNorm(hidden_dim * 3), nn.Linear(hidden_dim * 3, len(VIEW_NAMES)))
        self.location_router = nn.Sequential(nn.LayerNorm(hidden_dim * 3), nn.Linear(hidden_dim * 3, len(VIEW_NAMES)))
        self.heads = TaskHeads(hidden_dim, num_event_classes, num_location_classes, dropout)

    def _route(self, expert_features: torch.Tensor, context: torch.Tensor, router: nn.Module) -> torch.Tensor:
        weights = torch.softmax(router(context), dim=-1).unsqueeze(-1)
        return (weights * expert_features).sum(dim=1)

    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        features = self.views(inputs)
        context = torch.cat([features[name] for name in VIEW_NAMES], dim=-1)
        experts = torch.stack([self.experts[name](features[name]) for name in VIEW_NAMES], dim=1)
        return self.heads(
            self._route(experts, context, self.event_router),
            self._route(experts, context, self.location_router),
        )


class DASMAEConditionBaseline(nn.Module):
    """Masked raw-signal autoencoding with STF/GAF downstream fusion."""

    def __init__(
        self,
        num_event_classes: int,
        num_location_classes: int,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        patch_size: int = 100,
        mask_ratio: float = 0.5,
        mae_loss_weight: float = 0.1,
        **_: Any,
    ) -> None:
        super().__init__()
        self.patch_size = int(patch_size)
        self.mask_ratio = float(mask_ratio)
        self.mae_loss_weight = float(mae_loss_weight)
        self.patch_embed = nn.Linear(self.patch_size, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.raw_encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.reconstruction = nn.Linear(hidden_dim, self.patch_size)
        self.stf_encoder = MapViewEncoder(hidden_dim, dropout)
        self.gaf_encoder = MapViewEncoder(hidden_dim, dropout)
        self.fusion = nn.Sequential(nn.LayerNorm(hidden_dim * 3), nn.Linear(hidden_dim * 3, hidden_dim), nn.GELU())
        self.heads = TaskHeads(hidden_dim, num_event_classes, num_location_classes, dropout)

    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, Any]:
        _validate_views(inputs)
        raw = inputs["raw"].float()
        usable = (raw.size(-1) // self.patch_size) * self.patch_size
        patches = raw[..., :usable].unfold(-1, self.patch_size, self.patch_size).squeeze(1)
        mask = torch.rand(patches.shape[:2], device=patches.device) < self.mask_ratio if self.training else torch.zeros(
            patches.shape[:2], device=patches.device, dtype=torch.bool
        )
        encoded = self.raw_encoder(self.patch_embed(patches.masked_fill(mask.unsqueeze(-1), 0.0)))
        raw_feature = encoded.mean(dim=1)
        fused = self.fusion(
            torch.cat([raw_feature, self.stf_encoder(inputs["stf"]), self.gaf_encoder(inputs["gaf"])], dim=-1)
        )
        outputs: dict[str, Any] = self.heads(fused)
        if mask.any():
            error = (self.reconstruction(encoded) - patches).pow(2).mean(dim=-1)
            outputs["aux_losses"] = {"dasmae_reconstruction": self.mae_loss_weight * error[mask].mean()}
        return outputs


class VisualThreeViewBaseline(nn.Module):
    """ConvNeXt/Swin baseline over a three-channel Raw-STF-GAF image."""

    def __init__(
        self,
        num_event_classes: int,
        num_location_classes: int,
        backbone_name: str,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        image_size: int = 224,
        pretrained: bool = False,
        **_: Any,
    ) -> None:
        super().__init__()
        if timm is None:
            raise ImportError("timm is required for ConvNeXt and PipelineADWinT condition baselines.")
        self.image_size = int(image_size)
        try:
            self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0, global_pool="avg")
        except Exception as exc:  # pragma: no cover - cache/network dependent
            warnings.warn(f"Falling back to random initialization for {backbone_name}: {exc}")
            self.backbone = timm.create_model(backbone_name, pretrained=False, num_classes=0, global_pool="avg")
        feature_dim = int(getattr(self.backbone, "num_features"))
        self.proj = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, hidden_dim), nn.GELU())
        self.heads = TaskHeads(hidden_dim, num_event_classes, num_location_classes, dropout)

    def _stack_views(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        _validate_views(inputs)
        raw_map = F.interpolate(
            inputs["raw"].unsqueeze(2), size=(self.image_size, self.image_size), mode="bilinear", align_corners=False
        )
        maps = [raw_map]
        for name in ("stf", "gaf"):
            view = inputs[name]
            if view.shape[-2:] != (self.image_size, self.image_size):
                view = F.interpolate(view, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
            maps.append(view)
        return torch.cat(maps, dim=1).float()

    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        feature = self.backbone(self._stack_views(inputs))
        if feature.ndim > 2:
            feature = feature.flatten(1)
        return self.heads(self.proj(feature))


def build_condition_baseline(
    model_name: str,
    *,
    num_event_classes: int,
    num_location_classes: int,
    hidden_dim: int = 128,
    dropout: float = 0.2,
    image_size: int = 224,
    pretrained: bool = False,
) -> nn.Module:
    common = {
        "num_event_classes": num_event_classes,
        "num_location_classes": num_location_classes,
        "hidden_dim": hidden_dim,
        "dropout": dropout,
    }
    if model_name == "convnext_small":
        return VisualThreeViewBaseline(
            **common,
            backbone_name="convnext_small.fb_in22k_ft_in1k",
            image_size=image_size,
            pretrained=pretrained,
        )
    if model_name == "pipelineadwint":
        return VisualThreeViewBaseline(
            **common,
            backbone_name="swin_tiny_patch4_window7_224.ms_in1k",
            image_size=image_size,
            pretrained=pretrained,
        )
    if model_name == "multimodn":
        return MultiModNConditionBaseline(**common)
    if model_name == "m4oe":
        return M4oEConditionBaseline(**common)
    if model_name == "das_mae":
        return DASMAEConditionBaseline(**common)
    if model_name in {"aligned_mtl", "moco_mtl"}:
        return SharedThreeViewMTL(**common)
    raise KeyError(f"Unsupported condition baseline: {model_name}")


__all__ = [
    "DASMAEConditionBaseline",
    "M4oEConditionBaseline",
    "MultiModNConditionBaseline",
    "SharedThreeViewMTL",
    "VisualThreeViewBaseline",
    "build_condition_baseline",
]
