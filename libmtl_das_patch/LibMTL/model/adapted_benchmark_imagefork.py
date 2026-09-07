from __future__ import annotations

from typing import Any
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
except Exception:  # pragma: no cover - optional dependency at runtime
    timm = None

from LibMTL.model.pipemmtl import GAFEncoder, STFTEncoder, TimeStatisticsEncoder
from LibMTL.model.sensorfield_m3t_imagefork import build_location_image_encoder


def _extract_csv_signal(csv_inputs: torch.Tensor | None) -> torch.Tensor | None:
    if csv_inputs is None:
        return None
    if csv_inputs.ndim == 4:
        return csv_inputs[:, 0, :, :].float()
    if csv_inputs.ndim == 3:
        return csv_inputs.float()
    raise ValueError(f"Unsupported csv input shape: {tuple(csv_inputs.shape)}")


def _normalize_map(feature_map: torch.Tensor) -> torch.Tensor:
    reduce_dims = tuple(range(2, feature_map.ndim))
    min_value = feature_map.amin(dim=reduce_dims, keepdim=True)
    max_value = feature_map.amax(dim=reduce_dims, keepdim=True)
    return (feature_map - min_value) / (max_value - min_value + 1e-6)


class SignalFeatureExtractor(nn.Module):
    def __init__(
        self,
        input_rows: int,
        feature_dim: int,
        dropout: float,
        stft_n_fft: int,
        stft_hop_length: int,
        stft_win_length: int,
        gaf_size: int,
        image_size: int,
    ) -> None:
        super().__init__()
        self.input_rows = int(input_rows)
        self.image_size = int(image_size)
        self.time_encoder = TimeStatisticsEncoder(input_rows=input_rows, embed_dim=feature_dim, dropout=dropout)
        self.stft_encoder = STFTEncoder(
            input_rows=input_rows,
            embed_dim=feature_dim,
            tokens_per_row=8,
            n_fft=stft_n_fft,
            hop_length=stft_hop_length,
            win_length=stft_win_length,
            dropout=dropout,
        )
        self.gaf_encoder = GAFEncoder(
            input_rows=input_rows,
            embed_dim=feature_dim,
            tokens_per_row=8,
            gaf_size=gaf_size,
            dropout=dropout,
        )
        self.raw_proj = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.stf_proj = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.gaf_proj = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def _to_pseudo_image(self, signal: torch.Tensor, stf_maps: torch.Tensor, gaf_maps: torch.Tensor) -> torch.Tensor:
        raw_map = F.interpolate(signal.unsqueeze(1), size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        stf_map = stf_maps.mean(dim=(1, 2), keepdim=False).unsqueeze(1)
        stf_map = F.interpolate(stf_map, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        gaf_map = gaf_maps.mean(dim=(1, 2), keepdim=False).unsqueeze(1)
        gaf_map = F.interpolate(gaf_map, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        return torch.cat(
            [
                _normalize_map(raw_map),
                _normalize_map(stf_map),
                _normalize_map(gaf_map),
            ],
            dim=1,
        )

    def forward(self, signal: torch.Tensor) -> dict[str, torch.Tensor]:
        device_type = signal.device.type if signal.device.type in {"cuda", "cpu"} else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            signal = signal.float()
            raw_tokens, _ = self.time_encoder(signal)
            stf_tokens, stf_maps = self.stft_encoder(signal)
            gaf_tokens, gaf_maps = self.gaf_encoder(signal)
        return {
            "raw": self.raw_proj(raw_tokens.mean(dim=1)),
            "stf": self.stf_proj(stf_tokens.mean(dim=1)),
            "gaf": self.gaf_proj(gaf_tokens.mean(dim=1)),
            "pseudo_image": self._to_pseudo_image(signal, stf_maps, gaf_maps),
        }


class HybridImageHead(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        num_event_classes: int,
        num_location_classes: int,
        backbone_name: str,
        pretrained: bool,
        dropout: float,
    ) -> None:
        super().__init__()
        self.encoder = build_location_image_encoder(
            backbone_name=backbone_name,
            output_dim=feature_dim,
            dropout=dropout,
            pretrained=pretrained,
        )
        self.event_tower = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.location_tower = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.event_head = nn.Linear(feature_dim, num_event_classes)
        self.location_head = nn.Linear(feature_dim, num_location_classes)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feature = self.encoder(images.float())
        event_logits = self.event_head(self.event_tower(feature))
        location_logits = self.location_head(self.location_tower(feature))
        return feature, event_logits, location_logits


class MultiModNImageFork(nn.Module):
    """Adapted MultiModN baseline with sequential modality-conditioned state updates."""

    def __init__(
        self,
        num_event_classes: int,
        num_location_classes: int,
        input_rows: int = 6,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        stft_n_fft: int = 256,
        stft_hop_length: int = 128,
        stft_win_length: int = 256,
        gaf_size: int = 48,
        image_size: int = 224,
        image_backbone: str = "resnet18",
        image_pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.num_event_classes = int(num_event_classes)
        self.num_location_classes = int(num_location_classes)
        self.hidden_dim = int(hidden_dim)
        self.signal = SignalFeatureExtractor(
            input_rows=input_rows,
            feature_dim=hidden_dim,
            dropout=dropout,
            stft_n_fft=stft_n_fft,
            stft_hop_length=stft_hop_length,
            stft_win_length=stft_win_length,
            gaf_size=gaf_size,
            image_size=image_size,
        )
        self.image_head = HybridImageHead(
            feature_dim=hidden_dim,
            num_event_classes=num_event_classes,
            num_location_classes=num_location_classes,
            backbone_name=image_backbone,
            pretrained=image_pretrained,
            dropout=dropout,
        )
        self.modality_updates = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.LayerNorm(hidden_dim * 2),
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for name in ("raw", "stf", "gaf")
            }
        )
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.csv_event_head = nn.Linear(hidden_dim, num_event_classes)

    def _forward_csv(self, csv_inputs: torch.Tensor) -> torch.Tensor:
        signal = _extract_csv_signal(csv_inputs)
        views = self.signal(signal)
        state = torch.zeros(signal.size(0), self.hidden_dim, device=signal.device, dtype=signal.dtype)
        for name in ("raw", "stf", "gaf"):
            update = self.modality_updates[name](torch.cat([state, views[name]], dim=-1))
            state = self.final_norm(state + update)
        return self.csv_event_head(state)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, Any]:
        device = self.csv_event_head.weight.device
        batch_size = int(batch["batch_size"])
        event_logits = torch.zeros(batch_size, self.num_event_classes, device=device)
        location_logits = torch.zeros(batch_size, self.num_location_classes, device=device)

        csv_inputs = batch.get("csv_inputs")
        csv_indices = batch.get("csv_indices")
        if csv_inputs is not None and csv_indices is not None and csv_indices.numel() > 0:
            csv_indices = csv_indices.to(device, non_blocking=True)
            csv_inputs = csv_inputs.to(device, non_blocking=True)
            event_logits.index_copy_(0, csv_indices, self._forward_csv(csv_inputs).to(event_logits.dtype))

        image_inputs = batch.get("image_inputs")
        image_indices = batch.get("image_indices")
        if image_inputs is not None and image_indices is not None and image_indices.numel() > 0:
            image_indices = image_indices.to(device, non_blocking=True)
            image_inputs = image_inputs.to(device, non_blocking=True)
            _, image_event, image_location = self.image_head(image_inputs)
            event_logits.index_copy_(0, image_indices, image_event.to(event_logits.dtype))
            location_logits.index_copy_(0, image_indices, image_location.to(location_logits.dtype))

        return {"event_type": event_logits, "location": location_logits, "distance_cls": location_logits}


class M4oEImageFork(nn.Module):
    """Adapted M4oE baseline with task-specific routing over modality experts."""

    def __init__(
        self,
        num_event_classes: int,
        num_location_classes: int,
        input_rows: int = 6,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        stft_n_fft: int = 256,
        stft_hop_length: int = 128,
        stft_win_length: int = 256,
        gaf_size: int = 48,
        image_size: int = 224,
        image_backbone: str = "resnet18",
        image_pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.num_event_classes = int(num_event_classes)
        self.num_location_classes = int(num_location_classes)
        self.hidden_dim = int(hidden_dim)
        self.modalities = ("raw", "stf", "gaf", "image")
        self.signal = SignalFeatureExtractor(
            input_rows=input_rows,
            feature_dim=hidden_dim,
            dropout=dropout,
            stft_n_fft=stft_n_fft,
            stft_hop_length=stft_hop_length,
            stft_win_length=stft_win_length,
            gaf_size=gaf_size,
            image_size=image_size,
        )
        self.image_encoder = build_location_image_encoder(
            backbone_name=image_backbone,
            output_dim=hidden_dim,
            dropout=dropout,
            pretrained=image_pretrained,
        )
        self.experts = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.LayerNorm(hidden_dim),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for name in self.modalities
            }
        )
        router_input_dim = hidden_dim + len(self.modalities)
        self.event_router = nn.Sequential(
            nn.LayerNorm(router_input_dim),
            nn.Linear(router_input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, len(self.modalities)),
        )
        self.location_router = nn.Sequential(
            nn.LayerNorm(router_input_dim),
            nn.Linear(router_input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, len(self.modalities)),
        )
        self.event_head = nn.Linear(hidden_dim, num_event_classes)
        self.location_head = nn.Linear(hidden_dim, num_location_classes)

    def _fuse_task(self, features: dict[str, torch.Tensor], router: nn.Module) -> torch.Tensor:
        first_feature = next(iter(features.values()))
        presence = torch.zeros(first_feature.size(0), len(self.modalities), device=first_feature.device, dtype=first_feature.dtype)
        stacked = []
        for idx, name in enumerate(self.modalities):
            if name in features:
                presence[:, idx] = 1.0
                stacked.append(self.experts[name](features[name]))
            else:
                stacked.append(torch.zeros_like(first_feature))
        stacked_tensor = torch.stack(stacked, dim=1)
        mean_feature = stacked_tensor.sum(dim=1) / presence.sum(dim=1, keepdim=True).clamp_min(1.0)
        router_logits = router(torch.cat([mean_feature, presence], dim=-1))
        router_logits = router_logits.masked_fill(presence <= 0, -1e4)
        weights = torch.softmax(router_logits, dim=-1).unsqueeze(-1)
        return (stacked_tensor * weights).sum(dim=1)

    def _forward_csv(self, csv_inputs: torch.Tensor) -> torch.Tensor:
        signal = _extract_csv_signal(csv_inputs)
        views = self.signal(signal)
        features = {name: views[name] for name in ("raw", "stf", "gaf")}
        fused = self._fuse_task(features, self.event_router)
        return self.event_head(fused)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, Any]:
        device = self.event_head.weight.device
        batch_size = int(batch["batch_size"])
        event_logits = torch.zeros(batch_size, self.num_event_classes, device=device)
        location_logits = torch.zeros(batch_size, self.num_location_classes, device=device)

        csv_inputs = batch.get("csv_inputs")
        csv_indices = batch.get("csv_indices")
        if csv_inputs is not None and csv_indices is not None and csv_indices.numel() > 0:
            csv_indices = csv_indices.to(device, non_blocking=True)
            csv_inputs = csv_inputs.to(device, non_blocking=True)
            event_logits.index_copy_(0, csv_indices, self._forward_csv(csv_inputs).to(event_logits.dtype))

        image_inputs = batch.get("image_inputs")
        image_indices = batch.get("image_indices")
        if image_inputs is not None and image_indices is not None and image_indices.numel() > 0:
            image_indices = image_indices.to(device, non_blocking=True)
            image_inputs = image_inputs.to(device, non_blocking=True)
            image_feature = self.image_encoder(image_inputs.float())
            features = {"image": image_feature}
            image_event = self.event_head(self._fuse_task(features, self.event_router))
            image_location = self.location_head(self._fuse_task(features, self.location_router))
            event_logits.index_copy_(0, image_indices, image_event.to(event_logits.dtype))
            location_logits.index_copy_(0, image_indices, image_location.to(location_logits.dtype))

        return {"event_type": event_logits, "location": location_logits, "distance_cls": location_logits}


class DASMAEImageFork(nn.Module):
    """Adapted DAS-MAE baseline with masked signal reconstruction and downstream fine-tuning heads."""

    def __init__(
        self,
        num_event_classes: int,
        num_location_classes: int,
        input_rows: int = 6,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        patch_size: int = 128,
        mask_ratio: float = 0.5,
        mae_loss_weight: float = 0.1,
        image_backbone: str = "resnet18",
        image_pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.num_event_classes = int(num_event_classes)
        self.num_location_classes = int(num_location_classes)
        self.input_rows = int(input_rows)
        self.hidden_dim = int(hidden_dim)
        self.patch_size = int(patch_size)
        self.mask_ratio = float(mask_ratio)
        self.mae_loss_weight = float(mae_loss_weight)
        self.patch_embed = nn.Linear(self.input_rows * self.patch_size, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.event_tower = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.event_head = nn.Linear(hidden_dim, num_event_classes)
        self.recon_head = nn.Linear(hidden_dim, self.input_rows * self.patch_size)
        self.image_head = HybridImageHead(
            feature_dim=hidden_dim,
            num_event_classes=num_event_classes,
            num_location_classes=num_location_classes,
            backbone_name=image_backbone,
            pretrained=image_pretrained,
            dropout=dropout,
        )

    def _patchify(self, signal: torch.Tensor) -> torch.Tensor:
        total_length = signal.size(-1)
        usable_length = (total_length // self.patch_size) * self.patch_size
        if usable_length <= 0:
            raise ValueError("Signal length is too short for the selected patch size.")
        signal = signal[..., :usable_length]
        patches = signal.unfold(-1, self.patch_size, self.patch_size)
        patches = patches.permute(0, 2, 1, 3).contiguous()
        return patches.view(signal.size(0), patches.size(1), self.input_rows * self.patch_size)

    def _forward_csv(self, csv_inputs: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        signal = _extract_csv_signal(csv_inputs)
        patches = self._patchify(signal)
        if self.training:
            mask = torch.rand(patches.size(0), patches.size(1), device=patches.device) < self.mask_ratio
        else:
            mask = torch.zeros(patches.size(0), patches.size(1), device=patches.device, dtype=torch.bool)
        masked_patches = patches.masked_fill(mask.unsqueeze(-1), 0.0)
        tokens = self.patch_embed(masked_patches)
        encoded = self.encoder(tokens)
        pooled = encoded.mean(dim=1)
        event_logits = self.event_head(self.event_tower(pooled))
        aux_losses: dict[str, torch.Tensor] = {}
        if mask.any():
            recon = self.recon_head(encoded)
            recon_error = (recon - patches).pow(2).mean(dim=-1)
            mae_loss = recon_error.masked_select(mask).mean()
            aux_losses["dasmae_recon"] = self.mae_loss_weight * mae_loss
        return event_logits, aux_losses

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, Any]:
        device = self.event_head.weight.device
        batch_size = int(batch["batch_size"])
        event_logits = torch.zeros(batch_size, self.num_event_classes, device=device)
        location_logits = torch.zeros(batch_size, self.num_location_classes, device=device)
        aux_losses: dict[str, torch.Tensor] = {}

        csv_inputs = batch.get("csv_inputs")
        csv_indices = batch.get("csv_indices")
        if csv_inputs is not None and csv_indices is not None and csv_indices.numel() > 0:
            csv_indices = csv_indices.to(device, non_blocking=True)
            csv_inputs = csv_inputs.to(device, non_blocking=True)
            csv_event, csv_aux = self._forward_csv(csv_inputs)
            event_logits.index_copy_(0, csv_indices, csv_event.to(event_logits.dtype))
            aux_losses.update(csv_aux)

        image_inputs = batch.get("image_inputs")
        image_indices = batch.get("image_indices")
        if image_inputs is not None and image_indices is not None and image_indices.numel() > 0:
            image_indices = image_indices.to(device, non_blocking=True)
            image_inputs = image_inputs.to(device, non_blocking=True)
            _, image_event, image_location = self.image_head(image_inputs)
            event_logits.index_copy_(0, image_indices, image_event.to(event_logits.dtype))
            location_logits.index_copy_(0, image_indices, image_location.to(location_logits.dtype))

        return {
            "event_type": event_logits,
            "location": location_logits,
            "distance_cls": location_logits,
            "aux_losses": aux_losses,
        }


class TimmBackboneWrapper(nn.Module):
    def __init__(self, timm_name: str, output_dim: int, dropout: float, pretrained: bool = True) -> None:
        super().__init__()
        if timm is None:
            raise ImportError("timm is required for PipelineADWinTImageFork.")
        try:
            self.backbone = timm.create_model(timm_name, pretrained=pretrained, num_classes=0, global_pool="avg")
        except Exception as exc:  # pragma: no cover - depends on local cache/network
            warnings.warn(f"Falling back to randomly initialized {timm_name} because pretrained weights failed to load: {exc}")
            self.backbone = timm.create_model(timm_name, pretrained=False, num_classes=0, global_pool="avg")
        self.backbone_dim = int(getattr(self.backbone, "num_features"))
        self.proj = nn.Sequential(
            nn.LayerNorm(self.backbone_dim),
            nn.Dropout(dropout),
            nn.Linear(self.backbone_dim, output_dim),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images.float())
        if features.ndim > 2:
            features = features.flatten(1)
        return self.proj(features)


class PipelineADWinTImageFork(nn.Module):
    """Adapted PipelineADWinT baseline using a windowed vision transformer on signal pseudo-images and RGB images."""

    def __init__(
        self,
        num_event_classes: int,
        num_location_classes: int,
        input_rows: int = 6,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        stft_n_fft: int = 256,
        stft_hop_length: int = 128,
        stft_win_length: int = 256,
        gaf_size: int = 48,
        image_size: int = 224,
        timm_name: str = "swin_tiny_patch4_window7_224.ms_in1k",
    ) -> None:
        super().__init__()
        self.num_event_classes = int(num_event_classes)
        self.num_location_classes = int(num_location_classes)
        self.signal = SignalFeatureExtractor(
            input_rows=input_rows,
            feature_dim=hidden_dim,
            dropout=dropout,
            stft_n_fft=stft_n_fft,
            stft_hop_length=stft_hop_length,
            stft_win_length=stft_win_length,
            gaf_size=gaf_size,
            image_size=image_size,
        )
        self.visual_backbone = TimmBackboneWrapper(
            timm_name=timm_name,
            output_dim=hidden_dim,
            dropout=dropout,
            pretrained=True,
        )
        self.event_tower = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.location_tower = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.event_head = nn.Linear(hidden_dim, num_event_classes)
        self.location_head = nn.Linear(hidden_dim, num_location_classes)

    def _forward_visual(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feature = self.visual_backbone(images)
        return self.event_head(self.event_tower(feature)), self.location_head(self.location_tower(feature))

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, Any]:
        device = self.event_head.weight.device
        batch_size = int(batch["batch_size"])
        event_logits = torch.zeros(batch_size, self.num_event_classes, device=device)
        location_logits = torch.zeros(batch_size, self.num_location_classes, device=device)

        csv_inputs = batch.get("csv_inputs")
        csv_indices = batch.get("csv_indices")
        if csv_inputs is not None and csv_indices is not None and csv_indices.numel() > 0:
            csv_indices = csv_indices.to(device, non_blocking=True)
            csv_inputs = csv_inputs.to(device, non_blocking=True)
            signal = _extract_csv_signal(csv_inputs)
            pseudo_images = self.signal(signal)["pseudo_image"]
            csv_event, _ = self._forward_visual(pseudo_images)
            event_logits.index_copy_(0, csv_indices, csv_event.to(event_logits.dtype))

        image_inputs = batch.get("image_inputs")
        image_indices = batch.get("image_indices")
        if image_inputs is not None and image_indices is not None and image_indices.numel() > 0:
            image_indices = image_indices.to(device, non_blocking=True)
            image_inputs = image_inputs.to(device, non_blocking=True)
            image_event, image_location = self._forward_visual(image_inputs)
            event_logits.index_copy_(0, image_indices, image_event.to(event_logits.dtype))
            location_logits.index_copy_(0, image_indices, image_location.to(location_logits.dtype))

        return {"event_type": event_logits, "location": location_logits, "distance_cls": location_logits}


def build_multimodn_imagefork(**kwargs) -> MultiModNImageFork:
    return MultiModNImageFork(**kwargs)


def build_m4oe_imagefork(**kwargs) -> M4oEImageFork:
    return M4oEImageFork(**kwargs)


def build_dasmae_imagefork(**kwargs) -> DASMAEImageFork:
    return DASMAEImageFork(**kwargs)


def build_pipelineadwint_imagefork(**kwargs) -> PipelineADWinTImageFork:
    return PipelineADWinTImageFork(**kwargs)


__all__ = [
    "MultiModNImageFork",
    "M4oEImageFork",
    "DASMAEImageFork",
    "PipelineADWinTImageFork",
    "build_multimodn_imagefork",
    "build_m4oe_imagefork",
    "build_dasmae_imagefork",
    "build_pipelineadwint_imagefork",
]
