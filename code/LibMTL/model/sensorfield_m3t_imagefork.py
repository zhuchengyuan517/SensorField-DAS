from __future__ import annotations

from typing import Any
import warnings

import torch
import torch.nn as nn
try:
    from torchvision.models import ResNet18_Weights, ResNet34_Weights, resnet18, resnet34
except Exception:  # pragma: no cover - optional dependency at runtime
    ResNet18_Weights = None
    ResNet34_Weights = None
    resnet18 = None
    resnet34 = None
try:
    import timm
except Exception:  # pragma: no cover - optional dependency at runtime
    timm = None

from LibMTL.model.pipemmtl import LocationImageEncoder
from LibMTL.model.sensorfield_m3t import SensorFieldM3T


TIMM_LOCATION_BACKBONES = {
    "convnext_small": "convnext_small.fb_in22k_ft_in1k",
    "convnextv2_tiny": "convnextv2_tiny.fcmae_ft_in22k_in1k",
    "maxvit_tiny": "maxvit_tiny_tf_224.in1k",
    "coatnet_0": "coatnet_0_rw_224.sw_in1k",
}


class ResNetLocationImageEncoder(nn.Module):
    """Stronger RGB image encoder for location classification."""

    def __init__(
        self,
        output_dim: int,
        dropout: float,
        pretrained: bool = False,
        backbone_name: str = "resnet18",
    ) -> None:
        super().__init__()
        normalized_name = str(backbone_name).strip().lower()
        if normalized_name == "resnet18":
            backbone_builder = resnet18
            weights_enum = ResNet18_Weights
            weights_name = "ResNet18"
        elif normalized_name == "resnet34":
            backbone_builder = resnet34
            weights_enum = ResNet34_Weights
            weights_name = "ResNet34"
        else:
            raise ValueError(f"Unsupported ResNet image backbone: {backbone_name}")

        if backbone_builder is None:
            raise ImportError(f"torchvision is required to use the {normalized_name} location image backbone.")

        weights = None
        if pretrained:
            if weights_enum is None:
                warnings.warn(f"{weights_name} weights API is unavailable; falling back to randomly initialized weights.")
            else:
                weights = weights_enum.IMAGENET1K_V1
        try:
            backbone = backbone_builder(weights=weights)
        except Exception as exc:  # pragma: no cover - depends on local cache/network
            warnings.warn(
                f"Falling back to randomly initialized {weights_name} because pretrained weights failed to load: {exc}"
            )
            backbone = backbone_builder(weights=None)

        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.layer3_dim = 256
        self.backbone_dim = int(backbone.fc.in_features)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(backbone.fc.in_features, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.register_buffer("pixel_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("pixel_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

    def _normalize(self, images: torch.Tensor) -> torch.Tensor:
        images = (images - self.pixel_mean) / self.pixel_std
        return images

    def forward_backbone(self, images: torch.Tensor) -> torch.Tensor:
        images = self._normalize(images)
        features = self.stem(images)
        features = self.layer1(features)
        features = self.layer2(features)
        features = self.layer3(features)
        return self.layer4(features)

    def forward_with_multiscale(self, images: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        images = self._normalize(images)
        features = self.stem(images)
        features = self.layer1(features)
        features = self.layer2(features)
        layer3_map = self.layer3(features)
        layer4_map = self.layer4(layer3_map)
        return self.head(layer4_map), (layer3_map, layer4_map)

    def forward_with_expert(self, images: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        images = self._normalize(images)
        features = self.stem(images)
        features = self.layer1(features)
        features = self.layer2(features)
        layer3_map = self.layer3(features)
        layer4_map = self.layer4(layer3_map)
        backbone_feature = torch.flatten(nn.functional.adaptive_avg_pool2d(layer4_map, (1, 1)), 1)
        return self.head(layer4_map), (layer3_map, layer4_map), backbone_feature

    def forward_with_map(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pooled_feature, (_, layer4_map) = self.forward_with_multiscale(images)
        return pooled_feature, layer4_map

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        pooled_feature, _ = self.forward_with_map(images)
        return pooled_feature


class TimmLocationImageEncoder(nn.Module):
    """Timm-backed RGB image encoder for stronger location classification."""

    def __init__(
        self,
        output_dim: int,
        dropout: float,
        pretrained: bool = False,
        backbone_name: str = "convnext_small",
    ) -> None:
        super().__init__()
        normalized_name = str(backbone_name).strip().lower()
        timm_name = TIMM_LOCATION_BACKBONES.get(normalized_name)
        if timm_name is None:
            raise ValueError(f"Unsupported timm image backbone: {backbone_name}")
        if timm is None:
            raise ImportError(f"timm is required to use the {normalized_name} location image backbone.")

        try:
            backbone = timm.create_model(
                timm_name,
                pretrained=pretrained,
                num_classes=0,
                global_pool="",
                in_chans=3,
            )
        except Exception as exc:  # pragma: no cover - depends on local cache/network
            warnings.warn(
                f"Falling back to randomly initialized {normalized_name} because pretrained weights failed to load: {exc}"
            )
            backbone = timm.create_model(
                timm_name,
                pretrained=False,
                num_classes=0,
                global_pool="",
                in_chans=3,
            )

        self.backbone = backbone
        self.backbone_dim = int(getattr(backbone, "num_features"))
        self.head = nn.Sequential(
            nn.LayerNorm(self.backbone_dim),
            nn.Dropout(dropout),
            nn.Linear(self.backbone_dim, output_dim),
        )
        pretrained_cfg = getattr(backbone, "pretrained_cfg", {}) or {}
        mean = pretrained_cfg.get("mean", (0.485, 0.456, 0.406))
        std = pretrained_cfg.get("std", (0.229, 0.224, 0.225))
        self.register_buffer("pixel_mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("pixel_std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1), persistent=False)

    def _normalize(self, images: torch.Tensor) -> torch.Tensor:
        return (images - self.pixel_mean) / self.pixel_std

    def _forward_feature_map(self, images: torch.Tensor) -> torch.Tensor | None:
        features = self.backbone.forward_features(self._normalize(images))
        if isinstance(features, (list, tuple)):
            if not features:
                return None
            features = features[-1]
        if torch.is_tensor(features) and features.ndim == 4:
            return features
        return None

    def forward_with_map(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        feature_map = self._forward_feature_map(images)
        if feature_map is None:
            pooled = self.backbone(self._normalize(images))
            if pooled.ndim > 2:
                pooled = pooled.flatten(1)
        else:
            pooled = feature_map.mean(dim=(-2, -1))
        pooled_feature = self.head(pooled)
        return pooled_feature, feature_map

    def forward_with_expert(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        feature_map = self._forward_feature_map(images)
        if feature_map is None:
            pooled = self.backbone(self._normalize(images))
            if pooled.ndim > 2:
                pooled = pooled.flatten(1)
        else:
            pooled = feature_map.mean(dim=(-2, -1))
        pooled_feature = self.head(pooled)
        return pooled_feature, feature_map, pooled

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        pooled_feature, _ = self.forward_with_map(images)
        return pooled_feature


def build_location_image_encoder(
    backbone_name: str,
    output_dim: int,
    dropout: float,
    pretrained: bool,
) -> nn.Module:
    normalized_name = str(backbone_name).strip().lower()
    if normalized_name == "legacy_cnn":
        return LocationImageEncoder(output_dim=output_dim, dropout=dropout)
    if normalized_name in {"resnet18", "resnet34"}:
        return ResNetLocationImageEncoder(
            output_dim=output_dim,
            dropout=dropout,
            pretrained=pretrained,
            backbone_name=normalized_name,
        )
    if normalized_name in TIMM_LOCATION_BACKBONES:
        return TimmLocationImageEncoder(
            output_dim=output_dim,
            dropout=dropout,
            pretrained=pretrained,
            backbone_name=normalized_name,
        )
    raise ValueError(f"Unsupported location image backbone: {backbone_name}")


class SensorFieldM3TImageFork(nn.Module):
    """Hybrid image fork with SensorField-M3T on CSV samples and an image branch on flower photos."""

    def __init__(
        self,
        num_event_classes: int,
        num_location_classes: int,
        hidden_dim: int = 128,
        num_anchors: int = 8,
        num_heads: int = 4,
        fusion_dim: int = 256,
        raw_tokens: int = 16,
        stf_tokens: int = 16,
        gaf_tokens: int = 16,
        gaf_size: int = 64,
        stft_n_fft: int = 256,
        stft_hop_length: int = 128,
        stft_win_length: int = 256,
        image_size: int = 224,
        fac_loss_weight: float = 0.1,
        taef_loss_weight: float = 0.0,
        gcti_loss_weight: float = 0.0,
        view_drop_prob: float = 0.0,
        enable_view_consistency: bool = False,
        disable_fac: bool = False,
        disable_complement: bool = False,
        disable_taef: bool = False,
        disable_gcti: bool = False,
        disable_view_consistency: bool = False,
        enabled_views: str | tuple[str, ...] | list[str] = ("raw", "stf", "gaf"),
        view_consistency_weight: float = 0.0,
        view_noise_std: float = 0.01,
        location_image_backbone: str = "legacy_cnn",
        location_image_backbone_pretrained: bool = False,
        image_location_ensemble_weight: float = 0.0,
        image_location_specialist_blend: float = 1.0,
        image_event_expert_weight: float = 0.0,
        image_location_expert_weight: float = 0.0,
        dropout: float = 0.1,
        return_auxiliary: bool = False,
    ) -> None:
        super().__init__()
        self.num_event_classes = int(num_event_classes)
        self.num_location_classes = int(num_location_classes)
        self.image_size = int(image_size)
        self.return_auxiliary = bool(return_auxiliary)
        self.location_image_backbone = str(location_image_backbone).strip().lower()
        self.location_image_backbone_pretrained = bool(location_image_backbone_pretrained)
        self.image_location_ensemble_weight = float(image_location_ensemble_weight)
        self.image_location_specialist_blend = float(max(0.0, min(1.0, image_location_specialist_blend)))
        self.image_event_expert_weight = float(max(0.0, min(1.0, image_event_expert_weight)))
        self.image_location_expert_weight = float(max(0.0, min(1.0, image_location_expert_weight)))

        self.csv_backbone = SensorFieldM3T(
            task_output_dims={
                "event_type": self.num_event_classes,
                "distance_cls": self.num_location_classes,
            },
            hidden_dim=hidden_dim,
            num_anchors=num_anchors,
            num_heads=num_heads,
            raw_tokens=raw_tokens,
            stf_tokens=stf_tokens,
            gaf_tokens=gaf_tokens,
            stf_size=gaf_size,
            gaf_size=gaf_size,
            stft_n_fft=stft_n_fft,
            stft_hop_length=stft_hop_length,
            stft_win_length=stft_win_length,
            fac_loss_weight=fac_loss_weight,
            taef_loss_weight=taef_loss_weight,
            gcti_loss_weight=gcti_loss_weight,
            view_drop_prob=view_drop_prob,
            enable_view_consistency=enable_view_consistency,
            disable_fac=disable_fac,
            disable_complement=disable_complement,
            disable_taef=disable_taef,
            disable_gcti=disable_gcti,
            disable_view_consistency=disable_view_consistency,
            enabled_views=enabled_views,
            view_consistency_weight=view_consistency_weight,
            view_noise_std=view_noise_std,
            dropout=dropout,
            return_auxiliary=return_auxiliary,
        )

        self.location_image_encoder = build_location_image_encoder(
            backbone_name=self.location_image_backbone,
            output_dim=fusion_dim,
            dropout=dropout,
            pretrained=self.location_image_backbone_pretrained,
        )
        image_map_dim = int(getattr(self.location_image_encoder, "backbone_dim", fusion_dim))
        image_map_low_dim = int(getattr(self.location_image_encoder, "layer3_dim", image_map_dim))
        self.location_image_proj = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.location_image_aux_head = nn.Linear(fusion_dim, self.num_location_classes)
        self.image_event_tower = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.image_location_tower = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.image_location_spatial_proj_low = nn.Linear(image_map_low_dim, fusion_dim)
        self.image_location_spatial_proj_high = nn.Linear(image_map_dim, fusion_dim)
        self.image_event_expert_head = nn.Sequential(
            nn.LayerNorm(image_map_dim),
            nn.Dropout(dropout),
            nn.Linear(image_map_dim, self.num_event_classes),
        )
        self.image_location_expert_head = nn.Sequential(
            nn.LayerNorm(image_map_dim),
            nn.Dropout(dropout),
            nn.Linear(image_map_dim, self.num_location_classes),
        )
        self.image_location_query_proj = nn.Linear(fusion_dim, fusion_dim)
        self.image_location_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=max(1, min(num_heads, 4)),
            batch_first=True,
            dropout=dropout,
        )
        self.image_location_spatial_norm = nn.LayerNorm(fusion_dim)
        self.image_location_gate = nn.Sequential(
            nn.LayerNorm(fusion_dim * 2),
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, 2),
        )
        self.image_location_fusion = nn.Sequential(
            nn.LayerNorm(fusion_dim * 3),
            nn.Linear(fusion_dim * 3, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.image_event_head = nn.Linear(fusion_dim, self.num_event_classes)
        self.location_head = nn.Linear(fusion_dim, self.num_location_classes)
        self.image_location_classifier = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, self.num_location_classes),
        )
        self._initialize_location_specialists()

    def _initialize_location_specialists(self) -> None:
        gate_out = self.image_location_gate[-1]
        if isinstance(gate_out, nn.Linear):
            nn.init.zeros_(gate_out.weight)
            with torch.no_grad():
                gate_out.bias.copy_(torch.tensor([6.0, -6.0], dtype=gate_out.bias.dtype))

    def initialize_missing_location_specialists(self) -> None:
        classifier_out = self.image_location_classifier[-1]
        if isinstance(classifier_out, nn.Linear):
            if classifier_out.weight.shape == self.location_head.weight.shape:
                with torch.no_grad():
                    classifier_out.weight.copy_(self.location_head.weight)
                    classifier_out.bias.copy_(self.location_head.bias)
        self._initialize_location_specialists()

    def forward_location_image(self, images: torch.Tensor) -> torch.Tensor:
        image_feature, _, _ = self._encode_location_image(images.float())
        return self.location_image_aux_head(image_feature)

    def _encode_location_image(
        self,
        images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None, torch.Tensor | None]:
        if hasattr(self.location_image_encoder, "forward_with_expert"):
            image_feature, image_feature_map, backbone_feature = self.location_image_encoder.forward_with_expert(images)
        elif hasattr(self.location_image_encoder, "forward_with_multiscale"):
            image_feature, image_feature_map = self.location_image_encoder.forward_with_multiscale(images)
            backbone_feature = None
        elif hasattr(self.location_image_encoder, "forward_with_map"):
            image_feature, image_feature_map = self.location_image_encoder.forward_with_map(images)
            backbone_feature = None
        else:
            image_feature = self.location_image_encoder(images)
            image_feature_map = None
            backbone_feature = None
        image_feature = self.location_image_proj(image_feature)
        return image_feature, image_feature_map, backbone_feature

    def _forward_image_location(
        self,
        image_feature: torch.Tensor,
        image_feature_map: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        base_hidden = self.image_location_tower(image_feature)
        base_logits = self.location_head(base_hidden)
        if image_feature_map is None:
            dedicated_logits = self.image_location_classifier(base_hidden)
            dedicated_logits = dedicated_logits.to(base_logits.dtype)
            final_logits = torch.lerp(base_logits, dedicated_logits, self.image_location_specialist_blend)
            return final_logits, dedicated_logits

        # Multi-scale ResNet maps: [B, C, H, W] -> spatial tokens [B, N, D].
        if isinstance(image_feature_map, tuple):
            low_map, high_map = image_feature_map
            low_tokens = self.image_location_spatial_proj_low(low_map.flatten(2).transpose(1, 2))
            high_tokens = self.image_location_spatial_proj_high(high_map.flatten(2).transpose(1, 2))
            spatial_tokens = torch.cat([low_tokens, high_tokens], dim=1)
        else:
            spatial_tokens = self.image_location_spatial_proj_high(image_feature_map.flatten(2).transpose(1, 2))
        spatial_query = self.image_location_query_proj(image_feature).unsqueeze(1)
        attended_tokens, _ = self.image_location_attention(
            spatial_query,
            spatial_tokens,
            spatial_tokens,
            need_weights=False,
        )
        spatial_hidden = self.image_location_spatial_norm(
            attended_tokens.squeeze(1) + spatial_tokens.mean(dim=1)
        )
        fused_hidden = self.image_location_fusion(
            torch.cat([image_feature, base_hidden, spatial_hidden], dim=-1)
        )
        fusion_weights = torch.softmax(
            self.image_location_gate(torch.cat([base_hidden, spatial_hidden], dim=-1)),
            dim=-1,
        )
        dedicated_logits = self.image_location_classifier(fused_hidden + base_hidden)
        dedicated_logits = dedicated_logits.to(base_logits.dtype)
        specialist_logits = fusion_weights[:, :1] * base_logits + fusion_weights[:, 1:] * dedicated_logits
        specialist_logits = specialist_logits.to(base_logits.dtype)
        final_logits = torch.lerp(base_logits, specialist_logits, self.image_location_specialist_blend)
        return final_logits, dedicated_logits

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, Any]:
        if "batch_size" not in batch:
            raise ValueError("Hybrid model expects a collated batch dictionary with 'batch_size'.")

        device = self.location_head.weight.device
        batch_size = int(batch["batch_size"])
        event_logits = torch.zeros(batch_size, self.num_event_classes, device=device)
        location_logits = torch.zeros(batch_size, self.num_location_classes, device=device)
        aux_losses: dict[str, torch.Tensor] = {}
        csv_outputs: dict[str, Any] | None = None
        image_aux_location_logits: torch.Tensor | None = None
        image_dedicated_location_logits: torch.Tensor | None = None

        csv_inputs = batch.get("csv_inputs")
        csv_indices = batch.get("csv_indices")
        if csv_inputs is not None and csv_indices is not None and csv_indices.numel() > 0:
            csv_inputs = csv_inputs.to(device, non_blocking=True)
            csv_indices = csv_indices.to(device, non_blocking=True)
            csv_outputs = self.csv_backbone(csv_inputs)
            event_logits.index_copy_(0, csv_indices, csv_outputs["event_type"].to(event_logits.dtype))
            location_logits.index_copy_(0, csv_indices, csv_outputs["distance_cls"].to(location_logits.dtype))
            aux_losses.update(csv_outputs.get("aux_losses", {}))

        image_inputs = batch.get("image_inputs")
        image_indices = batch.get("image_indices")
        if image_inputs is not None and image_indices is not None and image_indices.numel() > 0:
            image_inputs = image_inputs.to(device, non_blocking=True)
            image_indices = image_indices.to(device, non_blocking=True)
            image_feature, image_feature_map, image_backbone_feature = self._encode_location_image(image_inputs.float())
            image_aux_location_logits = self.location_image_aux_head(image_feature)
            image_event_logits = self.image_event_head(self.image_event_tower(image_feature))
            image_location_logits, image_dedicated_location_logits = self._forward_image_location(
                image_feature,
                image_feature_map,
            )
            if image_backbone_feature is not None:
                if self.image_event_expert_weight > 0:
                    expert_event_logits = self.image_event_expert_head(image_backbone_feature).to(image_event_logits.dtype)
                    image_event_logits = torch.lerp(
                        image_event_logits,
                        expert_event_logits,
                        self.image_event_expert_weight,
                    )
                if self.image_location_expert_weight > 0:
                    expert_location_logits = self.image_location_expert_head(image_backbone_feature).to(
                        image_location_logits.dtype
                    )
                    image_location_logits = torch.lerp(
                        image_location_logits,
                        expert_location_logits,
                        self.image_location_expert_weight,
                    )
            if self.image_location_ensemble_weight > 0:
                ensemble_weight = max(0.0, min(1.0, self.image_location_ensemble_weight))
                image_aux_location_logits = image_aux_location_logits.to(image_location_logits.dtype)
                image_location_logits = (
                    (1.0 - ensemble_weight) * image_location_logits
                    + ensemble_weight * image_aux_location_logits
                )
            event_logits.index_copy_(0, image_indices, image_event_logits.to(event_logits.dtype))
            location_logits.index_copy_(0, image_indices, image_location_logits.to(location_logits.dtype))

        if not aux_losses:
            zero = event_logits.sum() * 0.0
            aux_losses = {
                "fac_loss": zero,
                "taef_loss": zero,
                "gcti_loss": zero,
            }

        outputs: dict[str, Any] = {
            "event_type": event_logits,
            "location": location_logits,
            "distance_cls": location_logits,
            "aux_losses": aux_losses,
            "image_aux_location_logits": image_aux_location_logits,
            "image_dedicated_location_logits": image_dedicated_location_logits,
        }
        if self.return_auxiliary:
            outputs["csv_outputs"] = csv_outputs
        return outputs


def build_sensorfield_m3t_imagefork(**kwargs: Any) -> SensorFieldM3TImageFork:
    return SensorFieldM3TImageFork(**kwargs)
