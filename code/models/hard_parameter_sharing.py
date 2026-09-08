from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.pipe_mmtl import LocationImageEncoder


class HPSImageFork(nn.Module):
    """Shared-backbone baseline for hybrid imagefork inputs.

    CSV samples are resized into 3-channel image-like tensors so both input domains can
    share the same CNN encoder, followed by one head per task.
    """

    def __init__(
        self,
        num_event_classes: int,
        num_location_classes: int,
        feature_dim: int = 256,
        hidden_dim: int = 128,
        image_size: int = 224,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.num_event_classes = int(num_event_classes)
        self.num_location_classes = int(num_location_classes)
        self.image_size = int(image_size)

        # Keep these attribute names aligned with the existing training loop hooks.
        self.location_image_encoder = LocationImageEncoder(output_dim=feature_dim, dropout=dropout)
        self.location_image_proj = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.event_tower = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.location_tower = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.event_head = nn.Linear(hidden_dim, self.num_event_classes)
        self.location_head = nn.Linear(hidden_dim, self.num_location_classes)

    def _csv_to_shared_image(self, csv_inputs: torch.Tensor) -> torch.Tensor:
        # csv_inputs: [B_csv, C, H, W] -> shared image tensor [B_csv, 3, image_size, image_size]
        if csv_inputs.ndim != 4:
            raise ValueError(f"Expected CSV batch with shape [B, C, H, W], got {tuple(csv_inputs.shape)}")
        csv_inputs = csv_inputs.float()
        if csv_inputs.size(1) == 1:
            csv_inputs = csv_inputs.repeat(1, 3, 1, 1)
        elif csv_inputs.size(1) != 3:
            csv_inputs = csv_inputs.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
        if csv_inputs.shape[-2:] != (self.image_size, self.image_size):
            csv_inputs = F.interpolate(
                csv_inputs,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
        return csv_inputs

    def _build_shared_batch(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        if "batch_size" not in batch:
            raise ValueError("HPSImageFork expects a collated batch dictionary with 'batch_size'.")

        device = self.event_head.weight.device
        shared_inputs = torch.zeros(
            int(batch["batch_size"]),
            3,
            self.image_size,
            self.image_size,
            device=device,
            dtype=torch.float32,
        )

        csv_inputs = batch.get("csv_inputs")
        csv_indices = batch.get("csv_indices")
        if csv_inputs is not None and csv_indices is not None and csv_indices.numel() > 0:
            shared_inputs.index_copy_(
                0,
                csv_indices.to(device, non_blocking=True),
                self._csv_to_shared_image(csv_inputs.to(device, non_blocking=True)),
            )

        image_inputs = batch.get("image_inputs")
        image_indices = batch.get("image_indices")
        if image_inputs is not None and image_indices is not None and image_indices.numel() > 0:
            image_inputs = image_inputs.to(device, non_blocking=True).float()
            if image_inputs.shape[-2:] != (self.image_size, self.image_size):
                image_inputs = F.interpolate(
                    image_inputs,
                    size=(self.image_size, self.image_size),
                    mode="bilinear",
                    align_corners=False,
                )
            shared_inputs.index_copy_(0, image_indices.to(device, non_blocking=True), image_inputs)

        return shared_inputs

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, Any]:
        # shared_inputs: [B, 3, image_size, image_size]
        shared_inputs = self._build_shared_batch(batch)
        shared_feature = self.location_image_proj(self.location_image_encoder(shared_inputs))  # [B, D]

        event_feature = self.event_tower(shared_feature)  # [B, H]
        location_feature = self.location_tower(shared_feature)  # [B, H]
        event_logits = self.event_head(event_feature)  # [B, C_event]
        location_logits = self.location_head(location_feature)  # [B, C_location]

        return {
            "event_type": event_logits,
            "location": location_logits,
            "distance_cls": location_logits,
        }


def build_hps_imagefork(**kwargs) -> HPSImageFork:
    return HPSImageFork(**kwargs)


__all__ = [
    "HPSImageFork",
    "build_hps_imagefork",
]
