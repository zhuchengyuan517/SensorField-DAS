from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from LibMTL.model.pipemmtl import (
    CrossAttentionBlock,
    GAFEncoder,
    LocationImageEncoder,
    SelfAttentionBlock,
    STFTEncoder,
    TimeStatisticsEncoder,
)


class PipeMMTLImageFork(nn.Module):
    """Hybrid fork: non-excavator samples use CSV, excavator samples use images."""

    def __init__(
        self,
        num_event_classes: int,
        num_location_classes: int,
        input_rows: int = 6,
        embed_dim: int = 128,
        fusion_dim: int = 256,
        time_tokens: int = 6,
        freq_tokens: int = 48,
        gaf_tokens: int = 48,
        gaf_size: int = 48,
        prior_tokens: int = 8,
        num_heads: int = 2,
        dropout: float = 0.2,
        stft_n_fft: int = 256,
        stft_hop_length: int = 128,
        stft_win_length: int = 256,
        image_size: int = 224,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads.")

        self.num_event_classes = num_event_classes
        self.num_location_classes = num_location_classes
        self.input_rows = input_rows
        self.embed_dim = embed_dim
        self.image_size = image_size

        self.time_encoder = TimeStatisticsEncoder(input_rows=input_rows, embed_dim=embed_dim, dropout=dropout)
        self.stft_encoder = STFTEncoder(
            input_rows=input_rows,
            embed_dim=embed_dim,
            tokens_per_row=max(1, freq_tokens // max(input_rows, 1)),
            n_fft=stft_n_fft,
            hop_length=stft_hop_length,
            win_length=stft_win_length,
            dropout=dropout,
        )
        self.gaf_encoder = GAFEncoder(
            input_rows=input_rows,
            embed_dim=embed_dim,
            tokens_per_row=max(1, gaf_tokens // max(input_rows, 1)),
            gaf_size=gaf_size,
            dropout=dropout,
        )

        self.prior_tokens = nn.Parameter(torch.randn(1, prior_tokens, embed_dim) * 0.02)
        self.spatial_cross_attention = CrossAttentionBlock(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        self.frequency_cross_attention = CrossAttentionBlock(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        self.temporal_self_attention = SelfAttentionBlock(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        self.cross_to_time = nn.Linear(embed_dim * 2, embed_dim)

        self.csv_event_fusion = nn.Sequential(
            nn.LayerNorm(embed_dim * 3),
            nn.Linear(embed_dim * 3, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.csv_event_tower = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.location_image_encoder = LocationImageEncoder(output_dim=fusion_dim, dropout=dropout)
        self.location_image_proj = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
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

        self.event_head = nn.Linear(fusion_dim, num_event_classes)
        self.location_head = nn.Linear(fusion_dim, num_location_classes)

    def _extract_csv_signal(self, csv_inputs: torch.Tensor) -> torch.Tensor:
        if csv_inputs is None:
            return None
        if csv_inputs.ndim == 4:
            return csv_inputs[:, 0, :, :].float()
        if csv_inputs.ndim == 3:
            return csv_inputs.float()
        raise ValueError(f"Unsupported csv input shape: {tuple(csv_inputs.shape)}")

    def _forward_csv_event(self, csv_inputs: torch.Tensor) -> torch.Tensor:
        signal = self._extract_csv_signal(csv_inputs)
        time_tokens, _ = self.time_encoder(signal)
        stft_tokens, _ = self.stft_encoder(signal)
        gaf_tokens, _ = self.gaf_encoder(signal)

        prior = self.prior_tokens.expand(signal.size(0), -1, -1)
        spatial_tokens = self.spatial_cross_attention(gaf_tokens, torch.cat([time_tokens, prior], dim=1))
        frequency_tokens = self.frequency_cross_attention(stft_tokens, torch.cat([time_tokens, spatial_tokens], dim=1))
        cross_summary = torch.cat([spatial_tokens.mean(dim=1), frequency_tokens.mean(dim=1)], dim=-1)
        time_tokens = self.temporal_self_attention(time_tokens + self.cross_to_time(cross_summary).unsqueeze(1))

        fused_feature = self.csv_event_fusion(
            torch.cat(
                [
                    time_tokens.mean(dim=1),
                    frequency_tokens.mean(dim=1),
                    spatial_tokens.mean(dim=1),
                ],
                dim=-1,
            )
        )
        return self.csv_event_tower(fused_feature)

    def forward_location_image(self, images: torch.Tensor) -> torch.Tensor:
        image_feature = self.location_image_proj(self.location_image_encoder(images.float()))
        image_feature = self.image_location_tower(image_feature)
        return self.location_head(image_feature)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, Any]:
        if "batch_size" not in batch:
            raise ValueError("Hybrid model expects a collated batch dictionary with 'batch_size'.")

        device = self.event_head.weight.device
        batch_size = int(batch["batch_size"])
        event_logits = torch.zeros(batch_size, self.num_event_classes, device=device)
        location_logits = torch.zeros(batch_size, self.num_location_classes, device=device)

        csv_inputs = batch.get("csv_inputs")
        csv_indices = batch.get("csv_indices")
        if csv_inputs is not None and csv_indices is not None and csv_indices.numel() > 0:
            csv_inputs = csv_inputs.to(device, non_blocking=True)
            csv_indices = csv_indices.to(device, non_blocking=True)
            csv_feature = self._forward_csv_event(csv_inputs)
            csv_event_logits = self.event_head(csv_feature)
            event_logits.index_copy_(0, csv_indices, csv_event_logits)

        image_inputs = batch.get("image_inputs")
        image_indices = batch.get("image_indices")
        if image_inputs is not None and image_indices is not None and image_indices.numel() > 0:
            image_inputs = image_inputs.to(device, non_blocking=True)
            image_indices = image_indices.to(device, non_blocking=True)
            image_feature = self.location_image_proj(self.location_image_encoder(image_inputs))
            image_event_logits = self.event_head(self.image_event_tower(image_feature))
            image_location_logits = self.location_head(self.image_location_tower(image_feature))
            event_logits.index_copy_(0, image_indices, image_event_logits)
            location_logits.index_copy_(0, image_indices, image_location_logits)

        return {
            "event_type": event_logits,
            "location": location_logits,
            "distance_cls": location_logits,
        }


def build_pipe_mmtl_imagefork(**kwargs) -> PipeMMTLImageFork:
    return PipeMMTLImageFork(**kwargs)


__all__ = [
    "PipeMMTLImageFork",
    "build_pipe_mmtl_imagefork",
]
