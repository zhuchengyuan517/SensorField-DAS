from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForwardBlock(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


class CrossAttentionBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(embed_dim)
        self.context_norm = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(embed_dim)
        self.ffn = FeedForwardBlock(embed_dim, int(embed_dim * mlp_ratio), dropout)

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(
            self.query_norm(query),
            self.context_norm(context),
            self.context_norm(context),
            need_weights=False,
        )
        query = query + attn_out
        query = query + self.ffn(self.ffn_norm(query))
        return query


class SelfAttentionBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(embed_dim)
        self.ffn = FeedForwardBlock(embed_dim, int(embed_dim * mlp_ratio), dropout)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        norm_tokens = self.input_norm(tokens)
        attn_out, _ = self.attn(norm_tokens, norm_tokens, norm_tokens, need_weights=False)
        tokens = tokens + attn_out
        tokens = tokens + self.ffn(self.ffn_norm(tokens))
        return tokens


class TimeStatisticsEncoder(nn.Module):
    def __init__(self, input_rows: int, embed_dim: int, dropout: float) -> None:
        super().__init__()
        self.input_rows = input_rows
        self.feature_dim = 12
        self.proj = nn.Sequential(
            nn.Linear(self.feature_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.row_pos = nn.Parameter(torch.randn(1, input_rows, embed_dim) * 0.02)

    def _extract_statistics(self, signal: torch.Tensor) -> torch.Tensor:
        eps = 1e-6
        mean = signal.mean(dim=-1)
        centered = signal - mean.unsqueeze(-1)
        std = centered.pow(2).mean(dim=-1).sqrt().clamp_min(eps)
        rms = signal.pow(2).mean(dim=-1).sqrt().clamp_min(eps)
        energy = signal.pow(2).sum(dim=-1)
        abs_mean = signal.abs().mean(dim=-1).clamp_min(eps)
        peak = signal.abs().amax(dim=-1)
        peak_to_peak = signal.amax(dim=-1) - signal.amin(dim=-1)
        crest = peak / rms
        impulse = peak / abs_mean
        shape = rms / abs_mean
        margin = peak / signal.abs().sqrt().mean(dim=-1).pow(2).clamp_min(eps)
        skewness = centered.pow(3).mean(dim=-1) / std.pow(3)
        kurtosis = centered.pow(4).mean(dim=-1) / std.pow(4)
        zero_cross = (signal[..., 1:] * signal[..., :-1] < 0).float().mean(dim=-1)
        features = torch.stack(
            [
                peak,
                energy,
                mean,
                std,
                rms,
                peak_to_peak,
                crest,
                impulse,
                shape,
                margin,
                skewness,
                kurtosis + zero_cross * 0.0,
            ],
            dim=-1,
        )
        return features

    def forward(self, signal: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        stats = self._extract_statistics(signal)
        tokens = self.proj(stats)
        if tokens.size(1) == self.row_pos.size(1):
            tokens = tokens + self.row_pos
        return tokens, stats


class STFTEncoder(nn.Module):
    def __init__(
        self,
        input_rows: int,
        embed_dim: int,
        tokens_per_row: int,
        n_fft: int,
        hop_length: int,
        win_length: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_rows = input_rows
        self.tokens_per_row = max(1, tokens_per_row)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.register_buffer("window", torch.hann_window(win_length), persistent=False)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, embed_dim // 4, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(embed_dim // 4),
            nn.GELU(),
            nn.Conv2d(embed_dim // 4, embed_dim // 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def _build_spectrogram(self, signal: torch.Tensor) -> torch.Tensor:
        batch_size, row_count, width = signal.shape
        rows = signal.reshape(batch_size * row_count, width)
        spec = torch.stft(
            rows,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(rows.device),
            return_complex=True,
            center=True,
        )
        magnitude = torch.log1p(torch.abs(spec))
        return magnitude.unsqueeze(1)

    def forward(self, signal: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, row_count, _ = signal.shape
        spectrogram = self._build_spectrogram(signal)
        feature_map = self.cnn(spectrogram)
        feature_map = feature_map.flatten(2)
        feature_map = F.adaptive_avg_pool1d(feature_map, self.tokens_per_row)
        tokens = feature_map.transpose(1, 2).reshape(batch_size, row_count * self.tokens_per_row, -1)
        return tokens, spectrogram.reshape(batch_size, row_count, *spectrogram.shape[1:])


class GAFEncoder(nn.Module):
    def __init__(self, input_rows: int, embed_dim: int, tokens_per_row: int, gaf_size: int, dropout: float) -> None:
        super().__init__()
        self.input_rows = input_rows
        self.tokens_per_row = max(1, tokens_per_row)
        self.gaf_size = gaf_size
        self.cnn = nn.Sequential(
            nn.Conv2d(1, embed_dim // 4, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(embed_dim // 4),
            nn.GELU(),
            nn.Conv2d(embed_dim // 4, embed_dim // 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def _gramian_angular_field(self, signal: torch.Tensor) -> torch.Tensor:
        batch_size, row_count, width = signal.shape
        rows = signal.reshape(batch_size * row_count, 1, width)
        pooled = F.adaptive_avg_pool1d(rows, self.gaf_size).squeeze(1)
        min_val = pooled.amin(dim=-1, keepdim=True)
        max_val = pooled.amax(dim=-1, keepdim=True)
        scaled = 2.0 * (pooled - min_val) / (max_val - min_val + 1e-6) - 1.0
        scaled = scaled.clamp(-0.999999, 0.999999)
        phase = torch.acos(scaled)
        gaf = torch.cos(phase.unsqueeze(2) + phase.unsqueeze(1))
        return gaf.unsqueeze(1)

    def forward(self, signal: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, row_count, _ = signal.shape
        gaf = self._gramian_angular_field(signal)
        feature_map = self.cnn(gaf)
        feature_map = feature_map.flatten(2)
        feature_map = F.adaptive_avg_pool1d(feature_map, self.tokens_per_row)
        tokens = feature_map.transpose(1, 2).reshape(batch_size, row_count * self.tokens_per_row, -1)
        return tokens, gaf.reshape(batch_size, row_count, *gaf.shape[1:])


class LocationImageEncoder(nn.Module):
    def __init__(self, output_dim: int, dropout: float) -> None:
        super().__init__()
        mid_dim = max(output_dim // 2, 64)
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, mid_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(mid_dim),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(mid_dim, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.head(self.cnn(images))


class PipeMMTL(nn.Module):
    def __init__(
        self,
        num_event_classes: int,
        num_location_classes: int,
        input_length: int = 10000,
        input_rows: int = 6,
        embed_dim: int = 128,
        fusion_dim: int = 256,
        time_tokens: int = 6,
        freq_tokens: int = 48,
        gaf_tokens: int = 48,
        gaf_size: int = 64,
        prior_tokens: int = 8,
        num_heads: int = 4,
        dropout: float = 0.1,
        stft_n_fft: int = 256,
        stft_hop_length: int = 128,
        stft_win_length: int = 256,
        location_image_size: int = 224,
        location_time_weight: float = 0.35,
        location_stft_weight: float = 2.0,
        location_gaf_weight: float = 1.0,
        location_image_weight: float = 1.0,
        return_auxiliary: bool = False,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads.")

        self.num_event_classes = num_event_classes
        self.num_location_classes = num_location_classes
        self.input_length = input_length
        self.input_rows = input_rows
        self.embed_dim = embed_dim
        self.location_image_size = location_image_size
        self.location_time_weight = float(location_time_weight)
        self.location_stft_weight = float(location_stft_weight)
        self.location_gaf_weight = float(location_gaf_weight)
        self.location_image_weight = float(location_image_weight)
        self.return_auxiliary = return_auxiliary

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

        self.event_fusion = nn.Sequential(
            nn.LayerNorm(embed_dim * 3),
            nn.Linear(embed_dim * 3, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.location_fusion = nn.Sequential(
            nn.LayerNorm(embed_dim * 3),
            nn.Linear(embed_dim * 3, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.event_tower = nn.Sequential(
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
        self.location_tower = nn.Sequential(
            nn.LayerNorm(fusion_dim * 2),
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.event_head = nn.Linear(fusion_dim, num_event_classes)
        self.location_head = nn.Linear(fusion_dim, num_location_classes)
        self.location_image_aux_head = nn.Linear(fusion_dim, num_location_classes)

    def _extract_raw_matrix(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim == 2:
            signal = inputs.unsqueeze(1)
        elif inputs.ndim == 3:
            signal = inputs
        elif inputs.ndim == 4:
            signal = inputs[:, 0, :, :]
        else:
            raise ValueError(f"Unsupported input shape {tuple(inputs.shape)} for PipeMMTL.")
        return signal.float()

    def _build_location_image(self, signal: torch.Tensor) -> torch.Tensor:
        batch_size, row_count, _ = signal.shape
        spectrogram = self.stft_encoder._build_spectrogram(signal).squeeze(1)
        freq_bins, time_bins = spectrogram.shape[-2], spectrogram.shape[-1]
        spectrogram = spectrogram.reshape(batch_size, row_count, freq_bins, time_bins)
        merged = spectrogram.permute(0, 2, 1, 3).reshape(batch_size, 1, freq_bins * row_count, time_bins)
        merged = F.interpolate(
            merged,
            size=(self.location_image_size, self.location_image_size),
            mode="bilinear",
            align_corners=False,
        )
        min_val = merged.amin(dim=(-2, -1), keepdim=True)
        max_val = merged.amax(dim=(-2, -1), keepdim=True)
        merged = (merged - min_val) / (max_val - min_val + 1e-6)
        return merged.repeat(1, 3, 1, 1)

    def forward_location_image(self, images: torch.Tensor) -> torch.Tensor:
        image_feature = self.location_image_proj(self.location_image_encoder(images.float()))
        return self.location_image_aux_head(image_feature)

    def forward(self, inputs: torch.Tensor) -> dict[str, Any]:
        signal = self._extract_raw_matrix(inputs)

        time_tokens, time_stats = self.time_encoder(signal)
        stft_tokens, stft_maps = self.stft_encoder(signal)
        gaf_tokens, gaf_maps = self.gaf_encoder(signal)

        prior = self.prior_tokens.expand(signal.size(0), -1, -1)
        spatial_tokens = self.spatial_cross_attention(gaf_tokens, torch.cat([time_tokens, prior], dim=1))
        frequency_tokens = self.frequency_cross_attention(stft_tokens, torch.cat([time_tokens, spatial_tokens], dim=1))
        cross_summary = torch.cat([spatial_tokens.mean(dim=1), frequency_tokens.mean(dim=1)], dim=-1)
        time_tokens = self.temporal_self_attention(time_tokens + self.cross_to_time(cross_summary).unsqueeze(1))

        time_summary = time_tokens.mean(dim=1)
        frequency_summary = frequency_tokens.mean(dim=1)
        spatial_summary = spatial_tokens.mean(dim=1)

        event_feature_in = self.event_fusion(
            torch.cat(
                [
                    time_summary,
                    frequency_summary,
                    spatial_summary,
                ],
                dim=-1,
            )
        )
        location_feature_in = self.location_fusion(
            torch.cat(
                [
                    time_summary * self.location_time_weight,
                    frequency_summary * self.location_stft_weight,
                    spatial_summary * self.location_gaf_weight,
                ],
                dim=-1,
            )
        )

        event_feature = self.event_tower(event_feature_in)
        event_logits = self.event_head(event_feature)

        location_image = self._build_location_image(signal)
        location_image_feature = self.location_image_proj(self.location_image_encoder(location_image))
        location_feature = self.location_tower(
            torch.cat([location_feature_in, location_image_feature * self.location_image_weight], dim=-1)
        )
        location_logits = self.location_head(location_feature)

        outputs: dict[str, Any] = {
            "event_type": event_logits,
            "location": location_logits,
            "distance_cls": location_logits,
        }
        if self.return_auxiliary:
            outputs.update(
                {
                    "event_fused_feature": event_feature_in,
                    "location_fused_feature": location_feature_in,
                    "time_tokens": time_tokens,
                    "time_stats": time_stats,
                    "frequency_tokens": frequency_tokens,
                    "stft_maps": stft_maps,
                    "spatial_tokens": spatial_tokens,
                    "gaf_maps": gaf_maps,
                    "location_image": location_image,
                    "location_image_feature": location_image_feature,
                }
            )
        return outputs


def build_pipe_mmtl(**kwargs: Any) -> PipeMMTL:
    return PipeMMTL(**kwargs)
