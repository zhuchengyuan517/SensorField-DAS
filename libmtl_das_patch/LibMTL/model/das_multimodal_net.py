from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock1D(nn.Module):
    """Simple 1D residual block for temporal feature extraction."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.act = nn.GELU()
        self.shortcut = (
            nn.Identity()
            if in_channels == out_channels and stride == 1
            else nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        x = self.act(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = self.act(x + identity)
        return x


class ResidualBlock2D(nn.Module):
    """Simple 2D residual block for spectrogram branches."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()
        self.shortcut = (
            nn.Identity()
            if in_channels == out_channels and stride == 1
            else nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        x = self.act(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = self.act(x + identity)
        return x


class TemporalBranch(nn.Module):
    """Raw 1D signal branch: 1D-CNN + residual blocks + GAP."""

    def __init__(self, in_channels: int = 1, hidden_dims: tuple[int, int, int] = (32, 64, 128)) -> None:
        super().__init__()
        c1, c2, c3 = hidden_dims
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, c1, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(c1),
            nn.GELU(),
        )
        self.res1 = ResidualBlock1D(c1, c2, stride=2)
        self.res2 = ResidualBlock1D(c2, c3, stride=2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.out_dim = c3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1, L]
        x = self.stem(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.pool(x).squeeze(-1)  # [B, C]
        return x


class SpectralBranch(nn.Module):
    """STFT 2D branch: 2D-CNN + residual blocks + GAP."""

    def __init__(self, in_channels: int = 1, hidden_dims: tuple[int, int, int] = (32, 64, 128)) -> None:
        super().__init__()
        c1, c2, c3 = hidden_dims
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(c1),
            nn.GELU(),
        )
        self.res1 = ResidualBlock2D(c1, c2, stride=2)
        self.res2 = ResidualBlock2D(c2, c3, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.out_dim = c3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1, H, W]
        x = self.stem(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.pool(x).flatten(1)  # [B, C]
        return x


class TransformerMLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, dropout: float = 0.1) -> None:
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = TransformerMLP(dim, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, D]
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class PatchEmbedding(nn.Module):
    """Patch embedding for a lightweight ViT-style GAF branch."""

    def __init__(self, in_channels: int = 1, embed_dim: int = 128, patch_size: int = 8) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1, H, W]
        x = self.proj(x)  # [B, D, H/P, W/P]
        x = x.flatten(2).transpose(1, 2)  # [B, N, D]
        return x


class GAFBranch(nn.Module):
    """Lightweight ViT-style branch for GAF images."""

    def __init__(
        self,
        in_channels: int = 1,
        embed_dim: int = 128,
        patch_size: int = 8,
        num_heads: int = 4,
        depth: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.patch_embed = PatchEmbedding(in_channels=in_channels, embed_dim=embed_dim, patch_size=patch_size)
        self.blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads=num_heads, dropout=dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.out_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1, H, W]
        x = self.patch_embed(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        x = x.mean(dim=1)  # [B, D]
        return x


class MLPProjection(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SharedPrivateDecomposer(nn.Module):
    """Decompose aligned feature into shared and private parts."""

    def __init__(self, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.shared_proj = MLPProjection(d_model, d_model, dropout=dropout)
        self.private_proj = MLPProjection(d_model, d_model, dropout=dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [B, D]
        shared = self.shared_proj(x)   # [B, D]
        private = self.private_proj(x)  # [B, D]
        return shared, private


class CrossModalAttention(nn.Module):
    """Single-token cross-modal attention with residual refinement."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(d_model)
        self.context_norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.out_norm = nn.LayerNorm(d_model)
        self.mlp = TransformerMLP(d_model, mlp_ratio=2.0, dropout=dropout)

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # query: [B, 1, D], context: [B, N_ctx, D]
        attn_out, _ = self.attn(
            self.query_norm(query),
            self.context_norm(context),
            self.context_norm(context),
            need_weights=False,
        )
        x = query + attn_out
        x = x + self.mlp(self.out_norm(x))
        return x


@dataclass
class DASLossOutput:
    total_loss: torch.Tensor
    cls_loss: torch.Tensor
    disentanglement_loss: torch.Tensor
    consistency_loss: torch.Tensor


class FocalLoss(nn.Module):
    """Multi-class focal loss."""

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        reduction: str = "mean",
        ignore_index: int = -100,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        if alpha is not None and not torch.is_tensor(alpha):
            alpha = torch.tensor(alpha, dtype=torch.float32)
        if torch.is_tensor(alpha):
            self.register_buffer("alpha", alpha.float(), persistent=False)
        else:
            self.alpha = None
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: [B, C], targets: [B]
        valid_mask = targets != self.ignore_index
        if not valid_mask.any():
            return logits.sum() * 0.0

        logits = logits[valid_mask]
        targets = targets[valid_mask]

        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        target_log_probs = log_probs.gather(dim=1, index=targets.unsqueeze(1)).squeeze(1)
        target_probs = probs.gather(dim=1, index=targets.unsqueeze(1)).squeeze(1)
        focal_weight = (1.0 - target_probs).pow(self.gamma)

        if self.alpha is not None:
            alpha = self.alpha.to(logits.device)
            alpha_weight = alpha.gather(dim=0, index=targets)
            loss = -alpha_weight * focal_weight * target_log_probs
        else:
            loss = -focal_weight * target_log_probs

        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        return loss.mean()


def shared_private_disentanglement_loss(
    shared_features: Dict[str, torch.Tensor],
    private_features: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """Encourage each modality's shared/private features to be orthogonal."""

    losses = []
    for key in shared_features.keys():
        shared = F.normalize(shared_features[key], dim=-1)
        private = F.normalize(private_features[key], dim=-1)
        cosine = (shared * private).sum(dim=-1)
        losses.append(cosine.pow(2).mean())
    return torch.stack(losses).mean()


def shared_consistency_loss(shared_features: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Encourage shared features from different modalities to stay semantically close."""

    names = list(shared_features.keys())
    pair_losses = []
    for idx in range(len(names)):
        for jdx in range(idx + 1, len(names)):
            a = F.normalize(shared_features[names[idx]], dim=-1)
            b = F.normalize(shared_features[names[jdx]], dim=-1)
            pair_losses.append((1.0 - F.cosine_similarity(a, b, dim=-1)).mean())
    if not pair_losses:
        device = next(iter(shared_features.values())).device
        return torch.zeros((), device=device)
    return torch.stack(pair_losses).mean()


def compute_total_loss(
    outputs: Dict[str, Dict[str, torch.Tensor] | torch.Tensor],
    targets: torch.Tensor,
    cls_criterion: nn.Module,
    lambda1: float = 0.1,
    lambda2: float = 0.1,
) -> DASLossOutput:
    logits = outputs["logits"]  # [B, 4]
    cls_loss = cls_criterion(logits, targets)
    dec_loss = shared_private_disentanglement_loss(outputs["shared_features"], outputs["private_features"])
    cons_loss = shared_consistency_loss(outputs["shared_features"])
    total = cls_loss + lambda1 * dec_loss + lambda2 * cons_loss
    return DASLossOutput(
        total_loss=total,
        cls_loss=cls_loss,
        disentanglement_loss=dec_loss,
        consistency_loss=cons_loss,
    )


class DASMultiModalNet(nn.Module):
    """Multimodal DAS classifier with shared-private decomposition and cross-modal attention."""

    def __init__(
        self,
        num_classes: int = 4,
        d_model: int = 128,
        tf_branch_dim: int = 128,
        temporal_branch_dim: int = 128,
        gaf_patch_size: int = 8,
        gaf_depth: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.d_model = d_model

        self.temporal_branch = TemporalBranch(in_channels=1, hidden_dims=(32, 64, temporal_branch_dim))
        self.spectral_branch = SpectralBranch(in_channels=1, hidden_dims=(32, 64, tf_branch_dim))
        self.gaf_branch = GAFBranch(
            in_channels=1,
            embed_dim=d_model,
            patch_size=gaf_patch_size,
            num_heads=num_heads,
            depth=gaf_depth,
            dropout=dropout,
        )

        self.temporal_align = nn.Linear(self.temporal_branch.out_dim, d_model)
        self.spectral_align = nn.Linear(self.spectral_branch.out_dim, d_model)
        self.gaf_align = nn.Linear(self.gaf_branch.out_dim, d_model)

        self.temporal_decomposer = SharedPrivateDecomposer(d_model, dropout=dropout)
        self.spectral_decomposer = SharedPrivateDecomposer(d_model, dropout=dropout)
        self.gaf_decomposer = SharedPrivateDecomposer(d_model, dropout=dropout)

        self.temporal_cross_attn = CrossModalAttention(d_model, num_heads=num_heads, dropout=dropout)
        self.spectral_cross_attn = CrossModalAttention(d_model, num_heads=num_heads, dropout=dropout)
        self.gaf_cross_attn = CrossModalAttention(d_model, num_heads=num_heads, dropout=dropout)

        fusion_input_dim = d_model * 6
        self.fusion_head = nn.Sequential(
            nn.LayerNorm(fusion_input_dim),
            nn.Linear(fusion_input_dim, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(
        self,
        raw_signal: torch.Tensor,
        stft_spectrogram: torch.Tensor,
        gaf_image: torch.Tensor,
    ) -> Dict[str, Dict[str, torch.Tensor] | torch.Tensor]:
        # raw_signal: [B, 1, L]
        # stft_spectrogram: [B, 1, H, W]
        # gaf_image: [B, 1, H, W]

        # Branch-specific global descriptors.
        temporal_feat = self.temporal_branch(raw_signal)          # [B, C_t]
        spectral_feat = self.spectral_branch(stft_spectrogram)    # [B, C_s]
        gaf_feat = self.gaf_branch(gaf_image)                     # [B, C_g]

        # Align all modalities into the same latent dimension d_model.
        temporal_feat = self.temporal_align(temporal_feat)        # [B, D]
        spectral_feat = self.spectral_align(spectral_feat)        # [B, D]
        gaf_feat = self.gaf_align(gaf_feat)                       # [B, D]

        # Shared-private decomposition.
        temporal_shared, temporal_private = self.temporal_decomposer(temporal_feat)  # [B, D], [B, D]
        spectral_shared, spectral_private = self.spectral_decomposer(spectral_feat)  # [B, D], [B, D]
        gaf_shared, gaf_private = self.gaf_decomposer(gaf_feat)                      # [B, D], [B, D]

        # Convert each shared feature into a single attention token.
        temporal_query = temporal_shared.unsqueeze(1)             # [B, 1, D]
        spectral_query = spectral_shared.unsqueeze(1)             # [B, 1, D]
        gaf_query = gaf_shared.unsqueeze(1)                       # [B, 1, D]

        temporal_context = torch.stack([spectral_shared, gaf_shared], dim=1)  # [B, 2, D]
        spectral_context = torch.stack([temporal_shared, gaf_shared], dim=1)  # [B, 2, D]
        gaf_context = torch.stack([temporal_shared, spectral_shared], dim=1)  # [B, 2, D]

        # Cross-modal shared-feature fusion.
        temporal_shared_enhanced = self.temporal_cross_attn(temporal_query, temporal_context).squeeze(1)  # [B, D]
        spectral_shared_enhanced = self.spectral_cross_attn(spectral_query, spectral_context).squeeze(1)  # [B, D]
        gaf_shared_enhanced = self.gaf_cross_attn(gaf_query, gaf_context).squeeze(1)                      # [B, D]

        # Shared/private fusion.
        enhanced_shared = torch.cat(
            [temporal_shared_enhanced, spectral_shared_enhanced, gaf_shared_enhanced],
            dim=-1,
        )  # [B, 3D]
        private_concat = torch.cat(
            [temporal_private, spectral_private, gaf_private],
            dim=-1,
        )  # [B, 3D]
        fusion_input = torch.cat([enhanced_shared, private_concat], dim=-1)  # [B, 6D]
        fused_feature = self.fusion_head(fusion_input)                        # [B, D]
        logits = self.classifier(fused_feature)                               # [B, 4]

        return {
            "logits": logits,
            "aligned_features": {
                "temporal": temporal_feat,
                "spectral": spectral_feat,
                "gaf": gaf_feat,
            },
            "shared_features": {
                "temporal": temporal_shared_enhanced,
                "spectral": spectral_shared_enhanced,
                "gaf": gaf_shared_enhanced,
            },
            "private_features": {
                "temporal": temporal_private,
                "spectral": spectral_private,
                "gaf": gaf_private,
            },
            "fusion_feature": fused_feature,
        }


def build_das_multimodal_net(**kwargs) -> DASMultiModalNet:
    return DASMultiModalNet(**kwargs)


def _demo() -> None:
    torch.manual_seed(7)

    batch_size = 4
    signal_length = 4096
    height = 128
    width = 128

    # Random demo inputs.
    raw_signal = torch.randn(batch_size, 1, signal_length)       # [B, 1, L]
    stft_spectrogram = torch.randn(batch_size, 1, height, width) # [B, 1, H, W]
    gaf_image = torch.randn(batch_size, 1, height, width)        # [B, 1, H, W]
    targets = torch.randint(low=0, high=4, size=(batch_size,))   # [B]

    model = DASMultiModalNet(
        num_classes=4,
        d_model=128,
        temporal_branch_dim=128,
        tf_branch_dim=128,
        gaf_patch_size=8,
        gaf_depth=2,
        num_heads=4,
        dropout=0.1,
    )
    outputs = model(raw_signal, stft_spectrogram, gaf_image)

    criterion = FocalLoss(gamma=2.0)
    loss_dict = compute_total_loss(outputs, targets, criterion, lambda1=0.1, lambda2=0.1)

    print("=== DASMultiModalNet demo ===")
    print("raw_signal shape:", tuple(raw_signal.shape))
    print("stft_spectrogram shape:", tuple(stft_spectrogram.shape))
    print("gaf_image shape:", tuple(gaf_image.shape))
    print("logits shape:", tuple(outputs["logits"].shape))
    print("temporal aligned feature shape:", tuple(outputs["aligned_features"]["temporal"].shape))
    print("spectral aligned feature shape:", tuple(outputs["aligned_features"]["spectral"].shape))
    print("gaf aligned feature shape:", tuple(outputs["aligned_features"]["gaf"].shape))
    print("fusion feature shape:", tuple(outputs["fusion_feature"].shape))
    print("classification loss:", float(loss_dict.cls_loss.item()))
    print("disentanglement loss:", float(loss_dict.disentanglement_loss.item()))
    print("consistency loss:", float(loss_dict.consistency_loss.item()))
    print("total loss:", float(loss_dict.total_loss.item()))


if __name__ == "__main__":
    _demo()
