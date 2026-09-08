from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


ALLOWED_VIEW_NAMES = ("raw", "stf", "gaf")


def _first_available_tensor(batch: dict[str, torch.Tensor | None]) -> torch.Tensor:
    for value in batch.values():
        if torch.is_tensor(value):
            return value
    raise ValueError("No tensor payload found in batch dictionary.")


def parse_enabled_views(enabled_views: str | tuple[str, ...] | list[str]) -> tuple[str, ...]:
    if isinstance(enabled_views, str):
        requested = [item.strip().lower() for item in enabled_views.split(",") if item.strip()]
    else:
        requested = [str(item).strip().lower() for item in enabled_views if str(item).strip()]
    if not requested:
        raise ValueError("enabled_views must include at least one view.")
    invalid = sorted({item for item in requested if item not in ALLOWED_VIEW_NAMES})
    if invalid:
        raise ValueError(f"Unsupported view names: {invalid}. Allowed views: {ALLOWED_VIEW_NAMES}")

    requested_set = set(requested)
    ordered = tuple(name for name in ALLOWED_VIEW_NAMES if name in requested_set)
    if len(ordered) != len(requested_set):
        raise ValueError(f"Duplicate or invalid enabled_views specification: {enabled_views}")
    return ordered


def build_evidence_bank(
    shared_anchors: torch.Tensor,
    complementary: dict[str, torch.Tensor],
    active_view_names: tuple[str, ...],
    use_complementary: bool,
    view_reliability: dict[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, tuple[str, ...], torch.Tensor]:
    evidence_names = ["shared"]
    evidence_tensors = [shared_anchors.mean(dim=1)]
    batch_size = shared_anchors.size(0)
    shared_reliability = shared_anchors.new_ones(batch_size, 1)
    evidence_reliability = [shared_reliability]
    if use_complementary:
        for name in active_view_names:
            evidence_names.append(f"{name}_complement")
            complement_summary = complementary[name].mean(dim=1)
            current_reliability = shared_reliability
            if view_reliability is not None and name in view_reliability:
                current_reliability = view_reliability[name].to(complement_summary.dtype)
                complement_summary = complement_summary * current_reliability
            evidence_tensors.append(complement_summary)
            evidence_reliability.append(current_reliability)
    evidence_bank = torch.stack(evidence_tensors, dim=1)
    evidence_reliability_tensor = torch.cat(evidence_reliability, dim=1)
    return evidence_bank, tuple(evidence_names), evidence_reliability_tensor


class FeedForwardBlock(nn.Module):
    def __init__(self, hidden_dim: int, mlp_ratio: float = 4.0, dropout: float = 0.1) -> None:
        super().__init__()
        inner_dim = max(int(hidden_dim * mlp_ratio), hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.net(tokens)


class CrossAttentionBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.context_norm = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.ffn = FeedForwardBlock(hidden_dim=hidden_dim, mlp_ratio=2.0, dropout=dropout)

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(
            self.query_norm(query),
            self.context_norm(context),
            self.context_norm(context),
            need_weights=False,
        )
        query = query + attn_out
        query = query + self.ffn(self.out_norm(query))
        return query


class RawSignalEncoder1D(nn.Module):
    """Encode a raw 1D temporal signal into tokens of shape [B, N_raw, D]."""

    def __init__(self, hidden_dim: int, num_tokens: int = 16, dropout: float = 0.1, in_channels: int = 1) -> None:
        super().__init__()
        self.num_tokens = max(int(num_tokens), 1)
        self.in_channels = max(int(in_channels), 1)
        mid_dim = max(hidden_dim // 2, 32)
        self.backbone = nn.Sequential(
            nn.Conv1d(self.in_channels, mid_dim, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(mid_dim),
            nn.GELU(),
            nn.Conv1d(mid_dim, hidden_dim, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, raw_signal: torch.Tensor) -> torch.Tensor:
        # raw_signal: [B, 1, L]
        feature_map = self.backbone(raw_signal)
        feature_map = F.adaptive_avg_pool1d(feature_map, self.num_tokens)
        tokens = feature_map.transpose(1, 2)
        return self.out_norm(tokens)


class ImageTokenEncoder2D(nn.Module):
    """Encode a 2D map or image into tokens of shape [B, N_img, D]."""

    def __init__(self, hidden_dim: int, num_tokens: int = 16, dropout: float = 0.1) -> None:
        super().__init__()
        self.num_tokens = max(int(num_tokens), 1)
        mid_dim = max(hidden_dim // 2, 32)
        self.backbone = nn.Sequential(
            nn.Conv2d(1, mid_dim, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(mid_dim),
            nn.GELU(),
            nn.Conv2d(mid_dim, hidden_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        # image: [B, 1, H, W]
        feature_map = self.backbone(image)
        feature_map = feature_map.flatten(2)
        feature_map = F.adaptive_avg_pool1d(feature_map, self.num_tokens)
        tokens = feature_map.transpose(1, 2)
        return self.out_norm(tokens)


class FAC(nn.Module):
    """Field-Anchor Complementation.

    Inputs:
    - view_tokens[name]: [B, N_v, D] for the currently enabled views

    Outputs:
    - shared_anchors: [B, K, D]
    - complementary[name]: [B, K, D]
    - agreement_weights: [B, V, K] where V is the active view count
    """

    def __init__(
        self,
        view_names: tuple[str, ...],
        hidden_dim: int,
        num_anchors: int,
        num_heads: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.view_names = view_names
        self.num_anchors = int(num_anchors)
        self.hidden_dim = int(hidden_dim)
        self.anchor_bank = nn.Parameter(torch.randn(self.num_anchors, self.hidden_dim) * 0.02)
        self.anchor_modulation = nn.Sequential(
            nn.LayerNorm(self.hidden_dim * len(self.view_names)),
            nn.Linear(self.hidden_dim * len(self.view_names), self.num_anchors * self.hidden_dim),
            nn.Tanh(),
        )
        self.cross_attention = nn.ModuleDict(
            {
                name: CrossAttentionBlock(hidden_dim=self.hidden_dim, num_heads=num_heads, dropout=dropout)
                for name in self.view_names
            }
        )

    def forward(self, view_tokens: dict[str, torch.Tensor]) -> dict[str, Any]:
        active_view_names = tuple(view_tokens.keys())
        if not active_view_names:
            raise ValueError("FAC requires at least one active view.")

        summaries = [view_tokens[name].mean(dim=1) for name in active_view_names]
        conditioning_source = torch.stack(summaries, dim=1).mean(dim=1)
        if len(active_view_names) != len(self.view_names):
            conditioning = conditioning_source.repeat(1, len(self.view_names))
        else:
            conditioning = torch.cat(summaries, dim=-1)
        anchor_delta = self.anchor_modulation(conditioning).view(-1, self.num_anchors, self.hidden_dim)
        anchors = self.anchor_bank.unsqueeze(0) + anchor_delta

        anchor_views = {
            name: self.cross_attention[name](anchors, view_tokens[name])
            for name in active_view_names
        }
        normalized = {name: F.normalize(tokens, dim=-1) for name, tokens in anchor_views.items()}

        if len(active_view_names) == 1:
            agreement_weights = torch.ones(
                anchor_views[active_view_names[0]].size(0),
                1,
                self.num_anchors,
                device=anchors.device,
                dtype=anchors.dtype,
            )
        else:
            agreement_terms = []
            for name in active_view_names:
                pair_scores = []
                for other_name in active_view_names:
                    if other_name == name:
                        continue
                    pair_scores.append((normalized[name] * normalized[other_name]).sum(dim=-1))
                agreement_terms.append(torch.stack(pair_scores, dim=1).mean(dim=1))
            agreement_scores = torch.stack(agreement_terms, dim=1)
            agreement_weights = torch.softmax(agreement_scores, dim=1)

        shared_anchors = 0.0
        for view_index, name in enumerate(active_view_names):
            shared_anchors = shared_anchors + agreement_weights[:, view_index].unsqueeze(-1) * anchor_views[name]

        complementary: dict[str, torch.Tensor] = {}
        shared_norm_sq = shared_anchors.pow(2).sum(dim=-1, keepdim=True).clamp_min(1e-6)
        for name in active_view_names:
            coeff = (anchor_views[name] * shared_anchors).sum(dim=-1, keepdim=True) / shared_norm_sq
            complementary[name] = anchor_views[name] - coeff * shared_anchors

        alignment_loss = torch.stack(
            [
                (1.0 - F.cosine_similarity(anchor_views[name], shared_anchors, dim=-1)).mean()
                for name in active_view_names
            ]
        ).mean()

        decorrelation_terms = []
        for name in active_view_names:
            complement_cos = F.cosine_similarity(complementary[name], shared_anchors, dim=-1)
            decorrelation_terms.append(complement_cos.pow(2).mean())
        for view_index, name in enumerate(active_view_names):
            for other_name in active_view_names[view_index + 1 :]:
                pair_cos = F.cosine_similarity(complementary[name], complementary[other_name], dim=-1)
                decorrelation_terms.append(pair_cos.pow(2).mean())
        complement_decorrelation_loss = torch.stack(decorrelation_terms).mean()

        return {
            "anchor_views": anchor_views,
            "shared_anchors": shared_anchors,
            "complementary": complementary,
            "agreement_weights": agreement_weights,
            "alignment_loss": alignment_loss,
            "complement_decorrelation_loss": complement_decorrelation_loss,
        }


class TAEF(nn.Module):
    """Task-Adaptive Evidence Fusion.

    Inputs:
    - shared_anchors: [B, K, D]
    - complementary[name]: [B, K, D]

    Outputs:
    - task_tokens: [B, T, D]
    - alpha: [B, T, E] where E depends on the enabled evidence sources
    """

    def __init__(
        self,
        task_names: tuple[str, ...],
        hidden_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.task_names = task_names
        self.hidden_dim = int(hidden_dim)
        self.evidence_norm = nn.LayerNorm(hidden_dim)
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.task_queries = nn.Parameter(torch.randn(len(task_names), hidden_dim) * 0.02)
        self.task_ffn = FeedForwardBlock(hidden_dim=hidden_dim, mlp_ratio=2.0, dropout=dropout)
        self.task_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        shared_anchors: torch.Tensor,
        complementary: dict[str, torch.Tensor],
        active_view_names: tuple[str, ...],
        use_complementary: bool,
        view_reliability: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        evidence_bank, evidence_names, evidence_reliability = build_evidence_bank(
            shared_anchors=shared_anchors,
            complementary=complementary,
            active_view_names=active_view_names,
            use_complementary=use_complementary,
            view_reliability=view_reliability,
        )
        evidence_bank = self.evidence_norm(evidence_bank)

        batch_size = evidence_bank.size(0)
        task_queries = self.task_queries.unsqueeze(0).expand(batch_size, -1, -1)
        query = self.query_proj(task_queries)
        key = self.key_proj(evidence_bank)
        value = self.value_proj(evidence_bank)
        scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.hidden_dim)
        evidence_mask = evidence_reliability <= 1e-6
        scores = scores.masked_fill(evidence_mask.unsqueeze(1), -1e4)
        alpha = torch.softmax(scores, dim=-1)
        alpha = alpha * (~evidence_mask).to(alpha.dtype).unsqueeze(1)
        alpha = alpha / alpha.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        task_tokens = torch.matmul(alpha, value)
        task_tokens = task_tokens + task_queries
        task_tokens = task_tokens + self.task_ffn(self.task_norm(task_tokens))

        diversity_loss = task_tokens.new_zeros(())
        if alpha.size(1) > 1 and alpha.size(-1) > 1:
            pair_losses = []
            normalized_alpha = F.normalize(alpha, dim=-1)
            for task_index in range(alpha.size(1)):
                for other_index in range(task_index + 1, alpha.size(1)):
                    pair_losses.append(
                        (normalized_alpha[:, task_index] * normalized_alpha[:, other_index]).sum(dim=-1).mean()
                    )
            diversity_loss = torch.stack(pair_losses).mean()

        return {
            "task_tokens": task_tokens,
            "alpha": alpha,
            "diversity_loss": diversity_loss,
            "evidence_names": evidence_names,
            "evidence_reliability": evidence_reliability,
        }


class SimpleViewAggregator(nn.Module):
    """Simplified view aggregation used by the FAC ablation."""

    def __init__(self, hidden_dim: int, num_anchors: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_anchors = int(num_anchors)
        self.anchor_bank = nn.Parameter(torch.randn(self.num_anchors, self.hidden_dim) * 0.02)
        self.summary_proj = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.num_anchors * self.hidden_dim),
        )

    def forward(self, view_tokens: dict[str, torch.Tensor]) -> dict[str, Any]:
        active_view_names = tuple(view_tokens.keys())
        if not active_view_names:
            raise ValueError("SimpleViewAggregator requires at least one active view.")

        summaries = {name: view_tokens[name].mean(dim=1) for name in active_view_names}
        anchor_views = {
            name: summaries[name].unsqueeze(1).expand(-1, self.num_anchors, -1)
            for name in active_view_names
        }
        pooled_summary = torch.stack([summaries[name] for name in active_view_names], dim=1).mean(dim=1)
        shared_anchors = self.anchor_bank.unsqueeze(0) + self.summary_proj(pooled_summary).view(
            -1, self.num_anchors, self.hidden_dim
        )
        complementary = {name: anchor_views[name] - shared_anchors for name in active_view_names}
        agreement_weights = shared_anchors.new_full(
            (shared_anchors.size(0), len(active_view_names), self.num_anchors),
            1.0 / max(len(active_view_names), 1),
        )
        zero = shared_anchors.sum() * 0.0
        return {
            "anchor_views": anchor_views,
            "shared_anchors": shared_anchors,
            "complementary": complementary,
            "agreement_weights": agreement_weights,
            "alignment_loss": zero,
            "complement_decorrelation_loss": zero,
        }


class TaskAgnosticEvidenceFusion(nn.Module):
    """Task-agnostic fusion shared by all tasks for the TAEF ablation."""

    def __init__(self, num_tasks: int, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.num_tasks = int(num_tasks)
        self.hidden_dim = int(hidden_dim)
        self.evidence_norm = nn.LayerNorm(hidden_dim)
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.shared_query = nn.Parameter(torch.randn(1, hidden_dim) * 0.02)
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_ffn = FeedForwardBlock(hidden_dim=hidden_dim, mlp_ratio=2.0, dropout=dropout)

    def forward(
        self,
        shared_anchors: torch.Tensor,
        complementary: dict[str, torch.Tensor],
        active_view_names: tuple[str, ...],
        use_complementary: bool,
        view_reliability: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        evidence_bank, evidence_names, evidence_reliability = build_evidence_bank(
            shared_anchors=shared_anchors,
            complementary=complementary,
            active_view_names=active_view_names,
            use_complementary=use_complementary,
            view_reliability=view_reliability,
        )
        evidence_bank = self.evidence_norm(evidence_bank)
        batch_size = evidence_bank.size(0)

        query_token = self.shared_query.unsqueeze(0).expand(batch_size, -1, -1)
        query = self.query_proj(query_token)
        key = self.key_proj(evidence_bank)
        value = self.value_proj(evidence_bank)
        scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.hidden_dim)
        evidence_mask = evidence_reliability <= 1e-6
        scores = scores.masked_fill(evidence_mask.unsqueeze(1), -1e4)
        alpha = torch.softmax(scores, dim=-1)
        alpha = alpha * (~evidence_mask).to(alpha.dtype).unsqueeze(1)
        alpha = alpha / alpha.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        fused_token = torch.matmul(alpha, value)
        fused_token = fused_token + query_token
        fused_token = fused_token + self.out_ffn(self.out_norm(fused_token))
        task_tokens = fused_token.expand(-1, self.num_tasks, -1).contiguous()

        return {
            "task_tokens": task_tokens,
            "alpha": alpha.expand(-1, self.num_tasks, -1),
            "diversity_loss": fused_token.sum() * 0.0,
            "evidence_names": evidence_names,
            "evidence_reliability": evidence_reliability,
        }


class GCTI(nn.Module):
    """Generalization-Consistent Task Interaction.

    Input:
    - task_tokens: [B, T, D]

    Output:
    - updated_task_tokens: [B, T, D]
    - relation_matrix: [B, T, T]
    """

    def __init__(self, num_tasks: int, hidden_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.num_tasks = int(num_tasks)
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.task_relation_bias = nn.Parameter(torch.zeros(self.num_tasks, self.num_tasks))

    def forward(self, task_tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        _, task_count, _ = task_tokens.shape
        query = self.q_proj(task_tokens)
        key = self.k_proj(task_tokens)
        scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.hidden_dim)
        scores = scores + self.task_relation_bias[:task_count, :task_count].unsqueeze(0)
        relation_matrix = torch.softmax(scores, dim=-1)
        interacted = torch.matmul(relation_matrix, task_tokens)
        updated_tokens = task_tokens + self.dropout(self.out_proj(interacted))
        zero = updated_tokens.sum() * 0.0
        return {
            "updated_task_tokens": updated_tokens,
            "relation_matrix": relation_matrix,
            "symmetry_loss": zero,
        }


class IdentityTaskInteraction(nn.Module):
    """No-op task interaction used by the GCTI ablation."""

    def __init__(self, num_tasks: int) -> None:
        super().__init__()
        self.num_tasks = int(num_tasks)

    def forward(self, task_tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size, task_count, _ = task_tokens.shape
        relation_matrix = torch.eye(task_count, device=task_tokens.device, dtype=task_tokens.dtype).unsqueeze(0)
        relation_matrix = relation_matrix.expand(batch_size, -1, -1)
        zero = task_tokens.sum() * 0.0
        return {
            "updated_task_tokens": task_tokens,
            "relation_matrix": relation_matrix,
            "symmetry_loss": zero,
        }


class SensorFieldM3T(nn.Module):
    """SensorField-M3T multitask model.

    Supported input formats:
    - tensor [B, 1, H, W]: existing multitask CSV loader payload
    - dict with keys {'raw', 'stft', 'gaf'}: existing MMIT-style multimodal loader
    - dict with key {'input_data'}: explicit raw matrix payload
    """

    def __init__(
        self,
        num_event_classes: int | None = None,
        num_location_classes: int | None = None,
        task_output_dims: dict[str, int] | None = None,
        hidden_dim: int = 128,
        num_anchors: int = 16,
        num_heads: int = 4,
        raw_tokens: int = 16,
        raw_in_channels: int = 1,
        stf_tokens: int = 16,
        gaf_tokens: int = 16,
        stf_size: int = 128,
        gaf_size: int = 64,
        stft_n_fft: int = 256,
        stft_hop_length: int = 128,
        stft_win_length: int = 256,
        fac_loss_weight: float = 0.1,
        taef_loss_weight: float = 0.0,
        gcti_loss_weight: float = 0.01,
        view_drop_prob: float = 0.3,
        enable_view_consistency: bool = True,
        disable_fac: bool = False,
        disable_complement: bool = False,
        disable_taef: bool = False,
        disable_gcti: bool = False,
        disable_view_consistency: bool = False,
        enabled_views: str | tuple[str, ...] | list[str] = ALLOWED_VIEW_NAMES,
        view_consistency_weight: float = 0.05,
        view_noise_std: float = 0.01,
        dropout: float = 0.1,
        return_auxiliary: bool = False,
    ) -> None:
        super().__init__()
        if task_output_dims is None:
            inferred_tasks: dict[str, int] = {}
            if num_event_classes is not None:
                inferred_tasks["event_type"] = int(num_event_classes)
            if num_location_classes is not None:
                inferred_tasks["distance_cls"] = int(num_location_classes)
            if not inferred_tasks:
                raise ValueError("Provide task_output_dims or at least one task output size.")
            task_output_dims = inferred_tasks

        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")

        self.hidden_dim = int(hidden_dim)
        self.stf_size = int(stf_size)
        self.gaf_size = int(gaf_size)
        self.stft_n_fft = int(stft_n_fft)
        self.stft_hop_length = int(stft_hop_length)
        self.stft_win_length = int(stft_win_length)
        self.fac_loss_weight = float(fac_loss_weight)
        self.taef_loss_weight = float(taef_loss_weight)
        self.gcti_loss_weight = float(gcti_loss_weight)
        self.view_drop_prob = max(float(view_drop_prob), 0.0)
        self.disable_fac = bool(disable_fac)
        self.disable_complement = bool(disable_complement)
        self.disable_taef = bool(disable_taef)
        self.disable_gcti = bool(disable_gcti)
        self.enable_view_consistency = bool(enable_view_consistency) and not bool(disable_view_consistency)
        self.view_consistency_weight = float(view_consistency_weight)
        self.view_noise_std = max(float(view_noise_std), 0.0)
        self.return_auxiliary = bool(return_auxiliary)

        self.task_output_dims = dict(task_output_dims)
        self.task_names = tuple(self.task_output_dims.keys())
        self.view_names = parse_enabled_views(enabled_views)
        self.register_buffer("_stft_window", torch.hann_window(self.stft_win_length), persistent=False)

        self.raw_encoder = RawSignalEncoder1D(
            hidden_dim=self.hidden_dim,
            num_tokens=raw_tokens,
            dropout=dropout,
            in_channels=raw_in_channels,
        )
        self.stf_encoder = ImageTokenEncoder2D(hidden_dim=self.hidden_dim, num_tokens=stf_tokens, dropout=dropout)
        self.gaf_encoder = ImageTokenEncoder2D(hidden_dim=self.hidden_dim, num_tokens=gaf_tokens, dropout=dropout)
        self.fac = (
            SimpleViewAggregator(hidden_dim=self.hidden_dim, num_anchors=num_anchors, dropout=dropout)
            if self.disable_fac
            else FAC(
                view_names=self.view_names,
                hidden_dim=self.hidden_dim,
                num_anchors=num_anchors,
                num_heads=num_heads,
                dropout=dropout,
            )
        )
        self.taef = (
            TaskAgnosticEvidenceFusion(num_tasks=len(self.task_names), hidden_dim=self.hidden_dim, dropout=dropout)
            if self.disable_taef
            else TAEF(task_names=self.task_names, hidden_dim=self.hidden_dim, dropout=dropout)
        )
        self.gcti = (
            IdentityTaskInteraction(num_tasks=len(self.task_names))
            if self.disable_gcti
            else GCTI(num_tasks=len(self.task_names), hidden_dim=self.hidden_dim, num_heads=num_heads, dropout=dropout)
        )
        self.task_heads = nn.ModuleDict(
            {task_name: nn.Linear(self.hidden_dim, task_dim) for task_name, task_dim in self.task_output_dims.items()}
        )

    def _extract_raw_matrix(self, inputs: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
        if isinstance(inputs, dict):
            if "input_data" in inputs and torch.is_tensor(inputs["input_data"]):
                source = inputs["input_data"]
            elif "raw" in inputs and torch.is_tensor(inputs["raw"]):
                source = inputs["raw"]
            else:
                source = _first_available_tensor(inputs)
        else:
            source = inputs

        if source.ndim == 4:
            if source.size(1) == 1:
                return source[:, 0].float()
            return source[:, 0].float()
        if source.ndim == 3:
            return source.float()
        if source.ndim == 2:
            return source.unsqueeze(1).float()
        raise ValueError(f"Unsupported raw input shape: {tuple(source.shape)}")

    def _extract_raw_signal(self, inputs: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
        if isinstance(inputs, dict) and "raw" in inputs and torch.is_tensor(inputs["raw"]):
            raw = inputs["raw"]
            if raw.ndim == 3:
                if raw.size(1) != 1:
                    raise ValueError(f"SensorField-M3T raw view must be [B, 1, L], got {tuple(raw.shape)}")
                return raw.float()
            if raw.ndim == 4 and raw.size(1) == 1 and raw.size(2) == 1:
                return raw.squeeze(2).float()

        raw_matrix = self._extract_raw_matrix(inputs)
        return raw_matrix.flatten(start_dim=1).unsqueeze(1)

    def _extract_modality_mask(
        self,
        inputs: torch.Tensor | dict[str, torch.Tensor],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, torch.Tensor]:
        if not isinstance(inputs, dict) or "modality_mask" not in inputs:
            return {
                name: torch.ones(batch_size, 1, 1, device=device, dtype=dtype)
                for name in self.view_names
            }
        mask = inputs["modality_mask"]
        if not torch.is_tensor(mask):
            raise TypeError("modality_mask must be a tensor with columns [raw, stf, gaf].")
        mask = mask.to(device=device, dtype=dtype)
        if mask.ndim == 1:
            mask = mask.unsqueeze(0).expand(batch_size, -1)
        if mask.ndim != 2 or mask.size(1) < len(ALLOWED_VIEW_NAMES):
            raise ValueError(f"modality_mask must be [B, 3], got {tuple(mask.shape)}")
        return {
            name: mask[:, ALLOWED_VIEW_NAMES.index(name)].view(batch_size, 1, 1)
            for name in self.view_names
        }

    def _build_stf_map_from_raw(self, raw_matrix: torch.Tensor) -> torch.Tensor:
        batch_size, row_count, width = raw_matrix.shape
        flat_rows = raw_matrix.reshape(batch_size * row_count, width)
        spectrogram = torch.stft(
            flat_rows,
            n_fft=self.stft_n_fft,
            hop_length=self.stft_hop_length,
            win_length=self.stft_win_length,
            window=self._stft_window.to(flat_rows.device),
            return_complex=True,
            center=True,
        )
        magnitude = torch.log1p(torch.abs(spectrogram))
        freq_bins = magnitude.size(-2)
        time_bins = magnitude.size(-1)
        magnitude = magnitude.reshape(batch_size, row_count, freq_bins, time_bins)
        magnitude = magnitude.permute(0, 2, 1, 3).reshape(batch_size, 1, freq_bins * row_count, time_bins)
        return F.interpolate(magnitude, size=(self.stf_size, self.stf_size), mode="bilinear", align_corners=False)

    def _build_gaf_from_raw(self, raw_signal: torch.Tensor) -> torch.Tensor:
        if raw_signal.size(1) > 1:
            raw_signal = raw_signal.mean(dim=1, keepdim=True)
        pooled = F.adaptive_avg_pool1d(raw_signal, self.gaf_size).squeeze(1)
        min_val = pooled.amin(dim=-1, keepdim=True)
        max_val = pooled.amax(dim=-1, keepdim=True)
        scaled = 2.0 * (pooled - min_val) / (max_val - min_val + 1e-6) - 1.0
        scaled = scaled.clamp(-0.999999, 0.999999)
        phase = torch.acos(scaled)
        gaf = torch.cos(phase.unsqueeze(2) + phase.unsqueeze(1))
        return gaf.unsqueeze(1)

    def _extract_stf_map(
        self,
        inputs: torch.Tensor | dict[str, torch.Tensor],
        raw_matrix: torch.Tensor | None,
    ) -> torch.Tensor:
        if isinstance(inputs, dict):
            for key in ("stft", "stf"):
                if key in inputs and torch.is_tensor(inputs[key]):
                    stf = inputs[key].float()
                    if stf.ndim == 3:
                        return stf.unsqueeze(1)
                    if stf.ndim == 4:
                        return stf
        if raw_matrix is None:
            raw_matrix = self._extract_raw_matrix(inputs)
        return self._build_stf_map_from_raw(raw_matrix)

    def _extract_gaf_map(
        self,
        inputs: torch.Tensor | dict[str, torch.Tensor],
        raw_signal: torch.Tensor | None,
    ) -> torch.Tensor:
        if isinstance(inputs, dict) and "gaf" in inputs and torch.is_tensor(inputs["gaf"]):
            gaf = inputs["gaf"].float()
            if gaf.ndim == 3:
                return gaf.unsqueeze(1)
            if gaf.ndim == 4:
                return gaf
        if raw_signal is None:
            raw_signal = self._extract_raw_signal(inputs)
        return self._build_gaf_from_raw(raw_signal)

    def _encode_views(self, inputs: torch.Tensor | dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        encoded: dict[str, torch.Tensor] = {}
        raw_signal: torch.Tensor | None = None
        raw_matrix: torch.Tensor | None = None
        stf_map: torch.Tensor | None = None
        gaf_map: torch.Tensor | None = None
        modality_mask: dict[str, torch.Tensor] | None = None

        if "raw" in self.view_names:
            raw_signal = self._extract_raw_signal(inputs)
            encoded["raw"] = self.raw_encoder(raw_signal)
            modality_mask = self._extract_modality_mask(
                inputs,
                batch_size=raw_signal.size(0),
                device=raw_signal.device,
                dtype=raw_signal.dtype,
            )
            encoded["raw"] = encoded["raw"] * modality_mask["raw"]
        if "stf" in self.view_names:
            stf_map = self._extract_stf_map(inputs, raw_matrix)
            encoded["stf"] = self.stf_encoder(stf_map)
            if modality_mask is None:
                modality_mask = self._extract_modality_mask(
                    inputs,
                    batch_size=stf_map.size(0),
                    device=stf_map.device,
                    dtype=stf_map.dtype,
                )
            encoded["stf"] = encoded["stf"] * modality_mask["stf"]
            if raw_matrix is None and not (isinstance(inputs, dict) and any(key in inputs for key in ("stft", "stf"))):
                raw_matrix = self._extract_raw_matrix(inputs)
        if "gaf" in self.view_names:
            if raw_signal is None:
                raw_signal = self._extract_raw_signal(inputs) if not (isinstance(inputs, dict) and "gaf" in inputs) else None
            gaf_map = self._extract_gaf_map(inputs, raw_signal)
            encoded["gaf"] = self.gaf_encoder(gaf_map)
            if modality_mask is None:
                modality_mask = self._extract_modality_mask(
                    inputs,
                    batch_size=gaf_map.size(0),
                    device=gaf_map.device,
                    dtype=gaf_map.dtype,
                )
            encoded["gaf"] = encoded["gaf"] * modality_mask["gaf"]
            if raw_signal is None and not (isinstance(inputs, dict) and "gaf" in inputs):
                raw_signal = self._extract_raw_signal(inputs)

        if self.return_auxiliary:
            if modality_mask is not None:
                encoded["_modality_mask"] = modality_mask
            if raw_signal is not None:
                encoded["_raw_signal"] = raw_signal
            if raw_matrix is not None:
                encoded["_raw_matrix"] = raw_matrix
            if stf_map is not None:
                encoded["_stf_map"] = stf_map
            if gaf_map is not None:
                encoded["_gaf_map"] = gaf_map
        return encoded

    def _forward_heads(
        self,
        view_tokens: dict[str, torch.Tensor],
        view_presence: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, Any]:
        active_view_names = tuple(name for name in self.view_names if name in view_tokens)
        fac_outputs = self.fac({name: view_tokens[name] for name in active_view_names})
        view_reliability: dict[str, torch.Tensor] = {}
        if active_view_names:
            reliability_scale = float(len(active_view_names))
            for view_index, name in enumerate(active_view_names):
                mean_agreement = fac_outputs["agreement_weights"][:, view_index].mean(dim=-1, keepdim=True)
                reliability = (mean_agreement * reliability_scale).clamp(0.25, 2.0)
                if view_presence is not None and name in view_presence:
                    reliability = reliability * view_presence[name].squeeze(-1).to(reliability.dtype)
                view_reliability[name] = reliability
        taef_outputs = self.taef(
            fac_outputs["shared_anchors"],
            fac_outputs["complementary"],
            active_view_names=active_view_names,
            use_complementary=not self.disable_complement,
            view_reliability=view_reliability,
        )
        gcti_outputs = self.gcti(taef_outputs["task_tokens"])

        outputs: dict[str, Any] = {}
        for task_index, task_name in enumerate(self.task_names):
            outputs[task_name] = self.task_heads[task_name](gcti_outputs["updated_task_tokens"][:, task_index])
        if "distance_cls" in outputs and "location" not in outputs:
            outputs["location"] = outputs["distance_cls"]

        fac_loss = (
            fac_outputs["alignment_loss"] + fac_outputs["complement_decorrelation_loss"]
        ) * self.fac_loss_weight
        taef_loss = taef_outputs["diversity_loss"] * self.taef_loss_weight
        gcti_loss = gcti_outputs["updated_task_tokens"].sum() * 0.0
        outputs["aux_losses"] = {
            "fac_loss": fac_loss,
            "taef_loss": taef_loss,
            "gcti_loss": gcti_loss,
        }
        outputs["_active_views"] = active_view_names
        outputs["_view_reliability"] = view_reliability
        outputs["_task_representations"] = gcti_outputs["updated_task_tokens"]

        if self.return_auxiliary:
            outputs["fac_outputs"] = fac_outputs
            outputs["taef_outputs"] = taef_outputs
            outputs["gcti_outputs"] = gcti_outputs
            outputs["active_views"] = active_view_names
            outputs["view_reliability"] = view_reliability
        return outputs

    def _maybe_add_view_consistency(
        self,
        outputs: dict[str, Any],
        view_tokens: dict[str, torch.Tensor],
        task_mask: torch.Tensor | None = None,
    ) -> None:
        if not self.training:
            return
        if not self.enable_view_consistency:
            return
        if self.disable_gcti or self.view_drop_prob <= 0 or self.gcti_loss_weight <= 0:
            return
        active_view_names = tuple(outputs.get("_active_views", tuple(name for name in self.view_names if name in view_tokens)))
        if not active_view_names:
            return
        reference_view = active_view_names[0]
        batch_size = view_tokens[reference_view].size(0)
        device = view_tokens[reference_view].device
        activated = torch.rand(batch_size, device=device) < self.view_drop_prob
        if not activated.any():
            return

        selected_views = torch.randint(0, len(active_view_names), (batch_size,), device=device)
        use_mask = torch.rand(batch_size, device=device) < 0.5
        perturbed_views = {name: view_tokens[name].clone() for name in active_view_names}
        for view_index, view_name in enumerate(active_view_names):
            selected = activated & (selected_views == view_index)
            masked = selected & use_mask
            corrupted = selected & ~use_mask
            if masked.any():
                perturbed_views[view_name][masked] = 0.0
            if corrupted.any():
                noise = torch.randn_like(perturbed_views[view_name][corrupted]) * self.view_noise_std
                perturbed_views[view_name][corrupted] = perturbed_views[view_name][corrupted] + noise

        perturbed_outputs = self._forward_heads(perturbed_views)
        prediction_terms = []
        representation_terms = []
        temperature = 2.0
        full_representations = outputs["_task_representations"]
        perturbed_representations = perturbed_outputs["_task_representations"]
        for task_index, task_name in enumerate(self.task_names):
            target_prob = torch.softmax(outputs[task_name].detach() / temperature, dim=-1)
            perturbed_log_prob = torch.log_softmax(perturbed_outputs[task_name] / temperature, dim=-1)
            prediction_per_sample = F.kl_div(
                perturbed_log_prob,
                target_prob,
                reduction="none",
            ).sum(dim=-1) * (temperature * temperature)
            representation_per_sample = (
                full_representations[:, task_index].detach() - perturbed_representations[:, task_index]
            ).pow(2).mean(dim=-1)
            valid = activated
            if task_mask is not None:
                valid = valid & (task_mask[:, task_index].to(device=device) > 0)
            if valid.any():
                prediction_terms.append(prediction_per_sample[valid].mean())
                representation_terms.append(representation_per_sample[valid].mean())

        if not prediction_terms:
            return
        prediction_consistency = torch.stack(prediction_terms).sum()
        representation_consistency = torch.stack(representation_terms).sum()
        gcti_consistency = prediction_consistency + self.view_consistency_weight * representation_consistency
        outputs["aux_losses"]["gcti_loss"] = gcti_consistency * self.gcti_loss_weight
        outputs["gcti_consistency"] = {
            "prediction": prediction_consistency,
            "representation": representation_consistency,
            "activated_fraction": activated.float().mean(),
        }
        if self.return_auxiliary:
            outputs["perturbed_view_indices"] = selected_views
            outputs["perturbation_is_mask"] = use_mask
            outputs["perturbed_outputs"] = perturbed_outputs

    def forward(
        self,
        inputs: torch.Tensor | dict[str, torch.Tensor],
        task_mask: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        encoded_views = self._encode_views(inputs)
        view_tokens = {name: encoded_views[name] for name in self.view_names}
        view_presence = encoded_views.get("_modality_mask")
        outputs = self._forward_heads(view_tokens, view_presence=view_presence)
        self._maybe_add_view_consistency(outputs, view_tokens, task_mask=task_mask)

        if self.return_auxiliary:
            outputs["view_tokens"] = view_tokens
            if "_modality_mask" in encoded_views:
                outputs["modality_mask"] = encoded_views["_modality_mask"]
            if "_raw_signal" in encoded_views:
                outputs["raw_signal"] = encoded_views["_raw_signal"]
            if "_raw_matrix" in encoded_views:
                outputs["raw_matrix"] = encoded_views["_raw_matrix"]
            if "_stf_map" in encoded_views:
                outputs["stf_map"] = encoded_views["_stf_map"]
            if "_gaf_map" in encoded_views:
                outputs["gaf_map"] = encoded_views["_gaf_map"]
        return outputs


def build_sensorfield_m3t(**kwargs: Any) -> SensorFieldM3T:
    return SensorFieldM3T(**kwargs)


def _demo() -> None:
    torch.manual_seed(7)
    model = SensorFieldM3T(
        task_output_dims={"event_type": 4, "distance_cls": 3},
        hidden_dim=96,
        num_anchors=6,
        num_heads=4,
        view_drop_prob=0.5,
        enable_view_consistency=True,
        view_consistency_weight=0.1,
        return_auxiliary=True,
    )
    inputs = torch.randn(2, 1, 6, 256)
    outputs = model(inputs)
    print("event_type:", tuple(outputs["event_type"].shape))
    print("distance_cls:", tuple(outputs["distance_cls"].shape))
    print("shared_anchors:", tuple(outputs["fac_outputs"]["shared_anchors"].shape))
    print("task_tokens:", tuple(outputs["taef_outputs"]["task_tokens"].shape))


if __name__ == "__main__":
    _demo()
