from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from LibMTL.model.sensorfield_m3t import (
    ALLOWED_VIEW_NAMES,
    ImageTokenEncoder2D,
    RawSignalEncoder1D,
    _first_available_tensor,
    parse_enabled_views,
)


DEFAULT_TASK_NAMES = ("event_type", "radial_threat", "threat_condition")
RADIAL_TASK_NAME = "radial_threat"


def _edge_key(source: str, target: str) -> str:
    return f"{source}__to__{target}"


def _align_token_count(tokens: torch.Tensor, token_count: int) -> torch.Tensor:
    if tokens.size(1) == token_count:
        return tokens
    return F.adaptive_avg_pool1d(tokens.transpose(1, 2), token_count).transpose(1, 2)


def ordinal_logits_to_probs(logits: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Convert cumulative ordinal logits into a normalized level distribution."""

    cumulative = torch.sigmoid(logits)
    left = torch.cat([torch.ones_like(cumulative[:, :1]), cumulative], dim=-1)
    right = torch.cat([cumulative, torch.zeros_like(cumulative[:, :1])], dim=-1)
    probs = (left - right).clamp_min(eps)
    return probs / probs.sum(dim=-1, keepdim=True).clamp_min(eps)


def ordinal_predictions(logits: torch.Tensor) -> torch.Tensor:
    return (torch.sigmoid(logits) > 0.5).sum(dim=-1).long()


class FeedForwardBlock(nn.Module):
    def __init__(self, hidden_dim: int, mlp_ratio: float = 2.0, dropout: float = 0.1) -> None:
        super().__init__()
        inner_dim = max(int(hidden_dim * mlp_ratio), hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


class MultimodalEvidenceDecomposition(nn.Module):
    """MED: split each modality into shared and private evidence."""

    def __init__(
        self,
        view_names: tuple[str, ...],
        hidden_dim: int,
        shared_tokens: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.view_names = view_names
        self.shared_tokens = int(shared_tokens)
        self.gates = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.LayerNorm(hidden_dim),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for name in self.view_names
            }
        )
        self.modality_score = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, view_tokens: dict[str, torch.Tensor]) -> dict[str, Any]:
        active_view_names = tuple(name for name in self.view_names if name in view_tokens)
        if not active_view_names:
            raise ValueError("MED requires at least one encoded view.")

        shared_components: dict[str, torch.Tensor] = {}
        private_components: dict[str, torch.Tensor] = {}
        pooled_shared = []
        aligned_shared = []

        for name in active_view_names:
            tokens = view_tokens[name]
            gate = torch.sigmoid(self.gates[name](tokens))
            shared = gate * tokens
            private = (1.0 - gate) * tokens
            shared_components[name] = shared
            private_components[name] = private
            pooled_shared.append(shared.mean(dim=1))
            aligned_shared.append(_align_token_count(shared, self.shared_tokens))

        pooled_shared_tensor = torch.stack(pooled_shared, dim=1)
        alpha = torch.softmax(self.modality_score(pooled_shared_tensor).squeeze(-1), dim=-1)
        aligned_shared_tensor = torch.stack(aligned_shared, dim=1)
        shared_evidence = (alpha[:, :, None, None] * aligned_shared_tensor).sum(dim=1)
        shared_pool = shared_evidence.mean(dim=1)

        alignment_terms = [
            (shared_components[name].mean(dim=1) - shared_pool).pow(2).mean()
            for name in active_view_names
        ]
        orthogonality_terms = []
        for name in active_view_names:
            shared_norm = F.normalize(shared_components[name], dim=-1)
            private_norm = F.normalize(private_components[name], dim=-1)
            cross_cov = torch.bmm(shared_norm.transpose(1, 2), private_norm)
            orthogonality_terms.append(cross_cov.pow(2).mean())

        dec_loss = torch.stack(alignment_terms + orthogonality_terms).mean()
        return {
            "shared_evidence": shared_evidence,
            "shared_components": shared_components,
            "private_components": private_components,
            "modality_reliability": alpha,
            "active_view_names": active_view_names,
            "dec_loss": dec_loss,
        }


class HierarchicalThreatTokenization(nn.Module):
    """HTT: retrieve shared and private evidence into task-level tokens."""

    def __init__(
        self,
        task_names: tuple[str, ...],
        view_names: tuple[str, ...],
        hidden_dim: int,
        num_heads: int,
        task_levels: tuple[int, ...],
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.task_names = task_names
        self.view_names = view_names
        self.task_levels = torch.tensor(task_levels, dtype=torch.long)
        self.task_embeddings = nn.Embedding(len(task_names), hidden_dim)
        self.level_embeddings = nn.Embedding(int(max(task_levels)) + 1, hidden_dim)
        self.query_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in task_names])
        self.shared_norm = nn.LayerNorm(hidden_dim)
        self.private_norm = nn.LayerNorm(hidden_dim)
        self.shared_attention = nn.ModuleList(
            [
                nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
                for _ in task_names
            ]
        )
        self.private_attention = nn.ModuleList(
            [
                nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
                for _ in task_names
            ]
        )
        self.private_scores = nn.ModuleList([nn.Linear(hidden_dim, 1, bias=False) for _ in task_names])
        self.output_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in task_names])
        self.task_ffns = nn.ModuleList([FeedForwardBlock(hidden_dim, dropout=dropout) for _ in task_names])

    def forward(
        self,
        shared_evidence: torch.Tensor,
        private_components: dict[str, torch.Tensor],
    ) -> dict[str, Any]:
        batch_size = shared_evidence.size(0)
        device = shared_evidence.device
        task_indices = torch.arange(len(self.task_names), device=device)
        level_indices = self.task_levels.to(device)
        base_queries = self.task_embeddings(task_indices) + self.level_embeddings(level_indices)
        queries = base_queries.unsqueeze(0).expand(batch_size, -1, -1)

        active_private_names = tuple(name for name in self.view_names if name in private_components)
        pooled_private = torch.stack(
            [private_components[name].mean(dim=1) for name in active_private_names],
            dim=1,
        )

        task_states = []
        private_weights = []
        shared_retrievals = []
        private_retrievals = []
        for task_index, _task_name in enumerate(self.task_names):
            query = queries[:, task_index : task_index + 1]
            norm_query = self.query_norms[task_index](query)
            shared_context = self.shared_norm(shared_evidence)
            shared_out, _ = self.shared_attention[task_index](
                norm_query,
                shared_context,
                shared_context,
                need_weights=False,
            )
            shared_out = shared_out.squeeze(1)

            beta = torch.softmax(self.private_scores[task_index](pooled_private).squeeze(-1), dim=-1)
            private_sum = torch.zeros_like(shared_out)
            for view_index, name in enumerate(active_private_names):
                private_context = self.private_norm(private_components[name])
                private_out, _ = self.private_attention[task_index](
                    norm_query,
                    private_context,
                    private_context,
                    need_weights=False,
                )
                private_sum = private_sum + beta[:, view_index : view_index + 1] * private_out.squeeze(1)

            combined = query.squeeze(1) + shared_out + private_sum
            state = self.task_ffns[task_index](self.output_norms[task_index](combined))
            task_states.append(state)
            private_weights.append(beta)
            shared_retrievals.append(shared_out)
            private_retrievals.append(private_sum)

        return {
            "task_states": torch.stack(task_states, dim=1),
            "task_queries": queries,
            "private_weights": torch.stack(private_weights, dim=1),
            "shared_retrievals": torch.stack(shared_retrievals, dim=1),
            "private_retrievals": torch.stack(private_retrievals, dim=1),
            "private_view_names": active_private_names,
        }


class BidirectionalTaskInteraction(nn.Module):
    """BTI: build directed, relation-aware, confidence-gated messages."""

    def __init__(
        self,
        task_names: tuple[str, ...],
        hidden_dim: int,
        relation_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.task_names = task_names
        self.task_to_index = {name: index for index, name in enumerate(task_names)}
        self.relation_names = ("top_down", "bottom_up", "lateral")
        self.relation_to_index = {name: index for index, name in enumerate(self.relation_names)}
        self.edges = self._build_default_edges(task_names)
        self.relation_embedding = nn.Embedding(len(self.relation_names), relation_dim)
        message_dim = hidden_dim * 2 + relation_dim
        gate_dim = message_dim + 1
        self.message_mlps = nn.ModuleDict()
        self.gate_mlps = nn.ModuleDict()
        for source, target, _relation in self.edges:
            key = _edge_key(source, target)
            self.message_mlps[key] = nn.Sequential(
                nn.LayerNorm(message_dim),
                nn.Linear(message_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.gate_mlps[key] = nn.Sequential(
                nn.LayerNorm(gate_dim),
                nn.Linear(gate_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )

    @staticmethod
    def _build_default_edges(task_names: tuple[str, ...]) -> tuple[tuple[str, str, str], ...]:
        if set(DEFAULT_TASK_NAMES).issubset(set(task_names)):
            return (
                ("event_type", "radial_threat", "top_down"),
                ("event_type", "threat_condition", "top_down"),
                ("radial_threat", "event_type", "bottom_up"),
                ("threat_condition", "event_type", "bottom_up"),
                ("radial_threat", "threat_condition", "lateral"),
                ("threat_condition", "radial_threat", "lateral"),
            )
        edges = []
        for source in task_names:
            for target in task_names:
                if source != target:
                    edges.append((source, target, "lateral"))
        return tuple(edges)

    def forward(
        self,
        task_states: torch.Tensor,
        confidence: dict[str, torch.Tensor],
    ) -> dict[str, Any]:
        batch_size = task_states.size(0)
        messages: dict[str, dict[str, Any]] = {}
        for source, target, relation in self.edges:
            source_index = self.task_to_index[source]
            target_index = self.task_to_index[target]
            relation_index = self.relation_to_index[relation]
            source_state = task_states[:, source_index]
            target_state = task_states[:, target_index]
            relation_ids = torch.full(
                (batch_size,),
                relation_index,
                device=task_states.device,
                dtype=torch.long,
            )
            relation_vector = self.relation_embedding(relation_ids)
            key = _edge_key(source, target)
            message_input = torch.cat([source_state, target_state, relation_vector], dim=-1)
            candidate = self.message_mlps[key](message_input)
            source_confidence = confidence[source].to(task_states.dtype).view(batch_size, 1)
            gate_input = torch.cat([message_input, source_confidence], dim=-1)
            gated = torch.sigmoid(self.gate_mlps[key](gate_input)) * candidate
            messages[key] = {
                "source": source,
                "target": target,
                "relation": relation,
                "relation_index": relation_index,
                "source_index": source_index,
                "target_index": target_index,
                "candidate": candidate,
                "gated": gated,
                "relation_vector": relation_vector,
            }
        return {"messages": messages, "edges": self.edges}


class ConsistentEvidencePropagation(nn.Module):
    """CEP: score feature, prediction, and hierarchy consistency before updates."""

    def __init__(
        self,
        task_names: tuple[str, ...],
        task_output_dims: dict[str, int],
        hidden_dim: int,
        relation_names: tuple[str, ...],
        edges: tuple[tuple[str, str, str], ...],
    ) -> None:
        super().__init__()
        self.task_names = task_names
        self.task_output_dims = task_output_dims
        self.edges = edges
        self.feature_proj = nn.Linear(hidden_dim, hidden_dim)
        self.target_proj = nn.Linear(hidden_dim, hidden_dim)
        self.update_norms = nn.ModuleDict({name: nn.LayerNorm(hidden_dim) for name in task_names})
        self.relation_scores = nn.Embedding(len(relation_names), 1)
        self.relation_to_index = {name: index for index, name in enumerate(relation_names)}
        self.compatibility = nn.ParameterDict()
        for source, target, _relation in self.edges:
            parameter = nn.Parameter(torch.empty(task_output_dims[source], task_output_dims[target]))
            nn.init.xavier_uniform_(parameter)
            self.compatibility[_edge_key(source, target)] = parameter
        self.a_f = nn.Parameter(torch.tensor(1.0))
        self.a_p = nn.Parameter(torch.tensor(1.0))
        self.a_h = nn.Parameter(torch.tensor(0.5))
        self.bias = nn.Parameter(torch.tensor(0.0))

    def map_distribution(self, source: str, target: str, source_distribution: torch.Tensor) -> torch.Tensor:
        matrix = torch.softmax(self.compatibility[_edge_key(source, target)], dim=-1)
        mapped = source_distribution @ matrix
        return mapped / mapped.sum(dim=-1, keepdim=True).clamp_min(1e-6)

    @staticmethod
    def _kl_divergence(target_distribution: torch.Tensor, mapped_distribution: torch.Tensor) -> torch.Tensor:
        target = target_distribution.clamp_min(1e-6)
        mapped = mapped_distribution.clamp_min(1e-6)
        return (target * (target.log() - mapped.log())).sum(dim=-1, keepdim=True)

    def forward(
        self,
        task_states: torch.Tensor,
        messages: dict[str, dict[str, Any]],
        distributions: dict[str, torch.Tensor],
    ) -> dict[str, Any]:
        incoming = torch.zeros_like(task_states)
        reliabilities: dict[str, torch.Tensor] = {}
        message_norms: dict[str, torch.Tensor] = {}

        for key, payload in messages.items():
            source = payload["source"]
            target = payload["target"]
            relation = payload["relation"]
            target_index = payload["target_index"]
            gated_message = payload["gated"]
            target_state = task_states[:, target_index]

            feature_score = F.cosine_similarity(
                self.feature_proj(gated_message),
                self.target_proj(target_state),
                dim=-1,
            ).unsqueeze(-1)
            mapped_distribution = self.map_distribution(source, target, distributions[source])
            prediction_penalty = self._kl_divergence(distributions[target], mapped_distribution)
            relation_index = torch.tensor(
                self.relation_to_index[relation],
                device=task_states.device,
                dtype=torch.long,
            )
            hierarchy_score = torch.tanh(self.relation_scores(relation_index)).view(1, 1)
            eta = torch.sigmoid(
                F.softplus(self.a_f) * feature_score
                - F.softplus(self.a_p) * prediction_penalty
                + F.softplus(self.a_h) * hierarchy_score
                + self.bias
            )
            incoming[:, target_index] = incoming[:, target_index] + eta * gated_message
            reliabilities[key] = eta
            message_norms[key] = gated_message.norm(dim=-1)

        updated_states = []
        for task_index, task_name in enumerate(self.task_names):
            updated_states.append(self.update_norms[task_name](task_states[:, task_index] + incoming[:, task_index]))
        return {
            "updated_task_states": torch.stack(updated_states, dim=1),
            "propagation_reliability": reliabilities,
            "message_norms": message_norms,
        }

    def consistency_loss(
        self,
        distributions: dict[str, torch.Tensor],
        task_validity: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        losses = []
        for source, target, _relation in self.edges:
            mapped_distribution = self.map_distribution(source, target, distributions[source])
            per_sample = self._kl_divergence(distributions[target], mapped_distribution).squeeze(-1)
            if task_validity is not None:
                source_valid = task_validity.get(source)
                target_valid = task_validity.get(target)
                if source_valid is not None and target_valid is not None:
                    mask = source_valid.bool() & target_valid.bool()
                    if not mask.any():
                        continue
                    per_sample = per_sample[mask]
            losses.append(per_sample.mean())
        if not losses:
            first_distribution = next(iter(distributions.values()))
            return first_distribution.sum() * 0.0
        return torch.stack(losses).mean()


class SensorFieldMEDHTT(nn.Module):
    """Hierarchical multimodal threat model with MED, HTT, BTI, and CEP."""

    def __init__(
        self,
        task_output_dims: dict[str, int],
        hidden_dim: int = 128,
        num_heads: int = 4,
        shared_tokens: int = 8,
        raw_tokens: int = 16,
        raw_in_channels: int = 1,
        stf_tokens: int = 16,
        gaf_tokens: int = 16,
        stf_size: int = 96,
        gaf_size: int = 48,
        stft_n_fft: int = 128,
        stft_hop_length: int = 64,
        stft_win_length: int = 128,
        propagation_steps: int = 1,
        relation_dim: int | None = None,
        enabled_views: str | tuple[str, ...] | list[str] = ALLOWED_VIEW_NAMES,
        dropout: float = 0.1,
        dec_loss_weight: float = 0.05,
        cep_loss_weight: float = 0.05,
        disable_med: bool = False,
        disable_htt: bool = False,
        disable_bti: bool = False,
        disable_cep: bool = False,
        return_auxiliary: bool = False,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")
        for task_name in DEFAULT_TASK_NAMES:
            if task_name not in task_output_dims:
                raise ValueError(f"task_output_dims must include '{task_name}'.")
        if task_output_dims[RADIAL_TASK_NAME] < 2:
            raise ValueError("radial_threat must have at least two ordinal levels.")

        self.hidden_dim = int(hidden_dim)
        self.task_output_dims = dict(task_output_dims)
        self.task_names = DEFAULT_TASK_NAMES
        self.task_to_index = {name: index for index, name in enumerate(self.task_names)}
        self.view_names = parse_enabled_views(enabled_views)
        self.stf_size = int(stf_size)
        self.gaf_size = int(gaf_size)
        self.stft_n_fft = int(stft_n_fft)
        self.stft_hop_length = int(stft_hop_length)
        self.stft_win_length = int(stft_win_length)
        self.propagation_steps = max(int(propagation_steps), 0)
        self.shared_tokens = int(shared_tokens)
        self.dec_loss_weight = float(dec_loss_weight)
        self.cep_loss_weight = float(cep_loss_weight)
        self.disable_med = bool(disable_med)
        self.disable_htt = bool(disable_htt)
        self.disable_bti = bool(disable_bti)
        self.disable_cep = bool(disable_cep)
        self.return_auxiliary = bool(return_auxiliary)
        self.register_buffer("_stft_window", torch.hann_window(self.stft_win_length), persistent=False)

        self.raw_encoder = RawSignalEncoder1D(
            hidden_dim=hidden_dim,
            num_tokens=raw_tokens,
            dropout=dropout,
            in_channels=raw_in_channels,
        )
        self.stf_encoder = ImageTokenEncoder2D(hidden_dim=hidden_dim, num_tokens=stf_tokens, dropout=dropout)
        self.gaf_encoder = ImageTokenEncoder2D(hidden_dim=hidden_dim, num_tokens=gaf_tokens, dropout=dropout)
        self.med = MultimodalEvidenceDecomposition(
            view_names=self.view_names,
            hidden_dim=hidden_dim,
            shared_tokens=shared_tokens,
            dropout=dropout,
        )
        self.htt = HierarchicalThreatTokenization(
            task_names=self.task_names,
            view_names=self.view_names,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            task_levels=(0, 1, 1),
            dropout=dropout,
        )
        actual_relation_dim = int(relation_dim or max(hidden_dim // 4, 16))
        self.bti = BidirectionalTaskInteraction(
            task_names=self.task_names,
            hidden_dim=hidden_dim,
            relation_dim=actual_relation_dim,
            dropout=dropout,
        )
        self.cep = ConsistentEvidencePropagation(
            task_names=self.task_names,
            task_output_dims=self.task_output_dims,
            hidden_dim=hidden_dim,
            relation_names=self.bti.relation_names,
            edges=self.bti.edges,
        )
        self.task_heads = nn.ModuleDict()
        for task_name in self.task_names:
            output_dim = self.task_output_dims[task_name]
            if task_name == RADIAL_TASK_NAME:
                output_dim -= 1
            self.task_heads[task_name] = nn.Linear(hidden_dim, output_dim)

    def _simple_med(self, view_tokens: dict[str, torch.Tensor]) -> dict[str, Any]:
        active_view_names = tuple(name for name in self.view_names if name in view_tokens)
        if not active_view_names:
            raise ValueError("Simple MED fallback requires at least one encoded view.")

        aligned_shared = [_align_token_count(view_tokens[name], self.shared_tokens) for name in active_view_names]
        shared_evidence = torch.stack(aligned_shared, dim=1).mean(dim=1)
        batch_size = shared_evidence.size(0)
        modality_reliability = shared_evidence.new_full(
            (batch_size, len(active_view_names)),
            1.0 / float(len(active_view_names)),
        )
        zero = shared_evidence.sum() * 0.0
        return {
            "shared_evidence": shared_evidence,
            "shared_components": {name: view_tokens[name] for name in active_view_names},
            "private_components": {name: view_tokens[name] for name in active_view_names},
            "modality_reliability": modality_reliability,
            "active_view_names": active_view_names,
            "dec_loss": zero,
        }

    def _simple_htt(
        self,
        shared_evidence: torch.Tensor,
        private_components: dict[str, torch.Tensor],
    ) -> dict[str, Any]:
        batch_size = shared_evidence.size(0)
        device = shared_evidence.device
        task_indices = torch.arange(len(self.task_names), device=device)
        level_indices = self.htt.task_levels.to(device)
        queries = (
            self.htt.task_embeddings(task_indices) + self.htt.level_embeddings(level_indices)
        ).unsqueeze(0).expand(batch_size, -1, -1)

        shared_pool = shared_evidence.mean(dim=1)
        active_private_names = tuple(name for name in self.view_names if name in private_components)
        if active_private_names:
            pooled_private = torch.stack(
                [private_components[name].mean(dim=1) for name in active_private_names],
                dim=1,
            )
            private_pool = pooled_private.mean(dim=1)
            private_weights = shared_evidence.new_full(
                (batch_size, len(self.task_names), len(active_private_names)),
                1.0 / float(len(active_private_names)),
            )
        else:
            private_pool = torch.zeros_like(shared_pool)
            private_weights = shared_evidence.new_zeros(batch_size, len(self.task_names), 0)

        shared_retrievals = []
        private_retrievals = []
        task_states = []
        combined_context = shared_pool + private_pool
        for task_index, _task_name in enumerate(self.task_names):
            combined = queries[:, task_index] + combined_context
            state = self.htt.task_ffns[task_index](self.htt.output_norms[task_index](combined))
            task_states.append(state)
            shared_retrievals.append(shared_pool)
            private_retrievals.append(private_pool)

        return {
            "task_states": torch.stack(task_states, dim=1),
            "task_queries": queries,
            "private_weights": private_weights,
            "shared_retrievals": torch.stack(shared_retrievals, dim=1),
            "private_retrievals": torch.stack(private_retrievals, dim=1),
            "private_view_names": active_private_names,
        }

    def _direct_update_without_cep(
        self,
        task_states: torch.Tensor,
        messages: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        if not messages:
            return {
                "updated_task_states": task_states,
                "propagation_reliability": {},
                "message_norms": {},
            }

        incoming = torch.zeros_like(task_states)
        reliability: dict[str, torch.Tensor] = {}
        message_norms: dict[str, torch.Tensor] = {}
        for key, payload in messages.items():
            gated_message = payload["gated"]
            target_index = payload["target_index"]
            incoming[:, target_index] = incoming[:, target_index] + gated_message
            reliability[key] = torch.ones(
                gated_message.size(0),
                1,
                device=gated_message.device,
                dtype=gated_message.dtype,
            )
            message_norms[key] = gated_message.norm(dim=-1)

        updated_states = []
        for task_index, task_name in enumerate(self.task_names):
            updated_states.append(self.cep.update_norms[task_name](task_states[:, task_index] + incoming[:, task_index]))
        return {
            "updated_task_states": torch.stack(updated_states, dim=1),
            "propagation_reliability": reliability,
            "message_norms": message_norms,
        }

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
                return raw.float()
            if raw.ndim == 4 and raw.size(1) == 1 and raw.size(2) == 1:
                return raw.squeeze(2).float()
        raw_matrix = self._extract_raw_matrix(inputs)
        return raw_matrix.flatten(start_dim=1).unsqueeze(1)

    def _build_stf_map_from_raw(self, raw_matrix: torch.Tensor) -> torch.Tensor:
        batch_size, row_count, width = raw_matrix.shape
        flat_rows = raw_matrix.reshape(batch_size * row_count, width)
        n_fft = min(self.stft_n_fft, max(width, 2))
        win_length = min(self.stft_win_length, n_fft)
        hop_length = min(self.stft_hop_length, max(win_length // 2, 1))
        window = torch.hann_window(win_length, device=flat_rows.device, dtype=flat_rows.dtype)
        spectrogram = torch.stft(
            flat_rows,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
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

        if "raw" in self.view_names:
            raw_signal = self._extract_raw_signal(inputs)
            encoded["raw"] = self.raw_encoder(raw_signal)
        if "stf" in self.view_names:
            raw_matrix = self._extract_raw_matrix(inputs)
            encoded["stf"] = self.stf_encoder(self._extract_stf_map(inputs, raw_matrix))
        if "gaf" in self.view_names:
            if raw_signal is None:
                raw_signal = self._extract_raw_signal(inputs)
            encoded["gaf"] = self.gaf_encoder(self._extract_gaf_map(inputs, raw_signal))
        return encoded

    def _heads_from_states(self, task_states: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = {}
        for task_name in self.task_names:
            task_index = self.task_to_index[task_name]
            outputs[task_name] = self.task_heads[task_name](task_states[:, task_index])
        return outputs

    def _distributions_from_outputs(self, outputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        distributions = {
            "event_type": torch.softmax(outputs["event_type"], dim=-1),
            "radial_threat": ordinal_logits_to_probs(outputs["radial_threat"]),
            "threat_condition": torch.softmax(outputs["threat_condition"], dim=-1),
        }
        return distributions

    def _confidence_from_distributions(self, distributions: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        confidence = {}
        for task_name, distribution in distributions.items():
            entropy = -(distribution.clamp_min(1e-6) * distribution.clamp_min(1e-6).log()).sum(dim=-1)
            normalizer = math.log(max(distribution.size(-1), 2))
            confidence[task_name] = (1.0 - entropy / normalizer).clamp(0.0, 1.0)
        return confidence

    def forward(
        self,
        inputs: torch.Tensor | dict[str, torch.Tensor],
        task_validity: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, Any]:
        view_tokens = self._encode_views(inputs)
        med_outputs = self._simple_med(view_tokens) if self.disable_med else self.med(view_tokens)
        htt_outputs = (
            self._simple_htt(
                med_outputs["shared_evidence"],
                med_outputs["private_components"],
            )
            if self.disable_htt
            else self.htt(
                med_outputs["shared_evidence"],
                med_outputs["private_components"],
            )
        )
        task_states = htt_outputs["task_states"]
        propagation_trace = []

        for _step in range(self.propagation_steps):
            step_outputs = self._heads_from_states(task_states)
            distributions = self._distributions_from_outputs(step_outputs)
            confidence = self._confidence_from_distributions(distributions)
            if self.disable_bti:
                bti_outputs = {"messages": {}, "edges": tuple()}
                cep_outputs = {
                    "updated_task_states": task_states,
                    "propagation_reliability": {},
                    "message_norms": {},
                }
            else:
                bti_outputs = self.bti(task_states, confidence)
                if self.disable_cep:
                    cep_outputs = self._direct_update_without_cep(task_states, bti_outputs["messages"])
                else:
                    cep_outputs = self.cep(task_states, bti_outputs["messages"], distributions)
                task_states = cep_outputs["updated_task_states"]
            if self.return_auxiliary:
                propagation_trace.append(
                    {
                        "confidence": confidence,
                        "bti_outputs": bti_outputs,
                        "cep_outputs": cep_outputs,
                    }
                )

        outputs = self._heads_from_states(task_states)
        distributions = self._distributions_from_outputs(outputs)
        outputs["radial_threat_probs"] = distributions["radial_threat"]
        outputs["distance_cls"] = outputs["radial_threat_probs"]
        outputs["task_states"] = task_states
        cep_loss = outputs["event_type"].sum() * 0.0
        if not self.disable_cep:
            cep_loss = self.cep.consistency_loss(distributions, task_validity) * self.cep_loss_weight
        outputs["aux_losses"] = {
            "med_loss": med_outputs["dec_loss"] * self.dec_loss_weight,
            "cep_loss": cep_loss,
        }
        if self.return_auxiliary:
            outputs["view_tokens"] = view_tokens
            outputs["med_outputs"] = med_outputs
            outputs["htt_outputs"] = htt_outputs
            outputs["propagation_trace"] = propagation_trace
            outputs["task_distributions"] = distributions
        return outputs


def build_sensorfield_medhtt(**kwargs: Any) -> SensorFieldMEDHTT:
    return SensorFieldMEDHTT(**kwargs)
