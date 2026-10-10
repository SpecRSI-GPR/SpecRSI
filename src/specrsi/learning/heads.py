"""One continuation scorer and a separately parameterized choice policy.

The ``action_head``/``batch_head`` properties retain the old adapter API, but
both name the same Q model. They are not independently trained critics.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import torch
from torch import Tensor, nn
from .records import DecisionRecord
from .work_graph import graph_schema, relation_count, validate_role_graph


class SurfaceHead(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1))

    def forward(self, hidden: Tensor) -> Tensor:
        return self.network(hidden).squeeze(-1)  # Raw binary logits; no third interrupted class.


class ContinuationValueHead(nn.Module):
    def __init__(self, state_dim: int, option_dim: int, hidden_dim: int):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(state_dim + option_dim, hidden_dim), nn.SiLU(),
                                     nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1))

    def forward(self, state: Tensor, options: Tensor) -> Tensor:
        if state.ndim == 1:
            state = state.expand(options.shape[0], -1)
        return self.network(torch.cat((state, options), dim=-1)).squeeze(-1)


class WholeBatchValueHead(ContinuationValueHead):
    """Scores the supplied UNION/completion features of an entire feasible batch.

    Callers must encode union coverage, remaining tasks, sources, dependencies,
    setup/caps and budget; this is intentionally not a sum of item predictions.
    """


class RelationalPlanBranch(nn.Module):
    """Typed message passing followed by four-head plan-to-set attention.

    Candidate nodes, observed job nodes, and their actual parent/scope links
    are inputs. A proposed plan marks its nodes; no ungenerated child enters
    this graph. Each role owns a separate instance of the complete branch.
    """
    def __init__(self, state_dim: int, option_dim: int, hidden_dim: int, node_dim: int = 12, heads: int = 4,
                 node_text_dim: int = 0, relations: int = 4):
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("Relational branch width must be divisible by its attention heads")
        self.state_projection = nn.Linear(state_dim, hidden_dim)
        self.option_projection = nn.Linear(option_dim, hidden_dim)
        self.node_projection = nn.Linear(node_dim, hidden_dim)
        self.node_text_dim = node_text_dim
        if node_text_dim:
            self.node_text_projection = nn.Linear(node_text_dim, hidden_dim, bias=False)
        self.messages = nn.ModuleList(nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(relations))
        self.node_norm = nn.LayerNorm(hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, heads, batch_first=True, dropout=0.)
        self.output = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1))

    def forward(self, state, options, graph_nodes=None, graph_edges=None, option_node_masks=None,
                graph_node_text_features=None):
        if self.node_text_dim:
            if (graph_nodes is None or graph_node_text_features is None or
                    graph_node_text_features.shape != (len(graph_nodes), self.node_text_dim)):
                raise ValueError("v3 branches require aligned frozen node text features")
        elif graph_node_text_features is not None:
            raise ValueError("v2 branches cannot silently consume v3 node text features")
        state_hidden = self.state_projection(state)
        queries = self.option_projection(options) + state_hidden
        if graph_nodes is None or graph_nodes.numel() == 0:
            nodes = state_hidden.reshape(1, -1)
        else:
            nodes = self.node_projection(graph_nodes)
            if self.node_text_dim:
                nodes = nodes + self.node_text_projection(graph_node_text_features.detach())
            messages = torch.zeros_like(nodes)
            counts = nodes.new_zeros((len(nodes), 1))
            if graph_edges is not None and graph_edges.numel():
                for relation, transform in enumerate(self.messages):
                    edges = graph_edges[graph_edges[:, 2] == relation]
                    if edges.numel():
                        messages.index_add_(0, edges[:, 1], transform(nodes[edges[:, 0]]))
                        counts.index_add_(0, edges[:, 1], nodes.new_ones((len(edges), 1)))
            nodes = self.node_norm(nodes + messages / counts.clamp_min(1)).tanh()
            if option_node_masks is not None and option_node_masks.numel():
                weights = option_node_masks / option_node_masks.sum(-1, keepdim=True).clamp_min(1)
                queries = queries + weights @ nodes
        context, _ = self.attention(queries.unsqueeze(0), nodes.unsqueeze(0), nodes.unsqueeze(0), need_weights=False)
        return self.output(torch.cat((queries, context.squeeze(0)), dim=-1)).squeeze(-1)


class LowRankAdapter(nn.Module):
    def __init__(self, dim: int, rank: int, alpha: float):
        super().__init__()
        self.a = nn.Linear(dim, rank, bias=False)
        self.b = nn.Linear(rank, dim, bias=False)
        nn.init.normal_(self.a.weight, std=.02)
        nn.init.zeros_(self.b.weight)
        self.scale = alpha / rank

    def forward(self, x):
        return x + self.b(self.a(x)) * self.scale


@dataclass
class TinyModelConfig:
    architecture_version: int = 2
    feature_dim: int = 8
    option_dim: int = 8
    hidden_dim: int = 32
    lora_rank: int = 4
    lora_alpha: float = 8.
    seed: int = 0
    graph_schema: str = "candidate_items_v1"
    role_observation_version: str = "legacy_v1"
    frozen_global_cache: bool = False

    def __post_init__(self):
        from .frozen_text import role_observation_contract
        graph_schema(self)
        role_observation_contract(self)


class TinyRoleModel(nn.Module):
    """Small role model with a frozen feature trunk for finite-action tasks.

    Separate trainable branches produce edit, policy, and value outputs.
    HFRoleModel provides the corresponding language-model LoRA implementation.
    """
    backend = "tiny"

    def __init__(self, config: TinyModelConfig):
        super().__init__()
        self.config = config
        if config.architecture_version not in {2, 3}:
            raise ValueError("PDF roles require architecture_version=2 or explicit semantic-node version 3")
        if config.architecture_version == 3 and (config.feature_dim < 8 or config.option_dim < 12):
            raise ValueError("v3 roles require feature_dim >= 8 and option_dim >= 12")
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(config.seed)
            self.backbone = nn.Linear(config.feature_dim, config.hidden_dim)
            self.backbone.requires_grad_(False)
            self.draft_adapter = LowRankAdapter(config.hidden_dim, config.lora_rank, config.lora_alpha)
            self.verify_adapter = LowRankAdapter(config.hidden_dim, config.lora_rank, config.lora_alpha)
            self.surface_head = SurfaceHead(config.hidden_dim)
            semantic = {"node_text_dim": config.hidden_dim} if config.architecture_version == 3 else {}
            semantic["relations"] = relation_count(graph_schema(config))
            self.value_head = RelationalPlanBranch(config.hidden_dim, config.option_dim, config.hidden_dim, **semantic)
            self.policy_head = RelationalPlanBranch(config.hidden_dim, config.option_dim, config.hidden_dim, **semantic)
            self.edit_head = ContinuationValueHead(config.hidden_dim, config.option_dim, config.hidden_dim)

    @property
    def device(self):
        return next(self.parameters()).device

    def encode(self, features, role="verify"):
        x = torch.as_tensor(features, dtype=torch.float32, device=self.device)
        if x.shape[-1] != self.config.feature_dim:
            raise ValueError(f"Expected {self.config.feature_dim} state features, got {x.shape[-1]}")
        hidden = self.backbone(x).tanh()
        if role in {"value", "policy"}:
            return hidden.detach()
        return (self.draft_adapter if role == "draft" else self.verify_adapter)(hidden)

    @property
    def action_head(self):
        return self.value_head

    @property
    def batch_head(self):
        return self.value_head

    def decision_values(self, decision: DecisionRecord):
        if decision.kind == "edit_token":
            raise ValueError("Token likelihoods require backend=hf; tiny demo is a finite-action model")
        if decision.kind != "edit":
            validate_role_graph(self, decision)
        state = self.encode(decision.state_features, "draft" if decision.kind == "edit" else "value")
        options = torch.as_tensor(decision.option_features, dtype=torch.float32, device=self.device)
        if options.shape != (len(decision.support), self.config.option_dim):
            raise ValueError("Stored option features disagree with configured option_dim")
        head = self.edit_head if decision.kind == "edit" else self.value_head
        if decision.option_item_features:
            return torch.stack([head(state, torch.tensor(items, dtype=torch.float32, device=self.device)).sum()
                                if items else state.sum()*0 for items in decision.option_item_features])
        if decision.kind == "edit":
            return head(state, options)
        values = head(state, options, **self._graph(decision))
        mask = torch.tensor([option == "ACCEPT" for option in decision.support], device=self.device)
        return values.masked_fill(mask, 0.)

    def _graph(self, decision):
        validate_role_graph(self, decision)
        semantic = {}
        if self.config.architecture_version == 3:
            from .frozen_text import node_text_cache
            if (not decision.graph_node_text_features or
                    decision.graph_text_snapshot != node_text_cache(self).identity):
                raise ValueError("v3 replay requires matching frozen encoder identity and node text features")
            semantic["graph_node_text_features"] = torch.tensor(decision.graph_node_text_features,
                dtype=torch.float32, device=self.device)
        elif decision.graph_text_snapshot or decision.graph_node_text_features or decision.graph_node_texts:
            raise ValueError("v2 roles cannot silently load v3 semantic graph records")
        if not decision.graph_nodes:
            return {}
        return {"graph_nodes": torch.tensor(decision.graph_nodes, dtype=torch.float32, device=self.device),
                "graph_edges": torch.tensor(decision.graph_edges, dtype=torch.long, device=self.device).reshape(-1, 3),
                "option_node_masks": torch.tensor(decision.option_node_masks, dtype=torch.float32, device=self.device)
                    if decision.option_node_masks else None, **semantic}

    def decision_logits(self, decision: DecisionRecord):
        if decision.kind in {"edit", "edit_token"}:
            return self.decision_values(decision)
        validate_role_graph(self, decision)
        state = self.encode(decision.state_features, "policy")
        options = torch.as_tensor(decision.option_features, dtype=torch.float32, device=self.device)
        if options.shape != (len(decision.support), self.config.option_dim):
            raise ValueError("Stored option features disagree with configured option_dim")
        logits = self.policy_head(state, options, **self._graph(decision))
        # ACCEPT is the fixed zero-logit reference action in the PDF policy.
        mask = torch.tensor([option == "ACCEPT" for option in decision.support], device=self.device)
        return logits.masked_fill(mask, 0.)

    def surface_logits(self, features, context_text=""):
        return self.surface_head(self.encode(features))

    def save(self, directory: str | Path):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        manifest = {"backend": self.backend, **asdict(self.config)}
        if self.config.architecture_version == 3:
            from .frozen_text import node_text_cache
            manifest["frozen_text_snapshot"] = node_text_cache(self).validate_identity()
        (directory / "model_config.json").write_text(json.dumps(manifest, indent=2))
        torch.save(self.state_dict(), directory / "roles.pt")

    @classmethod
    def load(cls, directory: str | Path, device="cpu"):
        directory = Path(directory)
        config = json.loads((directory / "model_config.json").read_text())
        config.pop("backend")
        expected_text = config.pop("frozen_text_snapshot", None)
        if config.get("architecture_version") not in {2, 3}:
            raise ValueError("Legacy coupled-Q checkpoints require explicit migration and fresh policy training")
        if config["architecture_version"] == 3 and not expected_text:
            raise ValueError("v3 checkpoint is missing its frozen text encoder identity")
        model = cls(TinyModelConfig(**config)).to(device)
        model.load_state_dict(torch.load(directory / "roles.pt", map_location=device, weights_only=True))
        if config["architecture_version"] == 3:
            from .frozen_text import node_text_cache
            if node_text_cache(model).validate_identity() != expected_text:
                raise ValueError("Checkpoint frozen text encoder identity does not match its weights")
        return model
