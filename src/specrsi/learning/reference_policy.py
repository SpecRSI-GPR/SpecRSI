"""Registered frozen actor for Eq. (21), sharing the immutable base backbone.

Only trainable draft/policy tensors are copied. Stateless replay substitutes
those tensors on the existing model; neither the critic nor another copy of
the frozen 35B MoE backbone is created. The stored support and sampling
transform belong to each decision. Eq. (21) differentiates the current actor
only: p_ref is detached, includes the registered token/plan temperature and
exploration transform, and is evaluated on the same recorded support Y_g,j.
"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
import hashlib
import json
from pathlib import Path

import torch

from .objectives import mixture_probabilities


def _contract(model):
    config = getattr(model, "config", {})
    config = asdict(config) if is_dataclass(config) else dict(config) if isinstance(config, dict) else vars(config)
    config = {key: value for key, value in config.items() if key != "device"}
    result = {"model_class": type(model).__qualname__, "model_config": config}
    if config.get("architecture_version") == 3:
        from .frozen_text import node_text_cache
        result["frozen_text_snapshot"] = node_text_cache(model).validate_identity()
    # JSON is the persisted identity representation (tuples become lists).
    return json.loads(json.dumps(result, sort_keys=True, default=str))


def _digest(contract, weights):
    digest = hashlib.sha256(json.dumps(contract, sort_keys=True, default=str).encode())
    for name, value in sorted(weights.items()):
        digest.update(json.dumps((name, list(value.shape), str(value.dtype))).encode())
        digest.update(value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return "frozen-actor-reference-v1:sha256:" + digest.hexdigest()


class _ActorReplay(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, decisions):
        if decisions[0].kind == "edit_token" and hasattr(self.model, "edit_token_logits"):
            return self.model.edit_token_logits(decisions)
        return tuple(self.model.decision_logits(decision) for decision in decisions)


class FrozenActorReference:
    schema = "frozen_actor_reference_v1"

    def __init__(self, model, *, weights=None, contract=None, provenance=None):
        from .trainer import _parameter_role
        self.contract = _contract(model)
        if contract is not None and contract != self.contract:
            raise ValueError("Frozen actor reference base/configuration differs from the live model")
        parameters = {name: parameter for name, parameter in model.named_parameters()
                      if parameter.requires_grad and _parameter_role(name) in {"draft", "policy"}}
        if not parameters:
            raise ValueError("A frozen actor reference needs registered trainable actor parameters")
        if weights is not None and set(weights) != set(parameters):
            raise ValueError("Frozen actor reference tensor names do not match the live actor")
        self.weights = {}
        for name, parameter in parameters.items():
            value = weights[name] if weights is not None else parameter
            if value.shape != parameter.shape or value.dtype != parameter.dtype or not torch.isfinite(value).all():
                raise ValueError("Frozen actor reference tensor shape/dtype/value is invalid")
            self.weights[name] = value.detach().to(parameter.device).clone()
        self.provenance = dict(provenance or {})
        self.identity = _digest(self.contract, self.weights)
        self._versions = {name: value._version for name, value in self.weights.items()}

    def validate_model(self, model):
        if _contract(model) != self.contract:
            raise ValueError("Frozen actor reference base/configuration changed")
        if any(self.weights[name]._version != version for name, version in self._versions.items()):
            raise RuntimeError("Frozen actor reference tensors changed")

    @torch.no_grad()
    def probabilities(self, model, decisions):
        if not decisions:
            return ()
        if any(self.weights[name]._version != version for name, version in self._versions.items()):
            raise RuntimeError("Frozen actor reference tensors changed")
        # Replay first, before any live forward graph. PEFT can change active
        # adapter/gradient flags; restore both even when a reference call fails.
        modes = [(module, module.training) for module in model.modules()]
        gradients = [(parameter, parameter.requires_grad) for parameter in model.parameters()]
        backbone = getattr(model, "backbone", None)
        active = getattr(backbone, "active_adapter", None)
        try:
            model.eval()
            rows = torch.func.functional_call(_ActorReplay(model),
                {"model." + name: value for name, value in self.weights.items()}, (tuple(decisions),), strict=False)
            result = []
            for decision, logits in zip(decisions, rows):
                exploration = logits.new_tensor(decision.exploration_probabilities) if decision.exploration_probabilities else None
                result.append(mixture_probabilities(logits, decision.temperature, decision.exploration, exploration).detach())
            if len(result) != len(decisions):
                raise ValueError("Frozen actor replay returned incomplete decision probabilities")
            return tuple(result)
        finally:
            if active is not None and hasattr(backbone, "set_adapter"):
                backbone.set_adapter(active)
            for parameter, requires_grad in gradients:
                parameter.requires_grad_(requires_grad)
            for value in self.weights.values():
                value.requires_grad_(False)
            for module, training in modes:
                module.training = training

    def manifest(self):
        return {"schema": self.schema, "identity": self.identity, "contract": self.contract,
                "provenance": self.provenance, "tensor_names": sorted(self.weights),
                "parameter_count": sum(value.numel() for value in self.weights.values()),
                "direction": "current_to_reference", "estimator": "full_recorded_support_sum",
                "scope": "actor_only_shared_immutable_backbone_no_critic_copy"}

    def save(self, directory):
        directory = Path(directory)
        if _digest(self.contract, self.weights) != self.identity:
            raise RuntimeError("Cannot save mutated frozen actor reference")
        torch.save({name: value.detach().cpu() for name, value in self.weights.items()}, directory / "actor_reference.pt")
        (directory / "actor_reference.json").write_text(json.dumps(self.manifest(), indent=2))

    @classmethod
    def load(cls, directory, model):
        directory = Path(directory)
        manifest = json.loads((directory / "actor_reference.json").read_text())
        if manifest.get("schema") != cls.schema:
            raise ValueError("Unknown frozen actor reference schema")
        weights = torch.load(directory / "actor_reference.pt", map_location="cpu", weights_only=True)
        reference = cls(model, weights=weights, contract=manifest["contract"], provenance=manifest["provenance"])
        if reference.identity != manifest["identity"] or reference.manifest() != manifest:
            raise ValueError("Frozen actor reference identity or manifest mismatch")
        return reference
