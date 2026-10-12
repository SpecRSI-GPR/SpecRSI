"""Native Qwen3.5 text backbone with separate Draft, Policy and Q roles.

Reference training loads the pinned BF16 Qwen3.5-35B-A3B text tower once. Its
shared weights, MoE experts and routing gates stay frozen; Draft theta uses
attention LoRA across both Gated DeltaNet and full-attention layers, while
Policy psi and Q phi have independent relation branches. Version-3 training
excludes the compatibility-only verify adapter and auxiliary heads from the
optimizer. The registered model configuration records the precise loading and
adapter contracts, and checkpoints retain their resolved native module paths.

Dependencies are imported lazily so CPU core tests need only PyTorch. Injected
tiny models exercise software contracts without downloading reference weights;
version-2 roles remain available for explicitly configured compatibility tests.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from contextlib import contextmanager
from functools import partial
import inspect
import json
import re
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .heads import SurfaceHead, ContinuationValueHead, RelationalPlanBranch, TinyRoleModel
from .records import DecisionRecord, EditExample


def _layer_state(layer, *, evaluation_gate=False):
    """Capture only one decoder layer, without retaining another weight copy."""
    from peft.tuners.tuners_utils import BaseTunerLayer

    modes = tuple((module, False if module is layer and evaluation_gate else module.training)
                  for module in layer.modules())
    adapters = tuple((module, tuple(module.active_adapters), module.disable_adapters)
                     for module in layer.modules() if isinstance(module, BaseTunerLayer))
    gradients = tuple((parameter, parameter.requires_grad) for parameter in layer.parameters())
    return modes, adapters, gradients


def _apply_layer_state(state):
    modes, adapters, gradients = state
    for module, training in modes:
        module.training = training
    for module, active, disabled in adapters:
        if tuple(module.active_adapters) != active:
            module.set_adapter(list(active))
        if module.disable_adapters != disabled:
            module.enable_adapters(not disabled)
    # PEFT adapter setters change these flags. Preserve the original graph's
    # trainable parameters without thawing frozen legacy adapter parameters.
    for parameter, requires_grad in gradients:
        if parameter.requires_grad != requires_grad:
            parameter.requires_grad_(requires_grad)


@contextmanager
def _replay_layer_state(layer, state):
    previous = _layer_state(layer)
    try:
        _apply_layer_state(state)
        yield
    finally:
        _apply_layer_state(previous)


def _checkpoint_with_layer_state(function, *args, **kwargs):
    """Bind dropout and PEFT role state across non-reentrant recomputation."""
    from torch.utils.checkpoint import checkpoint

    bound = function
    while isinstance(bound, partial):
        bound = bound.func
    layer = bound.__self__
    state = _layer_state(layer, evaluation_gate=getattr(layer, "_specrsi_evaluation_gate", False))

    def replay(*values, **options):
        # Scope state to the actual function call. A checkpoint context_fn's
        # forward generator can unwind late after a failing forward in Torch
        # 2.6; this explicit finally restores state before the caller proceeds.
        with _replay_layer_state(layer, state):
            return function(*values, **options)

    return checkpoint(replay, *args, use_reentrant=False, **kwargs)


@dataclass
class HFModelConfig:
    model_path: str
    architecture_version: int = 2
    option_dim: int = 8
    head_hidden_dim: int = 256
    feature_dim: int = 8
    lora_rank: int = 16
    lora_alpha: float = 32.
    lora_dropout: float = 0.
    target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    dtype: str = "bfloat16"
    device: str = "cuda:1"
    max_length: int = 2048
    gradient_checkpointing: bool = True
    revision: str | None = None
    semantic_budget_mode: str = "legacy_equal"
    semantic_global_tokens: int | None = None
    semantic_node_min_tokens: int | None = None
    semantic_node_max_tokens: int | None = None
    semantic_total_tokens: int | None = None
    semantic_truncation: str = "head_tail"
    graph_schema: str = "candidate_items_v1"
    role_observation_version: str = "legacy_v1"
    frozen_global_cache: bool = False
    # Explicit loading/adapter contracts are persisted with the learned roles.
    # In particular, requesting BF16 must never silently dequantize an FP8 repo.
    language_model_only: bool = False
    require_unquantized_backbone: bool = False
    lora_target_profile: str = "configured"
    # Freeze the implementation choice as well as the weights. Transformers
    # 5.17 can auto-select grouped_mm on a CUDA host even under Torch 2.6,
    # whose fallback backward allocates unused full expert-weight gradients.
    experts_implementation: str = "eager"
    # Storage transport only: full reads avoid parallel mmap page-fault stalls
    # on the project's shared filesystem. Tensor bytes/dtypes stay unchanged.
    disable_mmap: bool = True

    def __post_init__(self):
        from .frozen_text import semantic_budget, role_observation_contract
        from .work_graph import graph_schema
        semantic_budget(self)
        graph_schema(self)
        role_observation_contract(self)
        if self.lora_target_profile not in {"configured", "qwen35_hybrid_attention"}:
            raise ValueError("Unknown registered LoRA target profile")
        if type(self.language_model_only) is not bool or type(self.require_unquantized_backbone) is not bool:
            raise ValueError("Backbone loading contracts must be booleans")
        if type(self.disable_mmap) is not bool:
            raise ValueError("disable_mmap loading contract must be boolean")
        if self.experts_implementation != "eager":
            raise ValueError("This frozen-MoE training profile requires the registered eager expert implementation")


def _backbone_factory(model_type: str, transformers_module=None, *, language_model_only=False):
    """Choose only registered local model classes; old text-only models keep AutoModel."""
    if transformers_module is None:
        import transformers as transformers_module
    conditional = {
        "qwen3_5": "Qwen3_5ForConditionalGeneration",
        "qwen3_5_moe": "Qwen3_5MoeForConditionalGeneration",
    }
    if language_model_only:
        # The native causal classes load the publisher's text tower directly;
        # no randomly initialized replacement decoder or second base is built.
        conditional.update({"qwen3_5": "Qwen3_5ForCausalLM",
                            "qwen3_5_moe": "Qwen3_5MoeForCausalLM"})
    if model_type not in conditional:
        return transformers_module.AutoModelForCausalLM
    class_name = conditional[model_type]
    factory = getattr(transformers_module, class_name, None)
    if factory is None:
        raise RuntimeError(
            f"Installed Transformers does not provide {class_name}; use a separately "
            "validated Transformers/PEFT environment with Qwen3.5 support")
    return factory


def _text_hidden_size(model_config):
    """The role heads operate on decoder states, never on the vision tower width."""
    from collections.abc import Mapping
    text_config = getattr(model_config, "text_config", None)
    if text_config is None:
        text_config = model_config
    hidden = (text_config.get("hidden_size") if isinstance(text_config, Mapping)
              else getattr(text_config, "hidden_size", None))
    if type(hidden) is not int or hidden < 1:
        raise ValueError("Model config must declare a positive text hidden_size")
    return hidden


def _validate_unquantized_config(raw):
    """Reject quantized source identities before any large tensor is loaded."""
    for part in (raw, raw.get("text_config", {})):
        if isinstance(part, dict) and part.get("quantization_config"):
            raise ValueError("Unquantized backbone required: provision the official BF16 snapshot, not an FP8/INT4 checkpoint")


def _lora_targets(base_model, config):
    """Register language attention projections across both native layer types.

    Appendix F.5 freezes the shared representation and trains a rank-16 Draft
    adapter theta. In a Qwen3.5 hybrid decoder, q/k/v/o occur only in full-
    attention blocks; the other blocks use Gated DeltaNet input/output/gate
    projections. Selecting only the dense-model suffixes omits those layers.
    Exact module paths keep vision projections, MoE experts, routing gates and
    the vocabulary head frozen. This target set is an explicit implementation
    contract, not a change to the Eq. (18)/(20) likelihood or an added router loss.
    """
    if config.lora_target_profile == "configured":
        return list(config.target_modules)
    model_type = getattr(base_model.config, "model_type", "")
    if model_type not in {"qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text"}:
        raise ValueError("Qwen3.5 hybrid LoRA profile requires a native Qwen3.5 decoder")
    pattern = re.compile(r"(?:.*\.)?layers\.(\d+)\.(self_attn|linear_attn)\.([^.]+)$")
    targets, covered = [], set()
    for name, module in base_model.named_modules():
        match = pattern.fullmatch(name)
        if (match and isinstance(module, nn.Linear)
                and (match[3].endswith("_proj") or match[3].startswith("in_proj_"))):
            targets.append(name)
            covered.add(int(match[1]))
    text_config = getattr(base_model.config, "text_config", base_model.config)
    layer_count = getattr(text_config, "num_hidden_layers", None)
    if not targets or type(layer_count) is not int or covered != set(range(layer_count)):
        raise ValueError("Native hybrid LoRA targets must cover every language decoder layer")
    return targets


def _load_local_backbone(config, dtype):
    directory = Path(config.model_path)
    if not directory.exists():
        raise FileNotFoundError(f"Local model not found: {config.model_path}; provision it before training")
    # Read the local, pinned architecture descriptor before importing a concrete
    # conditional class. No remote architecture code or checkpoint rewrite.
    raw = json.loads((directory / "config.json").read_text())
    if config.require_unquantized_backbone:
        _validate_unquantized_config(raw)
    factory = (_backbone_factory(raw.get("model_type", ""), language_model_only=True)
               if config.language_model_only else _backbone_factory(raw.get("model_type", "")))
    options = ({"experts_implementation": config.experts_implementation}
               if raw.get("model_type") in {"qwen3_5_moe", "qwen3_5_moe_text"} else {})
    return factory.from_pretrained(config.model_path, local_files_only=True,
        revision=config.revision, torch_dtype=dtype, trust_remote_code=False,
        disable_mmap=config.disable_mmap, **options)


def _restore_registered_adapter(backbone, weights, adapter):
    """Restore only this implementation's exact native LoRA A/B tensor slots.

    save_pretrained removes the named adapter segment (draft/verify) from
    the keys. Insert that segment using the already registered parameter map;
    validate the complete key set and shapes before modifying any tensor.
    Our checkpoint already uses the loaded native decoder's module paths, so
    no publisher-to-native expert stacking/fusion conversion is appropriate.
    In particular this avoids PEFT 0.19.1's incompatible automatic MoE weight
    converter on Transformers 5.17 while never monkeypatching either library.
    """
    pattern = re.compile(r"^(.*\.lora_[AB])\." + re.escape(adapter) + r"\.(.+)$")
    slots = {}
    for name, parameter in backbone.named_parameters():
        match = pattern.fullmatch(name)
        if match:
            slots[match[1] + '.' + match[2]] = parameter
    if not slots or set(weights) != set(slots):
        raise ValueError(f"Checkpoint {adapter} must contain exactly the registered native LoRA tensors")
    for name, parameter in slots.items():
        if weights[name].shape != parameter.shape or weights[name].dtype != parameter.dtype:
            raise ValueError(f"Checkpoint {adapter} LoRA tensor shape/dtype mismatch: {name}")
        if not torch.isfinite(weights[name]).all():
            raise ValueError(f"Checkpoint {adapter} contains nonfinite LoRA tensor: {name}")
    with torch.no_grad():
        for name, parameter in slots.items():
            parameter.copy_(weights[name])


class HFRoleModel(nn.Module):
    backend = "hf"

    def __init__(self, config: HFModelConfig, *, base_model=None, tokenizer=None):
        super().__init__()
        try:
            from peft import LoraConfig, TaskType, get_peft_model
            from transformers import AutoTokenizer
        except (ImportError, ModuleNotFoundError) as exc:
            raise RuntimeError("Compatible transformers/peft/huggingface_hub are required; install the pinned project model dependencies") from exc
        self.config = config
        if config.architecture_version not in {2, 3}:
            raise ValueError("PDF roles require architecture_version=2 or explicit semantic-node version 3")
        if config.architecture_version == 3 and (config.feature_dim < 8 or config.option_dim < 12):
            raise ValueError("v3 roles require feature_dim >= 8 and option_dim >= 12")
        dtype = getattr(torch, config.dtype)
        if base_model is None:
            base_model = _load_local_backbone(config, dtype)
        if getattr(base_model.config, 'model_type', '') in {'qwen3_5_moe', 'qwen3_5_moe_text'}:
            # Also enforce the contract for explicitly injected native test
            # models. No per-token expert-weight replication or unnecessary
            # frozen grad_weight tensor is introduced by the eager loop.
            base_model.set_experts_implementation(config.experts_implementation)
        if config.require_unquantized_backbone:
            _validate_unquantized_config(base_model.config.to_dict())
            if getattr(base_model, "is_quantized", False):
                raise ValueError("Unquantized backbone required; injected model is quantized")
        # Qwen projects requested output positions; other HF models fall back.
        self._supports_logit_positions = "logits_to_keep" in inspect.signature(base_model.forward).parameters
        base_model.requires_grad_(False)
        targets = _lora_targets(base_model, config)
        self.resolved_lora_targets = tuple(targets)
        lora = LoraConfig(task_type=TaskType.CAUSAL_LM, r=config.lora_rank, lora_alpha=config.lora_alpha,
                          lora_dropout=config.lora_dropout, target_modules=targets, bias="none")
        self.backbone = get_peft_model(base_model, lora, adapter_name="draft")
        self.backbone.add_adapter("verify", lora)
        if tokenizer is not None:
            self.tokenizer = tokenizer
        elif Path(config.model_path).exists():
            self.tokenizer = AutoTokenizer.from_pretrained(config.model_path, local_files_only=True, trust_remote_code=False)
        else:
            self.tokenizer = None  # Injected tiny transformer tests use explicit input_ids.
        hidden = _text_hidden_size(base_model.config)
        self.surface_head = SurfaceHead(hidden)
        semantic = {"node_text_dim": hidden} if config.architecture_version == 3 else {}
        from .work_graph import graph_schema, relation_count
        semantic["relations"] = relation_count(graph_schema(config))
        self.value_head = RelationalPlanBranch(hidden + config.feature_dim, config.option_dim, config.head_hidden_dim, **semantic)
        self.policy_head = RelationalPlanBranch(hidden + config.feature_dim, config.option_dim, config.head_hidden_dim, **semantic)
        self.edit_head = ContinuationValueHead(hidden, config.option_dim, config.head_hidden_dim)
        if config.architecture_version == 3:
            # Paper F.5 / Eq. (23) train only Draft theta, Policy psi and Q phi.
            # Retain historical modules and tensor names for checkpoint/output
            # compatibility, but never register these unused auxiliaries with
            # the optimizer or the frozen SFT actor reference.
            self.surface_head.requires_grad_(False)
            self.edit_head.requires_grad_(False)
        self.to(config.device)
        if config.gradient_checkpointing:
            self.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            self.backbone.enable_input_require_grads()
            from transformers.modeling_layers import GradientCheckpointingLayer
            for module in self.backbone.modules():
                if isinstance(module, GradientCheckpointingLayer) and module.gradient_checkpointing:
                    module._gradient_checkpointing_func = _checkpoint_with_layer_state
        self._activate("verify")

    @property
    def device(self):
        return next(self.parameters()).device

    def _activate(self, adapter: str):
        self.backbone.set_adapter(adapter)
        # PEFT set_adapter toggles requires_grad. Eq. (23)'s v3 reference has
        # only the Draft adapter among trainable transformer parameters; reset
        # flags after EVERY switch so invoking a legacy verify path cannot
        # silently introduce another learned role. Preserve v2's old contract.
        for name, parameter in self.backbone.named_parameters():
            parameter.requires_grad_("lora_" in name and
                (self.config.architecture_version != 3 or ".draft." in name))

    @contextmanager
    def _gradient_forward(self):
        """Use block checkpointing while keeping replay's deterministic eval semantics.

        HF gates checkpointing on the decoder layer's ``training`` flag. Only
        that gate is opened here; its actual forward executes with the captured
        eval flags, including functional attention dropout and LoRA dropout.
        No-grad collection and frozen semantic encodings retain their old path.
        """
        if not self.config.gradient_checkpointing or not torch.is_grad_enabled() or self.training:
            yield
            return
        from transformers.modeling_layers import GradientCheckpointingLayer
        layers = [module for module in self.backbone.modules()
                  if isinstance(module, GradientCheckpointingLayer) and module.gradient_checkpointing]
        if not layers:
            raise RuntimeError("Deterministic checkpointing requires HF GradientCheckpointingLayer support")
        previous = [(layer, layer.training, getattr(layer, "_specrsi_evaluation_gate", False)) for layer in layers]
        try:
            for layer in layers:
                layer.training = True
                layer._specrsi_evaluation_gate = True
            yield
        finally:
            for layer, training, gate in previous:
                layer.training = training
                layer._specrsi_evaluation_gate = gate

    def _tokens(self, text: str = "", input_ids=()):
        if input_ids:
            if len(input_ids) > self.config.max_length:
                raise ValueError("Recorded context exceeds max_length; silently truncating changes policy likelihood")
            ids = torch.tensor([list(input_ids)], dtype=torch.long, device=self.device)
            return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
        if self.tokenizer is None or not text:
            raise ValueError("HF records require context_text or exact input_ids")
        tokens = self.tokenizer(text, return_tensors="pt", truncation=False)
        if tokens["input_ids"].shape[-1] > self.config.max_length:
            raise ValueError("Text context exceeds max_length; prepare explicit bounded contexts before collection")
        return {key: value.to(self.device) for key, value in tokens.items()}

    def _encode_frozen_text(self, text: str, input_ids=()):
        """One real frozen transformer forward, without a role-head cache."""
        tokens = self._tokens(text, input_ids)
        gradients = [(parameter, parameter.requires_grad) for name, parameter in self.backbone.named_parameters()
                     if "lora_" in name]
        modes = [(module, module.training) for module in self.backbone.modules()] if self.config.frozen_global_cache else ()
        try:
            # The opt-in cache stores a deterministic frozen representation,
            # including when an outer training caller left modules in train mode.
            # Restore each module individually; do not change actor dropout.
            for module, _ in modes:
                module.training = False
            with self.backbone.disable_adapter(), torch.no_grad():
                output = self.backbone.get_base_model().base_model(**tokens, use_cache=False, return_dict=True)
        finally:
            # PEFT reenables only the current adapter when leaving this context.
            for parameter, requires_grad in gradients:
                parameter.requires_grad_(requires_grad)
            for module, training in modes:
                module.training = training
        positions = tokens["attention_mask"].sum(-1) - 1
        return output.last_hidden_state[torch.arange(len(positions), device=self.device), positions].float().squeeze(0).detach()

    def encode(self, text: str, role="verify", input_ids=()):
        if role in {"value", "policy"}:
            if not self.config.frozen_global_cache:
                return self._encode_frozen_text(text, input_ids=input_ids)
            from .frozen_text import node_text_cache
            cache = node_text_cache(self)
            ids = tuple(input_ids) or tuple(self.tokenizer.encode(text))
            self._last_global_encoding = None
            try:
                values, receipt = cache.encode(text, input_ids=ids)
            except Exception as error:
                receipt = getattr(error, "node_text_receipt", None)
                if receipt is not None:
                    self._last_global_encoding = {"encoder_cache_hit": False, "encoder_forward_count": 1,
                        "encoder_snapshot": receipt["encoder_snapshot"], "context_input_token_ids": list(ids)}
                raise
            self._last_global_encoding = {"encoder_cache_hit": receipt is None,
                "encoder_forward_count": int(receipt is not None), "encoder_snapshot": cache.identity,
                "context_input_token_ids": list(ids)}
            return torch.tensor(values, dtype=torch.float32, device=self.device)
        self._activate(role)
        tokens = self._tokens(text, input_ids)
        # The injected LoRA modules also apply in the base transformer. Avoid
        # allocating vocabulary logits for surface and continuation value heads.
        with self._gradient_forward():
            output = self.backbone.get_base_model().base_model(**tokens, use_cache=False, return_dict=True)
        positions = tokens["attention_mask"].sum(-1) - 1
        return output.last_hidden_state[torch.arange(len(positions), device=self.device), positions].float().squeeze(0)

    def decision_values(self, decision: DecisionRecord):
        self._last_global_encoding = None
        self._last_role_head_started = False
        if decision.kind == "edit_token":
            return self.edit_token_logits((decision,))[0]
        if decision.kind != "edit":
            from .work_graph import validate_role_graph
            validate_role_graph(self, decision)
        hidden = self.encode(decision.context_text, "draft" if decision.kind == "edit" else "value", decision.input_ids)
        if decision.kind != "edit":
            hidden = self._with_numeric_state(hidden, decision)
        options = torch.as_tensor(decision.option_features, dtype=torch.float32, device=self.device)
        if options.shape != (len(decision.support), self.config.option_dim):
            raise ValueError("Stored option features disagree with configured option_dim")
        head = self.edit_head if decision.kind == "edit" else self.value_head
        if decision.option_item_features:
            return torch.stack([head(hidden, torch.tensor(items, dtype=torch.float32, device=self.device)).sum()
                                if items else hidden.sum()*0 for items in decision.option_item_features])
        if decision.kind == "edit":
            return head(hidden, options)
        graph = self._graph(decision)
        self._last_role_head_started = True
        values = head(hidden, options, **graph)
        mask = torch.tensor([option == "ACCEPT" for option in decision.support], device=self.device)
        return values.masked_fill(mask, 0.)

    _graph = TinyRoleModel._graph

    @property
    def action_head(self):
        return self.value_head

    @property
    def batch_head(self):
        return self.value_head

    def _with_numeric_state(self, hidden, decision):
        features = decision.state_features or (0.,) * self.config.feature_dim
        if len(features) != self.config.feature_dim:
            raise ValueError("Stored numeric state disagrees with configured feature_dim")
        return torch.cat((hidden, hidden.new_tensor(features)))

    def decision_logits(self, decision: DecisionRecord):
        if decision.kind in {"edit", "edit_token"}:
            return self.decision_values(decision)
        self._last_global_encoding = None
        self._last_role_head_started = False
        from .work_graph import validate_role_graph
        validate_role_graph(self, decision)
        hidden = self._with_numeric_state(self.encode(decision.context_text, "policy", decision.input_ids), decision)
        options = torch.as_tensor(decision.option_features, dtype=torch.float32, device=self.device)
        if options.shape != (len(decision.support), self.config.option_dim):
            raise ValueError("Stored option features disagree with configured option_dim")
        graph = self._graph(decision)
        self._last_role_head_started = True
        logits = self.policy_head(hidden, options, **graph)
        mask = torch.tensor([option == "ACCEPT" for option in decision.support], device=self.device)
        return logits.masked_fill(mask, 0.)

    def edit_token_logits(self, decisions):
        """Replay one recorded edit in one causal transformer forward.

        Every row retains its own prefix and frozen support. Causal masking
        makes its logits identical to replaying that prefix separately; future
        sampled tokens are inputs only to later rows. This avoids a complete
        forward/backward through every growing prefix during GVO.
        """
        if not decisions:
            return ()
        previous = None
        for decision in decisions:
            if decision.kind != "edit_token" or not decision.input_ids:
                raise ValueError("Token replay requires recorded edit-token prefixes")
            if previous is not None and decision.input_ids != previous.input_ids + (
                    int(previous.support[previous.selected_index]),):
                raise ValueError("Batched token replay must follow one contiguous sampled edit")
            previous = decision
        self._activate("draft")
        tokens = self._tokens(input_ids=decisions[-1].input_ids)
        logits = self._token_position_logits(tokens, tuple(len(decision.input_ids) - 1 for decision in decisions))
        rows = []
        for row, decision in zip(logits, decisions):
            indices = torch.tensor([int(x) for x in decision.support], dtype=torch.long, device=self.device)
            if indices.min() < 0 or indices.max() >= row.shape[-1]:
                raise ValueError("Stored token support is outside model vocabulary")
            rows.append(row[indices].float())
        return tuple(rows)

    def _token_position_logits(self, tokens, positions):
        """Preserve full causal context but project only observed output positions."""
        length = tokens["input_ids"].shape[-1]
        if not positions or any(type(position) is not int or not 0 <= position < length for position in positions):
            raise ValueError("Recorded token positions must lie inside their complete input context")
        indices = torch.tensor(positions, dtype=torch.long, device=self.device)
        options = {"logits_to_keep": indices} if self._supports_logit_positions else {}
        with self._gradient_forward():
            output = self.backbone(**tokens, use_cache=False, return_dict=True, **options)
        logits = output.logits[0] if self._supports_logit_positions else output.logits[0, indices]
        if logits.shape[0] != len(positions):
            raise ValueError("Backbone did not honor the recorded output-position contract")
        return logits

    def surface_logits(self, features=(), context_text=""):
        return self.surface_head(self.encode(context_text, "verify"))

    def edit_nll(self, example: EditExample):
        # Eq. (18), the edit term of
        #   L_SFT(theta,psi) = -E_{(H,e)~D_edit^sup}[log pi_theta(e|H)]
        #       - E_{(S,R,Omega,omega)~D_plan^sup}[log pi_psi(omega|S,R,Omega)].
        # For e=(y_1,...,y_M), with a sampled EOS included as y_M when present:
        #   pi_theta(e|H) = product_{j=1}^{M} pi_theta(y_j|H,y_<j,Y_j),
        #   -log pi_theta(e|H)
        #       = -sum_{j=1}^{M} log pi_theta(y_j|H,y_<j,Y_j).
        # This function returns that ONE edit's negative log likelihood. The
        # trainer averages over |D_edit^sup| edits, separately from the legal
        # plan mean; dividing here by M would reweight edits by token length.
        # ``input_ids`` retains the actual parent/history prompt and generated
        # prefix. Each stored support Y_j, temperature and exploration mixture
        # is replayed as collected. Forced grammar punctuation has probability
        # one and contributes log(1)=0. Sampled EOS is a real likelihood term,
        # whereas an unsampled output-cap termination contributes no extra token.
        self._activate("draft")
        tokens = self._tokens(input_ids=example.input_ids)
        if example.token_decisions:
            from .objectives import mixture_probabilities
            sampled = tuple(decision for decision in example.token_decisions if not decision.deterministic)
            positions = tuple(len(decision.input_ids) - 1 for decision in sampled)
            # All-forced edits retain their graph-connected zero objective.
            logits = self._token_position_logits(tokens, positions or (len(example.input_ids) - 1,))
            losses = []
            for position, decision in enumerate(sampled):
                ids = torch.tensor([int(x) for x in decision.support], device=self.device)
                row = logits[position, ids].float()
                exploration = row.new_tensor(decision.exploration_probabilities) if decision.exploration_probabilities else None
                probabilities = mixture_probabilities(row, decision.temperature, decision.exploration, exploration)
                losses.append(-probabilities[decision.selected_index].log())
            return torch.stack(losses).sum() if losses else logits.sum() * 0.
        # Eq. (18) still normalizes over the FULL vocabulary at every supervised
        # continuation position. Prompt labels -100 have no NLL term, so avoid
        # materializing their [prompt_length, vocab_size] logits. On the 35B
        # checkpoint, vocab_size=248320: projecting an 8192-token prompt would
        # otherwise allocate several GiB unrelated to the supervised objective.
        # This selects positions, never a vocabulary subset or token-average.
        labels = torch.tensor(example.labels[1:], dtype=torch.long, device=self.device)
        positions = tuple(torch.nonzero(labels != -100, as_tuple=False).flatten().tolist())
        if not positions:
            return self._token_position_logits(tokens, (len(example.input_ids) - 1,)).sum() * 0.
        logits = self._token_position_logits(tokens, positions).float()
        # An edit's log probability sums supervised continuation token logs.
        return F.cross_entropy(logits, labels[list(positions)], reduction="sum")

    def save(self, directory: str | Path):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        manifest = {"backend": self.backend, **asdict(self.config)}
        manifest["resolved_lora_targets"] = list(self.resolved_lora_targets)
        if self.config.architecture_version == 3:
            from .frozen_text import node_text_cache
            manifest["frozen_text_snapshot"] = node_text_cache(self).validate_identity()
        (directory / "model_config.json").write_text(json.dumps(manifest, indent=2))
        self.backbone.save_pretrained(directory / "adapters", selected_adapters=["draft", "verify"], safe_serialization=True)
        torch.save({key: value for key, value in self.state_dict().items() if not key.startswith("backbone.")}, directory / "heads.pt")
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(directory / "tokenizer")

    @classmethod
    def load(cls, directory: str | Path, device=None):
        directory = Path(directory)
        raw = json.loads((directory / "model_config.json").read_text())
        raw.pop("backend")
        expected_targets = raw.pop("resolved_lora_targets", None)
        expected_text = raw.pop("frozen_text_snapshot", None)
        if raw.get("architecture_version") not in {2, 3}:
            raise ValueError("Legacy coupled-Q checkpoints require explicit migration and fresh policy training")
        if raw["architecture_version"] == 3 and not expected_text:
            raise ValueError("v3 checkpoint is missing its frozen text encoder identity")
        if device is not None:
            raw["device"] = device
        config = HFModelConfig(**raw)
        if config.architecture_version == 3:
            from transformers import AutoTokenizer
            if not (directory / "tokenizer").is_dir():
                raise ValueError("v3 checkpoint requires its saved frozen tokenizer")
            tokenizer = AutoTokenizer.from_pretrained(directory / "tokenizer", local_files_only=True, trust_remote_code=False)
            model = cls(config, tokenizer=tokenizer)
        else:
            model = cls(config)
        if expected_targets is not None and list(model.resolved_lora_targets) != expected_targets:
            raise ValueError("Checkpoint LoRA target paths differ from the native backbone")
        # Load weights into existing named adapters without allocating a second
        # base model. load_peft_weights respects safetensors and local-only paths.
        from peft.utils.save_and_load import load_peft_weights
        for adapter in ("draft", "verify"):
            weights = load_peft_weights(str(directory / "adapters" / adapter), device=str(model.device))
            _restore_registered_adapter(model.backbone, weights, adapter)
        heads = torch.load(directory / "heads.pt", map_location=model.device, weights_only=True)
        if config.architecture_version == 2 and any("node_text_projection" in key for key in heads):
            raise ValueError("v2 checkpoint cannot silently load v3 semantic-head parameters")
        if config.architecture_version == 3:
            expected = {key for key in model.state_dict() if not key.startswith("backbone.")}
            if heads.keys() != expected:
                raise ValueError("v3 checkpoint requires its exact semantic-head parameter set")
            from .frozen_text import node_text_cache
            if node_text_cache(model).validate_identity() != expected_text:
                raise ValueError("Checkpoint frozen text encoder identity does not match its weights")
        model.load_state_dict(heads, strict=False)
        return model
