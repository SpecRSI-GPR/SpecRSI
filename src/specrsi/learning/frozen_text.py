"""Versioned, detached node text features for the explicit v3 role profile.

The cache belongs to one role-model instance. Its identity hashes actual frozen
weights, not just a model name, and a parameter-version guard invalidates entries
after an in-place weight update. Recorded vectors make replay independent of
cache residency. No task reference, target or unobserved outcome is accepted.
"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
import hashlib
import json
import math
import time

import torch


SEMANTIC_BUDGET_FIELDS = (
    "semantic_budget_mode", "semantic_global_tokens", "semantic_node_min_tokens",
    "semantic_node_max_tokens", "semantic_total_tokens", "semantic_truncation",
)

ROLE_OBSERVATION_FIELDS = ("role_observation_version", "frozen_global_cache")


def role_observation_contract(config):
    """Explicit opt-ins; old checkpoints keep their exact observation and bill."""
    read = config.get if isinstance(config, dict) else lambda key, default=None: getattr(config, key, default)
    version = read("role_observation_version", "legacy_v1")
    cached = read("frozen_global_cache", False)
    if version not in {"legacy_v1", "observed_results_v2"} or type(cached) is not bool:
        raise ValueError("Invalid registered role observation/cache contract")
    if (version != "legacy_v1" or cached) and read("architecture_version", 2) != 3:
        raise ValueError("New role observations/global text cache require architecture_version=3")
    if version != "legacy_v1" and read("model_path") is not None and read("semantic_budget_mode", "legacy_equal") != "redistribute":
        raise ValueError("Observed result text requires a registered semantic redistribution budget")
    return version, cached


def semantic_budget(config):
    """Validate the explicitly registered HF semantic observation contract.

    Legacy checkpoints omit every new field. Their old equal-share observation
    and encoder fingerprint remain exactly unchanged, including its truncation.
    A total budget may exceed one transformer context: each independent global
    or node encoding must still fit ``max_length``.
    """
    read = config.get if isinstance(config, dict) else lambda key, default=None: getattr(config, key, default)
    mode = read("semantic_budget_mode", "legacy_equal")
    fields = {key: read(key) for key in SEMANTIC_BUDGET_FIELDS[1:-1]}
    truncation = read("semantic_truncation", "head_tail")
    if mode == "legacy_equal":
        if any(value is not None for value in fields.values()) or truncation != "head_tail":
            raise ValueError("Semantic budget overrides require explicit semantic_budget_mode='redistribute'")
        return {"mode": mode, "total": read("max_length", 2048)}
    if mode != "redistribute" or read("architecture_version", 2) != 3:
        raise ValueError("Semantic redistribution requires architecture_version=3 and mode='redistribute'")
    if any(type(value) is not int or value < 1 for value in fields.values()):
        raise ValueError("Register positive integer semantic global, node minimum, node maximum and total tokens")
    if truncation not in {"head_tail", "prefix"}:
        raise ValueError("Register semantic_truncation='head_tail' or 'prefix'")
    global_cap, minimum, maximum, total = (fields[key] for key in SEMANTIC_BUDGET_FIELDS[1:-1])
    context_cap = read("max_length", 2048)
    if global_cap > context_cap or maximum > context_cap or minimum > maximum or global_cap >= total:
        raise ValueError("Semantic global/node caps must fit max_length, minimum <= maximum, and total > global")
    return {"mode": mode, "global": global_cap, "minimum": minimum,
            "maximum": maximum, "total": total, "truncation": truncation}


def semantic_input_cap(config):
    """Worst-case input-token reservation for one registered scorer/policy call."""
    return semantic_budget(config)["total"]


def allocate_node_tokens(lengths, *, available, minimum, maximum):
    """Stable water filling, using only observed token lengths and fixed caps.

    Short nodes keep their complete input and release unused minimum/equal
    shares. Remaining tokens go equally to unsatisfied nodes; ties follow the
    registered unique-node order. No outcome, reference or learned score enters.
    """
    if (type(available) is not int or available < 0 or type(minimum) is not int or minimum < 1
            or type(maximum) is not int or maximum < minimum
            or any(type(length) is not int or length < 1 for length in lengths)):
        raise ValueError("Node token allocation needs positive lengths and valid integer resource bounds")
    demands = [min(length, maximum) for length in lengths]
    assigned = [min(length, minimum) for length in demands]
    if sum(assigned) > available:
        raise ValueError("Semantic total cap cannot fit every unique node's registered minimum")
    remaining = available - sum(assigned)
    while remaining:
        active = [i for i in range(len(demands)) if assigned[i] < demands[i]]
        if not active:
            break
        share, remainder = divmod(remaining, len(active))
        for position, index in enumerate(active):
            extra = min(demands[index] - assigned[index], share + int(position < remainder))
            assigned[index] += extra
            remaining -= extra
    return tuple(assigned)


def bounded_semantic_text(tokenizer, text, cap, *, strategy="head_tail"):
    """Return the exact text and token IDs admitted by a declared encoding cap.

    Decode/re-encode is checked explicitly: BPE boundaries and special tokens
    can change the length of a string assembled from two excerpts. A decoded
    token fragment must be an actual source prefix/suffix; otherwise its length
    only estimates a slice of the original Unicode string. This prevents a
    byte-level token cut from inventing replacement characters.
    """
    if type(cap) is not int or cap < 1 or strategy not in {"head_tail", "prefix"}:
        raise ValueError("Invalid semantic text cap/truncation strategy")
    complete_ids = tuple(tokenizer.encode(text))
    if len(complete_ids) <= cap:
        return text, complete_ids
    raw = tokenizer.encode(text, add_special_tokens=False)
    def excerpt(ids, *, suffix=False):
        decoded = tokenizer.decode(ids, skip_special_tokens=False)
        if not decoded:
            return ""
        if (text.endswith(decoded) if suffix else text.startswith(decoded)):
            return decoded
        # Python string slicing preserves Unicode code points. Re-encoding
        # below still checks the full joined excerpt against the exact cap.
        return text[-len(decoded):] if suffix else text[:len(decoded)]
    count = min(len(raw), cap)
    while count > 0:
        if strategy == "prefix":
            clipped = excerpt(raw[:count])
        else:
            head, tail = (count + 1) // 2, count // 2
            clipped = excerpt(raw[:head]) + " … "
            if tail:
                clipped += excerpt(raw[-tail:], suffix=True)
        ids = tuple(tokenizer.encode(clipped))
        if clipped and ids and len(ids) <= cap:
            return clipped, ids
        count -= max(1, len(ids) - cap)
    raise ValueError("Semantic token cap cannot fit its explicit text excerpt")


class FrozenNodeTextCache:
    def __init__(self, model):
        self.model = model
        self._signature = None
        self._identity = None
        self._cache = {}
        self._tokenizer_signature = None
        self._tokenizer_digest = None

    def _tokenizer_contract(self, *, refresh=False):
        tokenizer = getattr(self.model, "tokenizer", None)
        if tokenizer is None:
            if self.model.backend == "hf":
                raise ValueError("v3 HF node semantics require a serialized fast tokenizer")
            return None
        backend = getattr(tokenizer, "backend_tokenizer", None)
        if backend is None:
            raise ValueError("v3 HF node semantics require a serialized fast tokenizer")
        settings = {name: getattr(tokenizer, name, None) for name in
                    ("special_tokens_map", "padding_side", "truncation_side", "model_max_length",
                     "clean_up_tokenization_spaces", "chat_template")}
        # These small component states catch ordinary tokenizer reconfiguration
        # without serializing a 250k-entry vocabulary at every scorer call.
        if backend is not None:
            settings["components"] = {name: component.__getstate__() if component is not None else None
                for name in ("normalizer", "pre_tokenizer", "post_processor", "decoder")
                for component in (getattr(backend, name, None),)}
        # Fast-tokenizer len() merges the entire base and added vocabularies;
        # avoid repeating that work for the Qwen3.5 248,320-token vocabulary
        # when the frozen node representation is already cached.
        # Read the base size directly and fingerprint only the small added-token
        # table. Keep these fast mutation hints out of the encoder contract so
        # unchanged tokenizers retain their exact historical identity. Same-size
        # internal base-vocabulary changes still need validate_identity().
        vocabulary_state = ((backend.get_vocab_size(with_added_tokens=False), tuple(
            (index, json.dumps(token.__getstate__(), sort_keys=True, default=str))
            for index, token in sorted(backend.get_added_tokens_decoder().items())))
            if backend is not None else len(tokenizer))
        signature = (id(tokenizer), id(backend), vocabulary_state,
                     json.dumps(settings, sort_keys=True, default=str))
        if refresh or signature != self._tokenizer_signature:
            state = {"backend": backend.to_str() if backend is not None else tokenizer.get_vocab(), **settings}
            self._tokenizer_digest = hashlib.sha256(json.dumps(state, sort_keys=True, default=str).encode()).hexdigest()
            self._tokenizer_signature = signature
        return self._tokenizer_digest

    def validate_identity(self):
        """Full tokenizer verification at search/training/checkpoint boundaries.

        Direct mutation of an internal same-size vocabulary has no cheap public
        version counter. Such mutation is unsupported inside a frozen run and
        detected here, rather than claiming automatic per-query detection.
        """
        self._tokenizer_contract(refresh=True)
        return self.identity

    def _weights(self):
        # LoRA belongs to the learned actors and is disabled for text features.
        return tuple((name, value) for name, value in self.model.backbone.named_parameters()
                     if "lora_" not in name) + tuple(
            ("buffer:" + name, value) for name, value in self.model.backbone.named_buffers())

    @property
    def identity(self):
        weights = self._weights()
        config = self.model.config
        config = asdict(config) if is_dataclass(config) else dict(vars(config))
        config.pop("device", None)
        # File I/O transport cannot change an encoder's mathematical identity.
        config.pop("disable_mmap", None)
        from .work_graph import graph_schema, LEGACY_GRAPH_SCHEMA
        if graph_schema(config) == LEGACY_GRAPH_SCHEMA:
            # Preserve the exact encoder identity of pre-schema checkpoints.
            config.pop("graph_schema", None)
        if semantic_budget(config)["mode"] == "legacy_equal":
            # Adding default dataclass fields must not invalidate an existing
            # checkpoint's saved v1 frozen encoder fingerprint or old replay.
            for key in SEMANTIC_BUDGET_FIELDS:
                config.pop(key, None)
        version, cached = role_observation_contract(config)
        # Merely adding default config fields must not relabel frozen legacy
        # vectors/checkpoints. Explicit opt-ins bind new replay/cache identities.
        if version == "legacy_v1":
            config.pop("role_observation_version", None)
        if not cached:
            config.pop("frozen_global_cache", None)
        encoder_contract = json.dumps({"encoder": "frozen_node_text_v1", "backend": self.model.backend,
            "config": config, "tokenizer_sha256": self._tokenizer_contract()}, sort_keys=True, default=str).encode()
        signature = (hashlib.sha256(encoder_contract).digest(), tuple(
            (name, id(value), value._version, tuple(value.shape), str(value.dtype)) for name, value in weights))
        if signature != self._signature:
            # Placement is not a different frozen encoder. Tensor dtypes and
            # actual contents below still distinguish different precisions.
            identity = hashlib.sha256(encoder_contract)
            for name, value in weights:
                identity.update(json.dumps((name, tuple(value.shape), str(value.dtype))).encode())
                flat = value.detach().reshape(-1)
                # Bound host copies even for a large frozen task backbone.
                chunk = max(1, 4 * 1024 * 1024 // value.element_size())
                for start in range(0, flat.numel(), chunk):
                    raw = flat[start:start + chunk].contiguous().cpu().view(torch.uint8)
                    identity.update(raw.numpy().tobytes())
            self._identity = "frozen_node_text_v1:sha256:" + identity.hexdigest()
            self._signature = signature
            self._cache.clear()
        return self._identity

    @torch.no_grad()
    def encode(self, text, *, input_ids=()):
        identity = self.identity
        if self.model.backend == "hf":
            if input_ids and tuple(self.model.tokenizer.encode(text)) != tuple(input_ids):
                raise ValueError("Recorded semantic text/token IDs disagree before encoding")
            tokens = self.model._tokens(text, input_ids)
            token_ids = tuple(tokens["input_ids"][0].tolist())
        else:
            token_ids = text.encode("utf-8")
        key = (identity, token_ids)
        if key in self._cache:
            return self._cache[key], None
        started = time.monotonic()
        receipt = {"kind": "learned_control", "role": "frozen_node_text",
            "gpu_count": int(str(self.model.device).startswith("cuda")),
            "prompt_tokens": len(token_ids) if self.model.backend == "hf" else 0,
            "completion_tokens": 0, "encoder_snapshot": identity,
            "input_token_ids": list(token_ids) if self.model.backend == "hf" else None,
            "text_sha256": hashlib.sha256(text.encode()).hexdigest()}
        if getattr(self.model.config, "frozen_global_cache", False):
            receipt.update(encoder_forward_count=1, head_forward_count=0, encoder_cache_hit=False)
        try:
            if self.model.backend == "hf":
                # Bypass the optional global cache wrapper: node and global
                # observations share this detached encoder cache, never heads.
                vector = (self.model._encode_frozen_text(text, input_ids=input_ids)
                          if getattr(self.model.config, "frozen_global_cache", False) else
                          self.model.encode(text, "value", input_ids=input_ids))
            else:
                # Disclosed CPU reference only: frozen signed byte-bigram
                # features through the small frozen feature trunk.
                features = torch.zeros(self.model.config.feature_dim, device=self.model.device)
                data = b"\x00" + bytes(token_ids) + b"\x00"
                for index in range(len(data) - 1):
                    value = int.from_bytes(hashlib.sha256(data[index:index + 2]).digest()[:4], "big")
                    features[value % len(features)] += 1. if value & 256 else -1.
                features /= features.norm().clamp_min(1.)
                vector = self.model.backbone(features).tanh()
            values = tuple(float(x) for x in vector.detach().float().cpu())
            if not values or not all(math.isfinite(x) for x in values):
                raise ValueError("Frozen node text encoder returned invalid features")
            # A changed encoder must never publish under the previous key.
            if self.identity != identity:
                raise RuntimeError("Frozen text encoder changed during inference")
        except Exception as error:
            error.node_text_receipt = {**receipt, "status": "error", "error": str(error),
                                      "elapsed_seconds": time.monotonic() - started}
            raise
        self._cache[key] = values
        receipt.update(status="complete", elapsed_seconds=time.monotonic() - started)
        return values, receipt


def node_text_cache(model):
    if model.config.architecture_version != 3:
        raise ValueError("Node text features require the explicit architecture_version=3 profile")
    cache = getattr(model, "_frozen_node_text_cache", None)
    if cache is None:
        cache = FrozenNodeTextCache(model)
        model._frozen_node_text_cache = cache
    return cache
