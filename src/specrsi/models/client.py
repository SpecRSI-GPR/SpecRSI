from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from typing import Any
import httpx
import math
import hashlib
import fcntl
from pathlib import Path


EXTENDED_SAMPLING_PARAMETERS = ("top_k", "min_p", "presence_penalty", "repetition_penalty")


def sampling_options(generation: Mapping) -> dict[str, int | float]:
    """Validate explicit optional sampler controls without changing old payloads.

    Bounds match the pinned vLLM contract: top_k=0 disables truncation (-1 is
    its legacy alias), min_p is a probability, and penalties must be finite.
    Other generation settings are intentionally left to their existing owners.
    """
    if not isinstance(generation, Mapping):
        raise ValueError("generation must be a mapping")
    result = {}
    for name in EXTENDED_SAMPLING_PARAMETERS:
        value = generation.get(name)
        if value is None:
            continue
        if name == "top_k":
            if type(value) is not int or value < -1:
                raise ValueError("top_k must be an integer >= -1; -1 and 0 disable top-k")
        else:
            try:
                finite = type(value) in (int, float) and math.isfinite(value)
            except OverflowError:
                finite = False
            if not finite:
                raise ValueError(f"{name} must be a finite number, excluding booleans")
            if name == "min_p" and not 0 <= value <= 1:
                raise ValueError("min_p must be in [0, 1]")
            if name == "presence_penalty" and not -2 <= value <= 2:
                raise ValueError("presence_penalty must be in [-2, 2]")
            if name == "repetition_penalty" and value <= 0:
                raise ValueError("repetition_penalty must be greater than zero")
        result[name] = value
    return result


@dataclass(frozen=True)
class ModelReply:
    text: str
    prompt_tokens: int
    completion_tokens: int
    elapsed_seconds: float
    finish_reason: str
    model_revision: str
    request_id: str
    gpu_count: int = 1
    reasoning_content: str | None = None
    reasoning_error: str | None = None

    def __post_init__(self):
        if any(type(n) is not int or n < 0 for n in (self.prompt_tokens, self.completion_tokens)):
            raise ValueError("model server returned invalid token metering")
        if not math.isfinite(self.elapsed_seconds) or self.elapsed_seconds < 0:
            raise ValueError("model server returned invalid elapsed time")
        if type(self.gpu_count) is not int or self.gpu_count < 1:
            raise ValueError("model server returned invalid GPU allocation count")
        if self.finish_reason not in {"stop", "length", "timeout"} or not self.request_id or not self.model_revision:
            raise ValueError("model server returned incomplete runtime identity/status")
        if self.reasoning_content is not None and not isinstance(self.reasoning_content, str):
            raise ValueError("model server returned invalid reasoning content")
        if self.reasoning_error not in {None, "invalid_reasoning_type", "conflicting_reasoning_fields"}:
            raise ValueError("model server returned unknown reasoning content error")
        if self.reasoning_error is not None and self.reasoning_content is not None:
            raise ValueError("rejected reasoning content cannot also be usable")


def _reasoning_content(message: dict[str, Any]) -> tuple[str | None, str | None]:
    """Normalize backend aliases without inventing or combining solution notes.

    Invalid optional notes do not discard an otherwise valid usage receipt.
    Their error is retained separately so an explicit extractor can fail closed
    while ordinary answer scoring continues to consume only final content.
    """
    values = [message[key] for key in ("reasoning_content", "reasoning")
              if message.get(key) is not None]
    if any(not isinstance(value, str) for value in values):
        return None, "invalid_reasoning_type"
    if len(values) == 2 and values[0] != values[1]:
        return None, "conflicting_reasoning_fields"
    return (values[0] if values else None), None


class LocalModelClient:
    def __init__(self, base_url: str = "http://127.0.0.1:18085", model: str = "Qwen3.5-35B-A3B"):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._sampling_capabilities: frozenset[str] | None = None

    def health(self) -> dict[str, Any]:
        response = httpx.get(self.base_url + "/health", timeout=10, trust_env=False)
        response.raise_for_status()
        return response.json()

    def ensure_sampling_support(self, options: Mapping) -> None:
        """Fail before generation if the endpoint cannot honor explicit controls.

        Campaign preflight may call this before reserving a physical request.
        Capabilities are scoped to this client instance and its frozen endpoint.
        """
        requested = sampling_options(options)
        if not requested:
            return
        if self._sampling_capabilities is None:
            declared = self.health().get("supported_sampling_parameters")
            if not isinstance(declared, list) or any(type(name) is not str for name in declared):
                raise ValueError("Endpoint does not declare supported_sampling_parameters")
            self._sampling_capabilities = frozenset(declared)
        missing = set(requested) - self._sampling_capabilities
        if missing:
            raise ValueError(f"Endpoint does not support explicit sampling parameters: {', '.join(sorted(missing))}")

    def generate(self, messages: list[dict[str, str]], *, max_tokens: int = 512,
                 temperature: float = 0.7, top_p: float = 0.8, seed: int = 0,
                 enable_thinking: bool = False, max_seconds: float = 120,
                 total_token_cap: int | None = None, top_k: int | None = None,
                 min_p: float | None = None, presence_penalty: float | None = None,
                 repetition_penalty: float | None = None) -> ModelReply:
        optional_sampling = sampling_options({"top_k": top_k, "min_p": min_p,
            "presence_penalty": presence_penalty, "repetition_penalty": repetition_penalty})
        payload = {
            "model": self.model, "messages": messages, "max_tokens": max_tokens,
            "temperature": temperature, "top_p": top_p, "seed": seed,
            "enable_thinking": enable_thinking, "max_seconds": max_seconds,
        }
        if total_token_cap is not None:
            if type(total_token_cap) is not int or total_token_cap < 1:
                raise ValueError("total_token_cap must be a positive integer")
            payload["total_token_cap"] = total_token_cap
        self.ensure_sampling_support(optional_sampling)
        payload.update(optional_sampling)
        # Multiple experiment processes share the local serialized GPU server.
        # Queue before admission so overlapping runs do not randomly fail429.
        directory=Path('/tmp/specrsi-model-client-locks'); directory.mkdir(exist_ok=True,mode=0o700)
        lock_path=directory/(hashlib.sha256(self.base_url.encode()).hexdigest()+'.lock')
        with lock_path.open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            response = httpx.post(self.base_url + "/v1/chat/completions", json=payload,
                                  timeout=max_seconds + 60, trust_env=False)
        response.raise_for_status()
        value = response.json()
        choice, usage, meter = value["choices"][0], value["usage"], value["specrsi_meter"]
        reasoning, reasoning_error = _reasoning_content(choice["message"])
        return ModelReply(choice["message"]["content"], usage["prompt_tokens"],
                          usage["completion_tokens"], meter["elapsed_seconds"],
                          choice["finish_reason"], meter["revision"], value["id"], meter.get("gpu_count", 1),
                          reasoning_content=reasoning, reasoning_error=reasoning_error)
