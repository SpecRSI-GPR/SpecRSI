#!/usr/bin/env python3
"""SpecRSI protocol-v2 gateway for a pinned, local OpenAI-compatible worker.

Reasoning fields and backend usage are preserved in full receipts. Only the
backend's final ``content`` is returned as the answer; an absent final answer is
empty, never reconstructed from a truncated reasoning trace. Token caps are
checked before work. Elapsed time is measured, not a hard execution limit.
"""
from __future__ import annotations

import argparse
import asyncio
from contextlib import asynccontextmanager
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
import httpx
from pydantic import BaseModel, ConfigDict, Field, field_validator

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from specrsi.tokenization import chat_token_ids


EXTENDED_SAMPLING_PARAMETERS = ("top_k", "min_p", "presence_penalty", "repetition_penalty")
# Freeze the loaded gateway artifact once. A later on-disk edit must not make
# an already-running process advertise a new implementation identity.
GATEWAY_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["system", "user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str
    messages: list[Message] = Field(min_length=1, max_length=128)
    max_tokens: int = Field(default=512, ge=1, le=32768)
    temperature: float = Field(default=.7, ge=0, le=2)
    top_p: float = Field(default=.8, gt=0, le=1)
    top_k: int | None = None
    min_p: float | None = None
    presence_penalty: float | None = None
    repetition_penalty: float | None = None
    seed: int = Field(default=0, ge=0, lt=2**32)
    enable_thinking: bool = False
    max_seconds: float = Field(default=120, gt=0, le=3600)
    total_token_cap: int | None = Field(default=None, ge=1)
    stream: Literal[False] = False

    @field_validator(*EXTENDED_SAMPLING_PARAMETERS, mode="before")
    @classmethod
    def validate_explicit_sampling(cls, value, info):
        # Validate before Pydantic coercion: strings and bools are not numbers
        # in the frozen request contract, even if Python would coerce them.
        if value is None:
            return value
        name = info.field_name
        if name == "top_k":
            if type(value) is not int or value < -1:
                raise ValueError("top_k must be an integer >= -1")
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
        return value


def create_app(lock_path: str, backend_url: str, log_path: str, *, tokenizer=None, transport=None):
    lock_bytes = Path(lock_path).read_bytes()
    lock = json.loads(lock_bytes)
    manifest_path = Path(lock["download_manifest"])
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if (manifest.get("status") != "publisher_verified_complete"
            or manifest.get("revision") != lock["revision"]
            or manifest.get("repo_id") != lock["repo_id"]
            or manifest.get("lock_sha256") != hashlib.sha256(lock_bytes).hexdigest()):
        raise ValueError("Serving requires the matching publisher-verified download manifest")
    served_name, context_cap = lock["served_model_name"], lock["context_length"]
    # The pinned vLLM worker implements these fields. A different backend must
    # not inherit an unverified declaration merely because it speaks OpenAI API.
    supported_sampling = ("temperature", "top_p", "seed") + (
        EXTENDED_SAMPLING_PARAMETERS if lock["engine_version"].startswith("vllm-") else ())
    thinking_key = lock.get("chat_template_thinking_key", "enable_thinking")
    if thinking_key not in {"enable_thinking", "thinking"}:
        raise ValueError("Unknown chat-template thinking switch")
    log = Path(log_path)
    log.parent.mkdir(parents=True, exist_ok=True)
    gate = asyncio.Lock()

    @asynccontextmanager
    async def lifespan(app):
        if tokenizer is None:
            from transformers import AutoTokenizer
            model_directory = Path(lock["local_path"])
            app.state.tokenizer = await asyncio.to_thread(
                AutoTokenizer.from_pretrained, model_directory, local_files_only=True, trust_remote_code=False)
        else:
            app.state.tokenizer = tokenizer
        app.state.http = httpx.AsyncClient(base_url=backend_url, trust_env=False, timeout=None, transport=transport)
        yield
        await app.state.http.aclose()

    app = FastAPI(title="SpecRSI pinned-model budget gateway", lifespan=lifespan)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(_request, error):
        # Python's JSON parser accepts NaN/Infinity, but the default validation
        # handler cannot serialize them back. Report the field and reason only;
        # rejected raw inputs must not turn a pre-inference 422 into a 500.
        return JSONResponse(status_code=422, content={"detail": [
            {key: item[key] for key in ("type", "loc", "msg")} for item in error.errors()]})

    @app.get("/health")
    async def health():
        try:
            response = await app.state.http.get("/health", timeout=10)
            response.raise_for_status()
        except httpx.HTTPError as error:
            raise HTTPException(503, "Pinned local backend is not ready") from error
        return {"status": "ok", "model": served_name, "revision": lock["revision"], "protocol_version": 2,
                "server_enforces_total_token_cap": True, "context_length": context_cap,
                "gpu_count": len(lock["gpu_ids"]), "gpu_ids": lock["gpu_ids"], "dtype": lock["format"],
                "engine": lock["engine_version"], "wall_time_enforced": False,
                "gateway_source_sha256": GATEWAY_SOURCE_SHA256,
                "supported_sampling_parameters": list(supported_sampling),
                "reasoning_policy": "separate_fields_preserved_full_usage_final_content_only",
                "precision_extension_not_paper_bf16": lock.get("precision_extension_not_paper_bf16", False),
                "integrity": {"repo_id": lock["repo_id"], "revision": lock["revision"],
                              "manifest": str(manifest_path),
                              "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest()}}

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": served_name, "object": "model", "owned_by": "local"}]}

    @app.post("/v1/chat/completions")
    async def completion(request: ChatRequest):
        if request.model not in {served_name, lock["repo_id"]}:
            raise HTTPException(404, "The requested model is not the pinned model at this endpoint")
        optional_sampling = {name: getattr(request, name) for name in EXTENDED_SAMPLING_PARAMETERS
                             if getattr(request, name) is not None}
        if set(optional_sampling) - set(supported_sampling):
            raise HTTPException(400, "Pinned backend does not support the requested sampling parameters")
        logged_request = request.model_dump(exclude={name for name in EXTENDED_SAMPLING_PARAMETERS
                                                    if getattr(request, name) is None})
        if gate.locked():
            raise HTTPException(429, "Inference is busy; no GPU work was admitted")
        async with gate:
            messages = [message.model_dump() for message in request.messages]
            thinking_kwargs = {thinking_key: request.enable_thinking}
            tokens = chat_token_ids(app.state.tokenizer,
                messages, tokenize=True, add_generation_prompt=True, **thinking_kwargs)
            prompt_tokens = len(tokens)
            full_cap = prompt_tokens + request.max_tokens
            if full_cap > context_cap:
                raise HTTPException(400, "The complete prompt/output request exceeds the serving context")
            if request.total_token_cap is not None and full_cap > request.total_token_cap:
                raise HTTPException(400, "The complete fixed generation request does not fit total_token_cap")
            payload = {"model": served_name, "messages": messages, "max_tokens": request.max_tokens,
                       "temperature": request.temperature, "top_p": request.top_p, "seed": request.seed,
                       "stream": False, "chat_template_kwargs": thinking_kwargs}
            payload.update(optional_sampling)
            started = time.monotonic()
            try:
                response = await app.state.http.post("/v1/chat/completions", json=payload)
                response.raise_for_status()
                result = response.json()
            except (httpx.HTTPError, ValueError) as error:
                raise HTTPException(502, "Local worker failed; actual token usage is unknown") from error
            elapsed = time.monotonic() - started
            usage = result.get("usage", {})
            actual_prompt, actual_completion = usage.get("prompt_tokens"), usage.get("completion_tokens")
            choices = result.get("choices", [])
            if (type(actual_prompt) is not int or type(actual_completion) is not int
                    or actual_prompt < 0 or actual_completion < 0 or len(choices) != 1
                    or not isinstance(choices[0].get("message"), dict)
                    or choices[0].get("finish_reason") not in {"stop", "length"}
                    or not isinstance(result.get("id"), str) or not result["id"]):
                with log.open("a") as handle:
                    handle.write(json.dumps({"request": logged_request, "invalid_response": result}) + "\n")
                raise HTTPException(502, "Local worker omitted a valid complete physical receipt")
            result["specrsi_meter"] = {"elapsed_seconds": elapsed, "gpu_count": len(lock["gpu_ids"]),
                "revision": lock["revision"], "dtype": lock["format"], "gpu_ids": lock["gpu_ids"],
                "gateway_source_sha256": GATEWAY_SOURCE_SHA256,
                "sampling_parameters": {"temperature": request.temperature, "top_p": request.top_p,
                                        "seed": request.seed, **optional_sampling},
                "seed": request.seed, "equality_certified": False, "wall_time_enforced": False,
                "timeout_policy": "late_bounded_token_response_is_incomplete_but_actual_usage_is_retained",
                "total_token_cap": request.total_token_cap, "gateway_prompt_tokens": prompt_tokens,
                "backend_finish_reason": choices[0]["finish_reason"],
                "backend_message_content_was_null": choices[0]["message"].get("content") is None}
            violations = []
            if actual_prompt != prompt_tokens:
                violations.append("tokenizer_mismatch")
            if actual_completion > request.max_tokens:
                violations.append("completion_token_cap_exceeded")
            if actual_prompt + actual_completion > full_cap:
                violations.append("request_token_cap_exceeded")
            if violations:
                # Known physical work must reach LocalModelClient, including an
                # over-cap total. Hiding it behind an HTTP error would leave the
                # caller with only its (now invalid) reserved upper bound.
                result["specrsi_meter"]["contract_violation"] = {
                    "reasons": violations, "requested_max_tokens": request.max_tokens,
                    "reserved_full_cap": full_cap,
                    "actual_total_tokens": actual_prompt + actual_completion}
            if elapsed > request.max_seconds or violations:
                choices[0]["finish_reason"] = "timeout"
            if choices[0]["message"].get("content") is None:
                choices[0]["message"]["content"] = ""
            with log.open("a") as handle:
                handle.write(json.dumps({"request": logged_request, "response": result}, ensure_ascii=False) + "\n")
            return result

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--log", required=True)
    args = parser.parse_args()
    import uvicorn
    uvicorn.run(create_app(args.lock, args.backend, args.log), host="127.0.0.1", port=args.port, workers=1)


if __name__ == "__main__":
    main()
