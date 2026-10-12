#!/usr/bin/env python3
"""SpecRSI protocol-v2 token-budget gateway for the local pinned Kimi vLLM service.

The token cap is checked before inference. Runtime is measured, not a hard GPU
wall-clock limit: a late bounded-token response is returned as a scored-interrupt
status with its actual usage, avoiding invented billing on cancelled work.
"""
from __future__ import annotations
import argparse
import asyncio
from contextlib import asynccontextmanager
import json
from pathlib import Path
import sys
import time
from typing import Literal

from fastapi import FastAPI, HTTPException
import httpx
from pydantic import BaseModel, ConfigDict, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from specrsi.tokenization import chat_token_ids


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["system", "user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str = "Kimi-K2.6"
    messages: list[Message] = Field(min_length=1, max_length=128)
    max_tokens: int = Field(default=512, ge=1, le=32768)
    temperature: float = Field(default=.6, ge=0, le=2)
    top_p: float = Field(default=.95, gt=0, le=1)
    seed: int = Field(default=0, ge=0, lt=2**32)
    enable_thinking: bool = False
    max_seconds: float = Field(default=120, gt=0, le=3600)
    total_token_cap: int | None = Field(default=None, ge=1)
    stream: Literal[False] = False


def create_app(lock_path: str, backend_url: str, log_path: str):
    lock = json.loads(Path(lock_path).read_text())
    served_name = lock.get("served_model_name", "Kimi-K2.6")
    context_cap = lock.get("context_length", 32768)
    log = Path(log_path)
    log.parent.mkdir(parents=True, exist_ok=True)
    gate = asyncio.Lock()

    @asynccontextmanager
    async def lifespan(app):
        from transformers import AutoTokenizer
        app.state.tokenizer = await asyncio.to_thread(AutoTokenizer.from_pretrained,
            lock["local_path"], local_files_only=True, trust_remote_code=True)
        app.state.http = httpx.AsyncClient(base_url=backend_url, trust_env=False, timeout=None)
        yield
        await app.state.http.aclose()

    app = FastAPI(title="SpecRSI local Kimi token-budget gateway", lifespan=lifespan)

    @app.get("/health")
    async def health():
        try:
            response = await app.state.http.get("/health", timeout=10)
            response.raise_for_status()
        except httpx.HTTPError as error:
            raise HTTPException(503, "Pinned Kimi backend is not ready") from error
        return {"status": "ok", "model": served_name, "revision": lock["revision"], "protocol_version": 2,
                "server_enforces_total_token_cap": True, "context_length": context_cap,
                "gpu_count": len(lock["gpu_ids"]), "gpu_ids": lock["gpu_ids"], "dtype": lock["format"],
                "engine": lock["engine_version"], "wall_time_enforced": False,
                "integrity": {"repo_id": lock["repo_id"], "revision": lock["revision"],
                              "manifest": "reports/kimi_download.json"}}

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": served_name, "object": "model", "owned_by": "local"}]}

    @app.post("/v1/chat/completions")
    async def completion(request: ChatRequest):
        if request.model not in {served_name, lock["repo_id"]}:
            raise HTTPException(404, "This endpoint serves the pinned Kimi-K2.6 model")
        if gate.locked():
            raise HTTPException(429, "Kimi inference is busy; no GPU work was admitted")
        async with gate:
            messages = [m.model_dump() for m in request.messages]
            tokens = chat_token_ids(app.state.tokenizer,messages, tokenize=True, add_generation_prompt=True,
                                                            thinking=request.enable_thinking, return_dict=False)
            prompt_tokens = len(tokens)
            full_cap = prompt_tokens + request.max_tokens
            if full_cap > context_cap:
                raise HTTPException(400, f"Prompt/output cap {full_cap} exceeds serving context {context_cap}")
            if request.total_token_cap is not None and full_cap > request.total_token_cap:
                raise HTTPException(400, "The complete fixed generation request does not fit total_token_cap")
            payload = {"model": served_name, "messages": messages, "max_tokens": request.max_tokens,
                       "temperature": request.temperature, "top_p": request.top_p, "seed": request.seed,
                       "stream": False, "chat_template_kwargs": {"thinking": request.enable_thinking}}
            started = time.monotonic()
            try:
                response = await app.state.http.post("/v1/chat/completions", json=payload)
                response.raise_for_status()
            except httpx.HTTPError as error:
                # There is no fabricated actual usage when a worker/transport fails.
                raise HTTPException(502, "Local Kimi worker failed; actual token usage is unknown") from error
            elapsed = time.monotonic() - started
            result = response.json()
            usage = result.get("usage", {})
            actual_prompt, actual_completion = usage.get("prompt_tokens"), usage.get("completion_tokens")
            if (type(actual_prompt) is not int or type(actual_completion) is not int
                    or actual_prompt < 0 or actual_completion < 0):
                raise HTTPException(502, "Local Kimi worker omitted valid actual token metering")
            result["specrsi_meter"] = {"elapsed_seconds": elapsed, "gpu_count": len(lock["gpu_ids"]),
                "revision": lock["revision"], "dtype": lock["format"], "gpu_ids": lock["gpu_ids"],
                "seed": request.seed, "equality_certified": False, "wall_time_enforced": False,
                "timeout_policy": "late_bounded_token_response_is_incomplete_but_actual_usage_is_retained",
                "total_token_cap": request.total_token_cap, "gateway_prompt_tokens": prompt_tokens}
            if elapsed > request.max_seconds:
                result["choices"][0]["finish_reason"] = "timeout"
            content = result["choices"][0]["message"].get("content")
            if content is None:
                result["choices"][0]["message"]["content"] = ""
            with log.open("a") as handle:
                handle.write(json.dumps({"request": request.model_dump(), "response": result}, ensure_ascii=False) + "\n")
            if actual_prompt != prompt_tokens or actual_completion > request.max_tokens or actual_prompt + actual_completion > full_cap:
                # The physical receipt above remains available for investigation.
                raise HTTPException(502, "Local Kimi worker violated its tokenizer/token-cap contract")
            return result

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", default="configs/models/kimi_k26.lock.json")
    parser.add_argument("--backend", default="http://127.0.0.1:18082")
    parser.add_argument("--port", type=int, default=18081)
    parser.add_argument("--log", default="logs/kimi_requests.jsonl")
    args = parser.parse_args()
    import uvicorn
    uvicorn.run(create_app(args.lock, args.backend, args.log), host="127.0.0.1", port=args.port, workers=1)


if __name__ == "__main__":
    main()
