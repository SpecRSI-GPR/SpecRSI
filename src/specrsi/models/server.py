"""Local, serialized Transformers inference with caller-supplied model settings.

The HTTP shape is a small chat-completion subset. A request seed is recorded for
reproducibility, and never treated as a SpecRSI equality certificate.
"""
from __future__ import annotations

import argparse
import asyncio
from contextlib import asynccontextmanager
import json
from pathlib import Path
import time
import uuid
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["system", "user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str = Field(min_length=1)
    messages: list[Message] = Field(min_length=1, max_length=128)
    max_tokens: int = Field(ge=1)
    temperature: float = Field(ge=0, le=2)
    top_p: float = Field(gt=0, le=1)
    seed: int = Field(ge=0, lt=2**32)
    enable_thinking: bool = False
    max_seconds: float = Field(gt=0)
    stream: Literal[False] = False
    total_token_cap: int | None = Field(default=None, ge=1)


def enforced_output_cap(prompt_tokens: int, requested_output: int, total_token_cap: int | None,
                        context_cap: int) -> int:
    """Apply token reservation using the server's actual tokenized prompt."""
    if (type(prompt_tokens) is not int or prompt_tokens < 0
            or type(requested_output) is not int or requested_output < 1
            or type(context_cap) is not int or context_cap < 1
            or (total_token_cap is not None and (type(total_token_cap) is not int or total_token_cap < 1))):
        raise ValueError("Token counts and caps must be integers within their declared range")
    output_cap = requested_output if total_token_cap is None else min(requested_output, total_token_cap - prompt_tokens)
    if output_cap < 1:
        raise ValueError("Prompt leaves no output token within total_token_cap")
    if prompt_tokens + output_cap > context_cap:
        raise ValueError(f"Prompt ({prompt_tokens}) plus output cap exceeds the configured {context_cap}-token context")
    return output_cap


class Backend:
    def __init__(self, lock_path: str, log_path: str, manifest_path: str, *,
                 model_name: str | None = None, device: str | None = None,
                 context_cap: int | None = None, dtype: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from .integrity import verify_model
        self.torch = torch
        self.lock = json.loads(Path(lock_path).read_text())
        self.model_name = model_name if model_name is not None else self.lock.get("model_name")
        configured_device = device if device is not None else self.lock.get("device")
        self.context_cap = context_cap if context_cap is not None else self.lock.get("context_cap")
        self.dtype = dtype if dtype is not None else self.lock.get("dtype")
        if not isinstance(self.model_name, str) or not self.model_name:
            raise ValueError("model_name must be supplied in the model lock or explicitly")
        if not isinstance(configured_device, str) or not configured_device:
            raise ValueError("device must be supplied in the model lock or explicitly")
        self.device = torch.device(configured_device)
        if self.device.type not in {"cpu", "cuda"}:
            raise ValueError("device must select CPU or CUDA for synchronized generation metering")
        if type(self.context_cap) is not int or self.context_cap < 1:
            raise ValueError("context_cap must be a positive integer supplied by the caller")
        if self.dtype not in {"float32", "float16", "bfloat16"}:
            raise ValueError("dtype must explicitly select float32, float16 or bfloat16")
        self.integrity = verify_model(lock_path, manifest_path)
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.lock["local_path"], local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.lock["local_path"], dtype=getattr(torch, self.dtype), device_map={"": str(self.device)},
            local_files_only=True,
        ).eval()

    @property
    def gpu_count(self):
        return int(self.device.type == "cuda")

    def synchronize(self):
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)

    def generate(self, request: ChatRequest):
        torch = self.torch
        inputs = self.tokenizer.apply_chat_template(
            [m.model_dump() for m in request.messages], add_generation_prompt=True,
            tokenize=True, return_dict=True, return_tensors="pt",
            enable_thinking=request.enable_thinking,
        ).to(self.model.device)
        count = inputs["input_ids"].shape[-1]
        output_cap = enforced_output_cap(count, request.max_tokens, request.total_token_cap, self.context_cap)
        torch.manual_seed(request.seed)
        self.synchronize()
        started = time.monotonic()
        kwargs = {"max_new_tokens": output_cap, "do_sample": request.temperature > 0,
                  "max_time": request.max_seconds, "pad_token_id": self.tokenizer.eos_token_id}
        if request.temperature > 0:
            kwargs.update(temperature=request.temperature, top_p=request.top_p)
        with torch.inference_mode():
            output = self.model.generate(**inputs, **kwargs)
        self.synchronize()
        elapsed = time.monotonic() - started
        generated = output[0, count:]
        answer = self.tokenizer.decode(generated, skip_special_tokens=True)
        eos = self.model.generation_config.eos_token_id
        eos = eos if isinstance(eos, list) else [eos]
        finished = len(generated) > 0 and int(generated[-1]) in eos
        reason = "stop" if finished else "length" if len(generated) >= output_cap else "timeout"
        result = {
            "id": "specrsi-" + uuid.uuid4().hex, "object": "chat.completion", "created": int(time.time()),
            "model": self.model_name,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": answer}, "finish_reason": reason}],
            "usage": {"prompt_tokens": count, "completion_tokens": len(generated), "total_tokens": count + len(generated)},
            "specrsi_meter": {"elapsed_seconds": elapsed, "gpu_count": self.gpu_count, "revision": self.lock["revision"],
                          "device": str(self.device), "dtype": self.dtype, "seed": request.seed,
                          "equality_certified": False, "timeout_is_cooperative": True,
                          "total_token_cap": request.total_token_cap, "effective_output_cap": output_cap},
        }
        with self.log_path.open("a") as handle:
            handle.write(json.dumps({"request": request.model_dump(), "response": result}, ensure_ascii=False) + "\n")
        return result


def create_app(lock_path: str, log_path: str, manifest_path: str, *,
               model_name: str | None = None, device: str | None = None,
               context_cap: int | None = None, dtype: str | None = None):
    @asynccontextmanager
    async def lifespan(app):
        app.state.backend = await asyncio.to_thread(Backend, lock_path, log_path, manifest_path,
            model_name=model_name, device=device, context_cap=context_cap, dtype=dtype)
        yield

    app = FastAPI(title="SpecRSI local inference", lifespan=lifespan)
    guard = asyncio.Lock()

    @app.get("/health")
    def health():
        backend = app.state.backend
        return {"status": "ok", "model": backend.model_name, "revision": backend.lock["revision"],
                "protocol_version": 2, "server_enforces_total_token_cap": True,
                "device": str(backend.device), "dtype": backend.dtype,
                "context_cap": backend.context_cap,
                "integrity": backend.integrity}

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [{"id": app.state.backend.model_name, "object": "model", "owned_by": "local"}]}

    @app.post("/v1/chat/completions")
    async def completions(request: ChatRequest):
        if request.model != app.state.backend.model_name:
            raise HTTPException(404, "Requested model is not served by this endpoint")
        if guard.locked():
            raise HTTPException(429, "Model is busy; retry after the current request finishes")
        async with guard:
            try:
                return await asyncio.to_thread(app.state.backend.generate, request)
            except ValueError as error:
                raise HTTPException(400, str(error)) from error
    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--requests-log", required=True)
    parser.add_argument("--model-name")
    parser.add_argument("--device")
    parser.add_argument("--context-cap", type=int)
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"))
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must lie between 1 and 65535")
    import uvicorn
    uvicorn.run(create_app(args.lock, args.requests_log, args.manifest, model_name=args.model_name,
                          device=args.device, context_cap=args.context_cap, dtype=args.dtype),
                host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
