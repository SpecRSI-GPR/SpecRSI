"""Local, serialized BF16 Transformers server. No external model API is used.

The HTTP shape is a small chat-completion subset. A request seed is recorded for
reproducibility, and never treated as an SpecRSI equality certificate.
"""
from __future__ import annotations

import argparse
import asyncio
from contextlib import asynccontextmanager
import hashlib
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
    model: str = "Qwen3.5-35B-A3B"
    messages: list[Message] = Field(min_length=1, max_length=128)
    max_tokens: int = Field(default=512, ge=1, le=32768)
    temperature: float = Field(default=0.7, ge=0, le=2)
    top_p: float = Field(default=0.8, gt=0, le=1)
    seed: int = Field(default=0, ge=0, lt=2**32)
    enable_thinking: bool = False
    max_seconds: float = Field(default=120, gt=0, le=3600)
    stream: Literal[False] = False
    total_token_cap: int | None = Field(default=None, ge=1)


def enforced_output_cap(prompt_tokens: int, requested_output: int, total_token_cap: int | None,
                        context_cap: int = 32768) -> int:
    """Apply token reservation using the server's actual tokenized prompt."""
    output_cap = requested_output if total_token_cap is None else min(requested_output, total_token_cap - prompt_tokens)
    if output_cap < 1:
        raise ValueError("Prompt leaves no output token within total_token_cap")
    if prompt_tokens + output_cap > context_cap:
        raise ValueError(f"Prompt ({prompt_tokens}) plus output cap exceeds the configured {context_cap}-token context")
    return output_cap


class Backend:
    def __init__(self, lock_path: str, log_path: str):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from .integrity import verify_model
        self.torch = torch
        self.integrity = verify_model(lock_path)
        self.lock = json.loads(Path(lock_path).read_text())
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.lock["local_path"], local_files_only=True)
        # Qwen3.5's publisher checkpoint includes a vision tower; text-only
        # science/system tasks use the native MoE causal language tower. This
        # is the same pretrained text model, not a dense-Qwen3 substitution.
        model_class = AutoModelForCausalLM
        if self.lock.get("language_model_only") and "Qwen3.5" in self.lock["repo_id"]:
            from transformers import Qwen3_5MoeForCausalLM
            model_class = Qwen3_5MoeForCausalLM
        self.model = model_class.from_pretrained(
            self.lock["local_path"], dtype=torch.bfloat16, device_map={"": "cuda:0"},
            attn_implementation="sdpa", local_files_only=True,
        ).eval()
        self.model_name = self.lock.get("served_model_name", self.lock["repo_id"].split("/")[-1])

    def generate(self, request: ChatRequest):
        torch = self.torch
        inputs = self.tokenizer.apply_chat_template(
            [m.model_dump() for m in request.messages], add_generation_prompt=True,
            tokenize=True, return_dict=True, return_tensors="pt",
            enable_thinking=request.enable_thinking,
        ).to(self.model.device)
        count = inputs["input_ids"].shape[-1]
        output_cap = enforced_output_cap(count, request.max_tokens, request.total_token_cap)
        torch.manual_seed(request.seed)
        torch.cuda.manual_seed_all(request.seed)
        torch.cuda.synchronize()
        started = time.monotonic()
        kwargs = {"max_new_tokens": output_cap, "do_sample": request.temperature > 0,
                  "max_time": request.max_seconds, "pad_token_id": self.tokenizer.eos_token_id}
        if request.temperature > 0:
            kwargs.update(temperature=request.temperature, top_p=request.top_p, top_k=20)
        with torch.inference_mode():
            output = self.model.generate(**inputs, **kwargs)
        torch.cuda.synchronize()
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
            "specrsi_meter": {"elapsed_seconds": elapsed, "gpu_count": 1, "revision": self.lock["revision"],
                          "device": torch.cuda.get_device_name(0), "dtype": "bfloat16", "seed": request.seed,
                          "equality_certified": False, "timeout_is_cooperative": True,
                          "total_token_cap": request.total_token_cap, "effective_output_cap": output_cap},
        }
        with self.log_path.open("a") as handle:
            handle.write(json.dumps({"request": request.model_dump(), "response": result}, ensure_ascii=False) + "\n")
        return result


def create_app(lock_path: str, log_path: str):
    @asynccontextmanager
    async def lifespan(app):
        app.state.backend = await asyncio.to_thread(Backend, lock_path, log_path)
        yield

    app = FastAPI(title="SpecRSI local Qwen inference", lifespan=lifespan)
    guard = asyncio.Lock()

    @app.get("/health")
    def health():
        backend = app.state.backend
        return {"status": "ok", "model": backend.model_name, "revision": backend.lock["revision"],
                "protocol_version": 2, "server_enforces_total_token_cap": True,
                "device": backend.torch.cuda.get_device_name(0), "dtype": "bfloat16",
                "integrity": backend.integrity}

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [{"id": app.state.backend.model_name, "object": "model", "owned_by": "local"}]}

    @app.post("/v1/chat/completions")
    async def completions(request: ChatRequest):
        backend = app.state.backend
        if request.model not in {backend.model_name, backend.lock["repo_id"]}:
            raise HTTPException(404, f"This endpoint serves {backend.model_name}")
        if guard.locked():
            raise HTTPException(429, "GPU is busy; retry after the current request finishes")
        async with guard:
            try:
                return await asyncio.to_thread(app.state.backend.generate, request)
            except ValueError as error:
                raise HTTPException(400, str(error)) from error
    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", default="configs/models/qwen35_35b_a3b_bf16.lock.json")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18085)
    parser.add_argument("--requests-log", default="logs/model_requests.jsonl")
    args = parser.parse_args()
    import uvicorn
    uvicorn.run(create_app(args.lock, args.requests_log), host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
