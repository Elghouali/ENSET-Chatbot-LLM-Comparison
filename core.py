"""Ollama integration and measurable, streamed text generation."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

from monitor import ResourceSampler, ResourceSummary


@dataclass(frozen=True)
class GenerationOptions:
    temperature: float = 0.7
    top_k: int = 40
    top_p: float = 0.9
    num_predict: int = 512
    seed: int = 42

    def as_dict(self) -> dict:
        return vars(self).copy()


@dataclass
class GenerationResult:
    model: str
    response: str
    time_s: float
    ttft_s: float | None
    generated_tokens: int | None
    tokens_s: float | None
    server_eval_s: float | None
    resources: ResourceSummary

    def as_dict(self) -> dict:
        return {
            "model": self.model,
            "response": self.response,
            "time_s": round(self.time_s, 3),
            "ttft_s": round(self.ttft_s, 3) if self.ttft_s is not None else None,
            "generated_tokens": self.generated_tokens,
            "tokens_s": round(self.tokens_s, 3) if self.tokens_s is not None else None,
            "server_eval_s": round(self.server_eval_s, 3) if self.server_eval_s is not None else None,
            **self.resources.as_dict(),
        }


def build_client(host: str):
    from ollama import Client
    return Client(host=host, timeout=600)


def installed_models(client) -> list[str]:
    """Use actual local installed model tags, not the original hard-coded list."""
    response = client.list()
    models = getattr(response, "models", None)
    if models is None and isinstance(response, dict):
        models = response.get("models", [])
    names = []
    for item in models or []:
        name = getattr(item, "model", None) or getattr(item, "name", None)
        if not name and isinstance(item, dict):
            name = item.get("model") or item.get("name")
        if name:
            names.append(str(name))
    return sorted(set(names))


def value(obj, attr: str, default=None):
    if isinstance(obj, dict):
        return obj.get(attr, default)
    return getattr(obj, attr, default)


def generate(
    client,
    model: str,
    messages: list[dict[str, str]],
    options: GenerationOptions,
    on_update: Callable[[str], None] | None = None,
    sampler_factory: Callable[[], ResourceSampler] = ResourceSampler,
) -> GenerationResult:
    """Stream once; use final Ollama response's real token/timing metrics.

    TTFT: first non-empty streamed text; wall time includes model load/queue.
    Tokens/sec: eval_count / eval_duration (Ollama server, excludes model load).
    """
    sample = sampler_factory()
    chunks: list[str] = []
    first_token_time: float | None = None
    last = None
    start = time.perf_counter()
    sample.start()
    try:
        for part in client.chat(
            model=model,
            messages=messages,
            options=options.as_dict(),
            stream=True,
        ):
            last = part
            msg = value(part, "message", {})
            content = value(msg, "content", "") or ""
            if content:
                if first_token_time is None:
                    first_token_time = time.perf_counter() - start
                chunks.append(content)
                if on_update:
                    on_update("".join(chunks))
    finally:
        elapsed = time.perf_counter() - start
        resources = sample.stop()

    tokens = value(last, "eval_count") if last is not None else None
    duration_ns = value(last, "eval_duration") if last is not None else None
    tokens = int(tokens) if tokens is not None else None
    eval_s = float(duration_ns) / 1e9 if duration_ns is not None else None
    tokens_s = tokens / eval_s if tokens is not None and eval_s and eval_s > 0 else None
    return GenerationResult(
        model=model, response="".join(chunks), time_s=elapsed,
        ttft_s=first_token_time, generated_tokens=tokens,
        tokens_s=tokens_s, server_eval_s=eval_s, resources=resources,
    )
