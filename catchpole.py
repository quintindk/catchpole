import os
from typing import AsyncIterator, Iterator

import litellm
from litellm import CustomLLM
from litellm.types.utils import GenericStreamingChunk


LOCAL_MODEL = os.getenv("CATCHPOLE_LOCAL_MODEL", "openai/google/gemma-4-26b-a4b")
LOCAL_API_BASE = os.getenv("CATCHPOLE_LOCAL_API_BASE", "http://localhost:1234/v1")
LOCAL_API_KEY = os.getenv("CATCHPOLE_LOCAL_API_KEY", "lm-studio")
ROUTER_MODEL = os.getenv("CATCHPOLE_ROUTER_MODEL", LOCAL_MODEL)
CLOUD_MODEL = os.getenv("CATCHPOLE_CLOUD_MODEL", "github_copilot/claude-sonnet-4.6")

STRIP_KWARGS = (
    "model",
    "api_base",
    "api_key",
    "custom_llm_provider",
    "metadata",
)

FORWARD_KWARGS = {
    "messages",
    "temperature",
    "top_p",
    "top_k",
    "n",
    "stop",
    "max_tokens",
    "max_completion_tokens",
    "presence_penalty",
    "frequency_penalty",
    "logit_bias",
    "logprobs",
    "top_logprobs",
    "user",
    "seed",
    "tools",
    "tool_choice",
    "parallel_tool_calls",
    "response_format",
    "reasoning_effort",
    "stream_options",
    "extra_headers",
    "extra_body",
}


def _forward(kwargs: dict) -> dict:
    return {k: v for k, v in kwargs.items() if k in FORWARD_KWARGS and v is not None}


def _summarize(messages):
    summarized = []
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str) and len(content) > 2000:
            content = (
                content[:500]
                + "\n\n... [MASSIVE CONTEXT OMITTED BY ROUTER] ...\n\n"
                + content[-1000:]
            )
        summarized.append({"role": msg.get("role"), "content": content})
    return summarized


def _flatten(summarized):
    parts = []
    for m in summarized:
        c = m["content"]
        if not isinstance(c, str):
            c = str(c)
        parts.append(f"{m['role']}: {c}")
    return "\n".join(parts)


import sys


def _log(msg: str) -> None:
    print(f"[catchpole] {msg}", file=sys.stderr, flush=True)


async def _decide(messages) -> str:
    summarized = _summarize(messages)
    routing_prompt = [
        {
            "role": "system",
            "content": (
                "You are a Catchpole, a system router. Assess the complexity "
                "of the following programming request. If it requires extensive "
                "context, deep architectural knowledge, or complex logic, reply "
                "ONLY with 'CLOUD'. If it is standard boilerplate, a simple "
                "function, or a basic question, reply ONLY with 'LOCAL'."
            ),
        },
        {"role": "user", "content": _flatten(summarized)},
    ]
    try:
        resp = await litellm.acompletion(
            model=ROUTER_MODEL,
            api_base=LOCAL_API_BASE,
            api_key=LOCAL_API_KEY,
            messages=routing_prompt,
            temperature=0.0,
        )
        raw = resp.choices[0].message.content or ""
        _log(f"router raw decision: {raw!r}")
        return raw.strip().upper()
    except Exception as e:
        _log(f"router error, defaulting CLOUD: {e}")
        return "CLOUD"


def _clean(kwargs: dict) -> dict:
    for k in STRIP_KWARGS:
        kwargs.pop(k, None)
    return kwargs


def _target(decision: str):
    if "LOCAL" in decision:
        print("Decision: LOCAL Inference")
        return {
            "model": LOCAL_MODEL,
            "api_base": LOCAL_API_BASE,
            "api_key": LOCAL_API_KEY,
        }
    print("Decision: CLOUD (GitHub Copilot)")
    return {"model": CLOUD_MODEL}


def _build_forward(kwargs: dict) -> dict:
    forward = {"messages": kwargs.get("messages", [])}
    optional = kwargs.get("optional_params") or {}
    for k, v in optional.items():
        if k in {"stream", "stream_options"}:
            continue
        forward[k] = v
    return forward


class CatchpoleRouter(CustomLLM):
    async def acompletion(self, *args, **kwargs) -> litellm.ModelResponse:
        messages = kwargs.get("messages", [])
        decision = await _decide(messages)
        target = _target(decision)
        forward = _build_forward(kwargs)
        _log(f"acompletion forwarding keys={sorted(forward.keys())} target={target.get('model')}")
        try:
            result = litellm.completion(**target, **forward)
            _log("acompletion upstream returned")
            return result
        except Exception as e:
            _log(f"acompletion upstream error: {type(e).__name__}: {e}")
            raise

    async def astreaming(
        self, *args, **kwargs
    ) -> AsyncIterator[GenericStreamingChunk]:
        messages = kwargs.get("messages", [])
        decision = await _decide(messages)
        target = _target(decision)
        forward = _build_forward(kwargs)
        forward["stream"] = True

        stream = litellm.completion(**target, **forward)
        for chunk in stream:
            try:
                delta = chunk.choices[0].delta
                text = getattr(delta, "content", None) or ""
                finish_reason = chunk.choices[0].finish_reason
            except (AttributeError, IndexError):
                text = ""
                finish_reason = None

            is_finished = finish_reason is not None
            out: GenericStreamingChunk = {
                "finish_reason": finish_reason or "stop" if is_finished else None,
                "index": 0,
                "is_finished": is_finished,
                "text": text,
                "tool_use": None,
                "usage": {
                    "completion_tokens": 0,
                    "prompt_tokens": 0,
                    "total_tokens": 0,
                },
            }
            yield out  # type: ignore

    def completion(self, *args, **kwargs) -> litellm.ModelResponse:
        messages = kwargs.get("messages", [])
        decision = "CLOUD"
        try:
            import asyncio as _asyncio
            decision = _asyncio.run(_decide(messages))
        except Exception:
            pass
        target = _target(decision)
        forward = _build_forward(kwargs)
        return litellm.completion(**target, **forward)

    def streaming(self, *args, **kwargs) -> Iterator[GenericStreamingChunk]:
        raise NotImplementedError("Use async streaming via the LiteLLM proxy.")


catchpole = CatchpoleRouter()
