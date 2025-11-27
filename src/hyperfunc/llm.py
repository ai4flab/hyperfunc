"""LiteLLM integration for hyperfunc.

Provides a built-in LLM completion hyperfunction that wraps litellm.completion()
with automatic rate limiting and ES optimization support.
"""

import asyncio
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Union

try:
    import litellm

    # Enable LiteLLM to return response headers for rate limiting
    litellm.return_response_headers = True
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False

from .core import LMParam, hyperfunction
from .rate_limit import AdaptiveRateLimiter

# Global rate limiter (shared across all calls)
_default_limiter = AdaptiveRateLimiter()

# Context var to detect if we're in optimization/evaluation (force stream=False)
_in_optimization: ContextVar[bool] = ContextVar("_in_optimization", default=False)


@dataclass
class TokenUsage:
    """Token usage statistics from an LLM response."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class LLMResponse:
    """Response from an LLM completion call."""

    content: str
    usage: TokenUsage
    model: str
    finish_reason: str

    def __str__(self) -> str:
        return self.content


def _check_litellm() -> None:
    """Check if litellm is available."""
    if not LITELLM_AVAILABLE:
        raise ImportError(
            "litellm is required for LLM hyperfunctions. "
            "Install with: pip install hyperfunc[llm]"
        )


def _extract_headers(response: Any) -> Dict[str, Any]:
    """Extract rate limit headers from LiteLLM response."""
    headers: Dict[str, Any] = {}
    if hasattr(response, "_hidden_params") and response._hidden_params:
        headers.update(response._hidden_params.get("additional_headers", {}))
    if hasattr(response, "_response_headers"):
        headers.update(response._response_headers)
    return headers


def _get_endpoint(model: str) -> str:
    """Extract endpoint identifier from model string."""
    if "/" in model:
        return model.split("/")[0]
    return model.split("-")[0]


async def _stream_completion(
    model: str,
    messages: List[Dict[str, str]],
    kwargs: Dict[str, Any],
    rate_limit: bool,
    endpoint: str,
) -> AsyncIterator[str]:
    """Stream completion chunks."""
    _check_litellm()
    kwargs["stream"] = True

    response = await asyncio.get_event_loop().run_in_executor(
        None, lambda: litellm.completion(model=model, messages=messages, **kwargs)
    )

    for chunk in response:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content

    # Update rate limiter after stream completes
    if rate_limit:
        headers = _extract_headers(response)
        if headers:
            _default_limiter.update_from_headers(endpoint, headers)


@hyperfunction(hp_type=LMParam, optimize_hparams=True)
async def llm_completion(
    model: str,
    messages: List[Dict[str, str]],
    hp: Optional[LMParam] = None,
    *,
    # LiteLLM params (override hp if provided)
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    max_tokens: Optional[int] = None,
    stop: Optional[List[str]] = None,
    presence_penalty: Optional[float] = None,
    frequency_penalty: Optional[float] = None,
    seed: Optional[int] = None,
    stream: bool = False,
    # Hyperfunc-specific
    rate_limit: bool = True,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    **extra_kwargs: Any,
) -> Union[LLMResponse, AsyncIterator[str]]:
    """
    LLM completion hyperfunction wrapping litellm.completion().

    Args:
        model: Model identifier (e.g., "gpt-4", "claude-3-opus", "openai/gpt-4")
        messages: List of message dicts [{"role": "user", "content": "..."}]
        hp: LMParam for ES optimization (optional)
        temperature, top_p, etc.: Override hp values if provided
        stream: If True, return async iterator of chunks (disabled during optimization)
        rate_limit: Enable automatic rate limiting (default: True)
        api_key, base_url: LiteLLM connection params
        **extra_kwargs: Additional litellm.completion() kwargs

    Returns:
        LLMResponse with content, usage, and metadata (or AsyncIterator if streaming)
    """
    _check_litellm()

    # Build kwargs from hp, then override with explicit params
    if hp is not None:
        kwargs = hp.to_litellm_kwargs()
    else:
        kwargs: Dict[str, Any] = {}

    # Explicit params override hp
    if temperature is not None:
        kwargs["temperature"] = temperature
    if top_p is not None:
        kwargs["top_p"] = top_p
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if stop is not None:
        kwargs["stop"] = stop
    if presence_penalty is not None:
        kwargs["presence_penalty"] = presence_penalty
    if frequency_penalty is not None:
        kwargs["frequency_penalty"] = frequency_penalty
    if seed is not None:
        kwargs["seed"] = seed
    if api_key is not None:
        kwargs["api_key"] = api_key
    if base_url is not None:
        kwargs["base_url"] = base_url
    kwargs.update(extra_kwargs)

    # Force stream=False during optimization/evaluation (need full response for metrics)
    effective_stream = stream and not _in_optimization.get()

    # Rate limiting - acquire before request
    endpoint = _get_endpoint(model)
    if rate_limit:
        estimated_tokens = sum(len(m.get("content", "")) // 4 for m in messages)
        await _default_limiter.acquire(endpoint, estimated_tokens)

    # Call LiteLLM
    if effective_stream:
        # Streaming mode - return async generator
        return _stream_completion(model, messages, kwargs, rate_limit, endpoint)

    # Non-streaming - call in executor (litellm.completion is sync)
    response = await asyncio.get_event_loop().run_in_executor(
        None, lambda: litellm.completion(model=model, messages=messages, **kwargs)
    )

    # Update rate limiter from headers
    if rate_limit:
        headers = _extract_headers(response)
        if headers:
            _default_limiter.update_from_headers(endpoint, headers)

    return LLMResponse(
        content=response.choices[0].message.content or "",
        usage=TokenUsage(
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens,
        ),
        model=response.model,
        finish_reason=response.choices[0].finish_reason or "stop",
    )


def make_llm_completion(
    model: str,
    *,
    system_prompt: Optional[str] = None,
    rate_limit: bool = True,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    **default_kwargs: Any,
) -> Callable[..., Any]:
    """
    Create a model-specific LLM hyperfunction using the completion API.

    Usage:
        gpt4 = make_llm_completion("gpt-4", system_prompt="You are helpful.")
        response = await gpt4("What is 2+2?")

        # In a HyperSystem, ES can optimize the hp:
        class MySystem(HyperSystem):
            async def run(self, query: str) -> str:
                return (await gpt4(query)).content

    Args:
        model: Model identifier
        system_prompt: Optional system message prepended to all calls
        rate_limit: Enable rate limiting
        api_key, base_url: Connection params
        **default_kwargs: Default kwargs for all calls

    Returns:
        Async hyperfunction that takes (prompt: str, hp: LMParam) -> LLMResponse
    """

    @hyperfunction(hp_type=LMParam, optimize_hparams=True)
    async def model_completion(
        prompt: str,
        hp: Optional[LMParam] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        merged_kwargs = {**default_kwargs, **kwargs}
        result = await llm_completion(
            model=model,
            messages=messages,
            hp=hp,
            rate_limit=rate_limit,
            api_key=api_key,
            base_url=base_url,
            **merged_kwargs,
        )
        # llm_completion returns LLMResponse when not streaming
        assert isinstance(result, LLMResponse)
        return result

    model_completion.__name__ = f"llm_{model.replace('/', '_').replace('-', '_')}"
    return model_completion


def get_rate_limiter() -> AdaptiveRateLimiter:
    """Get the global rate limiter instance."""
    return _default_limiter


def set_optimization_context(in_optimization: bool) -> Any:
    """Set the optimization context. Returns token for reset."""
    return _in_optimization.set(in_optimization)


def reset_optimization_context(token: Any) -> None:
    """Reset the optimization context."""
    _in_optimization.reset(token)
