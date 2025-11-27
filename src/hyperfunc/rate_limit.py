"""Adaptive rate limiting for HTTP-backed hyperfunctions (LLM APIs).

This module provides rate limiting that adapts based on X-RateLimit-* headers
from API responses (OpenAI, Anthropic, etc.).
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class RateLimitState:
    """State for a single endpoint's rate limits."""

    # Requests per minute
    requests_limit: Optional[int] = None
    requests_remaining: Optional[int] = None
    requests_reset_at: Optional[float] = None

    # Tokens per minute
    tokens_limit: Optional[int] = None
    tokens_remaining: Optional[int] = None
    tokens_reset_at: Optional[float] = None

    # Backoff state
    last_request_time: float = 0.0
    consecutive_429s: int = 0
    backoff_until: float = 0.0


@dataclass
class AdaptiveRateLimiter:
    """Adaptive rate limiter that learns from API response headers.

    Tracks rate limits per endpoint and adapts waiting behavior based on
    X-RateLimit-* headers from OpenAI, Anthropic, and similar APIs.

    Usage:
        limiter = AdaptiveRateLimiter()

        # Before making a request
        await limiter.acquire("openai", estimated_tokens=1000)

        # After receiving response
        limiter.update_from_headers("openai", response.headers)

    Features:
        - Parses X-RateLimit-* headers from major LLM providers
        - Tracks both request and token limits
        - Exponential backoff on 429 errors
        - Proactive waiting to avoid hitting limits
    """

    # Default limits if no headers received (conservative)
    default_requests_per_minute: int = 60
    default_tokens_per_minute: int = 90000

    # Backoff configuration
    initial_backoff_seconds: float = 1.0
    max_backoff_seconds: float = 60.0
    backoff_multiplier: float = 2.0

    # Safety margin (fraction of remaining capacity to use)
    safety_margin: float = 0.9

    # Per-endpoint state
    _endpoints: Dict[str, RateLimitState] = field(default_factory=dict)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def _get_state(self, endpoint: str) -> RateLimitState:
        """Get or create state for an endpoint."""
        if endpoint not in self._endpoints:
            self._endpoints[endpoint] = RateLimitState()
        return self._endpoints[endpoint]

    async def acquire(self, endpoint: str, estimated_tokens: int = 0) -> None:
        """Wait if necessary before making a request.

        Args:
            endpoint: The API endpoint identifier (e.g., "openai", "anthropic")
            estimated_tokens: Estimated tokens for this request (for token limits)
        """
        async with self._lock:
            state = self._get_state(endpoint)
            now = time.monotonic()

            # Check if we're in backoff
            if state.backoff_until > now:
                wait_time = state.backoff_until - now
                await asyncio.sleep(wait_time)
                now = time.monotonic()

            # Check request limits
            await self._wait_for_requests(state, now)

            # Check token limits
            if estimated_tokens > 0:
                await self._wait_for_tokens(state, estimated_tokens, now)

            state.last_request_time = time.monotonic()

    async def _wait_for_requests(self, state: RateLimitState, now: float) -> None:
        """Wait if request limit is about to be exceeded."""
        if state.requests_remaining is not None and state.requests_reset_at is not None:
            # We have rate limit info from headers
            if state.requests_remaining <= 1:
                # Wait until reset
                wait_time = max(0, state.requests_reset_at - now)
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
        else:
            # Use default rate limiting (simple spacing)
            min_interval = 60.0 / self.default_requests_per_minute
            elapsed = now - state.last_request_time
            if elapsed < min_interval:
                await asyncio.sleep(min_interval - elapsed)

    async def _wait_for_tokens(
        self, state: RateLimitState, tokens: int, now: float
    ) -> None:
        """Wait if token limit is about to be exceeded."""
        if state.tokens_remaining is not None and state.tokens_reset_at is not None:
            if state.tokens_remaining < tokens:
                # Wait until reset
                wait_time = max(0, state.tokens_reset_at - now)
                if wait_time > 0:
                    await asyncio.sleep(wait_time)

    def update_from_headers(self, endpoint: str, headers: Dict[str, Any]) -> None:
        """Update rate limit state from response headers.

        Supports headers from:
        - OpenAI: x-ratelimit-limit-requests, x-ratelimit-remaining-requests, etc.
        - Anthropic: Similar format

        Args:
            endpoint: The API endpoint identifier
            headers: Response headers (case-insensitive dict)
        """
        state = self._get_state(endpoint)
        now = time.monotonic()

        # Normalize header keys to lowercase
        headers_lower = {k.lower(): v for k, v in headers.items()}

        # Parse request limits
        if "x-ratelimit-limit-requests" in headers_lower:
            state.requests_limit = int(headers_lower["x-ratelimit-limit-requests"])
        if "x-ratelimit-remaining-requests" in headers_lower:
            state.requests_remaining = int(
                headers_lower["x-ratelimit-remaining-requests"]
            )
        if "x-ratelimit-reset-requests" in headers_lower:
            reset_str = headers_lower["x-ratelimit-reset-requests"]
            state.requests_reset_at = now + self._parse_reset_time(reset_str)

        # Parse token limits
        if "x-ratelimit-limit-tokens" in headers_lower:
            state.tokens_limit = int(headers_lower["x-ratelimit-limit-tokens"])
        if "x-ratelimit-remaining-tokens" in headers_lower:
            state.tokens_remaining = int(headers_lower["x-ratelimit-remaining-tokens"])
        if "x-ratelimit-reset-tokens" in headers_lower:
            reset_str = headers_lower["x-ratelimit-reset-tokens"]
            state.tokens_reset_at = now + self._parse_reset_time(reset_str)

        # Reset backoff on successful response
        state.consecutive_429s = 0

    def _parse_reset_time(self, reset_str: str) -> float:
        """Parse reset time string to seconds.

        Handles formats like:
        - "1s" (1 second)
        - "100ms" (100 milliseconds)
        - "1m30s" (1 minute 30 seconds)
        - "1h0m2.397s" (1 hour 0 minutes 2.397 seconds)
        - Unix timestamp
        """
        reset_str = str(reset_str).strip()

        # Check for ms suffix
        if reset_str.endswith("ms"):
            return float(reset_str[:-2]) / 1000.0

        # Check for s suffix - parse h/m/s components
        if reset_str.endswith("s") and not reset_str.endswith("ms"):
            total_seconds = 0.0
            remaining = reset_str

            # Parse hours if present
            if "h" in remaining:
                h_idx = remaining.find("h")
                total_seconds += float(remaining[:h_idx]) * 3600
                remaining = remaining[h_idx + 1 :]

            # Parse minutes if present
            if "m" in remaining:
                m_idx = remaining.find("m")
                total_seconds += float(remaining[:m_idx]) * 60
                remaining = remaining[m_idx + 1 :]

            # Parse seconds (remove trailing 's')
            if remaining.endswith("s"):
                remaining = remaining[:-1]
            if remaining:
                total_seconds += float(remaining)

            return total_seconds

        # Try parsing as float (seconds or unix timestamp)
        try:
            value = float(reset_str)
            # If it looks like a unix timestamp (large number), convert to relative
            if value > 1e9:
                return max(0, value - time.time())
            return value
        except ValueError:
            return 1.0  # Default to 1 second

    def record_429(self, endpoint: str) -> float:
        """Record a 429 (rate limited) response and return backoff time.

        Args:
            endpoint: The API endpoint identifier

        Returns:
            Number of seconds to wait before retrying
        """
        state = self._get_state(endpoint)
        state.consecutive_429s += 1

        # Exponential backoff
        backoff = self.initial_backoff_seconds * (
            self.backoff_multiplier ** (state.consecutive_429s - 1)
        )
        backoff = min(backoff, self.max_backoff_seconds)

        state.backoff_until = time.monotonic() + backoff
        return backoff

    def get_stats(self, endpoint: str) -> Dict[str, Any]:
        """Get current rate limit statistics for an endpoint.

        Args:
            endpoint: The API endpoint identifier

        Returns:
            Dict with current rate limit state
        """
        state = self._get_state(endpoint)
        now = time.monotonic()

        return {
            "requests_limit": state.requests_limit,
            "requests_remaining": state.requests_remaining,
            "requests_reset_in": (
                max(0, state.requests_reset_at - now)
                if state.requests_reset_at
                else None
            ),
            "tokens_limit": state.tokens_limit,
            "tokens_remaining": state.tokens_remaining,
            "tokens_reset_in": (
                max(0, state.tokens_reset_at - now) if state.tokens_reset_at else None
            ),
            "consecutive_429s": state.consecutive_429s,
            "in_backoff": state.backoff_until > now,
            "backoff_remaining": (
                max(0, state.backoff_until - now) if state.backoff_until > now else 0
            ),
        }

    def reset(self, endpoint: Optional[str] = None) -> None:
        """Reset rate limit state.

        Args:
            endpoint: If provided, reset only this endpoint. Otherwise reset all.
        """
        if endpoint:
            if endpoint in self._endpoints:
                self._endpoints[endpoint] = RateLimitState()
        else:
            self._endpoints.clear()
