"""Tests for LiteLLM integration."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import torch

from hyperfunc import LMParam

# Check if litellm is available
try:
    from hyperfunc.llm import (
        LITELLM_AVAILABLE,
        LLMResponse,
        TokenUsage,
        _in_optimization,
        get_rate_limiter,
        llm_completion,
        make_llm_completion,
        reset_optimization_context,
        set_optimization_context,
    )

    SKIP_LLM_TESTS = not LITELLM_AVAILABLE
except ImportError:
    SKIP_LLM_TESTS = True


pytestmark = pytest.mark.skipif(
    SKIP_LLM_TESTS, reason="litellm not installed"
)


class TestLMParam:
    """Test LMParam dataclass and tensor conversion."""

    def test_shape(self):
        """LMParam should have 5 ES-optimizable params."""
        assert LMParam.shape() == (5,)

    def test_default_values(self):
        """Test default parameter values."""
        p = LMParam()
        assert p.temperature == 0.7
        assert p.top_p == 1.0
        assert p.presence_penalty == 0.0
        assert p.frequency_penalty == 0.0
        assert p.max_tokens_frac == 0.25
        assert p.max_tokens is None
        assert p.stop is None
        assert p.seed is None

    def test_to_tensor(self):
        """Test conversion to tensor."""
        p = LMParam(temperature=0.5, top_p=0.9, presence_penalty=-0.5)
        t = p.to_tensor()
        assert t.shape == (5,)
        assert t[0].item() == pytest.approx(0.5)
        assert t[1].item() == pytest.approx(0.9)
        assert t[2].item() == pytest.approx(-0.5)

    def test_from_tensor(self):
        """Test conversion from tensor."""
        t = torch.tensor([0.8, 0.95, 0.1, -0.1, 0.5])
        p = LMParam.from_tensor(t)
        assert p.temperature == pytest.approx(0.8)
        assert p.top_p == pytest.approx(0.95)
        assert p.presence_penalty == pytest.approx(0.1)
        assert p.frequency_penalty == pytest.approx(-0.1)
        assert p.max_tokens_frac == pytest.approx(0.5)

    def test_roundtrip(self):
        """Test tensor roundtrip preserves values."""
        p = LMParam(
            temperature=0.6,
            top_p=0.85,
            presence_penalty=0.2,
            frequency_penalty=-0.3,
            max_tokens_frac=0.4,
        )
        t = p.to_tensor()
        p2 = LMParam.from_tensor(t)
        assert p2.temperature == pytest.approx(p.temperature)
        assert p2.top_p == pytest.approx(p.top_p)
        assert p2.presence_penalty == pytest.approx(p.presence_penalty)
        assert p2.frequency_penalty == pytest.approx(p.frequency_penalty)
        assert p2.max_tokens_frac == pytest.approx(p.max_tokens_frac)

    def test_to_litellm_kwargs_with_max_tokens(self):
        """Test conversion to litellm kwargs with explicit max_tokens."""
        p = LMParam(
            temperature=0.8,
            top_p=0.9,
            max_tokens=500,
            stop=["END"],
            seed=42,
        )
        kwargs = p.to_litellm_kwargs()
        assert kwargs["temperature"] == 0.8
        assert kwargs["top_p"] == 0.9
        assert kwargs["max_tokens"] == 500
        assert kwargs["stop"] == ["END"]
        assert kwargs["seed"] == 42

    def test_to_litellm_kwargs_with_max_tokens_frac(self):
        """Test conversion to litellm kwargs with max_tokens_frac."""
        p = LMParam(max_tokens_frac=0.5)
        kwargs = p.to_litellm_kwargs(model_max_tokens=8192)
        assert kwargs["max_tokens"] == 4096

    def test_to_litellm_kwargs_max_tokens_overrides_frac(self):
        """Explicit max_tokens should override max_tokens_frac."""
        p = LMParam(max_tokens=100, max_tokens_frac=0.5)
        kwargs = p.to_litellm_kwargs(model_max_tokens=8192)
        assert kwargs["max_tokens"] == 100


class TestTokenUsage:
    """Test TokenUsage dataclass."""

    def test_creation(self):
        usage = TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        assert usage.prompt_tokens == 100
        assert usage.completion_tokens == 50
        assert usage.total_tokens == 150


class TestLLMResponse:
    """Test LLMResponse dataclass."""

    def test_creation(self):
        usage = TokenUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        resp = LLMResponse(
            content="Hello!",
            usage=usage,
            model="gpt-4",
            finish_reason="stop",
        )
        assert resp.content == "Hello!"
        assert resp.model == "gpt-4"

    def test_str(self):
        """__str__ should return content."""
        usage = TokenUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        resp = LLMResponse(content="Test", usage=usage, model="gpt-4", finish_reason="stop")
        assert str(resp) == "Test"


class TestOptimizationContext:
    """Test the optimization context var."""

    def test_default_is_false(self):
        assert _in_optimization.get() is False

    def test_set_and_reset(self):
        token = set_optimization_context(True)
        assert _in_optimization.get() is True
        reset_optimization_context(token)
        assert _in_optimization.get() is False

    def test_nested_context(self):
        token1 = set_optimization_context(True)
        assert _in_optimization.get() is True
        token2 = set_optimization_context(False)
        assert _in_optimization.get() is False
        reset_optimization_context(token2)
        assert _in_optimization.get() is True
        reset_optimization_context(token1)
        assert _in_optimization.get() is False


class TestRateLimiter:
    """Test rate limiter integration."""

    def test_get_rate_limiter(self):
        limiter = get_rate_limiter()
        assert limiter is not None


@pytest.mark.asyncio
class TestLLMCompletion:
    """Test llm_completion hyperfunction."""

    async def test_llm_completion_basic(self):
        """Test basic completion with mocked litellm."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello, world!"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.model = "gpt-4"
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.usage.total_tokens = 15
        mock_response._hidden_params = {}

        with patch("hyperfunc.llm.litellm") as mock_litellm:
            mock_litellm.completion.return_value = mock_response

            result = await llm_completion(
                model="gpt-4",
                messages=[{"role": "user", "content": "Hi"}],
                rate_limit=False,
            )

            assert isinstance(result, LLMResponse)
            assert result.content == "Hello, world!"
            assert result.model == "gpt-4"
            assert result.usage.total_tokens == 15

    async def test_llm_completion_with_hp(self):
        """Test completion with LMParam."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Response"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.model = "gpt-4"
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.usage.total_tokens = 15
        mock_response._hidden_params = {}

        hp = LMParam(temperature=0.5, top_p=0.9)

        with patch("hyperfunc.llm.litellm") as mock_litellm:
            mock_litellm.completion.return_value = mock_response

            result = await llm_completion(
                model="gpt-4",
                messages=[{"role": "user", "content": "Test"}],
                hp=hp,
                rate_limit=False,
            )

            # Verify the completion was called with hp params
            call_kwargs = mock_litellm.completion.call_args[1]
            assert call_kwargs["temperature"] == 0.5
            assert call_kwargs["top_p"] == 0.9

    async def test_explicit_params_override_hp(self):
        """Explicit params should override hp values."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Response"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.model = "gpt-4"
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.usage.total_tokens = 15
        mock_response._hidden_params = {}

        hp = LMParam(temperature=0.5)

        with patch("hyperfunc.llm.litellm") as mock_litellm:
            mock_litellm.completion.return_value = mock_response

            await llm_completion(
                model="gpt-4",
                messages=[{"role": "user", "content": "Test"}],
                hp=hp,
                temperature=0.9,  # Override hp.temperature
                rate_limit=False,
            )

            call_kwargs = mock_litellm.completion.call_args[1]
            assert call_kwargs["temperature"] == 0.9

    async def test_stream_disabled_during_optimization(self):
        """Stream should be disabled when in optimization context."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Response"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.model = "gpt-4"
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.usage.total_tokens = 15
        mock_response._hidden_params = {}

        with patch("hyperfunc.llm.litellm") as mock_litellm:
            mock_litellm.completion.return_value = mock_response

            # Set optimization context
            token = set_optimization_context(True)
            try:
                result = await llm_completion(
                    model="gpt-4",
                    messages=[{"role": "user", "content": "Test"}],
                    stream=True,  # Request streaming
                    rate_limit=False,
                )
                # Should return LLMResponse, not async iterator
                assert isinstance(result, LLMResponse)
            finally:
                reset_optimization_context(token)


@pytest.mark.asyncio
class TestMakeLLMCompletion:
    """Test make_llm_completion factory."""

    async def test_factory_basic(self):
        """Test basic factory usage."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Factory response"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.model = "gpt-4"
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.usage.total_tokens = 15
        mock_response._hidden_params = {}

        with patch("hyperfunc.llm.litellm") as mock_litellm:
            mock_litellm.completion.return_value = mock_response

            gpt4 = make_llm_completion("gpt-4")
            result = await gpt4("Hello")

            assert isinstance(result, LLMResponse)
            assert result.content == "Factory response"

    async def test_factory_with_system_prompt(self):
        """Test factory with system prompt."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Response"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.model = "gpt-4"
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.usage.total_tokens = 15
        mock_response._hidden_params = {}

        with patch("hyperfunc.llm.litellm") as mock_litellm:
            mock_litellm.completion.return_value = mock_response

            gpt4 = make_llm_completion(
                "gpt-4",
                system_prompt="You are helpful.",
            )
            await gpt4("Test")

            call_args = mock_litellm.completion.call_args
            messages = call_args[1]["messages"]
            assert len(messages) == 2
            assert messages[0]["role"] == "system"
            assert messages[0]["content"] == "You are helpful."
            assert messages[1]["role"] == "user"
            assert messages[1]["content"] == "Test"

    async def test_factory_function_name(self):
        """Factory should set appropriate function name."""
        gpt4 = make_llm_completion("gpt-4")
        assert "gpt_4" in gpt4.__name__

        claude = make_llm_completion("anthropic/claude-3-opus")
        assert "anthropic_claude_3_opus" in claude.__name__
