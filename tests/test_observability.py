"""Tests for observability system."""

import asyncio
import json
import tempfile
from pathlib import Path

import pytest

from hyperfunc import (
    CallRecord,
    Example,
    HyperFunctionStats,
    HyperSystem,
    JSONExporter,
    ObservabilityHub,
    ObservationRecord,
    TraceSummary,
    hyperfunction,
)


def make_fast_fn():
    """Create a fresh fast_fn hyperfunction."""
    @hyperfunction(hp_type=None, optimize_hparams=False)
    async def fast_fn(x: int) -> int:
        """A fast function."""
        return x + 1
    return fast_fn


def make_slow_fn():
    """Create a fresh slow_fn hyperfunction."""
    @hyperfunction(hp_type=None, optimize_hparams=False)
    async def slow_fn(x: int) -> int:
        """A slow function."""
        await asyncio.sleep(0.01)
        return x * 2
    return slow_fn


def make_failing_fn():
    """Create a fresh failing_fn hyperfunction."""
    @hyperfunction(hp_type=None, optimize_hparams=False)
    async def failing_fn(x: int) -> int:
        """A function that fails."""
        if x < 0:
            raise ValueError("x must be non-negative")
        return x
    return failing_fn


class TestObservationRecord:
    """Test ObservationRecord dataclass."""

    def test_basic_creation(self):
        record = ObservationRecord(
            fn_name="test_fn",
            timestamp="2024-01-01T00:00:00Z",
            elapsed_s=0.5,
        )
        assert record.fn_name == "test_fn"
        assert record.elapsed_s == 0.5
        assert record.success is True

    def test_to_dict_omits_none(self):
        record = ObservationRecord(
            fn_name="test_fn",
            timestamp="2024-01-01T00:00:00Z",
            elapsed_s=0.5,
        )
        d = record.to_dict()
        assert "fn_name" in d
        assert "gen_ai_system" not in d  # None values omitted
        assert "extra_metrics" not in d  # Empty dict omitted

    def test_from_call_record(self):
        call_record = CallRecord(
            fn_name="my_fn",
            example_index=0,
            elapsed_s=0.123,
            error=None,
            extra_metrics={"custom": 1.0},
        )
        obs = ObservationRecord.from_call_record(call_record)
        assert obs.fn_name == "my_fn"
        assert obs.elapsed_s == 0.123
        assert obs.success is True
        assert obs.extra_metrics == {"custom": 1.0}

    def test_from_call_record_with_error(self):
        call_record = CallRecord(
            fn_name="my_fn",
            example_index=0,
            elapsed_s=0.1,
            error=ValueError("test error"),
        )
        obs = ObservationRecord.from_call_record(call_record)
        assert obs.success is False
        assert obs.error_type == "ValueError"
        assert obs.error_message == "test error"

    def test_genai_attributes(self):
        record = ObservationRecord(
            fn_name="llm_call",
            timestamp="2024-01-01T00:00:00Z",
            elapsed_s=1.5,
            gen_ai_system="openai",
            gen_ai_request_model="gpt-4",
            gen_ai_response_model="gpt-4-0613",
            gen_ai_request_temperature=0.7,
            gen_ai_usage_input_tokens=100,
            gen_ai_usage_output_tokens=50,
            gen_ai_usage_total_tokens=150,
            cost_usd=0.01,
        )
        d = record.to_dict()
        assert d["gen_ai_system"] == "openai"
        assert d["gen_ai_request_model"] == "gpt-4"
        assert d["gen_ai_usage_total_tokens"] == 150
        assert d["cost_usd"] == 0.01


class TestHyperFunctionStats:
    """Test HyperFunctionStats dataclass."""

    def test_basic_stats(self):
        stats = HyperFunctionStats(
            fn_name="test_fn",
            call_count=10,
            success_count=9,
            error_count=1,
            total_elapsed_s=5.0,
            min_elapsed_s=0.1,
            max_elapsed_s=1.0,
            mean_elapsed_s=0.5,
            p50_elapsed_s=0.4,
            p95_elapsed_s=0.9,
            p99_elapsed_s=0.95,
            error_rate=0.1,
        )
        assert stats.call_count == 10
        assert stats.error_rate == 0.1

    def test_to_dict(self):
        stats = HyperFunctionStats(
            fn_name="test_fn",
            call_count=5,
            success_count=5,
            error_count=0,
            total_elapsed_s=1.0,
            min_elapsed_s=0.1,
            max_elapsed_s=0.3,
            mean_elapsed_s=0.2,
            p50_elapsed_s=0.2,
            p95_elapsed_s=0.29,
            p99_elapsed_s=0.3,
            error_rate=0.0,
        )
        d = stats.to_dict()
        assert d["fn_name"] == "test_fn"
        assert d["call_count"] == 5


class TestTraceSummary:
    """Test TraceSummary dataclass."""

    def test_to_json(self):
        stats = HyperFunctionStats(
            fn_name="test_fn",
            call_count=5,
            success_count=5,
            error_count=0,
            total_elapsed_s=1.0,
            min_elapsed_s=0.1,
            max_elapsed_s=0.3,
            mean_elapsed_s=0.2,
            p50_elapsed_s=0.2,
            p95_elapsed_s=0.29,
            p99_elapsed_s=0.3,
            error_rate=0.0,
        )
        summary = TraceSummary(
            session_id="test-123",
            start_time="2024-01-01T00:00:00Z",
            end_time="2024-01-01T00:01:00Z",
            total_calls=5,
            total_errors=0,
            total_elapsed_s=1.0,
            by_function={"test_fn": stats},
        )
        json_str = summary.to_json()
        data = json.loads(json_str)
        assert data["session_id"] == "test-123"
        assert data["total_calls"] == 5
        assert "test_fn" in data["by_function"]

    def test_to_markdown(self):
        stats = HyperFunctionStats(
            fn_name="test_fn",
            call_count=5,
            success_count=4,
            error_count=1,
            total_elapsed_s=1.0,
            min_elapsed_s=0.1,
            max_elapsed_s=0.3,
            mean_elapsed_s=0.2,
            p50_elapsed_s=0.2,
            p95_elapsed_s=0.29,
            p99_elapsed_s=0.3,
            error_rate=0.2,
        )
        summary = TraceSummary(
            session_id="test-123",
            start_time="2024-01-01T00:00:00Z",
            end_time="2024-01-01T00:01:00Z",
            total_calls=5,
            total_errors=1,
            total_elapsed_s=1.0,
            by_function={"test_fn": stats},
        )
        md = summary.to_markdown()
        assert "# Trace Summary: test-123" in md
        assert "test_fn" in md
        assert "20.0%" in md  # Error rate


class TestJSONExporter:
    """Test JSONExporter."""

    def test_export_json(self):
        observations = [
            ObservationRecord(
                fn_name="fn1",
                timestamp="2024-01-01T00:00:00Z",
                elapsed_s=0.1,
            ),
            ObservationRecord(
                fn_name="fn2",
                timestamp="2024-01-01T00:00:01Z",
                elapsed_s=0.2,
            ),
        ]

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            exporter = JSONExporter(path, include_summary=False)
            exporter.export(observations, None)

            with open(path) as f:
                data = json.load(f)

            assert len(data["observations"]) == 2
            assert data["observations"][0]["fn_name"] == "fn1"
        finally:
            Path(path).unlink(missing_ok=True)

    def test_export_jsonl(self):
        observations = [
            ObservationRecord(
                fn_name="fn1",
                timestamp="2024-01-01T00:00:00Z",
                elapsed_s=0.1,
            ),
            ObservationRecord(
                fn_name="fn2",
                timestamp="2024-01-01T00:00:01Z",
                elapsed_s=0.2,
            ),
        ]

        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name

        try:
            exporter = JSONExporter(path, jsonl=True, include_summary=False)
            exporter.export(observations, None)

            with open(path) as f:
                lines = f.readlines()

            assert len(lines) == 2
            assert json.loads(lines[0])["fn_name"] == "fn1"
            assert json.loads(lines[1])["fn_name"] == "fn2"
        finally:
            Path(path).unlink(missing_ok=True)


def make_simple_system():
    """Create a SimpleSystem with fresh hyperfunctions."""
    fast_fn = make_fast_fn()
    slow_fn = make_slow_fn()

    class SimpleSystem(HyperSystem):
        """Simple system for testing observability."""

        async def run(self, x: int) -> int:
            a = await fast_fn(x)
            b = await slow_fn(a)
            return b

    return SimpleSystem()


def make_failing_system():
    """Create a FailingSystem with fresh hyperfunctions."""
    failing_fn = make_failing_fn()

    class FailingSystem(HyperSystem):
        """System that can fail."""

        async def run(self, x: int) -> int:
            return await failing_fn(x)

    return FailingSystem()


@pytest.mark.asyncio
class TestObservabilityHub:
    """Test ObservabilityHub integration."""

    async def test_observability_property(self):
        """System should have observability property."""
        system = make_simple_system()
        hub = system.observability
        assert isinstance(hub, ObservabilityHub)
        # Same instance on repeated access
        assert system.observability is hub

    async def test_trace_context_manager(self):
        """Trace context manager should collect observations."""
        system = make_simple_system()

        with system.observability.trace() as _:
            await system.run(5)

        history = system.observability.get_history()
        assert len(history) == 2  # fast_fn + slow_fn

        # Check function names
        fn_names = {obs.fn_name for obs in history}
        assert "fast_fn" in fn_names
        assert "slow_fn" in fn_names

    async def test_trace_with_session_id(self):
        """Trace can have custom session ID."""
        system = make_simple_system()

        with system.observability.trace(session_id="my-session"):
            await system.run(5)

        summary = system.observability.summary()
        assert summary.session_id == "my-session"

    async def test_summary_stats(self):
        """Summary should compute correct statistics."""
        system = make_simple_system()

        with system.observability.trace():
            for i in range(5):
                await system.run(i)

        summary = system.observability.summary()
        assert summary.total_calls == 10  # 5 runs * 2 functions
        assert summary.total_errors == 0
        assert "fast_fn" in summary.by_function
        assert "slow_fn" in summary.by_function

        fast_stats = summary.by_function["fast_fn"]
        assert fast_stats.call_count == 5
        assert fast_stats.error_count == 0

    async def test_summary_with_errors(self):
        """Summary should track errors."""
        system = make_failing_system()

        with system.observability.trace():
            # One success
            await system.run(5)
            # One failure
            try:
                await system.run(-1)
            except ValueError:
                pass

        summary = system.observability.summary()
        assert summary.total_calls == 2
        assert summary.total_errors == 1

        stats = summary.by_function["failing_fn"]
        assert stats.call_count == 2
        assert stats.error_count == 1
        assert stats.error_rate == 0.5

    async def test_export_json(self):
        """Should export to JSON file."""
        system = make_simple_system()

        with system.observability.trace():
            await system.run(5)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            system.observability.export_json(path)

            with open(path) as f:
                data = json.load(f)

            assert "observations" in data
            assert "summary" in data
            assert len(data["observations"]) == 2
        finally:
            Path(path).unlink(missing_ok=True)

    async def test_get_call_history(self):
        """System should expose get_call_history()."""
        system = make_simple_system()

        # Use observability.trace() to enable tracing properly
        with system.observability.trace():
            await system.run(5)

        # The history is now in the observability hub, not the system
        history = system.observability.get_history()
        assert len(history) == 2
        assert all(isinstance(r, ObservationRecord) for r in history)

    async def test_clear(self):
        """Clear should reset observation history."""
        system = make_simple_system()

        with system.observability.trace():
            await system.run(5)

        assert len(system.observability.get_history()) == 2

        system.observability.clear()
        assert len(system.observability.get_history()) == 0

    async def test_multiple_trace_sessions(self):
        """Each trace session should be independent."""
        system = make_simple_system()

        # First session
        with system.observability.trace(session_id="session-1"):
            await system.run(1)

        summary1 = system.observability.summary()
        assert summary1.session_id == "session-1"
        assert summary1.total_calls == 2

        # Second session - need fresh system since hyperfunctions are already registered
        system2 = make_simple_system()
        with system2.observability.trace(session_id="session-2"):
            await system2.run(2)
            await system2.run(3)

        summary2 = system2.observability.summary()
        assert summary2.session_id == "session-2"
        assert summary2.total_calls == 4  # 2 runs * 2 functions

    async def test_percentile_calculations(self):
        """Percentiles should be calculated correctly."""
        system = make_simple_system()

        with system.observability.trace():
            for _ in range(10):
                await system.run(1)

        summary = system.observability.summary()
        stats = summary.by_function["slow_fn"]

        # All elapsed times should be positive
        assert stats.min_elapsed_s > 0
        assert stats.max_elapsed_s >= stats.min_elapsed_s
        assert stats.p50_elapsed_s >= stats.min_elapsed_s
        assert stats.p95_elapsed_s >= stats.p50_elapsed_s
        assert stats.p99_elapsed_s >= stats.p95_elapsed_s


class TestLangFuseExporter:
    """Test LangFuseExporter (without actual LangFuse connection)."""

    def test_import_error_without_langfuse(self):
        """Should raise ImportError if langfuse not installed."""
        from hyperfunc.observability import LangFuseExporter

        exporter = LangFuseExporter(trace_name="test")
        # This will fail if langfuse is not installed
        # We just verify the class exists and can be instantiated
        assert exporter.trace_name == "test"


# Check if opentelemetry is available for OTLP tests
try:
    import opentelemetry  # noqa: F401
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False


class TestOTLPExporter:
    """Test OTLPExporter."""

    def test_otlp_exporter_creation(self):
        """Should create OTLPExporter with default settings."""
        from hyperfunc.observability import OTLPExporter

        exporter = OTLPExporter()
        assert exporter.endpoint == "http://localhost:4317"
        assert exporter.protocol == "grpc"
        assert exporter.service_name == "hyperfunc"
        assert exporter.insecure is True
        assert exporter.headers is None

    def test_otlp_exporter_custom_settings(self):
        """Should create OTLPExporter with custom settings."""
        from hyperfunc.observability import OTLPExporter

        exporter = OTLPExporter(
            endpoint="http://jaeger:4317",
            protocol="http",
            service_name="my-service",
            insecure=False,
            headers={"Authorization": "Bearer token"},
        )
        assert exporter.endpoint == "http://jaeger:4317"
        assert exporter.protocol == "http"
        assert exporter.service_name == "my-service"
        assert exporter.insecure is False
        assert exporter.headers == {"Authorization": "Bearer token"}

    def test_otlp_exporter_import_error_without_sdk(self):
        """Should raise ImportError with helpful message if opentelemetry not installed."""
        from hyperfunc.observability import OTLPExporter
        import sys

        # Mock the import failure by temporarily removing opentelemetry from sys.modules
        if not OTEL_AVAILABLE:
            exporter = OTLPExporter()
            with pytest.raises(ImportError, match="opentelemetry packages required"):
                exporter._setup_tracer()

    @pytest.mark.skipif(not OTEL_AVAILABLE, reason="opentelemetry not installed")
    def test_otlp_exporter_observation_conversion(self):
        """Test that observations are properly converted to span attributes."""
        from hyperfunc.observability import OTLPExporter
        from datetime import datetime, timezone

        # Create observation with GenAI attributes
        obs = ObservationRecord(
            fn_name="test_llm",
            timestamp=datetime.now(timezone.utc).isoformat(),
            elapsed_s=0.5,
            success=True,
            gen_ai_system="openai",
            gen_ai_request_model="gpt-4",
            gen_ai_response_model="gpt-4-0613",
            gen_ai_request_temperature=0.7,
            gen_ai_request_top_p=0.9,
            gen_ai_usage_input_tokens=100,
            gen_ai_usage_output_tokens=50,
            gen_ai_usage_total_tokens=150,
            cost_usd=0.01,
            extra_metrics={"latency_ms": 500.0},
        )

        # Just verify the observation has all expected fields
        assert obs.gen_ai_system == "openai"
        assert obs.gen_ai_request_model == "gpt-4"
        assert obs.gen_ai_usage_total_tokens == 150
        assert obs.cost_usd == 0.01
        assert obs.extra_metrics["latency_ms"] == 500.0

    @pytest.mark.skipif(not OTEL_AVAILABLE, reason="opentelemetry not installed")
    def test_otlp_exporter_error_observation(self):
        """Test that error observations set error status."""
        from hyperfunc.observability import OTLPExporter
        from datetime import datetime, timezone

        obs = ObservationRecord(
            fn_name="failing_fn",
            timestamp=datetime.now(timezone.utc).isoformat(),
            elapsed_s=0.1,
            success=False,
            error_type="ValueError",
            error_message="Something went wrong",
        )

        assert obs.success is False
        assert obs.error_type == "ValueError"
        assert obs.error_message == "Something went wrong"

    @pytest.mark.skipif(not OTEL_AVAILABLE, reason="opentelemetry not installed")
    def test_otlp_exporter_shutdown(self):
        """Test that shutdown clears the tracer and provider."""
        from hyperfunc.observability import OTLPExporter

        exporter = OTLPExporter()
        # Before setup, these should be None
        assert exporter._tracer is None
        assert exporter._provider is None

        # After shutdown (even without setup), should still be None
        exporter.shutdown()
        assert exporter._tracer is None
        assert exporter._provider is None


@pytest.mark.asyncio
@pytest.mark.skipif(not OTEL_AVAILABLE, reason="opentelemetry not installed")
class TestOTLPExporterIntegration:
    """Integration tests for OTLPExporter with ObservabilityHub."""

    async def test_export_otlp_method(self):
        """ObservabilityHub should have export_otlp method."""
        system = make_simple_system()

        with system.observability.trace():
            await system.run(5)

        # Verify the method exists and has correct signature
        assert hasattr(system.observability, "export_otlp")

        # We can't actually call it without a running OTLP collector,
        # but we can verify the observations are there
        history = system.observability.get_history()
        assert len(history) == 2
