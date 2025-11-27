"""Integration tests for OTLPExporter with real Jaeger instance.

These tests use testcontainers to spin up a Jaeger instance and verify
that spans are actually exported and visible via Jaeger's API.

Run with: pytest tests/test_otlp_integration.py -m integration -v
Skip with: pytest -m "not integration"
"""

import asyncio
import time
from datetime import datetime, timezone

import pytest
import requests

from hyperfunc import (
    Example,
    HyperSystem,
    ObservationRecord,
    OTLPExporter,
    hyperfunction,
)

# Check if testcontainers and docker are available
try:
    from testcontainers.core.container import DockerContainer
    from testcontainers.core.waiting_utils import wait_for_logs

    TESTCONTAINERS_AVAILABLE = True
except ImportError:
    TESTCONTAINERS_AVAILABLE = False

# Check if opentelemetry is available
try:
    import opentelemetry

    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False


class JaegerContainer(DockerContainer):
    """Jaeger all-in-one container for testing OTLP export."""

    def __init__(self):
        super().__init__("jaegertracing/all-in-one:1.54")
        self.with_exposed_ports(4317, 16686)  # OTLP gRPC, UI/API
        self.with_env("COLLECTOR_OTLP_ENABLED", "true")

    def get_otlp_endpoint(self) -> str:
        """Get the OTLP gRPC endpoint."""
        host = self.get_container_host_ip()
        port = self.get_exposed_port(4317)
        return f"{host}:{port}"

    def get_api_url(self) -> str:
        """Get the Jaeger API URL."""
        host = self.get_container_host_ip()
        port = self.get_exposed_port(16686)
        return f"http://{host}:{port}"


def make_test_fn():
    """Create a test hyperfunction."""

    @hyperfunction(hp_type=None, optimize_hparams=False)
    async def test_fn(x: int) -> int:
        await asyncio.sleep(0.01)
        return x * 2

    return test_fn


def make_test_system():
    """Create a test HyperSystem."""
    test_fn = make_test_fn()

    class TestSystem(HyperSystem):
        async def run(self, x: int) -> int:
            return await test_fn(x)

    return TestSystem()


@pytest.fixture(scope="module")
def jaeger_container():
    """Start Jaeger container for the test module."""
    if not TESTCONTAINERS_AVAILABLE:
        pytest.skip("testcontainers not available")

    container = JaegerContainer()
    container.start()

    # Wait for Jaeger to be ready
    wait_for_logs(container, "Starting HTTP server", timeout=30)
    time.sleep(2)  # Extra time for all services to start

    yield container

    container.stop()


@pytest.mark.integration
@pytest.mark.skipif(
    not TESTCONTAINERS_AVAILABLE or not OTEL_AVAILABLE,
    reason="testcontainers or opentelemetry not available",
)
class TestOTLPExporterWithJaeger:
    """Integration tests for OTLPExporter with real Jaeger instance."""

    def test_export_single_observation(self, jaeger_container):
        """Test exporting a single observation to Jaeger."""
        endpoint = jaeger_container.get_otlp_endpoint()
        api_url = jaeger_container.get_api_url()

        # Create observation
        obs = ObservationRecord(
            fn_name="single_test_fn",
            timestamp=datetime.now(timezone.utc).isoformat(),
            elapsed_s=0.123,
            success=True,
            extra_metrics={"test_value": 42.0},
        )

        # Export to Jaeger
        exporter = OTLPExporter(
            endpoint=endpoint,
            service_name="test-single",
            insecure=True,
        )
        exporter.export([obs])
        exporter.shutdown()

        # Wait for Jaeger to process
        time.sleep(2)

        # Query Jaeger API for the service
        response = requests.get(f"{api_url}/api/services")
        assert response.status_code == 200
        services = response.json().get("data", [])
        assert "test-single" in services, f"Service not found. Available: {services}"

    def test_export_genai_attributes(self, jaeger_container):
        """Test that GenAI attributes are properly exported."""
        endpoint = jaeger_container.get_otlp_endpoint()
        api_url = jaeger_container.get_api_url()

        # Create observation with GenAI attributes
        obs = ObservationRecord(
            fn_name="llm_call",
            timestamp=datetime.now(timezone.utc).isoformat(),
            elapsed_s=1.5,
            success=True,
            gen_ai_system="openai",
            gen_ai_request_model="gpt-4",
            gen_ai_response_model="gpt-4-0613",
            gen_ai_request_temperature=0.7,
            gen_ai_request_top_p=0.9,
            gen_ai_usage_input_tokens=100,
            gen_ai_usage_output_tokens=50,
            gen_ai_usage_total_tokens=150,
            cost_usd=0.015,
        )

        # Export to Jaeger
        exporter = OTLPExporter(
            endpoint=endpoint,
            service_name="test-genai",
            insecure=True,
        )
        exporter.export([obs])
        exporter.shutdown()

        # Wait for Jaeger to process
        time.sleep(2)

        # Query Jaeger for traces
        response = requests.get(
            f"{api_url}/api/traces",
            params={"service": "test-genai", "limit": 10},
        )
        assert response.status_code == 200
        traces = response.json().get("data", [])
        assert len(traces) > 0, "No traces found"

        # Check that the span has expected attributes
        trace = traces[0]
        spans = trace.get("spans", [])
        assert len(spans) > 0, "No spans in trace"

        span = spans[0]
        tags = {tag["key"]: tag["value"] for tag in span.get("tags", [])}

        # Verify GenAI attributes are present
        assert tags.get("gen_ai.system") == "openai"
        assert tags.get("gen_ai.request.model") == "gpt-4"
        assert tags.get("gen_ai.usage.input_tokens") == 100
        assert tags.get("gen_ai.usage.output_tokens") == 50

    def test_export_error_observation(self, jaeger_container):
        """Test that error observations have error status."""
        endpoint = jaeger_container.get_otlp_endpoint()
        api_url = jaeger_container.get_api_url()

        # Create error observation
        obs = ObservationRecord(
            fn_name="failing_fn",
            timestamp=datetime.now(timezone.utc).isoformat(),
            elapsed_s=0.05,
            success=False,
            error_type="ValueError",
            error_message="Something went wrong",
        )

        # Export to Jaeger
        exporter = OTLPExporter(
            endpoint=endpoint,
            service_name="test-error",
            insecure=True,
        )
        exporter.export([obs])
        exporter.shutdown()

        # Wait for Jaeger to process
        time.sleep(2)

        # Query Jaeger for traces
        response = requests.get(
            f"{api_url}/api/traces",
            params={"service": "test-error", "limit": 10},
        )
        assert response.status_code == 200
        traces = response.json().get("data", [])
        assert len(traces) > 0, "No traces found"

        # Check that the span has error attributes
        trace = traces[0]
        spans = trace.get("spans", [])
        span = spans[0]
        tags = {tag["key"]: tag["value"] for tag in span.get("tags", [])}

        assert tags.get("exception.type") == "ValueError"
        assert tags.get("exception.message") == "Something went wrong"
        # OTEL error status is typically represented as otel.status_code
        assert tags.get("otel.status_code") == "ERROR"

    def test_export_multiple_observations(self, jaeger_container):
        """Test exporting multiple observations."""
        endpoint = jaeger_container.get_otlp_endpoint()
        api_url = jaeger_container.get_api_url()

        # Create multiple observations
        observations = [
            ObservationRecord(
                fn_name=f"batch_fn_{i}",
                timestamp=datetime.now(timezone.utc).isoformat(),
                elapsed_s=0.1 * (i + 1),
                success=True,
                example_index=i,
            )
            for i in range(5)
        ]

        # Export to Jaeger
        exporter = OTLPExporter(
            endpoint=endpoint,
            service_name="test-batch",
            insecure=True,
        )
        exporter.export(observations)
        exporter.shutdown()

        # Wait for Jaeger to process
        time.sleep(2)

        # Query Jaeger for traces
        response = requests.get(
            f"{api_url}/api/traces",
            params={"service": "test-batch", "limit": 10},
        )
        assert response.status_code == 200
        traces = response.json().get("data", [])

        # Each observation becomes a separate trace (since they're not linked)
        # Just verify we got some traces
        assert len(traces) > 0, "No traces found"

    @pytest.mark.asyncio
    async def test_observability_hub_export_otlp(self, jaeger_container):
        """Test ObservabilityHub.export_otlp() method."""
        endpoint = jaeger_container.get_otlp_endpoint()
        api_url = jaeger_container.get_api_url()

        system = make_test_system()

        # Run with tracing
        with system.observability.trace(session_id="test-session"):
            await system.run(5)
            await system.run(10)

        # Export via ObservabilityHub
        system.observability.export_otlp(
            endpoint=endpoint,
            service_name="test-hub",
            insecure=True,
        )

        # Wait for Jaeger to process
        time.sleep(2)

        # Verify service was created
        response = requests.get(f"{api_url}/api/services")
        assert response.status_code == 200
        services = response.json().get("data", [])
        assert "test-hub" in services, f"Service not found. Available: {services}"

        # Verify traces exist
        response = requests.get(
            f"{api_url}/api/traces",
            params={"service": "test-hub", "limit": 10},
        )
        assert response.status_code == 200
        traces = response.json().get("data", [])
        assert len(traces) > 0, "No traces found"
