"""Observability system for hyperfunc.

Provides structured tracing, metrics, and exporters for hyperfunction calls.
Follows GenAI semantic conventions for LLM observability.

Usage:
    from hyperfunc import HyperSystem
    from hyperfunc.observability import TraceSummary

    system = MySystem()

    # Enable tracing
    with system.observability.trace():
        await system.run(...)

    # Get call history
    history = system.observability.get_history()

    # Get summary stats
    summary = system.observability.summary()
    print(summary.to_dict())

    # Export to JSON
    system.observability.export_json("trace.json")

    # Export to LangFuse (requires langfuse package)
    system.observability.export_langfuse(trace_id="my-trace")
"""

from __future__ import annotations

import json
import statistics
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Protocol,
    Sequence,
    TYPE_CHECKING,
    Union,
)

if TYPE_CHECKING:
    from .core import HyperSystem, CallRecord


@dataclass
class ObservationRecord:
    """Rich observation record with GenAI semantic conventions.

    Follows OpenTelemetry GenAI semantic conventions:
    https://opentelemetry.io/docs/specs/semconv/gen-ai/

    Attributes:
        fn_name: Name of the hyperfunction
        timestamp: ISO 8601 timestamp of the call
        elapsed_s: Duration in seconds
        example_index: Index of the example (if in evaluation)
        success: Whether the call succeeded
        error_type: Type of error if failed
        error_message: Error message if failed

        # GenAI attributes (for LLM hyperfunctions)
        gen_ai_system: LLM provider (e.g., "openai", "anthropic")
        gen_ai_request_model: Requested model name
        gen_ai_response_model: Actual model used
        gen_ai_request_temperature: Temperature setting
        gen_ai_request_top_p: Top-p setting
        gen_ai_usage_input_tokens: Input token count
        gen_ai_usage_output_tokens: Output token count
        gen_ai_usage_total_tokens: Total token count

        # Cost tracking
        cost_usd: Estimated cost in USD

        # Custom metrics
        extra_metrics: Additional metrics from otel_metric_funcs
    """

    # Core attributes
    fn_name: str
    timestamp: str
    elapsed_s: float
    example_index: Optional[int] = None
    success: bool = True
    error_type: Optional[str] = None
    error_message: Optional[str] = None

    # GenAI semantic conventions
    gen_ai_system: Optional[str] = None
    gen_ai_request_model: Optional[str] = None
    gen_ai_response_model: Optional[str] = None
    gen_ai_request_temperature: Optional[float] = None
    gen_ai_request_top_p: Optional[float] = None
    gen_ai_usage_input_tokens: Optional[int] = None
    gen_ai_usage_output_tokens: Optional[int] = None
    gen_ai_usage_total_tokens: Optional[int] = None

    # Cost
    cost_usd: Optional[float] = None

    # Custom metrics
    extra_metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, omitting None values."""
        d = asdict(self)
        return {k: v for k, v in d.items() if v is not None and v != {}}

    @classmethod
    def from_call_record(
        cls,
        record: "CallRecord",
        timestamp: Optional[str] = None,
    ) -> "ObservationRecord":
        """Create from a CallRecord."""
        return cls(
            fn_name=record.fn_name,
            timestamp=timestamp or datetime.now(timezone.utc).isoformat(),
            elapsed_s=record.elapsed_s,
            example_index=record.example_index,
            success=record.error is None,
            error_type=type(record.error).__name__ if record.error else None,
            error_message=str(record.error) if record.error else None,
            extra_metrics=dict(record.extra_metrics),
        )


@dataclass
class HyperFunctionStats:
    """Statistics for a single hyperfunction."""

    fn_name: str
    call_count: int
    success_count: int
    error_count: int
    total_elapsed_s: float
    min_elapsed_s: float
    max_elapsed_s: float
    mean_elapsed_s: float
    p50_elapsed_s: float
    p95_elapsed_s: float
    p99_elapsed_s: float
    error_rate: float

    # GenAI stats (if applicable)
    total_input_tokens: Optional[int] = None
    total_output_tokens: Optional[int] = None
    total_cost_usd: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        d = asdict(self)
        return {k: v for k, v in d.items() if v is not None}


@dataclass
class TraceSummary:
    """Summary of trace data with per-hyperfunction stats.

    Attributes:
        session_id: Unique session identifier
        start_time: Session start time (ISO 8601)
        end_time: Session end time (ISO 8601)
        total_calls: Total number of hyperfunction calls
        total_errors: Total number of errors
        total_elapsed_s: Total time across all calls
        by_function: Per-hyperfunction statistics
    """

    session_id: str
    start_time: str
    end_time: str
    total_calls: int
    total_errors: int
    total_elapsed_s: float
    by_function: Dict[str, HyperFunctionStats]

    # Aggregate GenAI stats
    total_input_tokens: Optional[int] = None
    total_output_tokens: Optional[int] = None
    total_cost_usd: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "session_id": self.session_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "total_calls": self.total_calls,
            "total_errors": self.total_errors,
            "total_elapsed_s": self.total_elapsed_s,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost_usd": self.total_cost_usd,
            "by_function": {
                name: stats.to_dict()
                for name, stats in self.by_function.items()
            },
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def to_markdown(self) -> str:
        """Generate markdown report."""
        lines = [
            f"# Trace Summary: {self.session_id}",
            "",
            f"**Start:** {self.start_time}",
            f"**End:** {self.end_time}",
            f"**Total Calls:** {self.total_calls}",
            f"**Total Errors:** {self.total_errors}",
            f"**Total Time:** {self.total_elapsed_s:.3f}s",
        ]

        if self.total_input_tokens is not None:
            lines.append(f"**Total Input Tokens:** {self.total_input_tokens:,}")
        if self.total_output_tokens is not None:
            lines.append(f"**Total Output Tokens:** {self.total_output_tokens:,}")
        if self.total_cost_usd is not None:
            lines.append(f"**Total Cost:** ${self.total_cost_usd:.4f}")

        lines.extend(["", "## Per-Function Statistics", ""])
        lines.append("| Function | Calls | Errors | Error Rate | Mean (s) | P50 (s) | P95 (s) | P99 (s) |")
        lines.append("|----------|-------|--------|------------|----------|---------|---------|---------|")

        for name, stats in sorted(self.by_function.items()):
            lines.append(
                f"| {name} | {stats.call_count} | {stats.error_count} | "
                f"{stats.error_rate:.1%} | {stats.mean_elapsed_s:.3f} | "
                f"{stats.p50_elapsed_s:.3f} | {stats.p95_elapsed_s:.3f} | "
                f"{stats.p99_elapsed_s:.3f} |"
            )

        return "\n".join(lines)


class Exporter(Protocol):
    """Protocol for trace exporters."""

    def export(
        self,
        observations: Sequence[ObservationRecord],
        summary: Optional[TraceSummary] = None,
    ) -> None:
        """Export observations and summary."""
        ...


class JSONExporter:
    """Export traces to JSON or JSONL files."""

    def __init__(
        self,
        path: Union[str, Path],
        jsonl: bool = False,
        include_summary: bool = True,
    ):
        """
        Args:
            path: Output file path
            jsonl: If True, write JSONL (one record per line)
            include_summary: If True, include summary at end of JSON output
        """
        self.path = Path(path)
        self.jsonl = jsonl
        self.include_summary = include_summary

    def export(
        self,
        observations: Sequence[ObservationRecord],
        summary: Optional[TraceSummary] = None,
    ) -> None:
        """Export to file."""
        if self.jsonl:
            with open(self.path, "w") as f:
                for obs in observations:
                    f.write(json.dumps(obs.to_dict()) + "\n")
                if summary and self.include_summary:
                    f.write(json.dumps({"_summary": summary.to_dict()}) + "\n")
        else:
            data: Dict[str, Any] = {
                "observations": [obs.to_dict() for obs in observations],
            }
            if summary and self.include_summary:
                data["summary"] = summary.to_dict()
            with open(self.path, "w") as f:
                json.dump(data, f, indent=2)


class LangFuseExporter:
    """Export traces to LangFuse.

    Requires: pip install langfuse

    Environment variables:
        LANGFUSE_PUBLIC_KEY: Your LangFuse public key
        LANGFUSE_SECRET_KEY: Your LangFuse secret key
        LANGFUSE_HOST: LangFuse host (default: https://cloud.langfuse.com)
    """

    def __init__(
        self,
        trace_name: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        public_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        host: Optional[str] = None,
    ):
        """
        Args:
            trace_name: Name for the trace in LangFuse
            user_id: User ID for attribution
            session_id: Session ID for grouping
            metadata: Additional metadata
            public_key: Override LANGFUSE_PUBLIC_KEY env var
            secret_key: Override LANGFUSE_SECRET_KEY env var
            host: Override LANGFUSE_HOST env var
        """
        self.trace_name = trace_name
        self.user_id = user_id
        self.session_id = session_id
        self.metadata = metadata or {}
        self.public_key = public_key
        self.secret_key = secret_key
        self.host = host
        self._langfuse = None

    def _get_client(self) -> Any:
        """Lazy-load LangFuse client."""
        if self._langfuse is None:
            try:
                from langfuse import Langfuse
            except ImportError:
                raise ImportError(
                    "langfuse is required for LangFuseExporter. "
                    "Install with: pip install langfuse"
                )

            kwargs: Dict[str, Any] = {}
            if self.public_key:
                kwargs["public_key"] = self.public_key
            if self.secret_key:
                kwargs["secret_key"] = self.secret_key
            if self.host:
                kwargs["host"] = self.host

            self._langfuse = Langfuse(**kwargs)
        return self._langfuse

    def export(
        self,
        observations: Sequence[ObservationRecord],
        summary: Optional[TraceSummary] = None,
    ) -> None:
        """Export to LangFuse."""
        client = self._get_client()

        # Create trace
        trace = client.trace(
            name=self.trace_name or "hyperfunc-trace",
            user_id=self.user_id,
            session_id=self.session_id,
            metadata={
                **self.metadata,
                "total_calls": len(observations),
                "summary": summary.to_dict() if summary else None,
            },
        )

        # Add spans for each observation
        for obs in observations:
            span_kwargs: Dict[str, Any] = {
                "name": obs.fn_name,
                "start_time": obs.timestamp,
                "end_time": obs.timestamp,  # Will be overwritten
                "metadata": obs.extra_metrics,
            }

            # Add GenAI-specific data if present
            if obs.gen_ai_request_model:
                span_kwargs["model"] = obs.gen_ai_request_model
            if obs.gen_ai_usage_input_tokens is not None:
                span_kwargs["usage"] = {
                    "input": obs.gen_ai_usage_input_tokens,
                    "output": obs.gen_ai_usage_output_tokens or 0,
                    "total": obs.gen_ai_usage_total_tokens or 0,
                }

            if obs.success:
                span_kwargs["level"] = "DEFAULT"
            else:
                span_kwargs["level"] = "ERROR"
                span_kwargs["status_message"] = obs.error_message

            trace.span(**span_kwargs)

        # Flush to ensure data is sent
        client.flush()


@dataclass
class OTLPExporter:
    """Export observations to OpenTelemetry OTLP backend.

    Supports any OTLP-compatible backend: Jaeger, Grafana Tempo, Honeycomb, etc.

    Requires: pip install opentelemetry-sdk opentelemetry-exporter-otlp-proto-grpc
              or: pip install opentelemetry-sdk opentelemetry-exporter-otlp-proto-http

    Example:
        exporter = OTLPExporter(endpoint="http://localhost:4317")
        exporter.export(observations, summary)

        # Or via ObservabilityHub:
        system.observability.export_otlp(endpoint="http://jaeger:4317")

    Environment variables (standard OpenTelemetry):
        OTEL_EXPORTER_OTLP_ENDPOINT: Default endpoint
        OTEL_EXPORTER_OTLP_HEADERS: Default headers (URL-encoded)
        OTEL_EXPORTER_OTLP_PROTOCOL: Default protocol (grpc or http/protobuf)
    """

    endpoint: str = "http://localhost:4317"
    protocol: str = "grpc"  # "grpc" or "http"
    service_name: str = "hyperfunc"
    insecure: bool = True
    headers: Optional[Dict[str, str]] = None

    def __post_init__(self) -> None:
        self._tracer: Any = None
        self._provider: Any = None

    def _setup_tracer(self) -> None:
        """Lazy setup of OpenTelemetry tracer."""
        if self._tracer is not None:
            return

        # Lazy import
        try:
            from opentelemetry import trace
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
            from opentelemetry.sdk.resources import Resource
        except ImportError:
            raise ImportError(
                "opentelemetry packages required. Install with:\n"
                "  pip install opentelemetry-sdk opentelemetry-exporter-otlp-proto-grpc\n"
                "  or: pip install opentelemetry-sdk opentelemetry-exporter-otlp-proto-http"
            )

        # Import protocol-specific exporter
        if self.protocol == "grpc":
            try:
                from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                    OTLPSpanExporter,
                )
            except ImportError:
                raise ImportError(
                    "gRPC exporter not found. Install with:\n"
                    "  pip install opentelemetry-exporter-otlp-proto-grpc"
                )
        else:
            try:
                from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                    OTLPSpanExporter,
                )
            except ImportError:
                raise ImportError(
                    "HTTP exporter not found. Install with:\n"
                    "  pip install opentelemetry-exporter-otlp-proto-http"
                )

        # Setup resource with service name
        resource = Resource.create({"service.name": self.service_name})

        # Setup provider
        self._provider = TracerProvider(resource=resource)

        # Setup exporter with appropriate kwargs
        exporter_kwargs: Dict[str, Any] = {"endpoint": self.endpoint}
        if self.protocol == "grpc":
            exporter_kwargs["insecure"] = self.insecure
        if self.headers:
            exporter_kwargs["headers"] = self.headers

        otlp_exporter = OTLPSpanExporter(**exporter_kwargs)
        self._provider.add_span_processor(BatchSpanProcessor(otlp_exporter))

        self._tracer = self._provider.get_tracer("hyperfunc")

    def export(
        self,
        observations: Sequence[ObservationRecord],
        summary: Optional[TraceSummary] = None,
    ) -> None:
        """Export observations as OTLP spans.

        Each ObservationRecord becomes a span with:
        - GenAI semantic conventions for LLM calls
        - Error status for failed calls
        - Custom metrics as attributes
        """
        self._setup_tracer()

        from opentelemetry.trace import Status, StatusCode, SpanKind
        from datetime import timedelta

        for obs in observations:
            # Parse timestamp and compute start/end times
            start_time = datetime.fromisoformat(obs.timestamp.replace("Z", "+00:00"))
            end_time = start_time + timedelta(seconds=obs.elapsed_s)

            # Convert to nanoseconds for OTel
            start_ns = int(start_time.timestamp() * 1e9)
            end_ns = int(end_time.timestamp() * 1e9)

            # Create span with explicit start time
            span = self._tracer.start_span(
                obs.fn_name,
                kind=SpanKind.INTERNAL,
                start_time=start_ns,
            )

            try:
                # Core attributes
                if obs.example_index is not None:
                    span.set_attribute("hyperfunc.example_index", obs.example_index)

                # GenAI semantic conventions
                if obs.gen_ai_system:
                    span.set_attribute("gen_ai.system", obs.gen_ai_system)
                if obs.gen_ai_request_model:
                    span.set_attribute("gen_ai.request.model", obs.gen_ai_request_model)
                if obs.gen_ai_response_model:
                    span.set_attribute("gen_ai.response.model", obs.gen_ai_response_model)
                if obs.gen_ai_request_temperature is not None:
                    span.set_attribute(
                        "gen_ai.request.temperature", obs.gen_ai_request_temperature
                    )
                if obs.gen_ai_request_top_p is not None:
                    span.set_attribute("gen_ai.request.top_p", obs.gen_ai_request_top_p)
                if obs.gen_ai_usage_input_tokens is not None:
                    span.set_attribute(
                        "gen_ai.usage.input_tokens", obs.gen_ai_usage_input_tokens
                    )
                if obs.gen_ai_usage_output_tokens is not None:
                    span.set_attribute(
                        "gen_ai.usage.output_tokens", obs.gen_ai_usage_output_tokens
                    )
                if obs.gen_ai_usage_total_tokens is not None:
                    span.set_attribute(
                        "gen_ai.usage.total_tokens", obs.gen_ai_usage_total_tokens
                    )

                # Cost tracking
                if obs.cost_usd is not None:
                    span.set_attribute("hyperfunc.cost_usd", obs.cost_usd)

                # Extra metrics
                for key, value in obs.extra_metrics.items():
                    span.set_attribute(f"hyperfunc.{key}", value)

                # Error handling
                if not obs.success:
                    span.set_status(
                        Status(StatusCode.ERROR, obs.error_message or "Unknown error")
                    )
                    if obs.error_type:
                        span.set_attribute("exception.type", obs.error_type)
                    if obs.error_message:
                        span.set_attribute("exception.message", obs.error_message)
            finally:
                # End span with explicit end time
                span.end(end_time=end_ns)

        # Force flush to ensure spans are sent
        if self._provider:
            self._provider.force_flush()

    def shutdown(self) -> None:
        """Shutdown the exporter and flush pending spans."""
        if self._provider:
            self._provider.shutdown()
            self._provider = None
            self._tracer = None


def _compute_percentile(values: List[float], percentile: float) -> float:
    """Compute percentile from sorted values."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    k = (len(sorted_vals) - 1) * percentile / 100
    f = int(k)
    c = f + 1 if f + 1 < len(sorted_vals) else f
    return sorted_vals[f] + (k - f) * (sorted_vals[c] - sorted_vals[f])


class ObservabilityHub:
    """Central hub for observability on a HyperSystem.

    Manages trace sessions, computes statistics, and exports data.

    Usage:
        system = MySystem()

        # Start a trace session
        with system.observability.trace():
            await system.run(...)

        # Get history and summary
        history = system.observability.get_history()
        summary = system.observability.summary()

        # Export
        system.observability.export_json("trace.json")
    """

    def __init__(self, system: "HyperSystem"):
        self._system = system
        self._session_id: Optional[str] = None
        self._session_start: Optional[str] = None
        self._observations: List[ObservationRecord] = []

    @contextmanager
    def trace(self, session_id: Optional[str] = None) -> Iterator[None]:
        """Context manager to enable tracing for a session.

        Args:
            session_id: Optional session identifier (auto-generated if not provided)

        Yields:
            None - tracing is active within the context
        """
        import uuid
        from .core import _CURRENT_SYSTEM

        self._session_id = session_id or str(uuid.uuid4())[:8]
        self._session_start = datetime.now(timezone.utc).isoformat()
        self._observations = []

        # Enable system tracing
        prev_enabled = self._system._trace_enabled
        prev_history = list(self._system._call_history)
        self._system._trace_enabled = True
        self._system._call_history.clear()

        # Set current system context so hyperfunctions auto-register
        import hyperfunc.core as core_module
        prev_system = core_module._CURRENT_SYSTEM
        core_module._CURRENT_SYSTEM = self._system

        try:
            yield
        finally:
            # Capture observations
            for record in self._system._call_history:
                self._observations.append(
                    ObservationRecord.from_call_record(record, self._session_start)
                )

            # Restore previous state
            self._system._trace_enabled = prev_enabled
            self._system._call_history = prev_history
            core_module._CURRENT_SYSTEM = prev_system

    def get_history(self) -> List[ObservationRecord]:
        """Get observation history from the last trace session."""
        return list(self._observations)

    def get_call_records(self) -> List["CallRecord"]:
        """Get raw CallRecords from the system (requires tracing enabled)."""
        return list(self._system._call_history)

    def summary(self) -> TraceSummary:
        """Compute summary statistics from the last trace session."""
        if not self._observations:
            return TraceSummary(
                session_id=self._session_id or "unknown",
                start_time=self._session_start or datetime.now(timezone.utc).isoformat(),
                end_time=datetime.now(timezone.utc).isoformat(),
                total_calls=0,
                total_errors=0,
                total_elapsed_s=0.0,
                by_function={},
            )

        # Group by function
        by_fn: Dict[str, List[ObservationRecord]] = {}
        for obs in self._observations:
            by_fn.setdefault(obs.fn_name, []).append(obs)

        # Compute per-function stats
        fn_stats: Dict[str, HyperFunctionStats] = {}
        total_input_tokens = 0
        total_output_tokens = 0
        total_cost = 0.0
        has_genai = False

        for fn_name, obs_list in by_fn.items():
            elapsed_times = [o.elapsed_s for o in obs_list]
            errors = [o for o in obs_list if not o.success]

            # GenAI aggregates
            fn_input_tokens = sum(
                o.gen_ai_usage_input_tokens or 0 for o in obs_list
            )
            fn_output_tokens = sum(
                o.gen_ai_usage_output_tokens or 0 for o in obs_list
            )
            fn_cost = sum(o.cost_usd or 0.0 for o in obs_list)

            if any(o.gen_ai_request_model for o in obs_list):
                has_genai = True
                total_input_tokens += fn_input_tokens
                total_output_tokens += fn_output_tokens
                total_cost += fn_cost

            fn_stats[fn_name] = HyperFunctionStats(
                fn_name=fn_name,
                call_count=len(obs_list),
                success_count=len(obs_list) - len(errors),
                error_count=len(errors),
                total_elapsed_s=sum(elapsed_times),
                min_elapsed_s=min(elapsed_times),
                max_elapsed_s=max(elapsed_times),
                mean_elapsed_s=statistics.mean(elapsed_times),
                p50_elapsed_s=_compute_percentile(elapsed_times, 50),
                p95_elapsed_s=_compute_percentile(elapsed_times, 95),
                p99_elapsed_s=_compute_percentile(elapsed_times, 99),
                error_rate=len(errors) / len(obs_list) if obs_list else 0.0,
                total_input_tokens=fn_input_tokens if fn_input_tokens else None,
                total_output_tokens=fn_output_tokens if fn_output_tokens else None,
                total_cost_usd=fn_cost if fn_cost else None,
            )

        total_errors = sum(1 for o in self._observations if not o.success)
        total_elapsed = sum(o.elapsed_s for o in self._observations)

        return TraceSummary(
            session_id=self._session_id or "unknown",
            start_time=self._session_start or self._observations[0].timestamp,
            end_time=datetime.now(timezone.utc).isoformat(),
            total_calls=len(self._observations),
            total_errors=total_errors,
            total_elapsed_s=total_elapsed,
            by_function=fn_stats,
            total_input_tokens=total_input_tokens if has_genai else None,
            total_output_tokens=total_output_tokens if has_genai else None,
            total_cost_usd=total_cost if has_genai and total_cost > 0 else None,
        )

    def export_json(
        self,
        path: Union[str, Path],
        jsonl: bool = False,
        include_summary: bool = True,
    ) -> None:
        """Export to JSON file.

        Args:
            path: Output file path
            jsonl: If True, write JSONL format
            include_summary: Include summary in output
        """
        exporter = JSONExporter(path, jsonl=jsonl, include_summary=include_summary)
        exporter.export(self._observations, self.summary() if include_summary else None)

    def export_langfuse(
        self,
        trace_name: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Export to LangFuse.

        Args:
            trace_name: Name for the trace
            user_id: User ID for attribution
            session_id: Session ID (defaults to current session)
            metadata: Additional metadata
            **kwargs: Additional LangFuse client options
        """
        exporter = LangFuseExporter(
            trace_name=trace_name,
            user_id=user_id,
            session_id=session_id or self._session_id,
            metadata=metadata,
            **kwargs,
        )
        exporter.export(self._observations, self.summary())

    def export_otlp(
        self,
        endpoint: str = "http://localhost:4317",
        protocol: str = "grpc",
        service_name: str = "hyperfunc",
        insecure: bool = True,
        headers: Optional[Dict[str, str]] = None,
    ) -> None:
        """Export to OTLP backend (Jaeger, Grafana Tempo, Honeycomb, etc.).

        Args:
            endpoint: OTLP endpoint (default: localhost:4317 for gRPC)
            protocol: "grpc" (default, port 4317) or "http" (port 4318)
            service_name: Service name for traces
            insecure: Use insecure connection (gRPC only)
            headers: Optional headers for authentication
        """
        exporter = OTLPExporter(
            endpoint=endpoint,
            protocol=protocol,
            service_name=service_name,
            insecure=insecure,
            headers=headers,
        )
        exporter.export(self._observations, self.summary())

    def clear(self) -> None:
        """Clear observation history."""
        self._observations = []
        self._session_id = None
        self._session_start = None
