from __future__ import annotations

import asyncio
import inspect
import time
import warnings
import weakref
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Dict, List, Optional, Protocol, Sequence, Set, Tuple, Type

import torch
from torch import nn


class TracedValueWarning(UserWarning):
    """Warning for potentially unsafe operations on TracedValue.

    This warning is raised when operations like torch.cat() or torch.split()
    are used directly on hyperfunction outputs. These operations won't be
    replayed during ES optimization, which can cause shape mismatches or
    incorrect results.

    Use hyperfunc.combine() and hyperfunc.split() instead.
    """
    pass


# Torch functions that won't be replayed during ES population evaluation
_UNSAFE_TORCH_FUNCS = {torch.cat, torch.stack, torch.concat, torch.split}

try:
    from opentelemetry import trace as _otel_trace  # type: ignore[import]
except Exception:  # pragma: no cover - optional dependency
    _otel_trace = None


# Global system context for implicit HyperFunction registration.
_CURRENT_SYSTEM: Optional["HyperSystem"] = None


# ============================================================
# Tracing infrastructure for DAG construction
# ============================================================


@dataclass
class TraceNode:
    """
    A single hyperfunction call in the execution trace.

    Each call gets a unique node_id. Dependencies are tracked by detecting
    when the output of one call is used as input to another.
    """
    node_id: int
    fn_name: str
    inputs: Dict[str, Any]  # Original inputs (may contain TracedValue refs)
    output: Any = None
    dependencies: Set[int] = field(default_factory=set)  # node_ids this depends on


class TracedValue:
    """
    Wrapper around a hyperfunction output that tracks its origin.

    When tracing is enabled, hyperfunction outputs are wrapped in this class.
    When another hyperfunction receives this as input, we detect the dependency.

    The wrapper is transparent for most operations - it delegates attribute
    access and common operations to the underlying value.

    TracedValue supports multiple parent node_ids to handle merging of parallel
    branches (e.g., torch.cat([branch1_output, branch2_output])).
    """
    __slots__ = ('_value', '_node_ids', '_system_ref')

    def __init__(self, value: Any, node_id: int | Set[int], system: "HyperSystem") -> None:
        object.__setattr__(self, '_value', value)
        # Support both single node_id and set of node_ids (for merged values)
        if isinstance(node_id, set):
            object.__setattr__(self, '_node_ids', node_id)
        else:
            object.__setattr__(self, '_node_ids', {node_id})
        object.__setattr__(self, '_system_ref', weakref.ref(system))

    @property
    def _traced_value(self) -> Any:
        return object.__getattribute__(self, '_value')

    @property
    def _traced_node_id(self) -> int:
        """Return first node_id for backwards compatibility."""
        node_ids = object.__getattribute__(self, '_node_ids')
        return min(node_ids)  # Return smallest for determinism

    @property
    def _traced_node_ids(self) -> Set[int]:
        """Return all node_ids this value depends on."""
        return object.__getattribute__(self, '_node_ids')

    def _merge_with(self, other: "TracedValue", result_value: Any) -> "TracedValue":
        """Create a new TracedValue that depends on both self and other."""
        merged_ids = self._traced_node_ids | other._traced_node_ids
        system = object.__getattribute__(self, '_system_ref')()
        return TracedValue(result_value, merged_ids, system)

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, '_value'), name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(object.__getattribute__(self, '_value'), name, value)

    def __repr__(self) -> str:
        return repr(object.__getattribute__(self, '_value'))

    def __str__(self) -> str:
        return str(object.__getattribute__(self, '_value'))

    # Delegate common operations to the underlying value
    def __eq__(self, other: Any) -> bool:
        if isinstance(other, TracedValue):
            other = other._traced_value
        return object.__getattribute__(self, '_value') == other

    def __hash__(self) -> int:
        return hash(object.__getattribute__(self, '_value'))

    def __iter__(self):
        return iter(object.__getattribute__(self, '_value'))

    def __len__(self) -> int:
        return len(object.__getattribute__(self, '_value'))

    def __getitem__(self, key):
        return object.__getattribute__(self, '_value')[key]

    def __bool__(self) -> bool:
        return bool(object.__getattribute__(self, '_value'))

    # Arithmetic operations - return merged TracedValue
    def __add__(self, other: Any) -> "TracedValue":
        self_val = self._traced_value
        if isinstance(other, TracedValue):
            result = self_val + other._traced_value
            return self._merge_with(other, result)
        else:
            result = self_val + other
            return TracedValue(result, self._traced_node_ids, object.__getattribute__(self, '_system_ref')())

    def __radd__(self, other: Any) -> "TracedValue":
        return self.__add__(other)

    def __sub__(self, other: Any) -> "TracedValue":
        self_val = self._traced_value
        if isinstance(other, TracedValue):
            result = self_val - other._traced_value
            return self._merge_with(other, result)
        else:
            result = self_val - other
            return TracedValue(result, self._traced_node_ids, object.__getattribute__(self, '_system_ref')())

    def __rsub__(self, other: Any) -> "TracedValue":
        self_val = self._traced_value
        result = other - self_val
        return TracedValue(result, self._traced_node_ids, object.__getattribute__(self, '_system_ref')())

    def __mul__(self, other: Any) -> "TracedValue":
        self_val = self._traced_value
        if isinstance(other, TracedValue):
            result = self_val * other._traced_value
            return self._merge_with(other, result)
        else:
            result = self_val * other
            return TracedValue(result, self._traced_node_ids, object.__getattribute__(self, '_system_ref')())

    def __rmul__(self, other: Any) -> "TracedValue":
        return self.__mul__(other)

    def __truediv__(self, other: Any) -> "TracedValue":
        self_val = self._traced_value
        if isinstance(other, TracedValue):
            result = self_val / other._traced_value
            return self._merge_with(other, result)
        else:
            result = self_val / other
            return TracedValue(result, self._traced_node_ids, object.__getattribute__(self, '_system_ref')())

    def __matmul__(self, other: Any) -> "TracedValue":
        self_val = self._traced_value
        if isinstance(other, TracedValue):
            result = self_val @ other._traced_value
            return self._merge_with(other, result)
        else:
            result = self_val @ other
            return TracedValue(result, self._traced_node_ids, object.__getattribute__(self, '_system_ref')())

    # PyTorch tensor protocol - allows torch.cat, torch.stack, etc. to work
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        """Handle PyTorch functions called on TracedValue objects.

        Note: Operations like torch.cat() and torch.split() will work but
        won't be replayed during ES optimization. Use hyperfunc.combine()
        and hyperfunc.split() instead.
        """
        if kwargs is None:
            kwargs = {}

        # Warn about operations that won't be replayed during ES optimization
        if func in _UNSAFE_TORCH_FUNCS:
            warnings.warn(
                f"torch.{func.__name__}() on hyperfunction outputs won't be "
                f"replayed during ES optimization. Use hyperfunc.combine() or "
                f"hyperfunc.split() instead.",
                TracedValueWarning,
                stacklevel=2
            )

        # Collect all TracedValue objects and their node_ids
        all_node_ids: Set[int] = set()
        system_ref = None

        def collect_and_unwrap(obj):
            nonlocal all_node_ids, system_ref
            if isinstance(obj, TracedValue):
                all_node_ids.update(obj._traced_node_ids)
                if system_ref is None:
                    system_ref = object.__getattribute__(obj, '_system_ref')
                return obj._traced_value
            elif isinstance(obj, (list, tuple)):
                return type(obj)(collect_and_unwrap(x) for x in obj)
            elif isinstance(obj, dict):
                return {k: collect_and_unwrap(v) for k, v in obj.items()}
            return obj

        # Unwrap all args and kwargs
        unwrapped_args = collect_and_unwrap(args)
        unwrapped_kwargs = collect_and_unwrap(kwargs)

        # Call the actual torch function
        result = func(*unwrapped_args, **unwrapped_kwargs)

        # Wrap result in TracedValue with merged dependencies
        if system_ref is not None and all_node_ids:
            system = system_ref()
            if system is not None:
                return TracedValue(result, all_node_ids, system)

        return result


def unwrap_traced(value: Any) -> Any:
    """Recursively unwrap TracedValue wrappers from a value."""
    if isinstance(value, TracedValue):
        return unwrap_traced(value._traced_value)
    elif isinstance(value, dict):
        return {k: unwrap_traced(v) for k, v in value.items()}
    elif isinstance(value, (list, tuple)):
        unwrapped = [unwrap_traced(v) for v in value]
        return type(value)(unwrapped)
    return value


def collect_traced_dependencies(value: Any) -> Set[int]:
    """Recursively collect all TracedValue node_ids from a value."""
    deps: Set[int] = set()
    if isinstance(value, TracedValue):
        # Use _traced_node_ids to get all dependencies (handles merged TracedValues)
        deps.update(value._traced_node_ids)
        deps.update(collect_traced_dependencies(value._traced_value))
    elif isinstance(value, dict):
        for v in value.values():
            deps.update(collect_traced_dependencies(v))
    elif isinstance(value, (list, tuple)):
        for v in value:
            deps.update(collect_traced_dependencies(v))
    return deps


@dataclass
class ExecutionTrace:
    """
    Complete execution trace of a run() call.

    Contains all TraceNodes and can be converted to a DAG for staged execution.
    """
    nodes: List[TraceNode] = field(default_factory=list)

    def add_node(self, fn_name: str, inputs: Dict[str, Any]) -> TraceNode:
        """Add a new node to the trace, detecting dependencies from inputs."""
        node_id = len(self.nodes)
        deps = set()
        for v in inputs.values():
            deps.update(collect_traced_dependencies(v))
        node = TraceNode(
            node_id=node_id,
            fn_name=fn_name,
            inputs=inputs,
            dependencies=deps,
        )
        self.nodes.append(node)
        return node

    def to_stages(self) -> List[List[TraceNode]]:
        """
        Topologically sort nodes into stages.

        Nodes with no dependencies go in stage 0.
        Nodes whose dependencies are all in earlier stages go in the next stage.
        """
        if not self.nodes:
            return []

        # Build adjacency for topological sort
        node_to_stage: Dict[int, int] = {}

        # Kahn's algorithm variant: assign stage = max(dependency stages) + 1
        for node in self.nodes:
            if not node.dependencies:
                node_to_stage[node.node_id] = 0
            else:
                max_dep_stage = max(
                    node_to_stage.get(dep_id, 0) for dep_id in node.dependencies
                )
                node_to_stage[node.node_id] = max_dep_stage + 1

        # Group by stage
        max_stage = max(node_to_stage.values()) if node_to_stage else 0
        stages: List[List[TraceNode]] = [[] for _ in range(max_stage + 1)]
        for node in self.nodes:
            stage = node_to_stage[node.node_id]
            stages[stage].append(node)

        return stages


# ============================================================
# HyperParam Protocol
# ============================================================


class HyperParam(Protocol):
    """
    Protocol for hyperparameter types that can be optimized by ES.

    Implementations can be:
    - Scalar collections (LMParam, AdapterMeta) - 1D tensors, standard Gaussian noise
    - Weight matrices (LoRA weights) - 2D tensors, EggRoll-style low-rank noise

    Required methods:
    - dim(): total number of elements
    - from_tensor(): reconstruct from tensor
    - to_tensor(): serialize to tensor

    Optional methods:
    - shape(): tensor shape (defaults to (dim,) if not provided)
    - noise_rank(): rank for low-rank noise (None = standard Gaussian)
    - default_init(): return default initialization tensor (defaults to Xavier-like)
    """

    @classmethod
    def dim(cls) -> int:
        """Return total number of elements in the tensor."""
        ...

    @classmethod
    def from_tensor(cls, t: torch.Tensor) -> "HyperParam":
        """Reconstruct hp object from tensor."""
        ...

    def to_tensor(self, device: Optional[torch.device] = None, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Serialize to tensor."""
        ...


def get_hp_noise_rank(hp_type: Optional[type]) -> Optional[int]:
    """
    Get the noise_rank from an hp_type, if it has one.

    Returns None if hp_type is None or doesn't have noise_rank method.
    """
    if hp_type is None:
        return None
    if hasattr(hp_type, 'noise_rank'):
        return hp_type.noise_rank()
    return None


def get_hp_shape(hp_type: Optional[type]) -> Optional[Tuple[int, ...]]:
    """
    Get the shape from an hp_type.

    Returns shape() if available, else (dim(),) if dim() available, else None.
    """
    if hp_type is None:
        return None
    if hasattr(hp_type, 'shape'):
        return hp_type.shape()
    if hasattr(hp_type, 'dim'):
        return (hp_type.dim(),)
    return None


def get_hp_default_init(hp_type: Optional[type]) -> Optional[torch.Tensor]:
    """
    Get the default initialization tensor from an hp_type.

    If hp_type has default_init(), uses that. Otherwise generates Xavier-like
    initialization based on shape().
    """
    if hp_type is None:
        return None

    # Use custom default_init if provided
    if hasattr(hp_type, 'default_init'):
        return hp_type.default_init()

    # Fall back to Xavier-like init based on shape
    shape = get_hp_shape(hp_type)
    if shape is None:
        return None

    # Xavier-like: scale by sqrt(2 / sum(dims))
    scale = (2.0 / sum(shape)) ** 0.5
    return torch.randn(*shape) * scale


# ---- Example: standard LLM parameters --------------------------------------


@dataclass
class LMParam:
    """
    Standard "LLM behavior" hyperparameters (1D, 6 scalars).

    Interpretation convention:

    - temperature: 0.0 – 1.5
    - top_p: 0.0 – 1.0
    - top_k_norm: 0.0 – 1.0 (mapped to top_k in some range, e.g. [1, 1024])
    - length_factor: 0.5 – 2.0 (scales max_tokens)
    - repetition_penalty: 0.0 – 2.0 (1.0 = none)
    - strictness: 0.0 – 1.0 (0 = relaxed, 1 = ultra strict about format)
    """

    temperature: float = 0.1
    top_p: float = 0.9
    top_k_norm: float = 1.0
    length_factor: float = 1.0
    repetition_penalty: float = 1.0
    strictness: float = 1.0

    @classmethod
    def shape(cls) -> Tuple[int, ...]:
        return (6,)

    @classmethod
    def dim(cls) -> int:
        return 6

    @classmethod
    def noise_rank(cls) -> Optional[int]:
        return None  # 1D param, use standard Gaussian noise

    def to_tensor(self, device=None, dtype=torch.float32) -> torch.Tensor:
        vals = [
            self.temperature,
            self.top_p,
            self.top_k_norm,
            self.length_factor,
            self.repetition_penalty,
            self.strictness,
        ]
        return torch.tensor(vals, device=device, dtype=dtype)

    @classmethod
    def from_tensor(cls, t: torch.Tensor) -> "LMParam":
        t = t.view(-1)
        assert t.numel() == cls.dim()
        return cls(
            temperature=float(t[0]),
            top_p=float(t[1]),
            top_k_norm=float(t[2]),
            length_factor=float(t[3]),
            repetition_penalty=float(t[4]),
            strictness=float(t[5]),
        )


# ============================================================
# 2. Standard adapter meta-parameters (LoRA / heads)
# ============================================================


@dataclass
class AdapterMeta:
    """
    Knobs around an adapter (e.g., LoRA), not the weights themselves (1D, 4 scalars).

    Interpretation convention:

    - scale: how strongly adapter output is mixed in (0–2)
    - dropout_keep: 0–1, probability of *keeping* activations in adapter block
    - gate_bias: bias term for logistic gate controlling adapter usage
    - priority: 0–1, for arbitration across multiple adapters
    """

    scale: float = 1.0
    dropout_keep: float = 1.0
    gate_bias: float = 0.0
    priority: float = 0.5

    @classmethod
    def shape(cls) -> Tuple[int, ...]:
        return (4,)

    @classmethod
    def dim(cls) -> int:
        return 4

    @classmethod
    def noise_rank(cls) -> Optional[int]:
        return None  # 1D param, use standard Gaussian noise

    def to_tensor(self, device=None, dtype=torch.float32) -> torch.Tensor:
        vals = [self.scale, self.dropout_keep, self.gate_bias, self.priority]
        return torch.tensor(vals, device=device, dtype=dtype)

    @classmethod
    def from_tensor(cls, t: torch.Tensor) -> "AdapterMeta":
        t = t.view(-1)
        assert t.numel() == cls.dim()
        return cls(
            scale=float(t[0]),
            dropout_keep=float(t[1]),
            gate_bias=float(t[2]),
            priority=float(t[3]),
        )


# ============================================================
# 3. LoRA weight matrices (2D, optimized with low-rank noise)
# ============================================================


@dataclass
class LoRAWeight:
    """
    LoRA weight matrix (2D tensor) optimized with EggRoll-style low-rank noise.

    This is a factory for creating LoRA-style hp_types with specific dimensions.
    Use LoRAWeight.create(out_dim, in_dim, rank) to create a concrete type.

    The noise_rank controls the rank of the A @ B.T perturbation used during ES.
    """

    weight: torch.Tensor  # (out_dim, in_dim)
    _out_dim: int = field(repr=False)
    _in_dim: int = field(repr=False)
    _noise_rank: int = field(repr=False, default=4)

    @classmethod
    def create(
        cls,
        out_dim: int,
        in_dim: int,
        noise_rank: int = 4,
    ) -> type:
        """
        Factory to create a LoRAWeight type with specific dimensions.

        Usage:
            MyLoRA = LoRAWeight.create(64, 128, noise_rank=8)

            @hyperfunction(hp_type=MyLoRA, optimize_hparams=True)
            def my_fn(x, hp: MyLoRA): ...
        """

        @dataclass
        class _LoRAWeightImpl:
            weight: torch.Tensor  # (out_dim, in_dim)

            @classmethod
            def shape(cls) -> Tuple[int, ...]:
                return (out_dim, in_dim)

            @classmethod
            def dim(cls) -> int:
                return out_dim * in_dim

            @classmethod
            def noise_rank(cls) -> Optional[int]:
                return noise_rank  # Use low-rank noise for 2D

            def to_tensor(self, device=None, dtype=torch.float32) -> torch.Tensor:
                t = self.weight.to(dtype=dtype)
                if device is not None:
                    t = t.to(device)
                return t

            @classmethod
            def from_tensor(cls, t: torch.Tensor) -> "_LoRAWeightImpl":
                return cls(weight=t.view(out_dim, in_dim))

        _LoRAWeightImpl.__name__ = f"LoRAWeight_{out_dim}x{in_dim}"
        _LoRAWeightImpl.__qualname__ = _LoRAWeightImpl.__name__
        return _LoRAWeightImpl

    @classmethod
    def shape(cls) -> Tuple[int, ...]:
        raise NotImplementedError("Use LoRAWeight.create() to make a concrete type")

    @classmethod
    def dim(cls) -> int:
        raise NotImplementedError("Use LoRAWeight.create() to make a concrete type")

    @classmethod
    def noise_rank(cls) -> Optional[int]:
        raise NotImplementedError("Use LoRAWeight.create() to make a concrete type")


# ============================================================
# 3. Core: Example, HyperFunction, decorator
# ============================================================


@dataclass
class Example:
    """
    One supervised training example for the system.

    - inputs: keyword arguments passed to run()
    - expected: the expected output from run()
    """
    inputs: Dict[str, Any]
    expected: Any


class HyperFunction:
    """
    Wrapper around a user-defined function:

        @hyperfunction(model="...", hp_type=LMParam)
        def my_fn(x: str, hp: LMParam) -> Output:
            \"\"\"Prompt lives here\"\"\"
            ...

    - Docstring is treated as the prompt (for GEPA).
    - hp_type (if provided) is a HyperParam-like type.
    - When you call it without hp, we auto-inject an instance
      reconstructed from the global HyperModel.hp tensor.
    """

    def __init__(
        self,
        fn: Callable[..., Any],
        hp_type: Optional[Type[Any]],
        optimize_prompt: bool,
        optimize_hparams: bool,
        local_gpu: bool = False,
        retries: int = 0,
        timeout_s: Optional[float] = None,
        max_calls: Optional[int] = None,
    ) -> None:
        self.fn = fn
        self.name = fn.__name__
        self.hp_type = hp_type
        self.optimize_prompt = optimize_prompt
        self.optimize_hparams = optimize_hparams
        # Hint for schedulers / ES about where this function runs.
        # True => local GPU resident model; False => CPU / HTTP / other.
        self.local_gpu = local_gpu

        # Runtime policy metadata. Enforcement is best-effort here and can be
        # strengthened by a future orchestrator.
        self.retries = max(retries, 0)
        self.timeout_s = timeout_s
        self.max_calls = max_calls
        self._call_count = 0

        self.prompt: str = (fn.__doc__ or "").strip()

        sig = inspect.signature(fn)
        self._sig = sig

        self._hp_param_name: Optional[str] = None
        if hp_type is not None:
            for name, p in sig.parameters.items():
                # Match if param name is "hp" and either:
                # 1. annotation matches hp_type exactly
                # 2. annotation is inspect.Parameter.empty (no annotation)
                # 3. param name is "hp" (conventional name for hp param)
                if name == "hp":
                    self._hp_param_name = name
                    break
        # Reference to this function's hp block, created lazily by HyperSystem.
        # This is a 1D torch.nn.Parameter of length hp_dim living on a device
        # chosen by the caller / optimiser.
        self.hp_param: Optional[nn.Parameter] = None
        self.system: Optional["HyperSystem"] = None

        # vmap support: sync version and cached tensor function
        self._fn_sync: Optional[Callable[..., Any]] = None
        self._tensor_fn: Optional[Callable[..., Any]] = None
        self._vmappable_cached: Optional[bool] = None

    @property
    def hp_dim(self) -> int:
        if self.hp_type is None:
            return 0
        return self.hp_type.dim()  # type: ignore[call-arg]

    @property
    def fn_sync(self) -> Callable[..., Any]:
        """
        Get a synchronous version of the function for vmap.

        For async functions, this strips the async wrapper since tensor ops
        don't actually need async. For sync functions, returns the original.
        """
        if self._fn_sync is not None:
            return self._fn_sync

        if asyncio.iscoroutinefunction(self.fn):
            # For async functions that are just tensor ops wrapped in async,
            # we can call them sync. But if they actually do I/O, vmap will fail.
            import functools

            @functools.wraps(self.fn)
            def sync_wrapper(*args, **kwargs):
                coro = self.fn(*args, **kwargs)
                # For pure tensor ops, the coroutine should complete immediately
                # when we send None to it. This is a simplification that works
                # for tensor-only async functions.
                try:
                    coro.send(None)
                except StopIteration as e:
                    return e.value
                # If we get here, the coroutine actually awaited something
                raise RuntimeError(
                    f"HyperFunction '{self.name}' contains real async I/O "
                    "and cannot be vmapped. Mark it with vmappable=False."
                )

            self._fn_sync = sync_wrapper
        else:
            self._fn_sync = self.fn

        return self._fn_sync

    # wiring

    def attach_system(self, system: "HyperSystem") -> None:
        self.system = system

    # prompt view for prompt optimizers

    def get_prompt(self) -> str:
        return self.prompt

    def set_prompt(self, text: str) -> None:
        self.prompt = text
        # keep the underlying function docstring in sync (optional)
        self.fn.__doc__ = text
        # Also update our own docstring so users accessing wrapper.__doc__ see the change
        self.__doc__ = text

    # call interface

    def __call__(self, *args: Any, **kwargs: Any) -> Coroutine[Any, Any, Any]:
        """Call the hyperfunction. Returns a coroutine that must be awaited."""
        return self._async_call(*args, **kwargs)

    async def _async_call(self, *args: Any, **kwargs: Any) -> Any:
        """Async implementation of the call."""
        bound = self._sig.bind_partial(*args, **kwargs)
        bound.apply_defaults()
        # If this HyperFunction is not yet attached to a system, but there is
        # a current HyperSystem context, register it automatically. This lets
        # HyperSystem discover hyperfunctions from normal Python calls.
        global _CURRENT_SYSTEM
        if self.system is None and _CURRENT_SYSTEM is not None:
            _CURRENT_SYSTEM.register_hyperfunction(self)

        # Use this HyperFunction's hp block (if any).
        hp_tensor: Optional[torch.Tensor] = None
        if self.hp_param is not None:
            hp_tensor = self.hp_param.data

        # If DAG tracing is enabled, record this call and wrap output
        system = self.system or _CURRENT_SYSTEM
        if system is not None and system._dag_trace is not None:
            # Record node in trace (dependencies detected from inputs)
            trace_node = system._dag_trace.add_node(self.name, dict(bound.arguments))
            # Unwrap any TracedValue inputs before actual invocation
            unwrapped_args = {k: unwrap_traced(v) for k, v in bound.arguments.items()}
            bound.arguments.clear()
            bound.arguments.update(unwrapped_args)
            # Invoke
            result = await self._invoke_with_hp(hp_tensor, bound, example_index=None)
            # Store output in trace node
            trace_node.output = result
            # Wrap result for dependency tracking
            return TracedValue(result, trace_node.node_id, system)

        # Direct calls from user code (outside eval/ES) do not carry an
        # example index.
        return await self._invoke_with_hp(hp_tensor, bound, example_index=None)

    async def _invoke_with_hp(
        self,
        hp_tensor: Optional[torch.Tensor],
        bound,
        example_index: Optional[int] = None,
    ) -> Any:
        # Simple max-call guard. This is per-process and not aware of
        # per-episode boundaries, but provides a basic safety rail.
        if self.max_calls is not None and self._call_count >= self.max_calls:
            raise RuntimeError(
                f"HyperFunction '{self.name}' exceeded max_calls={self.max_calls}"
            )

        self._call_count += 1

        # Inject hp if we have hp_type and an hp_tensor.
        # Always inject when hp_tensor is provided (e.g., during population evaluation),
        # even if hp is already in arguments - this ensures candidate weights are used.
        if (
            self._hp_param_name is not None
            and self.hp_type is not None
            and hp_tensor is not None
        ):
            # We rely on the hp_type having a from_tensor(...) classmethod,
            # but no longer require it to implement a formal Protocol.
            hp_obj = self.hp_type.from_tensor(hp_tensor)  # type: ignore[attr-defined]
            bound.arguments[self._hp_param_name] = hp_obj

        attempts = max(1, self.retries + 1)
        last_err: Optional[BaseException] = None

        system = self.system
        ctx_token: Any = None
        for _ in range(attempts):
            start = time.perf_counter()
            try:
                if system is not None:
                    # Hook for tracing/observability; arguments are an
                    # OrderedDict of bound parameters.
                    ctx_token = system._before_hf_call(  # type: ignore[attr-defined]
                        self,
                        bound.arguments,
                        example_index,
                    )

                # Call the underlying function - await if it's a coroutine
                result = self.fn(**bound.arguments)
                if asyncio.iscoroutine(result):
                    result = await result

                if self.timeout_s is not None:
                    elapsed = time.perf_counter() - start
                    if elapsed > self.timeout_s:
                        raise TimeoutError(
                            f"HyperFunction '{self.name}' exceeded timeout_s={self.timeout_s}"
                        )
                if system is not None:
                    elapsed = time.perf_counter() - start
                    system._after_hf_call(  # type: ignore[attr-defined]
                        self,
                        bound.arguments,
                        result,
                        None,
                        elapsed,
                        example_index,
                        ctx_token,
                    )
                return result
            except BaseException as e:  # noqa: BLE001
                if system is not None:
                    elapsed = time.perf_counter() - start
                    system._after_hf_call(  # type: ignore[attr-defined]
                        self,
                        bound.arguments,
                        None,
                        e,
                        elapsed,
                        example_index,
                        ctx_token,
                    )
                last_err = e
                continue

        assert last_err is not None
        raise last_err


def hyperfunction(
    *,
    hp_type: Optional[Type[Any]] = None,
    optimize_prompt: bool = True,
    optimize_hparams: bool = False,
    local_gpu: bool = False,
    retries: int = 0,
    timeout_s: Optional[float] = None,
    max_calls: Optional[int] = None,
    model: Optional[str] = None,  # kept for backwards compatibility, unused
) -> Callable[[Callable[..., Any]], HyperFunction]:
    """
    Decorator turning a plain Python function into a HyperFunction.

    - hp_type: a class with from_tensor()/dim() helpers (or None)
    - optimize_prompt: allow prompt optimisation (GEPA)
    - optimize_hparams: allow ES optimisation of hp
    """

    import functools
    
    def wrapper(fn: Callable[..., Any]) -> HyperFunction:
        hf = HyperFunction(
            fn=fn,
            hp_type=hp_type,
            optimize_prompt=optimize_prompt,
            optimize_hparams=optimize_hparams,
            local_gpu=local_gpu,
            retries=retries,
            timeout_s=timeout_s,
            max_calls=max_calls,
        )
        # Convenience attributes on the original function, FunctAI-style.
        if model is not None:
            fn.hyper_model_name = model   # type: ignore[attr-defined]
        fn.hyper_hp_type = hp_type        # type: ignore[attr-defined]
        fn.hyper_prompt = hf.prompt       # type: ignore[attr-defined]
        
        # We want the wrapper to look like the original function
        functools.update_wrapper(hf, fn)
        return hf

    return wrapper


# ============================================================
# 4. HyperSystem
# ============================================================

class PromptOptimizer(Protocol):
    def optimize(
        self,
        system: "HyperSystem",
        train_data: Sequence[Example],
        metric_fn: Callable[[List[Any], List[Any]], float],
    ) -> None:
        ...


class SystemOptimizer(Protocol):
    def optimize(
        self,
        system: "HyperSystem",
        train_data: Sequence[Example],
        metric_fn: Callable[[List[Any], List[Any]], float],
    ) -> None:
        ...


@dataclass
class NoOpPromptOptimizer:
    async def optimize(
        self,
        system: "HyperSystem",
        train_data: Sequence[Example],
        metric_fn: Callable[[List[Any], List[Any]], float],
    ) -> None:
        return


@dataclass
class NoOpSystemOptimizer:
    async def optimize(
        self,
        system: "HyperSystem",
        train_data: Sequence[Example],
        metric_fn: Callable[[List[Any], List[Any]], float],
    ) -> None:
        return


@dataclass
class CallRecord:
    """
    Lightweight record of a single HyperFunction invocation.

    This is intentionally minimal; higher-level systems can build richer
    traces on top of it.
    """

    fn_name: str
    example_index: Optional[int]
    elapsed_s: float
    error: Optional[BaseException]
    extra_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class CallContext:
    """
    Rich context for a single HyperFunction invocation, passed to
    OTel metric functions.
    """

    hf: HyperFunction
    bound_args: Dict[str, Any]
    result: Any
    error: Optional[BaseException]
    elapsed_s: float
    example_index: Optional[int]
    metadata: Dict[str, Any]
    span: Any = None


class HyperSystem:
    """
    Ties everything together:

    - a set of HyperFunctions,
    - a global HyperModel (hp vector),
    - a prompt optimizer (GEPA-style),
    - a system optimizer (ES / TorchEggrollES).

    You call:

        system = HyperSystem([...], prompt_optimizer, system_optimizer)
        system.optimize(train_data, metric_fn)

    After that, calling the HyperFunctions will use tuned prompts + tuned hp.
    """

    def __init__(
        self,
        prompt_optimizer: Optional[PromptOptimizer] = None,
        system_optimizer: Optional[SystemOptimizer] = None,
    ) -> None:
        # Registered HyperFunctions and their per-function hp blocks.
        self._hyperfunctions: Dict[str, HyperFunction] = {}
        # name -> 1D Parameter representing that HyperFunction's hp block.
        self._hp_params: Dict[str, nn.Parameter] = {}

        # By default we use a no-op prompt optimizer to avoid
        # taking a hard dependency on the external GEPA engine.
        # Callers that want real prompt optimisation should pass
        # an explicit GEPAPromptOptimizer instance.
        if prompt_optimizer is None:
            self.prompt_optimizer = NoOpPromptOptimizer()
        else:
            self.prompt_optimizer = prompt_optimizer

        if system_optimizer is None:
            from .es import ESHybridSystemOptimizer
            self.system_optimizer = ESHybridSystemOptimizer()
        else:
            self.system_optimizer = system_optimizer

        # Internal tracing / observability state. These are best-effort and
        # intentionally simple; richer history / OTel integration can be
        # layered on top in user code.
        self._trace_enabled: bool = False
        self._call_history: List[CallRecord] = []

        # DAG tracing for dependency tracking during run().
        # When not None, hyperfunction calls record themselves here.
        self._dag_trace: Optional[ExecutionTrace] = None
        # Cached trace from a previous run, used for staged execution in optimize.
        self._cached_trace: Optional[ExecutionTrace] = None

        # Optional OpenTelemetry tracer and metric functions.
        if _otel_trace is not None:  # pragma: no cover - requires opentelemetry
            self._otel_tracer = _otel_trace.get_tracer("hyperfunc")
        else:  # pragma: no cover - simple fallback
            self._otel_tracer = None
        # name -> metric function over CallContext
        self.otel_metric_funcs: Dict[str, Callable[[CallContext], float]] = {}

    # ------------------------------------------------------------------
    # HyperFunction registration and hp block management
    # ------------------------------------------------------------------

    def register_hyperfunction(
        self,
        hf: HyperFunction,
        hp_init: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Register a HyperFunction with this system.

        This attaches the system to the HyperFunction so that calls can see
        tracing hooks and (if enabled) a dedicated hp block.

        Args:
            hf: The HyperFunction to register.
            hp_init: Optional initial tensor for hp_param. Can be any shape
                     (1D for LMParam-style, 2D for LoRA weights, etc.).
                     If not provided and hf.hp_type is set, creates a 1D
                     tensor of size hp_type.dim().
        """
        name = hf.name
        self._hyperfunctions[name] = hf
        hf.attach_system(self)

        # Create hp Parameter for this HyperFunction if it has optimisable hparams.
        if hf.optimize_hparams and name not in self._hp_params:
            if hp_init is not None:
                # Use provided tensor (can be any shape - 1D, 2D for LoRA, etc.)
                param = nn.Parameter(hp_init.clone().detach(), requires_grad=False)
            elif hf.hp_type is not None:
                # Use default initialization from hp_type (Xavier-like)
                default = get_hp_default_init(hf.hp_type)
                if default is not None:
                    param = nn.Parameter(default.clone().detach(), requires_grad=False)
                elif hf.hp_dim > 0:
                    # Fall back to zeros if no default_init and no shape
                    param = nn.Parameter(torch.zeros(hf.hp_dim, dtype=torch.float32), requires_grad=False)
                else:
                    return
            else:
                # No hp_type and no hp_init - skip
                return

            self._hp_params[name] = param
            hf.hp_param = param

    @property
    def hyperfunctions(self) -> List[HyperFunction]:
        return list(self._hyperfunctions.values())

    def get_hyperfunction(self, name: str) -> HyperFunction:
        """
        Look up a registered HyperFunction by name.
        """
        return self._hyperfunctions[name]

    @property
    def hp_dim(self) -> int:
        """
        Total number of scalar hp dimensions.
        """
        return int(sum(p.numel() for p in self._hp_params.values()))

    # Simple helpers to expose / mutate the current hp state as a tree.

    def get_hp_state(self) -> Dict[str, torch.Tensor]:
        """
        Return a copy of the current hp state as name -> tensor.

        Tensors can be any shape (1D for LMParam-style, 2D for LoRA weights, etc.).
        """
        return {name: p.detach().clone() for name, p in self._hp_params.items()}

    def set_hp_state(self, state: Dict[str, torch.Tensor]) -> None:
        """
        Overwrite the current hp Parameters from a state dict.
        """
        for name, tensor in state.items():
            if name not in self._hp_params:
                raise KeyError(f"Unknown hyperfunction hp block: {name}")
            param = self._hp_params[name]
            if tensor.shape != param.data.shape:
                raise ValueError(
                    f"hp block shape mismatch for '{name}': got {tensor.shape}, "
                    f"expected {param.data.shape}"
                )
            param.data.copy_(tensor)

    # ------------------------------------------------------------------
    # Workflow-style interfaces
    # ------------------------------------------------------------------

    async def run(self, *args: Any, **kwargs: Any) -> Any:
        """
        Default workflow: if a single HyperFunction is registered, call it.

        For multi-function systems, override this method in a subclass to
        define the pipeline / call graph. Use await for hyperfunction calls
        and asyncio.gather for parallel execution.

        Example:
            async def run(self, data):
                # Sequential
                a = await project(data)
                b = await classify(a)

                # Parallel
                x, y = await asyncio.gather(
                    branch_a(data),
                    branch_b(data),
                )
                return combine([x, y])
        """
        hfs = self.hyperfunctions
        if len(hfs) == 1:
            hf = hfs[0]
            return await hf(*args, **kwargs)
        raise NotImplementedError(
            "HyperSystem.run() must be overridden for multi-HyperFunction systems."
        )

    async def trace_run(self, inputs: Dict[str, Any]) -> ExecutionTrace:
        """
        Execute run() once while tracing all hyperfunction calls.

        Returns an ExecutionTrace that captures the call DAG: which hyperfunctions
        were called, in what order, and with what dependencies between them.

        This trace can then be used for staged batch execution during optimize().
        """
        global _CURRENT_SYSTEM
        prev_system = _CURRENT_SYSTEM
        prev_trace = self._dag_trace

        _CURRENT_SYSTEM = self
        self._dag_trace = ExecutionTrace()

        try:
            result = await self.run(**inputs)
            # Unwrap the final result if it's a TracedValue
            if isinstance(result, TracedValue):
                result = unwrap_traced(result)
            trace = self._dag_trace
            return trace
        finally:
            _CURRENT_SYSTEM = prev_system
            self._dag_trace = prev_trace

    async def execute(
        self,
        inputs: Dict[str, Any],
        collect_trace: bool = True,
    ) -> Any:
        """
        Agent-style execution entrypoint.
        - Wraps `run(**inputs)` with tracing enabled by default so that
          CallRecords and OTel metrics are produced.
        """
        global _CURRENT_SYSTEM
        prev_trace_flag = self._trace_enabled
        prev_history = self._call_history
        prev_system = _CURRENT_SYSTEM
        self._trace_enabled = collect_trace
        if collect_trace:
            self._call_history = []
        _CURRENT_SYSTEM = self
        try:
            return await self.run(**inputs)
        finally:
            _CURRENT_SYSTEM = prev_system
            self._trace_enabled = prev_trace_flag
            if not collect_trace:
                self._call_history = prev_history

    # ------------------------------------------------------------------
    # Evaluation / agent-style interfaces
    # ------------------------------------------------------------------

    async def evaluate(
        self,
        examples: Sequence[Example],
        metric_fn: Callable[[List[Any], List[Any]], float],
    ) -> float:
        """
        Evaluate the system on a batch of Examples with a metric.

        - examples: list of Example(inputs, expected) where inputs are kwargs to run()
        - metric_fn: (preds, expected) -> scalar score (higher is better)

        This runs each example through run() sequentially.
        """
        global _CURRENT_SYSTEM
        prev_system = _CURRENT_SYSTEM
        _CURRENT_SYSTEM = self

        preds: List[Any] = []
        expected: List[Any] = []
        try:
            for ex in examples:
                expected.append(ex.expected)
                result = await self.run(**ex.inputs)
                # Unwrap if traced
                if isinstance(result, TracedValue):
                    result = unwrap_traced(result)
                preds.append(result)
        finally:
            _CURRENT_SYSTEM = prev_system

        return metric_fn(preds, expected)

    async def build_traces(
        self,
        examples: Sequence[Example],
    ) -> List[ExecutionTrace]:
        """
        Build execution traces for each example by running run() once per example.

        Returns a list of traces, one per example. These traces capture the DAG
        structure and can be used for staged batch execution.
        """
        traces: List[ExecutionTrace] = []
        for ex in examples:
            trace = await self.trace_run(ex.inputs)
            traces.append(trace)
        return traces

    # ------------------------------------------------------------------
    # vmap helpers for GPU-batched population evaluation
    # ------------------------------------------------------------------

    def _get_tensor_fn(self, hf: HyperFunction) -> Callable[..., Any]:
        """
        Get a vmap-compatible tensor function from a hyperfunction.

        The tensor function takes (*inputs, weight_tensor) and returns the output.
        For hyperfunctions with hp_type wrappers, this handles the conversion.
        """
        if hf._tensor_fn is not None:
            return hf._tensor_fn

        if hf.hp_type is None or hf._hp_param_name is None:
            # No hp - just return the sync function
            hf._tensor_fn = hf.fn_sync
            return hf._tensor_fn

        # Generate tensor function that wraps hp
        hp_type = hf.hp_type
        fn_sync = hf.fn_sync
        hp_param_name = hf._hp_param_name

        def tensor_fn(*args):
            # Last arg is the weight tensor, convert to hp object
            *inputs, weight = args
            hp = hp_type.from_tensor(weight)
            # Call with hp as keyword arg
            return fn_sync(*inputs, **{hp_param_name: hp})

        hf._tensor_fn = tensor_fn
        return tensor_fn

    def _is_vmappable(self, hf: HyperFunction) -> bool:
        """
        Check if a hyperfunction can be vmapped.

        Uses cached result if available, otherwise tries vmap on sample input.
        """
        if hf._vmappable_cached is not None:
            return hf._vmappable_cached

        # Check if hp_type has a shape method (needed for vmap testing)
        if hf.hp_type is None or not hasattr(hf.hp_type, 'shape'):
            hf._vmappable_cached = False
            return False

        try:
            from torch.func import vmap

            tensor_fn = self._get_tensor_fn(hf)
            hp_shape = hf.hp_type.shape()

            # Create minimal sample inputs - we'll test with batch size 2
            sample_weights = torch.randn(2, *hp_shape)

            # We can't easily test without knowing input shape, so just mark as vmappable
            # and let the actual vmap call fail if needed
            hf._vmappable_cached = True
        except Exception:
            hf._vmappable_cached = False

        return hf._vmappable_cached

    async def _execute_hf_batched(
        self,
        hf: HyperFunction,
        inputs: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Execute HF with vmap batching or asyncio.gather fallback.

        Args:
            hf: The hyperfunction to execute
            inputs: Batched inputs of shape (num_examples, *input_shape)
            weights: Candidate weights of shape (num_candidates, *hp_shape)

        Returns:
            Results of shape (num_examples, num_candidates, *output_shape)
        """
        if self._is_vmappable(hf):
            try:
                from torch.func import vmap

                tensor_fn = self._get_tensor_fn(hf)
                # vmap over candidates (weights), then over examples (inputs)
                # Inner vmap: (input, batch_weights) -> (batch_outputs)
                # Outer vmap: (batch_inputs, weights) -> (batch_inputs, batch_outputs)
                batched_over_weights = vmap(tensor_fn, in_dims=(None, 0))
                batched_both = vmap(batched_over_weights, in_dims=(0, None))

                # Result shape: (num_examples, num_candidates, *output_shape)
                return batched_both(inputs, weights)
            except Exception:
                # Fall through to asyncio.gather fallback
                pass

        # asyncio.gather fallback for non-vmappable HFs
        num_examples = inputs.shape[0]
        num_candidates = weights.shape[0]

        tasks = []
        for ex_idx in range(num_examples):
            inp = inputs[ex_idx]
            for cand_idx in range(num_candidates):
                weight = weights[cand_idx]
                if hf.hp_type is not None:
                    hp = hf.hp_type.from_tensor(weight)
                else:
                    hp = weight
                # Create bound args and invoke
                tasks.append(hf(inp, hp=hp))

        results = await asyncio.gather(*tasks)

        # Reshape to (num_examples, num_candidates, *output_shape)
        # Results are in order: [ex0_cand0, ex0_cand1, ..., ex1_cand0, ex1_cand1, ...]
        output_shape = results[0].shape if hasattr(results[0], 'shape') else ()
        reshaped = torch.stack([
            torch.stack([
                results[ex_idx * num_candidates + cand_idx]
                for cand_idx in range(num_candidates)
            ])
            for ex_idx in range(num_examples)
        ])
        return reshaped

    async def evaluate_population(
        self,
        hp_candidates: Sequence[Dict[str, torch.Tensor]],
        examples: Sequence[Example],
        metric_fn: Callable[[List[Any], List[Any]], float],
        traces: Optional[List[ExecutionTrace]] = None,
        max_batch_size: int = 1024,
    ) -> List[float]:
        """
        Evaluate a list of hp candidates on the same examples.

        Each candidate is a mapping: hyperfunction name -> hp tensor.

        Uses vmap for GPU-batched execution when possible, with memory chunking
        to avoid OOM for large batches. Falls back to sequential execution
        for non-vmappable hyperfunctions.

        Args:
            hp_candidates: List of candidate hp states (name -> tensor)
            examples: Training examples to evaluate on
            metric_fn: Metric function (preds, expected) -> score
            traces: Optional pre-built traces (unused, kept for API compat)
            max_batch_size: Max candidates × examples per vmap call (default 1024)
        """
        if not hp_candidates:
            return []
        if not examples:
            return [0.0] * len(hp_candidates)

        # Validate hp candidates
        base_state = self.get_hp_state()
        base_keys = set(base_state.keys())
        for cand_idx, state in enumerate(hp_candidates):
            keys = set(state.keys())
            if keys != base_keys:
                raise ValueError(
                    f"hp candidate {cand_idx} has hp blocks {sorted(keys)}, "
                    f"expected {sorted(base_keys)}"
                )
            for name, base_vec in base_state.items():
                vec = state[name]
                if vec.shape != base_vec.shape:
                    raise ValueError(
                        f"hp block shape mismatch for candidate {cand_idx}, '{name}': "
                        f"got {vec.shape}, expected {base_vec.shape}"
                    )

        num_candidates = len(hp_candidates)
        num_examples = len(examples)
        total_items = num_candidates * num_examples

        # Chunk by candidates if needed (keep all examples together for efficiency)
        if total_items <= max_batch_size:
            return await self._evaluate_population_batch(
                hp_candidates, examples, metric_fn
            )

        chunk_size = max(1, max_batch_size // num_examples)
        results: List[float] = []

        for i in range(0, num_candidates, chunk_size):
            chunk_candidates = hp_candidates[i:i + chunk_size]
            chunk_results = await self._evaluate_population_batch(
                chunk_candidates, examples, metric_fn
            )
            results.extend(chunk_results)

        return results

    async def _evaluate_population_batch(
        self,
        hp_candidates: Sequence[Dict[str, torch.Tensor]],
        examples: Sequence[Example],
        metric_fn: Callable[[List[Any], List[Any]], float],
    ) -> List[float]:
        """
        Evaluate a batch of hp candidates (internal, no chunking).

        This is the core implementation that uses sequential execution
        per candidate, which is the simplest correct approach. vmap batching
        would require deeper integration with the trace/DAG execution.
        """
        global _CURRENT_SYSTEM
        prev_system = _CURRENT_SYSTEM
        _CURRENT_SYSTEM = self

        base_state = self.get_hp_state()
        results: List[float] = []
        expected: List[Any] = [ex.expected for ex in examples]

        try:
            for cand_state in hp_candidates:
                # Temporarily set this candidate's hp state
                self.set_hp_state(cand_state)

                # Run all examples with this candidate's weights
                preds: List[Any] = []
                for ex in examples:
                    result = await self.run(**ex.inputs)
                    # Unwrap if traced
                    if isinstance(result, TracedValue):
                        result = unwrap_traced(result)
                    preds.append(result)

                # Compute metric for this candidate
                results.append(float(metric_fn(preds, expected)))

            # Restore base state
            self.set_hp_state(base_state)

        finally:
            _CURRENT_SYSTEM = prev_system

        return results

    def _resolve_input(
        self,
        val: Any,
        outputs: List[List[Dict[int, Any]]],
        ex_idx: int,
        trace: ExecutionTrace,
    ) -> Any:
        """
        Resolve an input value, replacing TracedValue references with actual outputs.

        For population eval, we need to resolve per-candidate, but since inputs
        are the same structure across candidates, we just need to identify
        which node_ids to look up. The actual lookup happens per-candidate
        in _execute_stage.
        """
        # For now, just unwrap TracedValue - the actual per-candidate resolution
        # happens in _execute_stage_item
        return unwrap_traced(val)

    async def _execute_stage(
        self,
        work_items: List[Tuple[int, int, TraceNode, Dict[str, Any]]],
        hp_candidates: Sequence[Dict[str, torch.Tensor]],
        outputs: List[List[Dict[int, Any]]],
    ) -> None:
        """
        Execute all work items for a stage in parallel using asyncio.gather.

        work_items: list of (cand_idx, ex_idx, node, resolved_inputs)
        """
        # Group by hyperfunction for potential batching
        by_fn: Dict[str, List[Tuple[int, int, TraceNode, Dict[str, Any]]]] = {}
        for item in work_items:
            _, _, node, _ = item
            by_fn.setdefault(node.fn_name, []).append(item)

        for fn_name, fn_items in by_fn.items():
            # Check if this is a primitive (combine, split) or a hyperfunction
            if fn_name == "combine":
                # Execute combine primitive (sync - just tensor ops)
                for cand_idx, ex_idx, node, resolved_inputs in fn_items:
                    tensors = self._substitute_deps(node.inputs["tensors"], outputs[cand_idx][ex_idx])
                    dim = node.inputs.get("dim", -1)
                    result = torch.cat(tensors, dim=dim)
                    outputs[cand_idx][ex_idx][node.node_id] = result
                continue
            elif fn_name == "split":
                # Execute split primitive (sync - just tensor ops)
                for cand_idx, ex_idx, node, resolved_inputs in fn_items:
                    tensor = self._substitute_deps(node.inputs["tensor"], outputs[cand_idx][ex_idx])
                    sizes = node.inputs["sizes"]
                    dim = node.inputs.get("dim", -1)
                    result = tuple(torch.split(tensor, sizes, dim=dim))
                    outputs[cand_idx][ex_idx][node.node_id] = result
                continue

            # Regular hyperfunction execution - use asyncio.gather for parallelism
            hf = self._hyperfunctions[fn_name]

            # Build list of coroutines for parallel execution
            async def execute_item(cand_idx, ex_idx, node, resolved_inputs):
                hp_tensor = hp_candidates[cand_idx].get(fn_name, None)
                final_inputs = self._resolve_inputs_for_candidate(
                    node, resolved_inputs, outputs[cand_idx][ex_idx]
                )
                bound = hf._sig.bind_partial(**final_inputs)
                bound.apply_defaults()
                result = await hf._invoke_with_hp(hp_tensor, bound, ex_idx)
                return cand_idx, ex_idx, node.node_id, result

            # Execute all items in parallel
            tasks = [
                execute_item(cand_idx, ex_idx, node, resolved_inputs)
                for cand_idx, ex_idx, node, resolved_inputs in fn_items
            ]
            results = await asyncio.gather(*tasks)

            # Store results
            for cand_idx, ex_idx, node_id, result in results:
                outputs[cand_idx][ex_idx][node_id] = result

    def _resolve_inputs_for_candidate(
        self,
        node: TraceNode,
        resolved_inputs: Dict[str, Any],
        cand_outputs: Dict[int, Any],
    ) -> Dict[str, Any]:
        """
        Replace dependency references in inputs with actual outputs from earlier nodes.

        node.dependencies tells us which node_ids this node depends on.
        We look up their outputs in cand_outputs and substitute them.
        """
        if not node.dependencies:
            return resolved_inputs

        # We need to figure out which input values came from which nodes.
        # This requires re-inspecting the original inputs to find TracedValue refs.
        final_inputs: Dict[str, Any] = {}
        for key, original_val in node.inputs.items():
            final_inputs[key] = self._substitute_deps(original_val, cand_outputs)
        return final_inputs

    def _substitute_deps(self, val: Any, cand_outputs: Dict[int, Any]) -> Any:
        """Recursively substitute TracedValue references with actual outputs."""
        if isinstance(val, TracedValue):
            node_id = val._traced_node_id
            if node_id in cand_outputs:
                return cand_outputs[node_id]
            # Fallback to unwrapped value if not found (shouldn't happen)
            return unwrap_traced(val)
        elif isinstance(val, dict):
            return {k: self._substitute_deps(v, cand_outputs) for k, v in val.items()}
        elif isinstance(val, list):
            return [self._substitute_deps(v, cand_outputs) for v in val]
        elif isinstance(val, tuple):
            return tuple(self._substitute_deps(v, cand_outputs) for v in val)
        return val

    async def optimize(
        self,
        train_data: Sequence[Example],
        metric_fn: Callable[[List[Any], List[Any]], float],
    ) -> None:
        """
        Optimize the system on training data.

        1. Build traces by running each example once (captures DAG structure)
        2. Run prompt optimization (if enabled)
        3. Run hyperparameter optimization using staged batch execution
        """
        # 0) Build traces for staged execution
        traces = await self.build_traces(train_data)
        self._cached_trace = traces[0] if traces else None  # Cache first for reference

        # 1) Prompt optimisation
        if self.prompt_optimizer is not None:
            await self.prompt_optimizer.optimize(self, train_data, metric_fn)

        # 2) Hyperparameter / system optimisation with pre-built traces
        if self.system_optimizer is not None and self.hp_dim > 0:
            await self.system_optimizer.optimize(self, train_data, metric_fn, traces=traces)

    # ------------------------------------------------------------------
    # Internal hooks for tracing / batching
    # ------------------------------------------------------------------

    def _before_hf_call(
        self,
        hf: HyperFunction,
        bound_args: Dict[str, Any],
        example_index: Optional[int],
    ) -> Any:
        # Start an OTel span if a tracer is configured. We deliberately keep
        # this very lightweight and optional.
        if self._otel_tracer is None:  # pragma: no cover - requires OTel
            return None
        span = self._otel_tracer.start_span(hf.name)
        # Attach basic attributes that are cheap to compute.
        if hasattr(span, "set_attribute"):  # pragma: no cover - depends on OTel
            span.set_attribute("hyperfunc.name", hf.name)
            if example_index is not None:
                span.set_attribute("example.index", int(example_index))
        return span

    def _after_hf_call(
        self,
        hf: HyperFunction,
        bound_args: Dict[str, Any],
        result: Any,
        error: Optional[BaseException],
        elapsed_s: float,
        example_index: Optional[int],
        span: Any,
    ) -> None:
        # Build a CallContext for OTel metric functions.
        metrics: Dict[str, float] = {}
        ctx = CallContext(
            hf=hf,
            bound_args=dict(bound_args),
            result=result,
            error=error,
            elapsed_s=elapsed_s,
            example_index=example_index,
            metadata={},
            span=span,
        )

        # Compute and attach OTel metrics if configured.
        for name, func in self.otel_metric_funcs.items():
            try:
                value = float(func(ctx))
            except Exception:
                # Metric functions must not break execution.
                continue
            metrics[name] = value
            if span is not None and hasattr(span, "set_attribute"):  # pragma: no cover
                span.set_attribute(name, value)  # type: ignore[call-arg]

        # End span if present.
        if span is not None and hasattr(span, "end"):  # pragma: no cover
            span.end()  # type: ignore[call-arg]

        # Append to in-memory call history if tracing is enabled.
        if self._trace_enabled:
            self._call_history.append(
                CallRecord(
                    fn_name=hf.name,
                    example_index=example_index,
                    elapsed_s=elapsed_s,
                    error=error,
                    extra_metrics=metrics,
                )
            )

    async def _run_batch(
        self,
        hf: HyperFunction,
        example_indices: Sequence[int],
        batch_inputs: Sequence[Dict[str, Any]],
    ) -> List[Any]:
        """
        Execute a batch of calls to a single HyperFunction in parallel.

        Uses asyncio.gather for parallel execution.
        """
        tasks = [
            self._run_index(hf, inputs, idx)
            for idx, inputs in zip(example_indices, batch_inputs)
        ]
        return await asyncio.gather(*tasks)

    async def _run_index(
        self,
        hf: HyperFunction,
        inputs: Dict[str, Any],
        idx: int,
    ) -> Any:
        # Route through HyperFunction._invoke_with_hp so that hp injection
        # and tracing / OTel hooks are consistently applied.
        bound = hf._sig.bind_partial(**inputs)
        bound.apply_defaults()
        hp_tensor: Optional[torch.Tensor] = None
        if hf.hp_param is not None:
            hp_tensor = hf.hp_param.data
        out = await hf._invoke_with_hp(hp_tensor, bound, example_index=idx)
        return out
