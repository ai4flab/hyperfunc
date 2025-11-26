"""Built-in primitives that auto-trace without @hyperfunction.

These primitives (combine, split) are automatically traced when called inside
a HyperSystem.run() method. They don't require explicit registration and
don't have any trainable parameters.
"""

from typing import List, Tuple

import torch

from . import core
from .core import TracedValue, unwrap_traced


def combine(tensors: List[torch.Tensor], dim: int = -1) -> torch.Tensor:
    """Combine tensors via concatenation. Auto-traced in run().

    This is automatically recorded in the DAG when called inside a HyperSystem's
    run() method, ensuring proper replay during ES optimization.

    Args:
        tensors: List of tensors to concatenate
        dim: Dimension along which to concatenate (default: -1)

    Returns:
        Concatenated tensor

    Example:
        >>> clip_features = clip_encoder(image)  # 512-dim
        >>> pixel_features = extract_pixels(image)  # 192-dim
        >>> combined = combine([clip_features, pixel_features])  # 704-dim
    """
    # Check if we're in a traced context
    system = core._CURRENT_SYSTEM
    if system is not None and system._dag_trace is not None:
        # Record in DAG
        trace_node = system._dag_trace.add_node("combine", {"tensors": tensors, "dim": dim})
        # Unwrap inputs
        unwrapped = [unwrap_traced(t) for t in tensors]
        result = torch.cat(unwrapped, dim=dim)
        # Wrap output
        trace_node.output = result
        return TracedValue(result, trace_node.node_id, system)

    # Outside traced context - just do the operation
    unwrapped = [unwrap_traced(t) for t in tensors]
    return torch.cat(unwrapped, dim=dim)


def split(tensor: torch.Tensor, sizes: List[int], dim: int = -1) -> Tuple[torch.Tensor, ...]:
    """Split tensor into parts. Auto-traced in run().

    This is automatically recorded in the DAG when called inside a HyperSystem's
    run() method, ensuring proper replay during ES optimization.

    Args:
        tensor: Input tensor to split
        sizes: List of sizes for each output
        dim: Dimension along which to split (default: -1)

    Returns:
        Tuple of tensors

    Example:
        >>> features = encoder(data)  # 1024-dim
        >>> left, right = split(features, sizes=[512, 512])
    """
    system = core._CURRENT_SYSTEM
    if system is not None and system._dag_trace is not None:
        trace_node = system._dag_trace.add_node("split", {"tensor": tensor, "sizes": sizes, "dim": dim})
        unwrapped = unwrap_traced(tensor)
        result = tuple(torch.split(unwrapped, sizes, dim=dim))
        trace_node.output = result
        # Wrap each output tensor with the same node_id
        wrapped = tuple(TracedValue(t, trace_node.node_id, system) for t in result)
        return wrapped

    unwrapped = unwrap_traced(tensor)
    return tuple(torch.split(unwrapped, sizes, dim=dim))
