"""Tests for DAG tracing and staged execution."""

import pytest
import torch

from hyperfunc import (
    Example,
    HyperSystem,
    LMParam,
    hyperfunction,
    unwrap_traced,
)


@pytest.mark.asyncio
async def test_trace_captures_call_order():
    """Test that trace_run captures hyperfunction calls in order."""

    @hyperfunction()
    async def step_a(x: int) -> int:
        return x + 1

    @hyperfunction()
    async def step_b(y: int) -> int:
        return y * 2

    class PipelineSystem(HyperSystem):
        async def run(self, x: int):
            a_result = await step_a(x)
            b_result = await step_b(a_result)
            return b_result

    system = PipelineSystem()
    system.register_hyperfunction(step_a)
    system.register_hyperfunction(step_b)

    trace = await system.trace_run({"x": 5})

    assert len(trace.nodes) == 2
    assert trace.nodes[0].fn_name == "step_a"
    assert trace.nodes[1].fn_name == "step_b"


@pytest.mark.asyncio
async def test_trace_captures_dependencies():
    """Test that dependencies are correctly detected from data flow."""

    @hyperfunction()
    async def step_a(x: int) -> int:
        return x + 1

    @hyperfunction()
    async def step_b(y: int) -> int:
        return y * 2

    class PipelineSystem(HyperSystem):
        async def run(self, x: int):
            a_result = await step_a(x)
            b_result = await step_b(a_result)  # b depends on a
            return b_result

    system = PipelineSystem()
    system.register_hyperfunction(step_a)
    system.register_hyperfunction(step_b)

    trace = await system.trace_run({"x": 5})

    # step_b should depend on step_a (node_id 0)
    assert trace.nodes[1].dependencies == {0}


@pytest.mark.asyncio
async def test_trace_stages_parallel_calls():
    """Test that independent calls are placed in the same stage."""
    import asyncio

    @hyperfunction()
    async def step_a(x: int) -> int:
        return x + 1

    @hyperfunction()
    async def step_b(x: int) -> int:
        return x * 2

    @hyperfunction()
    async def step_c(a: int, b: int) -> int:
        return a + b

    class ParallelSystem(HyperSystem):
        async def run(self, x: int):
            # a and b are independent, can run in parallel
            a_result, b_result = await asyncio.gather(
                step_a(x),
                step_b(x),
            )
            # c depends on both
            c_result = await step_c(a_result, b_result)
            return c_result

    system = ParallelSystem()
    system.register_hyperfunction(step_a)
    system.register_hyperfunction(step_b)
    system.register_hyperfunction(step_c)

    trace = await system.trace_run({"x": 5})

    stages = trace.to_stages()

    # Should have 2 stages: [a, b] and [c]
    assert len(stages) == 2
    assert len(stages[0]) == 2  # a and b in stage 0
    assert len(stages[1]) == 1  # c in stage 1

    stage_0_names = {node.fn_name for node in stages[0]}
    assert stage_0_names == {"step_a", "step_b"}
    assert stages[1][0].fn_name == "step_c"


@pytest.mark.asyncio
async def test_evaluate_with_multi_stage_system():
    """Test that evaluate works correctly with multi-stage systems."""

    @hyperfunction()
    async def double(x: int) -> int:
        return x * 2

    @hyperfunction()
    async def add_one(x: int) -> int:
        return x + 1

    class ChainedSystem(HyperSystem):
        async def run(self, x: int):
            doubled = await double(x)
            result = await add_one(doubled)
            return result

    system = ChainedSystem()
    system.register_hyperfunction(double)
    system.register_hyperfunction(add_one)

    examples = [
        Example({"x": 1}, 3),   # 1*2 + 1 = 3
        Example({"x": 5}, 11),  # 5*2 + 1 = 11
        Example({"x": 10}, 21), # 10*2 + 1 = 21
    ]

    def accuracy(preds, expected):
        return sum(1 for p, e in zip(preds, expected) if p == e) / len(expected)

    score = await system.evaluate(examples, accuracy)
    assert score == 1.0  # All correct


@pytest.mark.asyncio
async def test_population_eval_with_hp():
    """Test that population evaluation works with hyperparameters."""

    @hyperfunction(hp_type=LMParam, optimize_hparams=True)
    async def scaled_add(x: float, hp: LMParam) -> float:
        # Use temperature as a scaling factor
        return x + hp.temperature

    class ScaledSystem(HyperSystem):
        async def run(self, x: float):
            return await scaled_add(x)

    system = ScaledSystem()
    system.register_hyperfunction(scaled_add)

    examples = [Example({"x": 1.0}, 1.5)]  # We want x + 0.5 = 1.5

    def metric_fn(preds, expected):
        # Negative absolute error
        return -sum(abs(p - e) for p, e in zip(preds, expected))

    # Create candidates with different temperatures
    with torch.no_grad():
        base = system.get_hp_state()["scaled_add"].clone()

        # Candidate with temp=0.0
        hp_a = base.clone()
        hp_a[0] = 0.0

        # Candidate with temp=0.5 (optimal)
        hp_b = base.clone()
        hp_b[0] = 0.5

        # Candidate with temp=1.0
        hp_c = base.clone()
        hp_c[0] = 1.0

    scores = await system.evaluate_population(
        [
            {"scaled_add": hp_a},
            {"scaled_add": hp_b},
            {"scaled_add": hp_c},
        ],
        examples,
        metric_fn,
    )

    # hp_b should have the best (highest) score since it's optimal
    assert scores[1] > scores[0]
    assert scores[1] > scores[2]
    assert scores[1] == 0.0  # Perfect match


def test_unwrap_traced():
    """Test that unwrap_traced correctly unwraps nested structures."""
    from hyperfunc.core import TracedValue

    # Create a mock system for TracedValue
    system = HyperSystem()

    # Test simple value
    wrapped = TracedValue(42, node_id=0, system=system)
    assert unwrap_traced(wrapped) == 42

    # Test nested dict
    nested = {
        "a": TracedValue(1, node_id=0, system=system),
        "b": {"c": TracedValue(2, node_id=1, system=system)},
    }
    unwrapped = unwrap_traced(nested)
    assert unwrapped == {"a": 1, "b": {"c": 2}}

    # Test nested list
    nested_list = [TracedValue(1, node_id=0, system=system), [TracedValue(2, node_id=1, system=system)]]
    unwrapped_list = unwrap_traced(nested_list)
    assert unwrapped_list == [1, [2]]
