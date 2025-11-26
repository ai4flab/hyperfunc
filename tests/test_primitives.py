"""Tests for hyperfunc primitives (combine, split)."""

import asyncio
import warnings

import pytest
import torch

from hyperfunc import (
    HyperSystem,
    LoRAWeight,
    TracedValueWarning,
    combine,
    hyperfunction,
    split,
)


class TestCombine:
    """Tests for the combine primitive."""

    def test_combine_two_tensors(self):
        """combine should concatenate two tensors."""
        a = torch.randn(10)
        b = torch.randn(20)
        result = combine([a, b])
        assert result.shape == (30,)
        assert torch.allclose(result[:10], a)
        assert torch.allclose(result[10:], b)

    def test_combine_three_tensors(self):
        """combine should work with three or more tensors."""
        a = torch.randn(10)
        b = torch.randn(20)
        c = torch.randn(30)
        result = combine([a, b, c])
        assert result.shape == (60,)

    def test_combine_with_dim(self):
        """combine should respect the dim parameter."""
        a = torch.randn(2, 3)
        b = torch.randn(2, 4)
        result = combine([a, b], dim=1)
        assert result.shape == (2, 7)

    def test_combine_dim0(self):
        """combine should work with dim=0."""
        a = torch.randn(2, 5)
        b = torch.randn(3, 5)
        result = combine([a, b], dim=0)
        assert result.shape == (5, 5)


class TestSplit:
    """Tests for the split primitive."""

    def test_split_two_parts(self):
        """split should divide tensor into two parts."""
        x = torch.randn(30)
        a, b = split(x, sizes=[10, 20])
        assert a.shape == (10,)
        assert b.shape == (20,)
        assert torch.allclose(a, x[:10])
        assert torch.allclose(b, x[10:])

    def test_split_three_parts(self):
        """split should work with three or more parts."""
        x = torch.randn(60)
        a, b, c = split(x, sizes=[10, 20, 30])
        assert a.shape == (10,)
        assert b.shape == (20,)
        assert c.shape == (30,)

    def test_split_with_dim(self):
        """split should respect the dim parameter."""
        x = torch.randn(2, 7)
        a, b = split(x, sizes=[3, 4], dim=1)
        assert a.shape == (2, 3)
        assert b.shape == (2, 4)


class TestCombineSplitRoundtrip:
    """Tests for combine and split working together."""

    def test_roundtrip(self):
        """split(combine([a, b])) should recover original tensors."""
        a = torch.randn(10)
        b = torch.randn(20)
        combined = combine([a, b])
        a2, b2 = split(combined, sizes=[10, 20])
        assert torch.allclose(a, a2)
        assert torch.allclose(b, b2)


class TestTracedValueWarning:
    """Tests for TracedValueWarning when using torch.cat/split directly."""

    @pytest.mark.asyncio
    async def test_torch_cat_warns(self):
        """Using torch.cat on hyperfunction outputs should warn."""
        # Create a simple system with a hyperfunction
        WeightType = LoRAWeight.create(out_dim=10, in_dim=5, noise_rank=2)

        @hyperfunction(hp_type=WeightType, optimize_hparams=True)
        async def project(x, hp):
            return x @ hp.weight.T

        class TestSystem(HyperSystem):
            async def run(self, x, y):
                out1 = await project(x)
                out2 = await project(y)
                # This should trigger a warning
                return torch.cat([out1, out2])

        system = TestSystem()
        system.register_hyperfunction(project, hp_init=torch.randn(10, 5))

        # Run with tracing and check for warning
        x = torch.randn(5)
        y = torch.randn(5)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            trace = await system.trace_run({"x": x, "y": y})
            # Check that a TracedValueWarning was raised
            traced_warnings = [
                warning for warning in w
                if issubclass(warning.category, TracedValueWarning)
            ]
            assert len(traced_warnings) >= 1
            assert "torch.cat" in str(traced_warnings[0].message)

    @pytest.mark.asyncio
    async def test_combine_no_warning(self):
        """Using combine() should not warn."""
        WeightType = LoRAWeight.create(out_dim=10, in_dim=5, noise_rank=2)

        @hyperfunction(hp_type=WeightType, optimize_hparams=True)
        async def project(x, hp):
            return x @ hp.weight.T

        class TestSystem(HyperSystem):
            async def run(self, x, y):
                out1 = await project(x)
                out2 = await project(y)
                # Using combine should not warn
                return combine([out1, out2])

        system = TestSystem()
        system.register_hyperfunction(project, hp_init=torch.randn(10, 5))

        x = torch.randn(5)
        y = torch.randn(5)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            trace = await system.trace_run({"x": x, "y": y})
            # Check that no TracedValueWarning was raised
            traced_warnings = [
                warning for warning in w
                if issubclass(warning.category, TracedValueWarning)
            ]
            assert len(traced_warnings) == 0


class TestPrimitivesInDAG:
    """Tests for primitives being properly recorded in the DAG.

    Primitives (combine, split) auto-trace without needing explicit
    registration - they detect when they're called inside a traced context.
    """

    @pytest.mark.asyncio
    async def test_combine_recorded_in_trace(self):
        """combine should create a node without explicit registration."""
        WeightType = LoRAWeight.create(out_dim=10, in_dim=15, noise_rank=2)

        @hyperfunction(hp_type=WeightType, optimize_hparams=True)
        async def project(x, hp):
            return x @ hp.weight.T

        class TestSystem(HyperSystem):
            async def run(self, a, b):
                combined = combine([a, b])  # Auto-traced
                return await project(combined)

        system = TestSystem()
        system.register_hyperfunction(project, hp_init=torch.randn(10, 15))
        # No need to register combine - it auto-traces

        a = torch.randn(5)
        b = torch.randn(10)

        trace = await system.trace_run({"a": a, "b": b})

        # Should have 2 nodes: combine and project
        assert len(trace.nodes) == 2
        assert trace.nodes[0].fn_name == "combine"
        assert trace.nodes[1].fn_name == "project"

        # project should depend on combine
        assert 0 in trace.nodes[1].dependencies

    @pytest.mark.asyncio
    async def test_split_recorded_in_trace(self):
        """split should create a node without explicit registration."""
        WeightType = LoRAWeight.create(out_dim=5, in_dim=10, noise_rank=2)

        @hyperfunction(hp_type=WeightType, optimize_hparams=True)
        async def project_left(x, hp):
            return x @ hp.weight.T

        @hyperfunction(hp_type=WeightType, optimize_hparams=True)
        async def project_right(x, hp):
            return x @ hp.weight.T

        class TestSystem(HyperSystem):
            async def run(self, x):
                left, right = split(x, sizes=[10, 10])
                out_left, out_right = await asyncio.gather(
                    project_left(left),
                    project_right(right),
                )
                return combine([out_left, out_right])

        system = TestSystem()
        system.register_hyperfunction(project_left, hp_init=torch.randn(5, 10))
        system.register_hyperfunction(project_right, hp_init=torch.randn(5, 10))
        # No need to register split or combine - they auto-trace

        x = torch.randn(20)
        trace = await system.trace_run({"x": x})

        # Should have 4 nodes: split, project_left, project_right, combine
        assert len(trace.nodes) == 4
        node_names = [n.fn_name for n in trace.nodes]
        assert "split" in node_names
        assert "project_left" in node_names
        assert "project_right" in node_names
        assert "combine" in node_names
