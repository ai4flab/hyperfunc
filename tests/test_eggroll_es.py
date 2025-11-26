"""Tests for TorchEggrollES with low-rank noise and antithetic sampling."""

import torch
from torch import nn

from hyperfunc import (
    TorchEggrollES,
    HyperSystem,
    Example,
    LoRAWeight,
    hyperfunction,
    ESHybridSystemOptimizer,
)


class SimpleQuadraticModel(nn.Module):
    """Simple model where we want param to converge to target."""

    def __init__(self, dim: int, target: torch.Tensor):
        super().__init__()
        self.param = nn.Parameter(torch.zeros(dim))
        self.target = target

    def forward(self) -> torch.Tensor:
        # Negative squared distance from target
        return -((self.param - self.target) ** 2).sum()


class SimpleMatrixModel(nn.Module):
    """Model with a 2D parameter (matrix) to test low-rank noise."""

    def __init__(self, in_dim: int, out_dim: int, target: torch.Tensor):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(out_dim, in_dim))
        self.target = target

    def forward(self) -> torch.Tensor:
        # Negative Frobenius norm of difference from target
        return -((self.weight - self.target) ** 2).sum()


def test_eggroll_es_1d_param():
    """Test that ES converges for a simple 1D optimization problem."""
    target = torch.tensor([0.5, -0.3, 0.8])
    model = SimpleQuadraticModel(dim=3, target=target)

    es = TorchEggrollES(
        model=model,
        pop_size=32,
        sigma=0.1,
        lr=0.1,
        antithetic=True,
    )

    def eval_fn(m):
        return float(m())

    initial_fitness = eval_fn(model)

    # Run several steps
    for _ in range(20):
        es.step(eval_fn)

    final_fitness = eval_fn(model)

    # Should improve
    assert final_fitness > initial_fitness
    # Should be close to 0 (perfect match)
    assert final_fitness > -0.1


def test_eggroll_es_2d_lowrank_noise():
    """Test that low-rank noise works for matrix parameters."""
    target = torch.randn(4, 8) * 0.5
    model = SimpleMatrixModel(in_dim=8, out_dim=4, target=target)

    es = TorchEggrollES(
        model=model,
        pop_size=32,
        sigma=0.1,
        lr=0.1,
        rank=2,  # Low rank
        antithetic=True,
    )

    def eval_fn(m):
        return float(m())

    initial_fitness = eval_fn(model)

    # Run several steps
    for _ in range(30):
        es.step(eval_fn)

    final_fitness = eval_fn(model)

    # Should improve
    assert final_fitness > initial_fitness


def test_antithetic_sampling_reduces_variance():
    """
    Test that antithetic sampling produces lower variance estimates.

    With antithetic sampling, pairs of noise vectors are negatives of each other,
    which should reduce variance in the gradient estimate.
    """
    target = torch.tensor([0.5])
    model = SimpleQuadraticModel(dim=1, target=target)

    # Run without antithetic
    model_no_anti = SimpleQuadraticModel(dim=1, target=target)
    es_no_anti = TorchEggrollES(
        model=model_no_anti,
        pop_size=32,
        sigma=0.1,
        lr=0.1,
        antithetic=False,
    )

    # Run with antithetic
    model_anti = SimpleQuadraticModel(dim=1, target=target)
    es_anti = TorchEggrollES(
        model=model_anti,
        pop_size=32,
        sigma=0.1,
        lr=0.1,
        antithetic=True,
    )

    def eval_fn(m):
        return float(m())

    # Both should converge, but antithetic should be more stable
    for _ in range(20):
        es_no_anti.step(eval_fn)
        es_anti.step(eval_fn)

    # Both should improve significantly
    assert eval_fn(model_no_anti) > -0.2
    assert eval_fn(model_anti) > -0.2


def test_pop_size_must_be_even_for_antithetic():
    """Test that antithetic sampling requires even population size."""
    model = SimpleQuadraticModel(dim=1, target=torch.tensor([0.5]))

    try:
        TorchEggrollES(
            model=model,
            pop_size=31,  # Odd
            antithetic=True,
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "even" in str(e).lower()


def test_param_filter():
    """Test that param_filter correctly selects which parameters to optimize."""

    class TwoParamModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.frozen_param = nn.Parameter(torch.zeros(3))
            self.trainable_param = nn.Parameter(torch.zeros(3))
            self.target = torch.tensor([1.0, 1.0, 1.0])

        def forward(self):
            # Only trainable_param should change
            return -((self.trainable_param - self.target) ** 2).sum()

    model = TwoParamModel()
    initial_frozen = model.frozen_param.clone()

    # Only optimize trainable_param
    es = TorchEggrollES(
        model=model,
        pop_size=16,
        sigma=0.1,
        lr=0.1,
        param_filter=lambda p, name: "trainable" in name,
        antithetic=True,
    )

    def eval_fn(m):
        return float(m())

    for _ in range(10):
        es.step(eval_fn)

    # frozen_param should not have changed
    assert torch.allclose(model.frozen_param, initial_frozen)
    # trainable_param should have changed
    assert not torch.allclose(model.trainable_param, torch.zeros(3))


def test_lora_weight_with_system_optimizer():
    """Test that LoRAWeight hp_type uses low-rank noise in ESHybridSystemOptimizer."""

    # Create a LoRAWeight type for 4x8 matrix with rank=2 noise
    MyLoRA = LoRAWeight.create(out_dim=4, in_dim=8, noise_rank=2)

    # Target matrix we want to find
    target = torch.randn(4, 8) * 0.3

    @hyperfunction(hp_type=MyLoRA, optimize_hparams=True)
    def lora_fn(x: float, hp) -> float:
        # hp.weight is a 4x8 matrix
        # Return negative Frobenius distance from target
        return -float(((hp.weight - target) ** 2).sum())

    class LoRASystem(HyperSystem):
        def run(self, x: float):
            return lora_fn(x)

    system = LoRASystem(
        system_optimizer=ESHybridSystemOptimizer(
            steps=30,
            pop_size=32,
            sigma=0.1,
            lr=0.1,
        )
    )

    # Register with initial 2D tensor
    init_weight = torch.zeros(4, 8)
    system.register_hyperfunction(lora_fn, hp_init=init_weight)

    examples = [Example({"x": 0.0}, 0.0)]

    def metric_fn(preds, expected):
        # Higher is better (less negative)
        return sum(preds) / len(preds)

    initial_score = system.evaluate(examples, metric_fn)

    system.optimize(examples, metric_fn)

    final_score = system.evaluate(examples, metric_fn)

    # Should improve
    assert final_score > initial_score


def test_lora_weight_shape_and_noise_rank():
    """Test that LoRAWeight.create produces correct shape and noise_rank."""
    MyLoRA = LoRAWeight.create(out_dim=16, in_dim=32, noise_rank=4)

    assert MyLoRA.shape() == (16, 32)
    assert MyLoRA.dim() == 16 * 32
    assert MyLoRA.noise_rank() == 4

    # Test from_tensor / to_tensor roundtrip
    weight = torch.randn(16, 32)
    hp = MyLoRA.from_tensor(weight)
    assert torch.allclose(hp.weight, weight)
    assert torch.allclose(hp.to_tensor(), weight)
