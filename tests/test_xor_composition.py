"""
Test that composed LoRA hyperfunctions can solve XOR,
which a single linear layer mathematically cannot.

XOR truth table:
  (0,0) -> 0
  (0,1) -> 1
  (1,0) -> 1
  (1,1) -> 0

This is not linearly separable - requires at least one hidden layer with nonlinearity.
"""

import torch
from hyperfunc import (
    HyperSystem,
    Example,
    LoRAWeight,
    hyperfunction,
    ESHybridSystemOptimizer,
)


# Hidden layer: 2 inputs -> 4 hidden units
HiddenWeights = LoRAWeight.create(out_dim=4, in_dim=2, noise_rank=2)
# Output layer: 4 hidden -> 1 output
OutputWeights = LoRAWeight.create(out_dim=1, in_dim=4, noise_rank=2)


@hyperfunction(hp_type=HiddenWeights, optimize_hparams=True)
def hidden_layer(x: torch.Tensor, hp) -> torch.Tensor:
    """Hidden layer with ReLU activation."""
    return torch.relu(x @ hp.weight.T)


@hyperfunction(hp_type=OutputWeights, optimize_hparams=True)
def output_layer(x: torch.Tensor, hp) -> torch.Tensor:
    """Output layer with sigmoid for binary classification."""
    return torch.sigmoid(x @ hp.weight.T)


class XORSystem(HyperSystem):
    """Two-layer network: hidden(ReLU) -> output(sigmoid)"""

    def run(self, x: torch.Tensor) -> float:
        hidden = hidden_layer(x)
        output = output_layer(hidden)
        return float(output.squeeze())


# XOR training data
XOR_EXAMPLES = [
    Example({"x": torch.tensor([0., 0.])}, 0.0),
    Example({"x": torch.tensor([0., 1.])}, 1.0),
    Example({"x": torch.tensor([1., 0.])}, 1.0),
    Example({"x": torch.tensor([1., 1.])}, 0.0),
]


def binary_cross_entropy_metric(preds, expected):
    """Negative BCE (higher is better)."""
    import math
    total_bce = 0.0
    eps = 1e-7
    for pred, exp in zip(preds, expected):
        p = max(min(pred, 1 - eps), eps)  # Clamp to avoid log(0)
        bce = -(exp * math.log(p) + (1 - exp) * math.log(1 - p))
        total_bce += bce
    return -total_bce / len(preds)


def accuracy_metric(preds, expected):
    """Classification accuracy (higher is better)."""
    correct = sum(1 for p, e in zip(preds, expected) if (p > 0.5) == (e > 0.5))
    return correct / len(preds)


def test_xor_composition():
    """
    Test that a two-layer network can learn XOR.
    """
    system = XORSystem(
        system_optimizer=ESHybridSystemOptimizer(
            steps=200,
            pop_size=64,
            sigma=0.3,
            lr=0.2,
            antithetic=True,
        )
    )

    # Initialize with small random weights
    torch.manual_seed(42)
    system.register_hyperfunction(hidden_layer, hp_init=torch.randn(4, 2) * 0.5)
    system.register_hyperfunction(output_layer, hp_init=torch.randn(1, 4) * 0.5)

    # Evaluate before optimization
    initial_acc = accuracy_metric(
        [system.run(**ex.inputs) for ex in XOR_EXAMPLES],
        [ex.expected for ex in XOR_EXAMPLES]
    )

    # Optimize
    system.optimize(XOR_EXAMPLES, binary_cross_entropy_metric)

    # Evaluate after optimization
    final_preds = [system.run(**ex.inputs) for ex in XOR_EXAMPLES]
    final_acc = accuracy_metric(final_preds, [ex.expected for ex in XOR_EXAMPLES])

    print(f"Initial accuracy: {initial_acc:.2%}")
    print(f"Final accuracy: {final_acc:.2%}")
    print(f"\nPredictions:")
    for ex, pred in zip(XOR_EXAMPLES, final_preds):
        x = ex.inputs["x"].tolist()
        print(f"  {x} -> {pred:.3f} (expected {ex.expected})")

    assert final_acc == 1.0, f"Should achieve 100% accuracy on XOR, got {final_acc:.2%}"


def test_single_layer_cannot_solve_xor():
    """
    Prove that a single linear layer cannot solve XOR.
    This is the mathematical foundation for why composition matters.
    """
    # Single layer: 2 inputs -> 1 output (no hidden layer, no nonlinearity)
    SingleWeights = LoRAWeight.create(out_dim=1, in_dim=2, noise_rank=2)

    @hyperfunction(hp_type=SingleWeights, optimize_hparams=True)
    def single_layer(x: torch.Tensor, hp) -> float:
        return float(torch.sigmoid(x @ hp.weight.T).squeeze())

    class SingleLayerSystem(HyperSystem):
        def run(self, x: torch.Tensor) -> float:
            return single_layer(x)

    system = SingleLayerSystem(
        system_optimizer=ESHybridSystemOptimizer(
            steps=200,  # Give it many chances
            pop_size=64,
            sigma=0.3,
            lr=0.2,
            antithetic=True,
        )
    )
    torch.manual_seed(42)
    system.register_hyperfunction(single_layer, hp_init=torch.randn(1, 2) * 0.5)

    # Optimize
    system.optimize(XOR_EXAMPLES, binary_cross_entropy_metric)

    # Evaluate
    final_preds = [system.run(**ex.inputs) for ex in XOR_EXAMPLES]
    final_acc = accuracy_metric(final_preds, [ex.expected for ex in XOR_EXAMPLES])

    print(f"Single layer accuracy: {final_acc:.2%}")
    print(f"Predictions: {[f'{p:.3f}' for p in final_preds]}")

    # Single layer CANNOT achieve 100% on XOR - it's mathematically impossible
    assert final_acc < 1.0, "Single layer should NOT be able to solve XOR perfectly"
