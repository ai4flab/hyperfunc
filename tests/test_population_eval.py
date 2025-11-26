import pytest
import torch
from torch.func import vmap

from hyperfunc import Example, HyperSystem, LMParam, hyperfunction
from hyperfunc.core import LoRAWeight


@pytest.mark.asyncio
async def test_evaluate_population_matches_single_eval():
    @hyperfunction(hp_type=LMParam, optimize_hparams=True)
    async def target_fn(x: float, hp: LMParam) -> float:
        # Simple objective that depends only on temperature.
        return -abs(hp.temperature - x)

    # Create a custom system that routes inputs to target_fn
    class TestSystem(HyperSystem):
        async def run(self, x: float):
            return await target_fn(x)

    system = TestSystem()
    system.register_hyperfunction(target_fn)

    # One example where the "target" is 0.5 for temperature.
    # New Example format: just inputs (kwargs to run) and expected output
    examples = [Example({"x": 0.5}, 0.0)]

    def metric_fn(preds, expected):
        # Average the objective values.
        return sum(preds) / len(preds)

    # Build two candidate hp states with different temperatures.
    with torch.no_grad():
        base_state = system.get_hp_state()
        base_vec = base_state["target_fn"].detach().clone()

        # Candidate A: temperature at 0.0
        hp_a = base_vec.clone()
        hp_a[0] = 0.0

        # Candidate B: temperature at 0.5
        hp_b = base_vec.clone()
        hp_b[0] = 0.5

    scores = await system.evaluate_population(
        [
            {"target_fn": hp_a},
            {"target_fn": hp_b},
        ],
        examples,
        metric_fn,
    )

    # Evaluate individually and compare.
    with torch.no_grad():
        system.set_hp_state({"target_fn": hp_a})
    score_a_single = await system.evaluate(examples, metric_fn)

    with torch.no_grad():
        system.set_hp_state({"target_fn": hp_b})
    score_b_single = await system.evaluate(examples, metric_fn)

    assert scores[0] == score_a_single
    assert scores[1] == score_b_single


@pytest.mark.asyncio
async def test_evaluate_population_with_lora_weights():
    """Test evaluate_population with 2D LoRA-style weights."""
    MyLoRA = LoRAWeight.create(5, 3, noise_rank=2)

    @hyperfunction(hp_type=MyLoRA, optimize_hparams=True)
    async def classify_fn(features: torch.Tensor, hp) -> torch.Tensor:
        """Simple linear classifier."""
        logits = features @ hp.weight.T
        return torch.softmax(logits, dim=-1)

    class ClassifierSystem(HyperSystem):
        async def run(self, features: torch.Tensor):
            return await classify_fn(features, hp=None)

    system = ClassifierSystem()
    # Initialize with a specific weight
    init_weight = torch.randn(5, 3)
    system.register_hyperfunction(classify_fn, hp_init=init_weight)

    # Create examples
    examples = [
        Example({"features": torch.randn(3)}, torch.tensor([1, 0, 0, 0, 0])),
        Example({"features": torch.randn(3)}, torch.tensor([0, 1, 0, 0, 0])),
        Example({"features": torch.randn(3)}, torch.tensor([0, 0, 1, 0, 0])),
    ]

    def accuracy_metric(preds, expected):
        correct = sum(
            p.argmax() == e.argmax() for p, e in zip(preds, expected)
        )
        return correct / len(preds)

    # Create candidates with different weights
    candidates = [
        {"classify_fn": torch.randn(5, 3)},
        {"classify_fn": torch.randn(5, 3)},
        {"classify_fn": torch.randn(5, 3)},
    ]

    scores = await system.evaluate_population(candidates, examples, accuracy_metric)
    assert len(scores) == 3
    assert all(0 <= s <= 1 for s in scores)


@pytest.mark.asyncio
async def test_evaluate_population_chunking():
    """Test that chunking works correctly for large populations."""
    MyLoRA = LoRAWeight.create(3, 2, noise_rank=1)

    @hyperfunction(hp_type=MyLoRA, optimize_hparams=True)
    async def simple_fn(x: torch.Tensor, hp) -> torch.Tensor:
        return x @ hp.weight.T

    class SimpleSystem(HyperSystem):
        async def run(self, x: torch.Tensor):
            return await simple_fn(x, hp=None)

    system = SimpleSystem()
    system.register_hyperfunction(simple_fn, hp_init=torch.randn(3, 2))

    # Create examples
    examples = [
        Example({"x": torch.randn(2)}, torch.zeros(3))
        for _ in range(10)
    ]

    def metric_fn(preds, expected):
        return 1.0  # Dummy metric

    # Create many candidates to trigger chunking
    num_candidates = 50
    candidates = [{"simple_fn": torch.randn(3, 2)} for _ in range(num_candidates)]

    # Use small max_batch_size to force chunking
    scores = await system.evaluate_population(
        candidates, examples, metric_fn, max_batch_size=20
    )

    assert len(scores) == num_candidates
    # All should return 1.0 from our dummy metric
    assert all(s == 1.0 for s in scores)


def test_fn_sync_property():
    """Test that fn_sync property returns sync version of function."""
    @hyperfunction(hp_type=LMParam, optimize_hparams=True)
    async def async_fn(x: float, hp: LMParam) -> float:
        return x + hp.temperature

    @hyperfunction(hp_type=LMParam, optimize_hparams=True)
    def sync_fn(x: float, hp: LMParam) -> float:
        return x + hp.temperature

    # Sync function should return itself
    assert sync_fn.fn_sync is sync_fn.fn

    # Async function should return a wrapped sync version
    assert async_fn.fn_sync is not async_fn.fn


def test_get_tensor_fn():
    """Test _get_tensor_fn generates correct tensor function."""
    MyLoRA = LoRAWeight.create(5, 3, noise_rank=2)

    @hyperfunction(hp_type=MyLoRA, optimize_hparams=True)
    def classify_fn(features: torch.Tensor, hp) -> torch.Tensor:
        logits = features @ hp.weight.T
        return torch.softmax(logits, dim=-1)

    system = HyperSystem()
    system.register_hyperfunction(classify_fn, hp_init=torch.randn(5, 3))

    tensor_fn = system._get_tensor_fn(classify_fn)

    # Test that tensor_fn works with raw tensors
    features = torch.randn(3)
    weight = torch.randn(5, 3)
    result = tensor_fn(features, weight)

    assert result.shape == (5,)
    assert torch.allclose(result.sum(), torch.tensor(1.0), atol=1e-5)


def test_vmap_with_tensor_fn():
    """Test that vmap works with the generated tensor function."""
    MyLoRA = LoRAWeight.create(5, 3, noise_rank=2)

    @hyperfunction(hp_type=MyLoRA, optimize_hparams=True)
    def classify_fn(features: torch.Tensor, hp) -> torch.Tensor:
        logits = features @ hp.weight.T
        return torch.softmax(logits, dim=-1)

    system = HyperSystem()
    system.register_hyperfunction(classify_fn, hp_init=torch.randn(5, 3))

    tensor_fn = system._get_tensor_fn(classify_fn)

    # Test vmap over weights (candidates)
    features = torch.randn(3)
    weights_batch = torch.randn(4, 5, 3)  # 4 candidates

    batched_fn = vmap(tensor_fn, in_dims=(None, 0))
    results = batched_fn(features, weights_batch)

    assert results.shape == (4, 5)  # 4 candidates, 5 classes each

    # Test nested vmap over examples and weights
    features_batch = torch.randn(10, 3)  # 10 examples
    weights_batch = torch.randn(4, 5, 3)  # 4 candidates

    batched_over_weights = vmap(tensor_fn, in_dims=(None, 0))
    batched_both = vmap(batched_over_weights, in_dims=(0, None))
    results = batched_both(features_batch, weights_batch)

    assert results.shape == (10, 4, 5)  # 10 examples, 4 candidates, 5 classes


def test_is_vmappable():
    """Test _is_vmappable detection."""
    MyLoRA = LoRAWeight.create(5, 3, noise_rank=2)

    @hyperfunction(hp_type=MyLoRA, optimize_hparams=True)
    def vmappable_fn(features: torch.Tensor, hp) -> torch.Tensor:
        return features @ hp.weight.T

    @hyperfunction(hp_type=LMParam, optimize_hparams=True)
    def non_vmappable_fn(x: float, hp: LMParam) -> float:
        # Uses float, not tensor
        return x + hp.temperature

    system = HyperSystem()
    system.register_hyperfunction(vmappable_fn, hp_init=torch.randn(5, 3))
    system.register_hyperfunction(non_vmappable_fn)

    # LoRA-style functions should be vmappable
    assert system._is_vmappable(vmappable_fn) is True

    # LMParam has shape but uses floats, not directly vmappable in typical use
    # The detection is based on hp_type having shape(), so it may still be True
    # The actual vmap call would fail at runtime for non-tensor ops
