import pytest
import torch
from hyperfunc import Example, HyperSystem, LMParam, ESHybridSystemOptimizer, hyperfunction


@pytest.mark.asyncio
async def test_es_optimization():
    # Define a function where "temperature" is the target value we want to find
    # Let's say we want temperature to be 0.5

    @hyperfunction(hp_type=LMParam, optimize_hparams=True)
    async def target_fn(x: float, hp: LMParam) -> float:
        # We return the distance from 0.5
        return -abs(hp.temperature - 0.5)

    # Create a custom system that routes inputs to target_fn
    class TestSystem(HyperSystem):
        async def run(self, x: float):
            return await target_fn(x)

    # Data is irrelevant for this dummy function, but required by API
    # New Example format: just inputs (kwargs to run) and expected output
    train_data = [Example({"x": 0.0}, 0.0)]

    def metric_fn(preds, expected):
        # preds contains the return value of target_fn (negative distance)
        # We want to maximize this (closest to 0)
        return sum(preds) / len(preds)

    system = TestSystem(
        system_optimizer=ESHybridSystemOptimizer(
            steps=20,
            pop_size=20,
            sigma=0.1,
            lr=0.1,
        ),
    )
    system.register_hyperfunction(target_fn)

    # Initialize far from 0.5
    with torch.no_grad():
        state = system.get_hp_state()
        hp = state["target_fn"]
        hp[0] = 0.0
        system.set_hp_state({"target_fn": hp})

    initial_score = await system.evaluate(train_data, metric_fn)
    assert initial_score == -0.5

    await system.optimize(train_data, metric_fn)

    final_score = await system.evaluate(train_data, metric_fn)
    print(f"Initial: {initial_score}, Final: {final_score}")

    # Should have improved
    assert final_score > initial_score
    # Should be close to 0 (max score)
    assert final_score > -0.1
