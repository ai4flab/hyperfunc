import torch
from hyperfunc import (
    Example,
    HyperSystem,
    LMParam,
    TorchEggrollSystemOptimizer,
    hyperfunction,
)


def test_es_optimization():
    # Define a function where "temperature" is the target value we want to find
    # Let's say we want temperature to be 0.5
    
    @hyperfunction(model="test", hp_type=LMParam, optimize_hparams=True)
    def target_fn(x: float, hp: LMParam) -> float:
        # We return the distance from 0.5
        return -abs(hp.temperature - 0.5)

    # Data is irrelevant for this dummy function, but required by API
    train_data = [Example("target_fn", {"x": 0.0}, 0.0)]
    
    def metric_fn(preds, expected):
        # preds contains the return value of target_fn (negative distance)
        # We want to maximize this (closest to 0)
        return sum(preds) / len(preds)

    system = HyperSystem(
        [target_fn],
        system_optimizer=TorchEggrollSystemOptimizer(
            steps=20, pop_size=20, sigma=0.1, lr=0.1
        )
    )
    
    # Initialize far from 0.5
    with torch.no_grad():
        system.model.hp[0] = 0.0
        
    initial_score = system.eval_on_examples(train_data, metric_fn)
    assert initial_score == -0.5
    
    system.optimize(train_data, metric_fn)
    
    final_score = system.eval_on_examples(train_data, metric_fn)
    print(f"Initial: {initial_score}, Final: {final_score}")
    
    # Should have improved
    assert final_score > initial_score
    # Should be close to 0 (max score)
    assert final_score > -0.1
