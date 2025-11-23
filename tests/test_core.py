import torch
from hyperfunc import (
    Example,
    HyperParam,
    HyperSystem,
    LMParam,
    hyperfunction,
)


def test_hyperparam_protocol():
    # Test LMParam
    p = LMParam(temperature=0.5)
    t = p.to_tensor()
    assert t.shape == (6,)
    assert t[0] == 0.5
    
    import pytest
    p2 = LMParam.from_tensor(t)
    assert p2.temperature == pytest.approx(0.5)
    # We can't use direct equality because of float precision
    assert p2.top_p == pytest.approx(p.top_p)


def test_hyperfunction_decorator():
    @hyperfunction(model="test", hp_type=LMParam)
    def my_fn(x: str, hp: LMParam) -> str:
        """My prompt"""
        return f"{x}-{hp.temperature}"

    assert my_fn.fn.hyper_model_name == "test"
    assert my_fn.fn.hyper_hp_type == LMParam
    assert my_fn.fn.hyper_prompt == "My prompt"
    
    # Test direct call with explicit hp
    res = my_fn("hello", hp=LMParam(temperature=0.7))
    assert res == "hello-0.7"


def test_system_wiring():
    @hyperfunction(model="test", hp_type=LMParam, optimize_hparams=True)
    def fn1(x: int, hp: LMParam) -> float:
        return x * hp.temperature

    system = HyperSystem([fn1])
    
    # Check model hp size
    assert system.model.total_dim == 6
    assert system.model.hp.shape == (6,)
    
    # Initialize hp to something specific
    with torch.no_grad():
        system.model.hp[0] = 2.0 # temperature
        
    # Call function without hp - should inject from system
    res = fn1(10)
    assert res == 20.0
