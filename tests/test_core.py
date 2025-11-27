import asyncio
import torch
import pytest
from hyperfunc import Example, HyperSystem, LMParam, hyperfunction


def test_lmparam_roundtrip():
    # Basic sanity check for LMParam's torch helpers.
    p = LMParam(temperature=0.5)
    t = p.to_tensor()
    assert t.shape == (5,)  # 5 ES-optimizable params
    assert t[0] == 0.5

    p2 = LMParam.from_tensor(t)
    assert p2.temperature == pytest.approx(0.5)
    # We can't use direct equality because of float precision
    assert p2.top_p == pytest.approx(p.top_p)


@pytest.mark.asyncio
async def test_hyperfunction_decorator():
    @hyperfunction(hp_type=LMParam)
    async def my_fn(x: str, hp: LMParam) -> str:
        """My prompt"""
        return f"{x}-{hp.temperature}"

    assert my_fn.fn.hyper_hp_type == LMParam
    assert my_fn.fn.hyper_prompt == "My prompt"

    # Test direct call with explicit hp
    res = await my_fn("hello", hp=LMParam(temperature=0.7))
    assert res == "hello-0.7"


@pytest.mark.asyncio
async def test_system_wiring():
    @hyperfunction(hp_type=LMParam, optimize_hparams=True)
    async def fn1(x: int, hp: LMParam) -> float:
        return x * hp.temperature

    system = HyperSystem()
    system.register_hyperfunction(fn1)

    # Check hp size
    assert system.hp_dim == 5  # 5 ES-optimizable params
    state = system.get_hp_state()
    assert set(state.keys()) == {"fn1"}
    hp = state["fn1"]
    assert hp.shape == (5,)

    # Initialize hp to something specific
    with torch.no_grad():
        hp[0] = 2.0  # temperature
        system.set_hp_state({"fn1": hp})

    # Call function without hp - should inject from system
    res = await fn1(10)
    assert res == 20.0
