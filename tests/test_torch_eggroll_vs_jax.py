import math

import pytest
import torch
from torch import nn

from hyperfunc.es import TorchEggrollES


jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")


class JaxQuadraticModel:
    """
    Minimal JAX "model" with a single parameter vector.
    Mirrors the behaviour of a simple nn.Module with one Parameter.
    """

    def __init__(self, dim: int, key):
        self.dim = dim
        self.key = key
        # Single parameter vector
        self.w = jnp.zeros((dim,), dtype=jnp.float32)


class JaxEggrollES:
    """
    Minimal JAX reference implementation of the TorchEggrollES algorithm,
    specialised to the simple JaxQuadraticModel above.
    """

    def __init__(
        self,
        model: JaxQuadraticModel,
        pop_size: int = 32,
        sigma: float = 0.02,
        lr: float = 0.05,
        normalize_fitness: bool = True,
    ):
        self.model = model
        self.pop_size = pop_size
        self.sigma = sigma
        self.lr = lr
        self.normalize_fitness = normalize_fitness

    def step(self, eval_fn):
        key = self.model.key
        pop_size = self.pop_size

        # Sample Gaussian noise for each population member
        key, sub = jax.random.split(key)
        noise = jax.random.normal(sub, shape=(pop_size, self.model.dim), dtype=jnp.float32)
        noise = noise * self.sigma

        def eval_member(i, carry):
            w_base = self.model.w
            w_perturbed = w_base + noise[i]
            # Temporarily set parameter
            old_w = self.model.w
            self.model.w = w_perturbed
            reward = eval_fn(self.model)
            # Restore parameter
            self.model.w = old_w
            carry = carry.at[i].set(reward)
            return carry

        rewards = jnp.zeros((pop_size,), dtype=jnp.float32)
        for i in range(pop_size):
            rewards = eval_member(i, rewards)

        if self.normalize_fitness:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        else:
            rewards = rewards - rewards.mean()

        grad = jnp.zeros_like(self.model.w)
        for i in range(pop_size):
            grad = grad + rewards[i] * noise[i]
        grad = grad / (self.sigma * pop_size)

        self.model.w = self.model.w + self.lr * grad
        self.model.key = key

        return float(rewards.mean())


def _torch_and_jax_quadratic(dim: int = 5):
    """
    Shared setup for a simple quadratic objective:
        fitness(w) = -||w - w_star||^2
    Higher fitness is better; optimum at w_star.
    """

    # Target vector
    w_star_np = jnp.linspace(-1.0, 1.0, dim, dtype=jnp.float32)

    # Torch model: single parameter
    class TorchModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.zeros(dim, dtype=torch.float32))

    torch_model = TorchModel()

    # JAX model
    key = jax.random.PRNGKey(0)
    jax_model = JaxQuadraticModel(dim=dim, key=key)

    def torch_eval_fn(_model: nn.Module) -> float:
        w = _model.w
        target = torch.tensor(w_star_np, dtype=torch.float32)
        return float(-torch.sum((w - target) ** 2).item())

    def jax_eval_fn(_model: JaxQuadraticModel) -> float:
        w = _model.w
        return float(-(jnp.sum((w - w_star_np) ** 2)).item())

    return torch_model, torch_eval_fn, jax_model, jax_eval_fn, w_star_np


def test_torch_eggroll_es_matches_jax_on_quadratic():
    """
    Cross-implementation parity test:
    TorchEggrollES and JaxEggrollES should achieve similar fitness
    on a simple quadratic objective, starting from the same initial point.
    """

    dim = 5
    steps = 30
    pop_size = 32
    sigma = 0.1
    lr = 0.1

    (
        torch_model,
        torch_eval_fn,
        jax_model,
        jax_eval_fn,
        w_star_np,
    ) = _torch_and_jax_quadratic(dim=dim)

    # Initialise both models at the same point (all zeros).
    with torch.no_grad():
        torch_model.w.zero_()
    jax_model.w = jnp.zeros_like(w_star_np)

    torch_es = TorchEggrollES(
        model=torch_model,
        pop_size=pop_size,
        sigma=sigma,
        lr=lr,
        rank=1,
        normalize_fitness=True,
    )

    jax_es = JaxEggrollES(
        model=jax_model,
        pop_size=pop_size,
        sigma=sigma,
        lr=lr,
        normalize_fitness=True,
    )

    # Run both optimisers for a few steps
    for _ in range(steps):
        torch_es.step(torch_eval_fn)
        jax_es.step(jax_eval_fn)

    # Compute final fitness for both
    torch_fitness = torch_eval_fn(torch_model)
    jax_fitness = jax_eval_fn(jax_model)

    # Both should be close to the optimum and to each other.
    # Absolute scale is modest, so we use relatively loose tolerances.
    assert torch_fitness > -0.5
    assert jax_fitness > -0.5
    assert math.isclose(torch_fitness, jax_fitness, rel_tol=0.3, abs_tol=0.3)

