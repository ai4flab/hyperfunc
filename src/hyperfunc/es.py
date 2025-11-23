from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence

import torch
from torch import nn

from .core import Example, HyperSystem


# ============================================================
# 4. ES core: TorchEggrollES (EGGROLL-style in PyTorch)
# ============================================================


class TorchEggrollES:
    """
    Evolution Strategies trainer with EGGROLL-style low-rank noise
    for 2D parameters.

    Works with any nn.Module. We will use it on HyperModel (hp vector),
    or on a real model + param_filter to restrict to certain parameters
    (e.g. LoRA weights).
    """

    def __init__(
        self,
        model: nn.Module,
        pop_size: int = 32,
        sigma: float = 0.02,
        lr: float = 0.05,
        rank: int = 4,
        device: Optional[torch.device] = None,
        param_filter: Optional[Callable[[nn.Parameter, str], bool]] = None,
        normalize_fitness: bool = True,
    ) -> None:
        """
        Args:
            model: nn.Module to optimise.
            pop_size: population size per ES step.
            sigma: noise scale.
            lr: learning rate in ES update.
            rank: rank of low-rank noise for 2D params.
            device: device to run on; inferred from model if None.
            param_filter: optional (param, name) -> bool to choose which params
                          to include in ES updates.
            normalize_fitness: if True, z-score rewards; else only mean-center.
        """
        self.model = model
        self.pop_size = pop_size
        self.sigma = sigma
        self.lr = lr
        self.rank = rank
        self.normalize_fitness = normalize_fitness

        if device is None:
            try:
                device = next(model.parameters()).device
            except StopIteration:
                device = torch.device("cpu")
        self.device = torch.device(device)
        self.model.to(self.device)

        # Gather parameters (and names) to evolve
        self.params: List[nn.Parameter] = []
        self.names: List[str] = []
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if param_filter is not None and not param_filter(p, name):
                continue
            self.params.append(p)
            self.names.append(name)

        if not self.params:
            raise ValueError("TorchEggrollES: no parameters selected for ES.")

        # Base (mean) parameters
        self.base_params: List[torch.Tensor] = [
            p.detach().clone().to(self.device) for p in self.params
        ]

    def _refresh_base(self) -> None:
        for base, p in zip(self.base_params, self.params):
            base.copy_(p.detach())

    def step(self, eval_fn: Callable[[nn.Module], float]) -> float:
        """
        One ES step.

        eval_fn should:
            - run the model (or system around it),
            - return a scalar fitness (higher is better).

        Returns mean fitness across the population (for logging).
        """
        self._refresh_base()
        pop_size = self.pop_size
        sigma = self.sigma
        device = self.device

        # noises[i][j] = noise for population i, parameter j
        noises: List[List[torch.Tensor]] = [
            [torch.zeros_like(base, device=device) for base in self.base_params]
            for _ in range(pop_size)
        ]
        fitnesses = torch.empty(pop_size, device=device, dtype=torch.float32)

        self.model.eval()
        with torch.no_grad():
            for i in range(pop_size):
                # 1) Apply noise
                for j, (p, base) in enumerate(zip(self.params, self.base_params)):
                    if base.ndim == 2:
                        out_dim, in_dim = base.shape
                        r = min(self.rank, out_dim, in_dim)
                        A = torch.randn(out_dim, r, device=device)
                        B = torch.randn(in_dim, r, device=device)
                        noise = A @ B.t()
                    else:
                        noise = torch.randn_like(base, device=device)

                    noise = noise * sigma
                    p.data = base + noise
                    noises[i][j] = noise

                # 2) Evaluate fitness
                fitness = eval_fn(self.model)
                fitnesses[i] = float(fitness)

        # 3) Normalise rewards
        rewards = fitnesses
        if self.normalize_fitness:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        else:
            rewards = rewards - rewards.mean()

        # 4) ES gradient estimate and update
        with torch.no_grad():
            for j, base in enumerate(self.base_params):
                grad = torch.zeros_like(base, device=device)
                for i in range(pop_size):
                    grad.add_(rewards[i] * noises[i][j])
                grad /= (sigma * pop_size)

                base.add_(self.lr * grad)
                self.params[j].data.copy_(base)

        return float(fitnesses.mean().item())


# ============================================================
# 5. System optimizer using TorchEggrollES
# ============================================================


@dataclass
class TorchEggrollSystemOptimizer:
    """
    System-level optimizer that runs TorchEggrollES on HyperModel.hp.

    It doesn't know about HyperParam types; it just mutates model.hp and lets
    HyperSystem.eval_on_examples compute a scalar reward.
    """

    steps: int = 50
    pop_size: int = 32
    sigma: float = 0.05
    lr: float = 0.1
    rank: int = 4
    device: Optional[torch.device] = None

    def optimize(
        self,
        system: HyperSystem,
        train_data: Sequence[Example],
        metric_fn: Callable[[List[Any], List[Any]], float],
    ) -> None:
        # Our "model" is just the HyperModel (with hp parameter)
        model = system.model

        # We only want to evolve the hp parameter
        def param_filter(p: nn.Parameter, name: str) -> bool:
            return name == "hp"

        es = TorchEggrollES(
            model=model,
            pop_size=self.pop_size,
            sigma=self.sigma,
            lr=self.lr,
            rank=self.rank,
            device=self.device,
            param_filter=param_filter,
        )

        def eval_fn(m: nn.Module) -> float:
            # m is the HyperModel; hp is already updated by ES.
            # HyperFunctions see hp via their slices on system.model.hp.
            return system.eval_on_examples(train_data, metric_fn)

        for _ in range(self.steps):
            es.step(eval_fn)
