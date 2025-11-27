"""Evolution Strategies optimizers for hyperfunc.

This module re-exports TorchEggrollES from the torcheggroll package and provides
ESHybridSystemOptimizer for system-level optimization with HyperSystem.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, TYPE_CHECKING

import torch
from torch import nn

# Re-export from torcheggroll
from torcheggroll import (
    TorchEggrollES,
    generate_lora_noise,
    generate_standard_noise,
)

from .core import Example, HyperSystem, get_hp_noise_rank

if TYPE_CHECKING:
    from .core import ExecutionTrace


# ============================================================
# Hybrid System Optimizer: ES for hp + TorchEggrollES for GPU params
# ============================================================


@dataclass
class ESHybridSystemOptimizer:
    """
    Hybrid system-level optimizer that uses:
    - Simple ES for CPU-based hyperparameters (LMParam blocks)
    - TorchEggrollES for GPU-resident parameters (LoRA weights, small models)

    For hyperfunctions with local_gpu=True, uses TorchEggrollES with low-rank
    noise for efficient gradient estimation on matrices.

    For other hyperfunctions, uses standard ES with Gaussian noise on the
    1D hp vectors, with proper gradient-based updates (not hill-climbing).
    """

    steps: int = 50
    pop_size: int = 32
    sigma: float = 0.05
    lr: float = 0.1
    rank: int = 4
    device: Optional[torch.device] = None
    normalize_fitness: bool = True
    antithetic: bool = True

    # GPU model optimizers, keyed by hyperfunction name
    _gpu_optimizers: Dict[str, TorchEggrollES] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        if self.antithetic and self.pop_size % 2 != 0:
            raise ValueError("pop_size must be even when using antithetic sampling")

    def _register_gpu_model(
        self,
        name: str,
        model: nn.Module,
        param_filter: Optional[Callable[[nn.Parameter, str], bool]] = None,
    ) -> None:
        """
        Register a GPU-resident model for TorchEggrollES optimization.

        Call this for each hyperfunction that has local_gpu=True and owns
        a PyTorch model you want to optimize.
        """
        self._gpu_optimizers[name] = TorchEggrollES(
            model=model,
            pop_size=self.pop_size,
            sigma=self.sigma,
            lr=self.lr,
            rank=self.rank,
            device=self.device,
            param_filter=param_filter,
            normalize_fitness=self.normalize_fitness,
            antithetic=self.antithetic,
        )

    def _generate_hp_noise(
        self,
        base_tensor: torch.Tensor,
        pop_idx: int,
        param_idx: int,
        epoch: int,
        noise_rank: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate noise for an hp tensor with optional antithetic sampling.

        For 2D tensors with noise_rank specified, uses EggRoll-style low-rank noise:
        noise = A @ B.T * sigma / sqrt(rank)

        For 1D tensors or when noise_rank is None, uses standard Gaussian noise.
        """
        device = base_tensor.device

        if self.antithetic:
            noise_idx = pop_idx // 2
            sign = 1.0 if pop_idx % 2 == 0 else -1.0
        else:
            noise_idx = pop_idx
            sign = 1.0

        # Deterministic seed
        seed = hash((epoch, noise_idx, param_idx)) & 0x7FFFFFFF

        # Use low-rank noise for 2D tensors with noise_rank specified
        if base_tensor.ndim == 2 and noise_rank is not None:
            noise = generate_lora_noise(
                base_tensor,
                rank=noise_rank,
                sigma=self.sigma,
                seed=seed,
                device=device,
            )
        else:
            noise = generate_standard_noise(
                base_tensor,
                sigma=self.sigma,
                seed=seed,
                device=device,
            )

        return noise * sign

    async def optimize(
        self,
        system: HyperSystem,
        train_data: Sequence[Example],
        metric_fn: Callable[[List[Any], List[Any]], float],
        traces: Optional[List["ExecutionTrace"]] = None,
    ) -> None:
        """
        Run ES optimization over the system's hp blocks.

        Uses proper gradient-based ES update (not hill-climbing):
        - Sample population with Gaussian noise (antithetic if enabled)
        - For 2D params with noise_rank, uses EggRoll-style low-rank noise
        - Evaluate all candidates
        - Compute gradient as weighted average of noise by normalized fitness
        - Update parameters with learning rate

        If traces are provided, uses them for staged batch execution.
        """
        if system.hp_dim == 0:
            return

        # Get parameter names and their noise_ranks from hp_types
        base_state = system.get_hp_state()
        param_names = list(base_state.keys())

        # Build a map of name -> noise_rank from hyperfunction hp_types
        noise_ranks: Dict[str, Optional[int]] = {}
        for name in param_names:
            hf = system.get_hyperfunction(name)
            noise_ranks[name] = get_hp_noise_rank(hf.hp_type)

        with torch.no_grad():
            for epoch in range(self.steps):
                # 1) Generate population with noise
                hp_candidates: List[Dict[str, torch.Tensor]] = []
                noises: List[Dict[str, torch.Tensor]] = []

                for pop_idx in range(self.pop_size):
                    cand_state: Dict[str, torch.Tensor] = {}
                    noise_state: Dict[str, torch.Tensor] = {}

                    for param_idx, name in enumerate(param_names):
                        base_tensor = base_state[name]
                        noise = self._generate_hp_noise(
                            base_tensor,
                            pop_idx,
                            param_idx,
                            epoch,
                            noise_rank=noise_ranks[name],
                        )
                        cand_state[name] = base_tensor + noise
                        noise_state[name] = noise

                    hp_candidates.append(cand_state)
                    noises.append(noise_state)

                # 2) Evaluate all candidates (async)
                rewards_list = await system.evaluate_population(
                    hp_candidates,
                    train_data,
                    metric_fn,
                    traces=traces,
                )
                rewards = torch.tensor(rewards_list, dtype=torch.float32)

                # 3) Normalize rewards
                if self.normalize_fitness:
                    std = rewards.std()
                    if std > 1e-8:
                        rewards = (rewards - rewards.mean()) / std
                    else:
                        rewards = rewards - rewards.mean()
                else:
                    rewards = rewards - rewards.mean()

                # 4) Compute gradient and update
                for name in param_names:
                    base_tensor = base_state[name]
                    grad = torch.zeros_like(base_tensor)

                    for pop_idx in range(self.pop_size):
                        grad.add_(rewards[pop_idx] * noises[pop_idx][name])

                    grad /= self.pop_size
                    grad *= (self.pop_size ** 0.5)  # Scale as in HyperscaleES

                    # Update base state
                    base_state[name] = base_tensor + self.lr * grad

            # Apply final state to system
            system.set_hp_state(base_state)
