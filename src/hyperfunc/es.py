from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, TYPE_CHECKING

import torch
from torch import nn

from .core import Example, HyperSystem, get_hp_noise_rank

if TYPE_CHECKING:
    from .core import ExecutionTrace


# ============================================================
# EggRoll-style ES: Low-rank noise for efficient gradient estimation
# ============================================================


def generate_lora_noise(
    param: torch.Tensor,
    rank: int,
    sigma: float,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Generate low-rank noise for a 2D parameter (matrix).

    For a matrix of shape (out_dim, in_dim), generates:
    - A: (out_dim, rank)
    - B: (in_dim, rank)
    - noise = A @ B.T * sigma / sqrt(rank)

    The division by sqrt(rank) normalizes the variance.
    """
    out_dim, in_dim = param.shape
    r = min(rank, out_dim, in_dim)

    # Use deterministic seeding for reproducibility (needed for antithetic sampling)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    # Generate A and B factors
    total_elements = out_dim + in_dim
    lora_params = torch.randn(total_elements, r, generator=gen, device=device, dtype=param.dtype)
    B = lora_params[:in_dim]  # (in_dim, r)
    A = lora_params[in_dim:]  # (out_dim, r)

    # noise = A @ B.T, scaled
    noise = (A @ B.t()) * (sigma / (r ** 0.5))
    return noise


def generate_standard_noise(
    param: torch.Tensor,
    sigma: float,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Generate standard Gaussian noise for non-matrix parameters.
    """
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    noise = torch.randn(param.shape, generator=gen, device=device, dtype=param.dtype)
    return noise * sigma


class TorchEggrollES:
    """
    Evolution Strategies trainer with EggRoll-style low-rank noise
    for 2D parameters (matrices).

    Key features:
    - Low-rank noise for matrices: reduces variance in gradient estimates
    - Antithetic sampling: half population uses +noise, half uses -noise
    - Proper ES gradient estimate: weighted average of noise by fitness

    Works with any nn.Module. Use param_filter to select which parameters
    to optimize (e.g., only LoRA weights).
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
        antithetic: bool = True,
    ) -> None:
        """
        Args:
            model: nn.Module to optimise.
            pop_size: population size per ES step. If antithetic=True, must be even.
            sigma: noise scale.
            lr: learning rate for ES update.
            rank: rank of low-rank noise for 2D params.
            device: device to run on; inferred from model if None.
            param_filter: optional (param, name) -> bool to choose which params
                          to include in ES updates.
            normalize_fitness: if True, z-score fitnesses; else only mean-center.
            antithetic: if True, use antithetic (mirrored) sampling for lower variance.
        """
        self.model = model
        self.pop_size = pop_size
        self.sigma = sigma
        self.lr = lr
        self.rank = rank
        self.normalize_fitness = normalize_fitness
        self.antithetic = antithetic

        if antithetic and pop_size % 2 != 0:
            raise ValueError("pop_size must be even when using antithetic sampling")

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

        # Base (mean) parameters - the center of our search distribution
        self.base_params: List[torch.Tensor] = [
            p.detach().clone().to(self.device) for p in self.params
        ]

        # Epoch counter for seeding
        self.epoch = 0

    def _refresh_base(self) -> None:
        """Copy current model params to base_params."""
        for base, p in zip(self.base_params, self.params):
            base.copy_(p.detach())

    def _generate_noise(self, param_idx: int, pop_idx: int) -> torch.Tensor:
        """
        Generate noise for a specific parameter and population member.

        With antithetic sampling:
        - pop_idx 0,1 share the same base noise (1 is negated)
        - pop_idx 2,3 share the same base noise (3 is negated)
        - etc.
        """
        base = self.base_params[param_idx]
        device = self.device

        if self.antithetic:
            # Pairs share noise seeds
            noise_idx = pop_idx // 2
            sign = 1.0 if pop_idx % 2 == 0 else -1.0
        else:
            noise_idx = pop_idx
            sign = 1.0

        # Create deterministic seed from epoch, noise_idx, and param_idx
        seed = hash((self.epoch, noise_idx, param_idx)) & 0x7FFFFFFF

        if base.ndim == 2:
            noise = generate_lora_noise(base, self.rank, self.sigma, seed, device)
        else:
            noise = generate_standard_noise(base, self.sigma, seed, device)

        return noise * sign

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
        device = self.device

        # Store noises for gradient computation
        noises: List[List[torch.Tensor]] = [
            [torch.zeros_like(base, device=device) for base in self.base_params]
            for _ in range(pop_size)
        ]
        fitnesses = torch.empty(pop_size, device=device, dtype=torch.float32)

        self.model.eval()
        with torch.no_grad():
            for i in range(pop_size):
                # 1) Apply noise to each parameter
                for j, (p, base) in enumerate(zip(self.params, self.base_params)):
                    noise = self._generate_noise(j, i)
                    p.data = base + noise
                    noises[i][j] = noise

                # 2) Evaluate fitness
                fitness = eval_fn(self.model)
                fitnesses[i] = float(fitness)

        # 3) Normalize fitnesses (z-score or mean-center)
        rewards = fitnesses
        if self.normalize_fitness:
            std = rewards.std()
            if std > 1e-8:
                rewards = (rewards - rewards.mean()) / std
            else:
                rewards = rewards - rewards.mean()
        else:
            rewards = rewards - rewards.mean()

        # 4) ES gradient estimate and update
        # grad ≈ (1/N) * Σ fitness_i * noise_i
        # Then we scale by sqrt(N) as in HyperscaleES
        with torch.no_grad():
            for j, base in enumerate(self.base_params):
                grad = torch.zeros_like(base, device=device)
                for i in range(pop_size):
                    grad.add_(rewards[i] * noises[i][j])
                grad /= pop_size
                # Scale by sqrt(pop_size) as in HyperscaleES
                grad *= (pop_size ** 0.5)

                # Apply update
                base.add_(self.lr * grad)
                self.params[j].data.copy_(base)

        self.epoch += 1
        return float(fitnesses.mean().item())


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
