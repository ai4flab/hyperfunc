Here’s a clean v0 of the “HyperFunctions” core you can drop straight into a repo, plus a short write-up of the architecture so you can extend it later (agents, more HyperParam types, real GEPA integration, etc.).

---

## High-level design

**Goal:** declarative, FunctAI-style hyperfunctions that can be optimised on two axes:

1. **Prompt optimisation (GEPA-style)**
   – mutate docstrings (prompts) to improve task metrics.

2. **System / hyperparameter optimisation (Eggroll-style ES)**
   – mutate typed hyperparameters (`HyperParam`s) and/or LoRA/etc. to improve **end-to-end** metrics.

### Core concepts

* **`HyperParam` (Protocol)**
  A *shape* for anything optimisable as a vector:

  * `dim() -> int`
  * `to_tensor() -> torch.Tensor`
  * `from_tensor(t: torch.Tensor) -> instance`

  You define your own types (`LMParam`, `AdapterMeta`, `AgentParam`, …) as plain dataclasses that implement these three methods. No base class needed.

* **`HyperFunction`**
  A decorated function:

  ```python
  @hyperfunction(model="qwen-7b", hp_type=LMParam, optimize_prompt=True, optimize_hparams=True)
  def extract_invoice(doc: str, hp: LMParam) -> Invoice:
      """docstring is the prompt..."""
      ...
  ```

  * Docstring = prompt (optimised by GEPA).
  * `hp` = typed `HyperParam` object (backed by a slice of a global `hp` tensor).
  * When you call `extract_invoice("...")`, the framework auto-injects `hp` constructed from the latest global `hp` tensor.

* **`HyperModel` (nn.Module)**

  * Holds a single parameter `hp: nn.Parameter` (1D tensor).
  * Concatenation of all `HyperFunction` hyperparameters.
  * Each `HyperFunction` knows its slice into this vector.

* **`HyperSystem`**

  * Owns:

    * a list of `HyperFunction`s,
    * the `HyperModel`,
    * a `PromptOptimizer` (GEPA backend stub),
    * a `SystemOptimizer` (ES backend using `TorchEggrollES`).
  * Provides:

    * `eval_on_examples(examples, metric_fn)` – run HyperFunctions on data and compute metric.
    * `optimize(train_data, metric_fn)` – run prompt optimisation then ES optimisation.

* **`TorchEggrollES`**

  * ES trainer for any `nn.Module`:

    * low-rank Gaussian noise for 2D params (EGGROLL-style),
    * standard Gaussian for 1D/scalars,
    * ES gradient estimate from fitness,
    * updates base params in place.

* **`TorchEggrollSystemOptimizer`**

  * Wraps `TorchEggrollES` to optimise `HyperModel.hp` using `HyperSystem.eval_on_examples`.

* **`PromptOptimizer` / `GEPAPromptOptimizer` (stub)**

  * Protocol + no-op implementation in this file.
  * Real GEPA integration: you’ll plug in `gepa` and implement `optimize(system, train_data, metric_fn)` using `hf.get_prompt()/set_prompt()` and `system.eval_on_examples`.

---

## Single-file library: `hyper_core.py`

```python
# hyper_core.py
from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Sequence,
    Type,
    TypeVar,
)

import torch
from torch import nn


# ============================================================
# 1. HyperParam protocol and standard param types
# ============================================================

HP = TypeVar("HP", bound="HyperParam")


class HyperParam(Protocol):
    """
    Structural protocol for any optimisable hyperparameter object.

    Anything that implements these three methods is a HyperParam:
    - dim(): how many scalars
    - to_tensor(): flatten to a 1D tensor
    - from_tensor(): reconstruct from a 1D tensor
    """

    @classmethod
    def dim(cls) -> int:
        ...

    def to_tensor(self, device=None, dtype=torch.float32) -> torch.Tensor:
        ...

    @classmethod
    def from_tensor(cls: Type[HP], t: torch.Tensor) -> HP:
        ...


# ---- Example: standard LLM parameters --------------------------------------


@dataclass
class LMParam:
    """
    Standard "LLM behavior" hyperparameters.

    Interpretation convention:

    - temperature: 0.0 – 1.5
    - top_p: 0.0 – 1.0
    - top_k_norm: 0.0 – 1.0 (mapped to top_k in some range, e.g. [1, 1024])
    - length_factor: 0.5 – 2.0 (scales max_tokens)
    - repetition_penalty: 0.0 – 2.0 (1.0 = none)
    - strictness: 0.0 – 1.0 (0 = relaxed, 1 = ultra strict about format)
    """

    temperature: float = 0.1
    top_p: float = 0.9
    top_k_norm: float = 1.0
    length_factor: float = 1.0
    repetition_penalty: float = 1.0
    strictness: float = 1.0

    @classmethod
    def dim(cls) -> int:
        return 6

    def to_tensor(self, device=None, dtype=torch.float32) -> torch.Tensor:
        vals = [
            self.temperature,
            self.top_p,
            self.top_k_norm,
            self.length_factor,
            self.repetition_penalty,
            self.strictness,
        ]
        return torch.tensor(vals, device=device, dtype=dtype)

    @classmethod
    def from_tensor(cls, t: torch.Tensor) -> "LMParam":
        t = t.view(-1)
        assert t.numel() == cls.dim()
        return cls(
            temperature=float(t[0]),
            top_p=float(t[1]),
            top_k_norm=float(t[2]),
            length_factor=float(t[3]),
            repetition_penalty=float(t[4]),
            strictness=float(t[5]),
        )


# ---- Example: adapter meta-parameters for LoRA / heads ---------------------


@dataclass
class AdapterMeta:
    """
    Knobs around an adapter (e.g., LoRA), not the weights themselves.

    Interpretation convention:

    - scale: how strongly adapter output is mixed in (0–2)
    - dropout_keep: 0–1, probability of *keeping* activations in adapter block
    - gate_bias: bias term for logistic gate controlling adapter usage
    - priority: 0–1, for arbitration across multiple adapters
    """

    scale: float = 1.0
    dropout_keep: float = 1.0
    gate_bias: float = 0.0
    priority: float = 0.5

    @classmethod
    def dim(cls) -> int:
        return 4

    def to_tensor(self, device=None, dtype=torch.float32) -> torch.Tensor:
        vals = [self.scale, self.dropout_keep, self.gate_bias, self.priority]
        return torch.tensor(vals, device=device, dtype=dtype)

    @classmethod
    def from_tensor(cls, t: torch.Tensor) -> "AdapterMeta":
        t = t.view(-1)
        assert t.numel() == cls.dim()
        return cls(
            scale=float(t[0]),
            dropout_keep=float(t[1]),
            gate_bias=float(t[2]),
            priority=float(t[3]),
        )


# You can define more HyperParam types in your own code:
#
#   @dataclass
#   class AgentParam:
#       tool_thresh: float = 0.5
#       human_bias: float = 0.0
#       max_steps_norm: float = 0.5
#
#       @classmethod
#       def dim(cls) -> int: ...
#       def to_tensor(self, ...): ...
#       @classmethod
#       def from_tensor(cls, t): ...


# ============================================================
# 2. Core: Example, HyperFunction, decorator
# ============================================================


@dataclass
class Example:
    """
    One supervised training example for a HyperFunction.
    """
    fn_name: str
    inputs: Dict[str, Any]
    expected: Any


class HyperFunction:
    """
    Wrapper around a user-defined function:

        @hyperfunction(model="...", hp_type=LMParam)
        def my_fn(x: str, hp: LMParam) -> Output:
            \"\"\"Prompt lives here\"\"\"
            ...

    - Docstring is treated as the prompt (for GEPA).
    - hp_type (if provided) is a HyperParam-like type.
    - When you call it without hp, we auto-inject an instance
      reconstructed from the global HyperModel.hp tensor.
    """

    def __init__(
        self,
        fn: Callable[..., Any],
        model: str,
        hp_type: Optional[Type[HyperParam]],
        optimize_prompt: bool,
        optimize_hparams: bool,
    ) -> None:
        self.fn = fn
        self.name = fn.__name__
        self.model_name = model
        self.hp_type = hp_type
        self.optimize_prompt = optimize_prompt
        self.optimize_hparams = optimize_hparams

        self.prompt: str = (fn.__doc__ or "").strip()

        sig = inspect.signature(fn)
        self._sig = sig

        self._hp_param_name: Optional[str] = None
        if hp_type is not None:
            for name, p in sig.parameters.items():
                if name == "hp" and p.annotation is hp_type:
                    self._hp_param_name = name
                    break

        self.hp_slice: Optional[slice] = None
        self.system: Optional["HyperSystem"] = None

    @property
    def hp_dim(self) -> int:
        if self.hp_type is None:
            return 0
        return self.hp_type.dim()  # type: ignore[call-arg]

    # wiring

    def attach_system(self, system: "HyperSystem") -> None:
        self.system = system

    def set_hp_slice(self, s: slice) -> None:
        self.hp_slice = s

    # prompt view for prompt optimizers

    def get_prompt(self) -> str:
        return self.prompt

    def set_prompt(self, text: str) -> None:
        self.prompt = text
        # keep the underlying function docstring in sync (optional)
        self.fn.__doc__ = text

    # call interface

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        bound = self._sig.bind_partial(*args, **kwargs)
        bound.apply_defaults()

        # Inject hp if:
        # - user didn't supply it,
        # - we have hp_type, hp_slice, and a system.
        if (
            self._hp_param_name is not None
            and self.hp_type is not None
            and self.hp_slice is not None
            and self.system is not None
            and self._hp_param_name not in bound.arguments
        ):
            vec = self.system.model.hp[self.hp_slice]  # type: ignore[index]
            hp_obj = self.hp_type.from_tensor(vec)  # type: ignore[call-arg]
            bound.arguments[self._hp_param_name] = hp_obj

        return self.fn(**bound.arguments)


def hyperfunction(
    *,
    model: str,
    hp_type: Optional[Type[HyperParam]] = None,
    optimize_prompt: bool = True,
    optimize_hparams: bool = False,
) -> Callable[[Callable[..., Any]], HyperFunction]:
    """
    Decorator turning a plain Python function into a HyperFunction.

    - model: model identifier (e.g. "qwen-7b", "deepseek-v2", etc.)
    - hp_type: a class implementing HyperParam protocol (or None)
    - optimize_prompt: allow prompt optimisation (GEPA)
    - optimize_hparams: allow ES optimisation of hp
    """

    def wrapper(fn: Callable[..., Any]) -> HyperFunction:
        hf = HyperFunction(
            fn=fn,
            model=model,
            hp_type=hp_type,
            optimize_prompt=optimize_prompt,
            optimize_hparams=optimize_hparams,
        )
        # Convenience attributes on the original function, FunctAI-style.
        fn.hyper_model_name = model       # type: ignore[attr-defined]
        fn.hyper_hp_type = hp_type        # type: ignore[attr-defined]
        fn.hyper_prompt = hf.prompt       # type: ignore[attr-defined]
        return hf

    return wrapper


# ============================================================
# 3. HyperModel and HyperSystem
# ============================================================

import inspect  # needs to be after __future__ in this file


class HyperModel(nn.Module):
    """
    A single nn.Module holding a global hp vector.

    - hp: nn.Parameter of shape [total_dim]
    - each HyperFunction gets a slice into this vector
    """

    def __init__(self, hyperfunctions: Sequence[HyperFunction]):
        super().__init__()
        self.hyperfunctions = list(hyperfunctions)

        total_dim = 0
        for hf in self.hyperfunctions:
            if hf.optimize_hparams and hf.hp_dim > 0:
                start = total_dim
                end = total_dim + hf.hp_dim
                hf.set_hp_slice(slice(start, end))
                total_dim = end
            else:
                hf.set_hp_slice(slice(0, 0))

        self.total_dim = total_dim
        if total_dim == 0:
            self.hp = nn.Parameter(torch.empty(0), requires_grad=False)
        else:
            # Initialise hp to zeros; you can inject your own init later.
            self.hp = nn.Parameter(torch.zeros(total_dim))


class PromptOptimizer(Protocol):
    def optimize(
        self,
        system: "HyperSystem",
        train_data: Sequence[Example],
        metric_fn: Callable[[List[Any], List[Any]], float],
    ) -> None:
        ...


class SystemOptimizer(Protocol):
    def optimize(
        self,
        system: "HyperSystem",
        train_data: Sequence[Example],
        metric_fn: Callable[[List[Any], List[Any]], float],
    ) -> None:
        ...


@dataclass
class NoOpPromptOptimizer:
    def optimize(
        self,
        system: "HyperSystem",
        train_data: Sequence[Example],
        metric_fn: Callable[[List[Any], List[Any]], float],
    ) -> None:
        return


@dataclass
class NoOpSystemOptimizer:
    def optimize(
        self,
        system: "HyperSystem",
        train_data: Sequence[Example],
        metric_fn: Callable[[List[Any], List[Any]], float],
    ) -> None:
        return


class HyperSystem:
    """
    Ties everything together:

    - a set of HyperFunctions,
    - a global HyperModel (hp vector),
    - a prompt optimizer (GEPA-style),
    - a system optimizer (ES / TorchEggrollES).

    You call:

        system = HyperSystem([...], prompt_optimizer, system_optimizer)
        system.optimize(train_data, metric_fn)

    After that, calling the HyperFunctions will use tuned prompts + tuned hp.
    """

    def __init__(
        self,
        hyperfunctions: Sequence[HyperFunction],
        prompt_optimizer: Optional[PromptOptimizer] = None,
        system_optimizer: Optional[SystemOptimizer] = None,
    ) -> None:
        self.hyperfunctions = list(hyperfunctions)
        self.model = HyperModel(self.hyperfunctions)

        for hf in self.hyperfunctions:
            hf.attach_system(self)

        self._by_name: Dict[str, HyperFunction] = {
            hf.name: hf for hf in self.hyperfunctions
        }

        self.prompt_optimizer = prompt_optimizer
        self.system_optimizer = system_optimizer

    def eval_on_examples(
        self,
        examples: Sequence[Example],
        metric_fn: Callable[[List[Any], List[Any]], float],
    ) -> float:
        preds: List[Any] = []
        expected: List[Any] = []

        for ex in examples:
            hf = self._by_name[ex.fn_name]
            out = hf(**ex.inputs)
            preds.append(out)
            expected.append(ex.expected)

        return metric_fn(preds, expected)

    def optimize(
        self,
        train_data: Sequence[Example],
        metric_fn: Callable[[List[Any], List[Any]], float],
    ) -> None:
        # 1) Prompt optimisation
        if self.prompt_optimizer is not None:
            self.prompt_optimizer.optimize(self, train_data, metric_fn)

        # 2) Hyperparameter / system optimisation
        if self.system_optimizer is not None and self.model.total_dim > 0:
            self.system_optimizer.optimize(self, train_data, metric_fn)


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


# ============================================================
# 6. Prompt optimizer stub (GEPA hook)
# ============================================================


@dataclass
class GEPAPromptOptimizer:
    """
    Stub for a real GEPA-based prompt optimizer.

    To make this real:
    - add `gepa` to your dependencies,
    - implement this.optimize(...) using:
        - hf.get_prompt() / hf.set_prompt()
        - system.eval_on_examples(...) for metric,
        - GEPA's search over prompt candidates.

    For now, this just acts as NoOp.
    """

    def optimize(
        self,
        system: HyperSystem,
        train_data: Sequence[Example],
        metric_fn: Callable[[List[Any], List[Any]], float],
    ) -> None:
        # TODO: integrate GEPA here.
        return
```

---

## How to use this in practice

### 1. Define your HyperFunctions

Example for a JSON extractor:

```python
from pydantic import BaseModel
from hyper_core import hyperfunction, Example, HyperSystem, LMParam, TorchEggrollSystemOptimizer, NoOpPromptOptimizer


class Invoice(BaseModel):
    invoice_number: str | None
    date: str | None
    total: float | None


@hyperfunction(
    model="qwen-7b",
    hp_type=LMParam,
    optimize_prompt=True,
    optimize_hparams=True,
)
def extract_invoice(doc: str, hp: LMParam) -> Invoice:
    """
    You are an invoice extraction model.

    Given OCR text for a single invoice, extract:

    - invoice_number: string or null
    - date: ISO YYYY-MM-DD or null
    - total: number or null (including taxes if mentioned)

    Rules:
    - Never hallucinate fields. Use null when unsure.
    - If multiple totals are present, prefer the grand total.
    """
    # Example: wiring to your own LLM call
    temp = hp.temperature
    top_p = hp.top_p
    # top_k, etc mapped as you like

    raw_json = call_your_llm(
        model="qwen-7b",
        prompt=extract_invoice.__doc__,
        doc=doc,
        temperature=temp,
        top_p=top_p,
    )
    return Invoice.model_validate_json(raw_json)
```

### 2. Build training data and metric

```python
train_data = [
    Example(
        fn_name="extract_invoice",
        inputs={"doc": "INVOICE #123\nDate: 2024-12-01\nTotal: 99.95 USD"},
        expected=Invoice(invoice_number="123", date="2024-12-01", total=99.95),
    ),
    # ...
]


def metric_fn(preds, expected):
    correct = sum(int(p == e) for p, e in zip(preds, expected))
    return correct / max(1, len(preds))
```

### 3. Create a HyperSystem and optimise

```python
system = HyperSystem(
    hyperfunctions=[extract_invoice],
    prompt_optimizer=NoOpPromptOptimizer(),          # or GEPAPromptOptimizer(...)
    system_optimizer=TorchEggrollSystemOptimizer(
        steps=50, pop_size=32, sigma=0.05, lr=0.1
    ),
)

system.optimize(train_data, metric_fn)
```

After `optimize`, calls to `extract_invoice("...")` will use the tuned:

* docstring prompt (once GEPA integration is real),
* `LMParam` values (temperature, top_p, etc.) found by ES.

---

## Next steps for you

* **Move this file into your library** as the “core”:

  * maybe split into `hyper_core.py` + `hyperparams.py` if you prefer.
* **Wire real LLM calls** in your HyperFunctions (Qwen/OLMo/DeepSeek/vLLM).
* **Add more HyperParam subclasses**:

  * `VisionParam` for OCR detection thresholds,
  * `AgentParam` for routing / cost / risk knobs.
* **Integrate real GEPA**:

  * add `gepa` dependency,
  * implement `GEPAPromptOptimizer.optimize` to:

    * consider candidates for each `hf.get_prompt()`,
    * evaluate via `system.eval_on_examples`,
    * choose best prompts.
* **Extend ES to LoRA weights** (optional):

  * use `TorchEggrollES` on your PyTorch model with a `param_filter` that selects LoRA parameters.

From here you can layer an “agent graph” on top: each agent is a HyperFunction with its own `HyperParam` type, and the same HyperSystem + ES loop can tune the entire fleet end-to-end.

