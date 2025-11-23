from __future__ import annotations

import inspect
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
        # Also update our own docstring so users accessing wrapper.__doc__ see the change
        self.__doc__ = text

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

    import functools
    
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
        
        # We want the wrapper to look like the original function
        functools.update_wrapper(hf, fn)
        return hf

    return wrapper


# ============================================================
# 3. HyperModel and HyperSystem
# ============================================================


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

        # By default we use a no-op prompt optimizer to avoid
        # taking a hard dependency on the external GEPA engine.
        # Callers that want real prompt optimisation should pass
        # an explicit GEPAPromptOptimizer instance.
        if prompt_optimizer is None:
            self.prompt_optimizer = NoOpPromptOptimizer()
        else:
            self.prompt_optimizer = prompt_optimizer

        if system_optimizer is None:
            from .es import TorchEggrollSystemOptimizer
            self.system_optimizer = TorchEggrollSystemOptimizer()
        else:
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
