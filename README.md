hyperfunc
=========

Hyperfunc is a small library for tuning compound AI systems.

Instead of tuning a single prompt or a single model call, you treat a whole graph of functions as one system and optimise its knobs together: prompts, sampling parameters, adapter scales, and whatever else you expose as hyperparameters.


Motivation: tuning compound AI systems
--------------------------------------

Real applications are not a single `llm(prompt)`. They are:

- multiple functions that call different models or tools
- retrieval, routing, and control flow in the middle
- adapters, LoRA blocks, and post‑processing logic

Each of these pieces has its own knobs:

- prompt wording and structure
- model choice
- temperature, top‑p, top‑k, max tokens
- penalties, thresholds, retry logic
- adapter / LoRA scales, gating, dropout

Manually tuning these is painful:

- you change one prompt and silently break another part
- you tune one function in isolation and ignore system‑level metrics
- grid search over all parameters explodes combinatorially

Hyperfunc gives you a single place to:

- declare the knobs you care about
- define a metric over real examples
- optimise prompts and hyperparameters jointly at the system level


Core idea: the HyperSystem
--------------------------

You describe your system as a set of Python functions, each representing a meaningful operation in your pipeline (e.g. “extract invoice”, “route ticket”, “write reply”).

You decorate them with `@hyperfunction`:

- the function’s docstring is treated as its prompt
- you can add a structured hyperparameter object (like `LMParam`) as an argument
- Hyperfunc wires all of these functions into a `HyperSystem`

`HyperSystem` owns:

- a single learnable vector `hp` (a `HyperModel`), sliced across your functions
- a prompt optimiser (GEPA‑style, for docstring prompts)
- a system optimiser (TorchEggrollES, for numeric hyperparameters)

You feed it supervised examples and a metric:

- `Example(fn_name, inputs, expected)` for each function you care about
- a `metric_fn(preds, expected)` that returns a scalar score (higher is better)

Then you call:

- `system.optimize(train_data, metric_fn)`

Hyperfunc will:

- evaluate your system on examples
- nudge prompts and hyperparameters to improve the metric


Relation to DSPy, GEPA, and Hyperscale Eggroll
----------------------------------------------

This project stands on a few ideas you may already know:

- DSPy: describing LLM pipelines as Python functions and letting a “teleprompter” tune prompts for a metric
- GEPA: a library that automates prompt mutation and reflection to improve performance
- Hyperscale Eggroll: a JAX implementation of an evolution strategies trainer (EggrollES) that can tune model parameters with low‑rank noise

Hyperfunc borrows from all three:

- like DSPy, you write normal Python functions and let the system optimise them
- it can plug into GEPA to optimise docstring prompts via reflection and mutation
- it ships `TorchEggrollES`, a PyTorch reimplementation of EggrollES, to tune numeric hyperparameters

Out of the box, the core library only uses evolution strategies for hyperparameters. Prompt optimisation via GEPA is opt‑in and aimed at experimenters who are happy to depend on GEPA and its evolving API.


What you can do with Hyperfunc
------------------------------

Hyperfunc is intentionally small, but you can build powerful workflows on top of it:

- Tune LLM sampling parameters so that downstream parsers stop breaking.
  - Example: minimise JSON parse failures or maximise exact‑match on structured outputs.
- Tune prompts and temperatures jointly for multi‑step tools.
  - Example: a router + summariser + generator pipeline that is graded on final answer quality.
- Tune adapter‑level knobs (like `AdapterMeta`) on top of a frozen model.
  - Example: adjust LoRA scales, dropout, or gating to optimise an application‑level metric.
- Run end‑to‑end system optimisation in a tight loop around a real test harness.
  - Example: use real integration tests as the metric; HyperSystem does the hill‑climbing.

The key constraint is that your metric must be:

- a scalar
- deterministic enough over a small population of runs

Beyond that, you can be creative about how you define “better”.


Quick tour of the API
----------------------

The core pieces live in `hyperfunc`:

- `@hyperfunction`: decorator that wraps a Python function as a `HyperFunction`.
- `HyperSystem`: connects a set of `HyperFunction`s with a shared hyperparameter vector.
- `Example`: a single training example for a specific function.
- `LMParam`: a standard LLM hyperparameter bundle (temperature, top‑p, etc.).
- `AdapterMeta`: a bundle of adapter‑level knobs.
- `TorchEggrollSystemOptimizer`: ES‑based optimiser for `HyperModel.hp`.
- `GEPAPromptOptimizer`: GEPA‑based prompt optimiser (experimental / opt‑in).


Minimal example
---------------

Here is a small example that tunes a temperature‑like knob so that a toy function behaves as desired.

```python
import torch
from hyperfunc import (
    Example,
    HyperSystem,
    LMParam,
    TorchEggrollSystemOptimizer,
    hyperfunction,
)


@hyperfunction(model="my-llm", hp_type=LMParam, optimize_hparams=True)
def score_answer(x: float, hp: LMParam) -> float:
    """
    A toy objective: we want hp.temperature to be close to 0.5.
    """
    return -abs(hp.temperature - 0.5)


train_data = [Example("score_answer", {"x": 0.0}, 0.0)]


def metric_fn(preds, expected):
    # We just average the objective values.
    return sum(preds) / len(preds)


system = HyperSystem(
    [score_answer],
    # Use ES to tune LMParam fields (through HyperModel.hp).
    system_optimizer=TorchEggrollSystemOptimizer(steps=20, pop_size=20, sigma=0.1, lr=0.1),
)

# Initialise temperature far from the optimum
with torch.no_grad():
    system.model.hp[0] = 0.0

print("Initial score:", system.eval_on_examples(train_data, metric_fn))
system.optimize(train_data, metric_fn)
print("Final score:", system.eval_on_examples(train_data, metric_fn))
```

In a real system, `score_answer` would call an LLM with `hp.temperature` and friends, and the metric would be something meaningful: accuracy, revenue, latency‑penalised quality, or any custom function over your outputs.


Example: tuning an extraction function
--------------------------------------

The repo includes `demo.py`, which simulates an invoice extraction function backed by a mock LLM.

In that demo:

- The function `extract_invoice(doc: str, hp: LMParam) -> Invoice` is decorated with `@hyperfunction`.
- The docstring is treated as the prompt.
- The system uses:
  - a mock GEPA‑style prompt optimiser that appends “Please output valid JSON.”
  - `TorchEggrollSystemOptimizer` to tune `LMParam.temperature`.
- The metric is “does the parsed `Invoice` exactly match the expected invoice”.

Even with a very simple objective, you get a feel for the workflow:

- define a function that represents the behaviour you want
- define how to score it
- let HyperSystem run a loop that finds better prompts and parameters


Prompt optimisation with GEPA
-----------------------------

GEPA is a richer system for prompt search and reflection. Hyperfunc’s `GEPAPromptOptimizer` is a thin adapter that:

- exposes a single `HyperFunction` (and a metric) as a GEPA optimisation problem
- lets GEPA mutate candidate prompts and evaluate them through your system

At the moment:

- `HyperSystem` defaults to a no‑op prompt optimiser (`NoOpPromptOptimizer`).
- If you want GEPA, you must pass it explicitly:

```python
from hyperfunc import HyperSystem, GEPAPromptOptimizer

system = HyperSystem(
    [extract_invoice],
    prompt_optimizer=GEPAPromptOptimizer(model="gpt-4o"),
)
```

The GEPA integration is for experimentation. Expect to adjust it as the `gepa` package evolves.


How TorchEggrollES works (at a glance)
--------------------------------------

`TorchEggrollES` is a PyTorch ES trainer inspired by Hyperscale’s EggrollES:

- works with any `nn.Module`
- samples Gaussian noise per population member
- uses low‑rank noise for 2D parameters (matrices) to cut cost
- estimates a gradient in parameter space from rewards
- applies an update with learning rate `lr`

Hyperfunc wraps this as `TorchEggrollSystemOptimizer`, which:

- focuses ES updates on the `HyperModel.hp` parameter
- calls your metric through `HyperSystem.eval_on_examples`

You do not have to think about the ES details to use it, but the behaviour is entirely transparent and implemented in `src/hyperfunc/es.py`.


Testing and development
-----------------------

This repo is set up with `pytest` and `uv`:

- create a venv with `python -m uv venv .venv`
- install dev deps with `python -m uv sync --group dev`
- run tests with `source .venv/bin/activate && pytest`

There is also a parity test that compares `TorchEggrollES` against a minimal JAX reference ES on a quadratic objective. This test runs only if JAX is installed and helps ensure the ES behaviour matches the original Eggroll design.


Status
------

Hyperfunc is a small, evolving library aimed at developers who want to treat their AI systems as optimisable programs, not as a bag of ad‑hoc prompts.

The core pieces are stable enough for experiments. The GEPA integration and higher‑level tooling are intentionally thin so that you can adapt them to your own stack.
