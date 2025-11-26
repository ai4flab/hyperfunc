hyperfunc
=========

Hyperfunc is a small library for training compound AI systems, not just individual models.

Most ML tooling assumes you can backpropagate gradients through the whole stack. In practice, real systems look very different: they are graphs of model calls, tools, databases, and control flow. Large parts of that graph are not differentiable and often not even visible (closed‑source APIs, external services). You still want those pieces to get better from data, but you cannot run standard backprop through them.


Table of contents
-----------------

- [hyperfunc](#hyperfunc)
  - [Table of contents](#table-of-contents)
  - [Motivation: why backprop is not enough](#motivation-why-backprop-is-not-enough)
  - [ES and Eggroll: tuning without backprop](#es-and-eggroll-tuning-without-backprop)
  - [DSPy and GEPA: tuning prompts in compound systems](#dspy-and-gepa-tuning-prompts-in-compound-systems)
  - [HyperSystem: combining ES and prompt optimisation](#hypersystem-combining-es-and-prompt-optimisation)
  - [Ablations and sensitivity](#ablations-and-sensitivity)
  - [Caveats and limits](#caveats-and-limits)
  - [Core idea: the HyperSystem](#core-idea-the-hypersystem)
  - [What you can do with Hyperfunc](#what-you-can-do-with-hyperfunc)
  - [Quick tour of the API](#quick-tour-of-the-api)
  - [Minimal example](#minimal-example)
  - [Example: tuning an extraction function](#example-tuning-an-extraction-function)
  - [Prompt optimisation with GEPA](#prompt-optimisation-with-gepa)
  - [How TorchEggrollES works (at a glance)](#how-torcheggrolles-works-at-a-glance)
  - [Testing and development](#testing-and-development)
  - [Status](#status)


Motivation: why backprop is not enough
--------------------------------------

A realistic “agent” or application typically includes:

- calls to large LMs behind an API
- small local models (vision, rerankers, classifiers)
- retrieval, routing, and branching logic
- adapters and LoRA blocks
- parsing, validation, and “if this then that” heuristics

Some of these parts expose weights, some only expose knobs:

- prompt text
- sampling parameters (temperature, top‑p, top‑k, max tokens)
- thresholds, penalties, retry counts
- adapter scales, dropout, gating

You can backprop inside your own models, but you cannot backprop through:

- the external LLM API
- the control flow and routing logic
- non‑differentiable metrics (human feedback, discrete checks, business KPIs)

If you want the entire system to improve on an end‑to‑end metric, you need something that does not rely on gradients.


ES and Eggroll: tuning without backprop
----------------------------------------

Evolution Strategies (ES) is a classic answer here:

- treat the whole system as a black box
- perturb parameters with noise
- evaluate a scalar score
- move parameters in the direction of higher scores

ES does not need gradients or differentiability. It only needs a way to sample parameter settings and score them. The problem is that naive ES gets expensive and noisy as parameter counts grow.

Eggroll (Hyperscale’s EggrollES) is a more efficient flavour of ES. It uses low‑rank noise for large 2D parameters (e.g. matrices), which:

- reduces variance in gradient estimates
- cuts the cost of exploring high‑dimensional parameter spaces

This makes it more realistic to tune meaningful parameter vectors (hyperparameters, adapter weights, small heads) without backprop and without blowing up compute.


DSPy and GEPA: tuning prompts in compound systems
--------------------------------------------------

In parallel, there is a line of work that treats compound systems as programs whose main knobs are prompts:

- DSPy lets you describe a pipeline as Python functions and automatically tune prompts (“teleprompting”) using supervised data and metrics.
- GEPA provides a generic prompt optimisation engine that mutates and reflects on prompts to improve performance.

These tools focus on prompt text as the primary control surface. They usually assume the model weights and most hyperparameters stay fixed or are hand‑tuned.


HyperSystem: combining ES and prompt optimisation
-------------------------------------------------

Hyperfunc’s HyperSystem ties these two ideas together:

- prompt optimisation (DSPy/GEPA style) for text knobs
- ES/Eggroll‑style optimisation for numeric knobs

You model your system as a set of Python functions that call LLMs, small models, tools, or anything else. For each function you can:

- expose a docstring prompt
- expose a structured hyperparameter object (sampling settings, adapter meta‑parameters, etc.)

**HyperSystem:**

- collects all numeric knobs into a single parameter vector
- tracks exactly which slice of that vector belongs to which function
- optionally lets GEPA optimise docstring prompts
- uses a system‑level metric over input/output examples as the training signal

You give it supervised examples:

- inputs that flow through the whole system
- desired outputs or scores
- a metric function that turns predictions and targets into a scalar

HyperSystem then runs a training loop that nudges both prompts and numeric hyperparameters to improve that scalar, even though the system as a whole is non‑differentiable and may involve external APIs.


Ablations and sensitivity
-------------------------

Because HyperSystem knows which hyperparameters belong to which function, you can do simple but useful ablations:

- zero out or reset the slice for one function and re‑evaluate the metric
- freeze a subset of functions while optimising others
- run ES with a filter that only updates certain groups of knobs

This lets you answer questions like:

- “If I turn off tuning for this adapter, how much does performance drop?”
- “Is the router’s threshold actually doing work, or is the summariser carrying everything?”
- “What happens if I lock prompts and only tune numeric parameters, or vice versa?”

The results are not perfect causal attributions, but they give you local sensitivity under your current metric and configuration. That is often enough to decide where to spend optimisation budget and engineering time.


Caveats and limits
------------------

This approach is powerful, but it comes with trade‑offs:

- Control‑flow invariance:
  - Hyperparameters are assumed to change *behaviour inside calls*, not the call graph itself.
  - In other words: which hyperfunctions run, in what order, and how many times (loops) should not depend on hp.
  - If hp starts gating branches or loops, the traced DAG that ES replays becomes invalid and optimisation can behave badly.
- Sample efficiency and cost:
  - ES evaluates many noisy variants of the system.
  - If each evaluation makes multiple LLM calls, the loop can get expensive.
- Credit assignment:
  - The optimiser only sees whether the overall metric went up or down.
  - It does not automatically know which component “caused” the change beyond what your parametrisation exposes.
- Scale:
  - ES is suitable for hyperparameter vectors, adapters, and small heads.
  - It is not a drop‑in replacement for backprop on massive models.
- Metric design:
  - You optimise whatever you measure.
  - Poorly chosen metrics can be gamed or lead to overfitting.
- Non‑stationarity:
  - If external APIs or data distributions drift, the optimiser may chase moving targets.

Hyperfunc is meant for developers who understand these caveats and still want a practical way to train the behaviour of a compound system — prompts, small models, and knobs — using supervised examples and a single optimisation loop.


Population evaluation and DAG staging
-------------------------------------

When you move from a single set of hyperparameters to ES over a *population* of candidates, Hyperfunc does not re‑run your Python control flow separately for every candidate. Instead, it:

1. **Traces `run` once per example.**
   - In evaluation mode you call `HyperSystem.execute(...)` / `evaluate(...)` with tracing enabled.
   - Every `@hyperfunction` call inside `run` is recorded as a `CallRecord` (plus richer context), including which function was called, for which example, and in what order.
   - Branches and loops are captured as “this hyperfunction was (or was not) called” and “it was called N times” in the trace.
2. **Builds a per‑example DAG of hyperfunction calls.**
   - From the trace, Hyperfunc derives a DAG whose nodes are hyperfunction invocations and whose edges are data dependencies (e.g. OCR → NER → LLM).
   - Nodes that only depend on raw inputs form the first stage; nodes that depend only on those form the next stage, and so on.
   - This DAG is assumed to be *independent of hp* for a fixed dataset (see the control‑flow invariance caveat above).
3. **Stages and batches work in `evaluate_population`.**
   - `evaluate_population` replays the DAG for many hp candidates at once rather than re‑interpreting the Python control flow:
     - For each stage and each hyperfunction in that stage, it collects all `(candidate, example)` calls that should run there.
     - For hyperfunctions marked `local_gpu=True`, these calls are intended to be executed as real batched GPU forwards across the population and examples.
     - For the rest, calls can be fanned out concurrently (e.g. via a thread pool) across CPU/HTTP work.
   - Intermediate results are stored per candidate per example and fed forward stage by stage through the DAG.
4. **Reduces to per‑candidate rewards.**
   - After the final stage, you have outputs `preds_k[i]` for each candidate `k` and example `i`.
   - `metric_fn(preds_k, expected)` is applied per candidate to produce ES rewards, which drive the optimiser.

The key point is that the *structure* of execution (the DAG and stages) comes from a single traced run of your `HyperSystem.run` on the dataset, while ES and `evaluate_population` focus on efficiently re‑using that structure across many hp candidates. Hyperparameters are free to change how each node behaves, but not which nodes exist or how they are wired. 


Core idea: the HyperSystem
--------------------------

You describe your system as a set of Python functions, each representing a meaningful operation in your pipeline (e.g. “extract invoice”, “route ticket”, “write reply”).

You decorate them with `@hyperfunction`:

- the function’s docstring is treated as its prompt
- you can add a structured hyperparameter object (like `LMParam`) as an argument
- Hyperfunc wires all of these functions into a `HyperSystem`

`HyperSystem` owns:

- a structured set of learnable hp blocks (one 1D tensor per `@hyperfunction`)
- a prompt optimiser (GEPA‑style, for docstring prompts)
- a system optimiser (ES / TorchEggrollES‑style, for numeric hyperparameters)

You feed it supervised examples and a metric:

- `Example(fn_name, inputs, expected)` for each function you care about
- a `metric_fn(preds, expected)` that returns a scalar score (higher is better)

Then you call:

- `system.optimize(train_data, metric_fn)`

Hyperfunc will:

- evaluate your system on examples
- nudge prompts and hyperparameters to improve the metric


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

print("Initial score:", system.evaluate(train_data, metric_fn))
system.optimize(train_data, metric_fn)
print("Final score:", system.evaluate(train_data, metric_fn))
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
- calls your metric through `HyperSystem.evaluate`

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
