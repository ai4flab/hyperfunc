hyperfunc
=========

Hyperfunc is a small library for training compound AI systems, not just individual models.

Most ML tooling assumes you can backpropagate gradients through the whole stack. In practice, real systems look very different: they are graphs of model calls, tools, databases, and control flow. Large parts of that graph are not differentiable and often not even visible (closed-source APIs, external services). You still want those pieces to get better from data, but you cannot run standard backprop through them.


Table of contents
-----------------

- [Quick start](#quick-start)
- [What you can do with Hyperfunc](#what-you-can-do-with-hyperfunc)
- [Core concepts](#core-concepts)
  - [The HyperSystem](#the-hypersystem)
  - [Quick tour of the API](#quick-tour-of-the-api)
  - [LoRA weights and 2D hyperparameters](#lora-weights-and-2d-hyperparameters)
  - [Composition example: XOR](#composition-example-xor)
  - [Parallel execution](#parallel-execution)
  - [Primitives: combine and split](#primitives-combine-and-split)
- [Background](#background)
  - [Motivation: why backprop is not enough](#motivation-why-backprop-is-not-enough)
  - [ES and Eggroll: tuning without backprop](#es-and-eggroll-tuning-without-backprop)
  - [DSPy and GEPA: tuning prompts in compound systems](#dspy-and-gepa-tuning-prompts-in-compound-systems)
  - [How ES with low-rank noise works](#how-es-with-low-rank-noise-works)
- [Advanced topics](#advanced-topics)
  - [Population evaluation and DAG staging](#population-evaluation-and-dag-staging)
  - [GPU batching with vmap](#gpu-batching-with-vmap)
  - [Prompt optimisation with GEPA](#prompt-optimisation-with-gepa)
  - [Ablations and sensitivity](#ablations-and-sensitivity)
  - [Caveats and limits](#caveats-and-limits)
- [Testing and development](#testing-and-development)
- [Status](#status)


Quick start
-----------

```python
import asyncio
import torch
from hyperfunc import (
    Example,
    HyperSystem,
    LMParam,
    ESHybridSystemOptimizer,
    hyperfunction,
)


@hyperfunction(hp_type=LMParam, optimize_hparams=True)
async def score_answer(x: float, hp) -> float:
    """A toy objective: we want hp.temperature to be close to 0.5."""
    return -abs(hp.temperature - 0.5)


class ScoreSystem(HyperSystem):
    async def run(self, x: float) -> float:
        return await score_answer(x)


train_data = [Example({"x": 0.0}, 0.0)]


def metric_fn(preds, expected):
    return sum(preds) / len(preds)  # Higher is better


async def main():
    system = ScoreSystem(
        system_optimizer=ESHybridSystemOptimizer(
            steps=20, pop_size=20, sigma=0.1, lr=0.1,
        )
    )
    # Hyperfunctions auto-register with default initialization when called
    print("Initial score:", await system.evaluate(train_data, metric_fn))
    await system.optimize(train_data, metric_fn)
    print("Final score:", await system.evaluate(train_data, metric_fn))


asyncio.run(main())
```


What you can do with Hyperfunc
------------------------------

Hyperfunc is intentionally small, but you can build powerful workflows on top of it:

- Tune LLM sampling parameters so that downstream parsers stop breaking.
  - Example: minimise JSON parse failures or maximise exact-match on structured outputs.
- Tune prompts and temperatures jointly for multi-step tools.
  - Example: a router + summariser + generator pipeline that is graded on final answer quality.
- Tune LoRA-style weight matrices for small models without backprop.
  - Example: adjust 2D weight matrices using ES with low-rank noise.
- Run end-to-end system optimisation in a tight loop around a real test harness.
  - Example: use real integration tests as the metric; HyperSystem does the optimisation.

The key constraint is that your metric must be:

- a scalar
- deterministic enough over a small population of runs

Beyond that, you can be creative about how you define "better".


Core concepts
=============


The HyperSystem
---------------

You describe your system by subclassing `HyperSystem` and implementing an `async def run()` method. Inside `run()`, you call hyperfunctions that represent meaningful operations in your pipeline.

You decorate functions with `@hyperfunction`:

- specify an `hp_type` (like `LMParam` for LLM settings, or `LoRAWeight` for 2D matrices)
- set `optimize_hparams=True` to include them in ES optimisation
- the function receives an `hp` argument with the current parameter values

All hyperfunctions are `async def` and must be `await`ed. This enables:

- Natural parallelism with `asyncio.gather`
- Consistent semantics for both IO-bound (LLM calls) and compute-bound (tensor ops) operations
- Clear execution order matching developer intent

Then you:

1. Create a `HyperSystem` subclass with an `async def run()` method
2. Call `await system.optimize()` with training examples and a metric

Hyperfunctions **auto-register** when first called inside `run()`, with Xavier-like initialization based on their `hp_type.shape()`. No explicit registration needed for basic usage.

```python
import asyncio
from hyperfunc import HyperSystem, Example, hyperfunction, LMParam, ESHybridSystemOptimizer

@hyperfunction(hp_type=LMParam, optimize_hparams=True)
async def my_function(x: str, hp) -> str:
    # hp.temperature, hp.top_p, etc. are available
    return await call_llm(x, temperature=hp.temperature)

class MySystem(HyperSystem):
    async def run(self, x: str) -> str:
        return await my_function(x)  # Auto-registers on first call

async def main():
    system = MySystem(system_optimizer=ESHybridSystemOptimizer(steps=50))
    await system.optimize(examples, metric_fn)

asyncio.run(main())
```

For custom initialization (e.g., pretrained weights or deterministic tests), use `register_hyperfunction`:

```python
system.register_hyperfunction(my_function, hp_init=loaded_weights)
```


Quick tour of the API
----------------------

The core pieces live in `hyperfunc`:

- `@hyperfunction`: decorator that wraps an async Python function, declaring its hp_type and whether to optimise it.
- `HyperSystem`: base class you subclass to define your pipeline via `async def run()`.
- `Example`: a training example with `inputs` (dict) and `expected` output.
- `LMParam`: a standard LLM hyperparameter bundle (temperature, top-p, etc.).
- `LoRAWeight`: factory for creating 2D weight matrix hp_types with low-rank noise support.
- `ESHybridSystemOptimizer`: ES-based optimiser with Eggroll-style low-rank noise for 2D params.
- `TorchEggrollES`: lower-level ES trainer for nn.Module (used internally).
- `GEPAPromptOptimizer`: GEPA-based prompt optimiser (experimental).
- `combine`, `split`: primitives for tensor operations that auto-trace in the DAG.


LoRA weights and 2D hyperparameters
-----------------------------------

For small models or adapters, you often want to optimise 2D weight matrices rather than scalar hyperparameters. Hyperfunc supports this via `LoRAWeight`:

```python
from hyperfunc import LoRAWeight, hyperfunction, HyperSystem

# Create a type for 4x8 weight matrices with rank-2 noise
MyWeights = LoRAWeight.create(out_dim=4, in_dim=8, noise_rank=2)

@hyperfunction(hp_type=MyWeights, optimize_hparams=True)
async def linear_layer(x: torch.Tensor, hp) -> torch.Tensor:
    # hp.weight is a 4x8 tensor
    return x @ hp.weight.T
```

Key features:

- `LoRAWeight.create(out_dim, in_dim, noise_rank)` creates an hp_type for 2D matrices
- `hp.weight` gives you direct access to the tensor
- ES uses **low-rank noise** (Eggroll-style) for efficient exploration of 2D parameters
- `noise_rank` controls the rank of the noise factorisation (lower = cheaper, less expressive)

This is useful for:

- Small classifier heads
- Adapter/LoRA-style weight matrices
- Any learnable 2D transformation


Composition example: XOR
------------------------

A powerful demonstration of composition: a single linear layer cannot solve XOR, but two composed layers can.

```python
import asyncio
import torch
from hyperfunc import (
    HyperSystem,
    Example,
    LoRAWeight,
    hyperfunction,
    ESHybridSystemOptimizer,
)

# Hidden layer: 2 inputs -> 4 hidden units
HiddenWeights = LoRAWeight.create(out_dim=4, in_dim=2, noise_rank=2)
# Output layer: 4 hidden -> 1 output
OutputWeights = LoRAWeight.create(out_dim=1, in_dim=4, noise_rank=2)


@hyperfunction(hp_type=HiddenWeights, optimize_hparams=True)
async def hidden_layer(x: torch.Tensor, hp) -> torch.Tensor:
    return torch.relu(x @ hp.weight.T)


@hyperfunction(hp_type=OutputWeights, optimize_hparams=True)
async def output_layer(x: torch.Tensor, hp) -> torch.Tensor:
    return torch.sigmoid(x @ hp.weight.T)


class XORSystem(HyperSystem):
    async def run(self, x: torch.Tensor) -> float:
        hidden = await hidden_layer(x)
        output = await output_layer(hidden)
        return float(output.squeeze())


# XOR training data
XOR_EXAMPLES = [
    Example({"x": torch.tensor([0., 0.])}, 0.0),
    Example({"x": torch.tensor([0., 1.])}, 1.0),
    Example({"x": torch.tensor([1., 0.])}, 1.0),
    Example({"x": torch.tensor([1., 1.])}, 0.0),
]


async def main():
    system = XORSystem(
        system_optimizer=ESHybridSystemOptimizer(
            steps=200,
            pop_size=64,
            sigma=0.3,
            lr=0.2,
            antithetic=True,
        )
    )

    # Optional: explicit init for reproducibility (auto-init works too)
    torch.manual_seed(42)
    system.register_hyperfunction(hidden_layer, hp_init=torch.randn(4, 2) * 0.5)
    system.register_hyperfunction(output_layer, hp_init=torch.randn(1, 4) * 0.5)

    # Optimize with binary cross-entropy metric
    await system.optimize(XOR_EXAMPLES, bce_metric)


asyncio.run(main())
```

This demonstrates:

- **Composition**: Two hyperfunctions working together to solve a problem neither could solve alone
- **2D weights**: Using `LoRAWeight` for weight matrices
- **Low-rank ES**: Efficient exploration of the combined parameter space
- **DAG tracing**: The system automatically tracks that `output_layer` depends on `hidden_layer`


Parallel execution
------------------

Hyperfunc uses standard Python async/await semantics for parallelism. Use `asyncio.gather` to run independent hyperfunctions concurrently:

```python
import asyncio
from hyperfunc import HyperSystem, hyperfunction, LMParam

@hyperfunction(hp_type=LMParam, optimize_hparams=True)
async def step_a(x: int, hp) -> int:
    return x + 1

@hyperfunction(hp_type=LMParam, optimize_hparams=True)
async def step_b(x: int, hp) -> int:
    return x * 2

@hyperfunction(hp_type=LMParam, optimize_hparams=True)
async def step_c(a: int, b: int, hp) -> int:
    return a + b

class ParallelSystem(HyperSystem):
    async def run(self, x: int):
        # a and b are independent, run in parallel
        a_result, b_result = await asyncio.gather(
            step_a(x),
            step_b(x),
        )
        # c depends on both a and b
        c_result = await step_c(a_result, b_result)
        return c_result
```

The DAG tracing automatically detects:
- `step_a` and `step_b` have no dependencies on each other (stage 0)
- `step_c` depends on both (stage 1)

This lets `evaluate_population()` batch operations efficiently.


Primitives: combine and split
-----------------------------

For tensor operations that need to participate in DAG tracing, use the built-in `combine` and `split` primitives:

```python
from hyperfunc import combine, split

class MySystem(HyperSystem):
    async def run(self, a, b):
        # Concatenate tensors (auto-traced)
        combined = combine([a, b])

        # Process...
        result = await some_hyperfunction(combined)

        # Split back (auto-traced)
        left, right = split(result, sizes=[10, 20])
        return left, right
```

These primitives:
- Auto-detect when they're called inside a traced context
- Create proper DAG nodes without needing `@hyperfunction` decoration
- Support `dim` parameter for multi-dimensional operations

Using raw `torch.cat` or `torch.split` on traced values will trigger a warning, since they bypass the tracing system.


Background
==========


Motivation: why backprop is not enough
--------------------------------------

A realistic "agent" or application typically includes:

- calls to large LMs behind an API
- small local models (vision, rerankers, classifiers)
- retrieval, routing, and branching logic
- adapters and LoRA blocks
- parsing, validation, and "if this then that" heuristics

Some of these parts expose weights, some only expose knobs:

- prompt text
- sampling parameters (temperature, top-p, top-k, max tokens)
- thresholds, penalties, retry counts
- adapter scales, dropout, gating

You can backprop inside your own models, but you cannot backprop through:

- the external LLM API
- the control flow and routing logic
- non-differentiable metrics (human feedback, discrete checks, business KPIs)

If you want the entire system to improve on an end-to-end metric, you need something that does not rely on gradients.


ES and Eggroll: tuning without backprop
----------------------------------------

Evolution Strategies (ES) is a classic answer here:

- treat the whole system as a black box
- perturb parameters with noise
- evaluate a scalar score
- move parameters in the direction of higher scores

ES does not need gradients or differentiability. It only needs a way to sample parameter settings and score them. The problem is that naive ES gets expensive and noisy as parameter counts grow.

Eggroll (from HyperscaleES) is a more efficient flavour of ES. It uses low-rank noise for large 2D parameters (e.g. matrices), which:

- reduces variance in gradient estimates
- cuts the cost of exploring high-dimensional parameter spaces

This makes it more realistic to tune meaningful parameter vectors (hyperparameters, adapter weights, small heads) without backprop and without blowing up compute.


DSPy and GEPA: tuning prompts in compound systems
--------------------------------------------------

In parallel, there is a line of work that treats compound systems as programs whose main knobs are prompts:

- DSPy lets you describe a pipeline as Python functions and automatically tune prompts ("teleprompting") using supervised data and metrics.
- GEPA provides a generic prompt optimisation engine that mutates and reflects on prompts to improve performance.

These tools focus on prompt text as the primary control surface. They usually assume the model weights and most hyperparameters stay fixed or are hand-tuned.


How ES with low-rank noise works
--------------------------------

`ESHybridSystemOptimizer` uses Evolution Strategies with Eggroll-style low-rank noise:

**For 1D parameters (like `LMParam`):**
- Standard Gaussian noise: `noise = randn(dim) * sigma`

**For 2D parameters (like `LoRAWeight`):**
- Low-rank factored noise: `noise = A @ B.T * sigma / sqrt(rank)`
- Where A is `(out_dim, rank)` and B is `(in_dim, rank)`
- This reduces variance in gradient estimates for matrices

**Key features:**
- **Antithetic sampling**: Pairs of population members use +noise and -noise for lower variance
- **Proper ES gradient**: Weighted average of noise by normalised fitness (not hill-climbing)
- **No backprop**: Works with any black-box function

The optimizer:

1. Generates a population of parameter candidates with noise
2. Evaluates all candidates on training examples
3. Computes gradient as fitness-weighted average of noise
4. Updates parameters with learning rate

You do not have to think about the ES details to use it, but the implementation is transparent in `src/hyperfunc/es.py`.


Advanced topics
===============


Population evaluation and DAG staging
-------------------------------------

When you move from a single set of hyperparameters to ES over a *population* of candidates, Hyperfunc does not re-run your Python control flow separately for every candidate. Instead, it:

1. **Traces `run()` once per example.**
   - Your `HyperSystem` subclass defines an `async def run()` method with plain Python.
   - Every `@hyperfunction` call inside `run()` is recorded, including which function was called and data dependencies.
   - The trace captures the DAG structure automatically.

2. **Builds a per-example DAG of hyperfunction calls.**
   - From the trace, Hyperfunc derives a DAG whose nodes are hyperfunction invocations and whose edges are data dependencies.
   - Dependencies are detected when one hyperfunction's output is passed as input to another.
   - Nodes that only depend on raw inputs form the first stage; nodes that depend on those form the next stage, and so on.

3. **Stages and batches work in `evaluate_population()`.**
   - `evaluate_population()` replays the DAG for many hp candidates at once:
     - For each stage, it collects all `(candidate, example)` calls that should run.
     - For hyperfunctions with 2D weights (like LoRA), low-rank noise is used for efficient exploration.
     - Intermediate results are stored per candidate per example and fed forward through the DAG.

4. **Reduces to per-candidate rewards.**
   - After the final stage, you have outputs for each candidate and example.
   - `metric_fn(preds, expected)` is applied per candidate to produce ES rewards.


GPU batching with vmap
----------------------

For tensor-based hyperfunctions (like those using `LoRAWeight`), Hyperfunc provides infrastructure for GPU-batched execution using PyTorch's `torch.vmap`. This enables true SIMD-style parallelism where a single GPU kernel processes all candidates × examples.

**Key concepts:**

1. **Tensor functions for vmap**
   - Each hyperfunction can generate a "tensor function" that takes raw tensors instead of wrapper objects
   - `system._get_tensor_fn(hf)` returns a vmap-compatible function
   - Wrapper objects (like `LoRAWeight`) are constructed during vmap's trace phase, not execution

2. **Nested vmap for candidates × examples**
   ```python
   from torch.func import vmap

   # Inner vmap: batch over candidates (weights)
   batched_over_weights = vmap(tensor_fn, in_dims=(None, 0))

   # Outer vmap: batch over examples (inputs)
   batched_both = vmap(batched_over_weights, in_dims=(0, None))

   # Single kernel: (num_examples, num_candidates, *output_shape)
   results = batched_both(inputs_batch, weights_batch)
   ```

3. **Memory chunking**
   - `evaluate_population()` accepts a `max_batch_size` parameter (default 1024)
   - Large batches are automatically chunked to avoid GPU OOM
   - Chunking is by candidates, keeping all examples together for efficiency

4. **Fallback for non-vmappable functions**
   - Functions with async I/O or non-tensor operations fall back to `asyncio.gather`
   - Vmappability is auto-detected and cached per hyperfunction

**Example: vmap-compatible classifier**

```python
import torch
from hyperfunc import LoRAWeight, hyperfunction

MyLoRA = LoRAWeight.create(5, 3, noise_rank=2)

@hyperfunction(hp_type=MyLoRA, optimize_hparams=True)
async def classify(features: torch.Tensor, hp) -> torch.Tensor:
    # Pure tensor ops - vmap compatible
    logits = features @ hp.weight.T
    return torch.softmax(logits, dim=-1)

# Get sync function for vmap (works directly on the hyperfunction)
tensor_fn = lambda features, weights: classify.fn_sync(features, MyLoRA.from_tensor(weights))

# Use with vmap directly
from torch.func import vmap
batched_fn = vmap(tensor_fn, in_dims=(None, 0))
candidate_weights = torch.randn(10, 5, 3)  # 10 candidates
features = torch.randn(3)
results = batched_fn(features, candidate_weights)  # (10, 5)
```

**Performance expectations:**

| Scenario | Sequential | With vmap |
|----------|------------|-----------|
| 32 candidates × 200 examples | 6400 kernel launches | ~2 per stage |
| GPU utilization | Low (launch overhead) | High (batched) |
| Speedup | Baseline | 10-100x for tensor ops |


Prompt optimisation with GEPA
-----------------------------

GEPA is a richer system for prompt search and reflection. Hyperfunc's `GEPAPromptOptimizer` is a thin adapter that:

- exposes a single `HyperFunction` (and a metric) as a GEPA optimisation problem
- lets GEPA mutate candidate prompts and evaluate them through your system

At the moment:

- `HyperSystem` defaults to a no-op prompt optimiser (`NoOpPromptOptimizer`).
- If you want GEPA, you must pass it explicitly:

```python
from hyperfunc import HyperSystem, GEPAPromptOptimizer

system = MySystem(
    prompt_optimizer=GEPAPromptOptimizer(model="gpt-4o"),
)
```

The GEPA integration is for experimentation. Expect to adjust it as the `gepa` package evolves.


Ablations and sensitivity
-------------------------

Because HyperSystem knows which hyperparameters belong to which function, you can do simple but useful ablations:

- zero out or reset the slice for one function and re-evaluate the metric
- freeze a subset of functions while optimising others
- run ES with a filter that only updates certain groups of knobs

This lets you answer questions like:

- "If I turn off tuning for this adapter, how much does performance drop?"
- "Is the router's threshold actually doing work, or is the summariser carrying everything?"
- "What happens if I lock prompts and only tune numeric parameters, or vice versa?"

The results are not perfect causal attributions, but they give you local sensitivity under your current metric and configuration. That is often enough to decide where to spend optimisation budget and engineering time.


Caveats and limits
------------------

This approach is powerful, but it comes with trade-offs:

- Control-flow invariance:
  - Hyperparameters are assumed to change *behaviour inside calls*, not the call graph itself.
  - In other words: which hyperfunctions run, in what order, and how many times (loops) should not depend on hp.
  - If hp starts gating branches or loops, the traced DAG that ES replays becomes invalid and optimisation can behave badly.
- Sample efficiency and cost:
  - ES evaluates many noisy variants of the system.
  - If each evaluation makes multiple LLM calls, the loop can get expensive.
- Credit assignment:
  - The optimiser only sees whether the overall metric went up or down.
  - It does not automatically know which component "caused" the change beyond what your parametrisation exposes.
- Scale:
  - ES is suitable for hyperparameter vectors, adapters, and small heads.
  - It is not a drop-in replacement for backprop on massive models.
- Metric design:
  - You optimise whatever you measure.
  - Poorly chosen metrics can be gamed or lead to overfitting.
- Non-stationarity:
  - If external APIs or data distributions drift, the optimiser may chase moving targets.

Hyperfunc is meant for developers who understand these caveats and still want a practical way to train the behaviour of a compound system — prompts, small models, and knobs — using supervised examples and a single optimisation loop.


Testing and development
-----------------------

This repo is set up with `pytest` and `uv`:

```bash
# Create venv and install deps
uv venv .venv
uv sync --group dev

# Run tests
source .venv/bin/activate
pytest
```

Key test files:

- `tests/test_core.py` - Core hyperfunction and system wiring
- `tests/test_dag_tracing.py` - DAG tracing and staged execution
- `tests/test_eggroll_es.py` - ES with low-rank noise
- `tests/test_population_eval.py` - Population evaluation, chunking, and vmap infrastructure
- `tests/test_affine_composition.py` - Composed 2x2 matrices learning transforms
- `tests/test_xor_composition.py` - XOR proving composition is required
- `tests/test_primitives.py` - combine/split primitives and auto-tracing


Status
------

Hyperfunc is a small, evolving library aimed at developers who want to treat their AI systems as optimisable programs, not as a bag of ad-hoc prompts.

The core pieces are stable enough for experiments:

- `HyperSystem` with `async def run()` and DAG tracing
- **Auto-registration**: hyperfunctions register with Xavier-like init when first called
- `ESHybridSystemOptimizer` with low-rank noise for 2D params
- `LoRAWeight` for optimising weight matrices
- `LMParam` for LLM hyperparameters
- `combine` and `split` primitives for tensor operations
- Full async/await API for explicit parallelism
- `vmap` infrastructure for GPU-batched population evaluation
- Memory chunking for large candidate × example batches

The GEPA integration and higher-level tooling are intentionally thin so that you can adapt them to your own stack.
