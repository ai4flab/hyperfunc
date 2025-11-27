hyperfunc
=========

[![PyPI version](https://badge.fury.io/py/hyperfunc.svg)](https://pypi.org/project/hyperfunc/)
[![Tests](https://github.com/ai4flab/hyperfunc/actions/workflows/test.yml/badge.svg)](https://github.com/ai4flab/hyperfunc/actions/workflows/test.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

Hyperfunc is a small library for training compound AI systems, not just individual models.

Most ML tooling assumes you can backpropagate gradients through the whole stack. In practice, real systems look very different: they are graphs of model calls, tools, databases, and control flow. Large parts of that graph are not differentiable and often not even visible (closed-source APIs, external services). You still want those pieces to get better from data, but you cannot run standard backprop through them.


Installation
------------

```bash
pip install hyperfunc
```

With optional dependencies:

```bash
pip install hyperfunc[llm]       # LiteLLM for 100+ LLM providers
pip install hyperfunc[otlp]      # OpenTelemetry export (gRPC)
pip install hyperfunc[otlp-http] # OpenTelemetry export (HTTP)
pip install hyperfunc[postgres]  # PostgreSQL memory backend
```

Or install multiple extras:

```bash
pip install hyperfunc[llm,otlp,postgres]
```


Table of contents
-----------------

- [Installation](#installation)
- [Quick start](#quick-start)
- [What you can do with Hyperfunc](#what-you-can-do-with-hyperfunc)
- [Core concepts](#core-concepts)
  - [The HyperSystem](#the-hypersystem)
  - [Quick tour of the API](#quick-tour-of-the-api)
  - [LoRA weights and 2D hyperparameters](#lora-weights-and-2d-hyperparameters)
  - [Composition example: XOR](#composition-example-xor)
  - [Parallel execution](#parallel-execution)
  - [Primitives: combine and split](#primitives-combine-and-split)
  - [LLM integration](#llm-integration)
  - [Memory for CHAT agents](#memory-for-chat-agents)
  - [Observability](#observability)
  - [Evaluation Framework](#evaluation-framework)
- [Background](#background)
  - [Motivation: why backprop is not enough](#motivation-why-backprop-is-not-enough)
  - [ES and Eggroll: tuning without backprop](#es-and-eggroll-tuning-without-backprop)
  - [DSPy and Prompt Learning: tuning prompts in compound systems](#dspy-and-prompt-learning-tuning-prompts-in-compound-systems)
  - [How ES with low-rank noise works](#how-es-with-low-rank-noise-works)
- [Advanced topics](#advanced-topics)
  - [Population evaluation and DAG staging](#population-evaluation-and-dag-staging)
  - [GPU batching with vmap](#gpu-batching-with-vmap)
  - [DSPy-style signatures](#dspy-style-signatures)
  - [Prompt optimisation with Prompt Learning](#prompt-optimisation-with-prompt-learning)
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
- `PromptLearningOptimizer`: meta-prompting-based prompt optimiser with rich textual feedback.
- `Signature`, `InputField`, `OutputField`: DSPy-style semantic task definitions.
- `Predict`: create hyperfunctions from signatures.
- `combine`, `split`: primitives for tensor operations that auto-trace in the DAG.
- `llm_completion`: hyperfunction wrapping LiteLLM for 100+ LLM providers.
- `make_llm_completion`: factory for model-specific LLM hyperfunctions.
- `Memory`: SQLite/PostgreSQL-backed persistent memory for CHAT agents.
- `AgentType`: enum for agent patterns (`FLOW`, `CHAT`, `GAME`).
- `Scorer`, `ScoreResult`: evaluation framework with rich feedback.
- `ExactMatch`, `NumericDistance`, `ContainsMatch`, etc.: built-in scorers.
- `LLMJudge`: LLM-as-judge evaluation with structured feedback.
- `NoOpSystemOptimizer`: disable ES for prompt-only optimization.
- `OTLPExporter`: export traces to Jaeger, Grafana Tempo, or any OTLP backend.


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


LLM integration
---------------

Hyperfunc includes built-in LLM support via LiteLLM, with ES-optimizable parameters and automatic rate limiting.

**Installation:**

```bash
pip install hyperfunc[llm]
```

**Basic usage:**

```python
from hyperfunc import llm_completion, LMParam

# Direct call with explicit parameters
response = await llm_completion(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello!"}],
    temperature=0.7,
)
print(response.content)

# With ES-optimizable LMParam
hp = LMParam(temperature=0.5, top_p=0.9)
response = await llm_completion(
    model="anthropic/claude-3-haiku",
    messages=[{"role": "user", "content": "Hello!"}],
    hp=hp,
)
```

**Factory pattern for model-specific functions:**

```python
from hyperfunc import make_llm_completion, HyperSystem

# Create a model-specific hyperfunction
gpt4 = make_llm_completion(
    "gpt-4o",
    system_prompt="You are a helpful assistant.",
)

class QASystem(HyperSystem):
    async def run(self, question: str) -> str:
        response = await gpt4(question)  # LMParam is ES-optimizable
        return response.content

# ES will optimize temperature, top_p, etc.
await system.optimize(examples, metric_fn)
```

**Key features:**

- Works with 100+ LLM providers via LiteLLM (OpenAI, Anthropic, Ollama, etc.)
- `LMParam` has 5 ES-optimizable parameters: `temperature`, `top_p`, `presence_penalty`, `frequency_penalty`, `max_tokens_frac`
- Automatic rate limiting (enabled by default, disable with `rate_limit=False`)
- Streaming automatically disabled during `evaluate()` and `optimize()` to get full responses for metrics


Memory for CHAT agents
----------------------

For conversational agents, Hyperfunc provides a persistent memory system with full-text search.

**Basic usage:**

```python
from hyperfunc import Memory, MemoryEntry, MemoryType

# SQLite (default, local)
memory = Memory("chat.db")
memory = Memory(":memory:")  # In-memory for testing

# PostgreSQL (multi-tenant, shared)
memory = Memory("postgresql://user:pass@localhost/mydb")
```

**Storing and retrieving memories:**

```python
# Store a conversation turn (auto-extracts facts, preferences, entities)
await memory.store_turn(
    user_message="I'm building a FastAPI project in Python",
    assistant_response="Great! What features do you need?",
    user_id="user123",  # Optional: for multi-tenant isolation
)

# Retrieve relevant memories using full-text search
memories = memory.retrieve("FastAPI", user_id="user123")
context = memory.format_context(memories)  # Format for prompt injection
```

**Integration with CHAT agents:**

```python
from hyperfunc import HyperSystem, AgentType, ChatResponse, Memory

class ChatBot(HyperSystem):
    agent_type = AgentType.CHAT

    async def run(self, message, history=None, memory_context=None):
        # memory_context is automatically injected when memory is enabled
        prompt = f"{memory_context}\n\nUser: {message}" if memory_context else message
        response = await self.llm(prompt)
        return ChatResponse(message=response, done=False)

# Create with memory
memory = Memory("chat.db", extraction_model="gpt-4o-mini")
bot = ChatBot(memory=memory)
```

**Key features:**

- **SQLite FTS5** for local full-text search with BM25 ranking
- **PostgreSQL TSVector** for shared database deployments
- **Multi-tenant** via `user_id` field (isolate memories per user)
- **Auto-extraction** of facts, preferences, entities from conversations
- **Relevance scoring** combining text match, importance, and recency
- Memory types: `FACT`, `PREFERENCE`, `CONTEXT`, `EVENT`, `ENTITY`

**Installation for PostgreSQL:**

```bash
pip install hyperfunc[postgres]
```


Observability
-------------

Hyperfunc includes a built-in observability system that traces hyperfunction calls with OpenTelemetry GenAI semantic conventions.

**Basic usage:**

```python
from hyperfunc import HyperSystem, hyperfunction

class MySystem(HyperSystem):
    async def run(self, x):
        return await my_hyperfunction(x)

system = MySystem()

# Enable tracing with a context manager
with system.observability.trace(session_id="my-session"):
    result = await system.run(5)

# Get call history as ObservationRecord objects
history = system.observability.get_history()
for obs in history:
    print(f"{obs.fn_name}: {obs.elapsed_s:.3f}s")

# Get summary with per-function statistics
summary = system.observability.summary()
print(summary.to_markdown())  # Formatted report
```

**Export to JSON:**

```python
# Export observations and summary to JSON
system.observability.export_json("trace.json")

# Or use JSONExporter for more control
from hyperfunc import JSONExporter

exporter = JSONExporter("trace.jsonl", jsonl=True, include_summary=False)
exporter.export(history, summary)
```

**Export to OpenTelemetry (OTLP):**

Export traces to any OTLP-compatible backend: Jaeger, Grafana Tempo, Honeycomb, etc.

```bash
# Install OTLP support
pip install hyperfunc[otlp]       # gRPC (default, port 4317)
pip install hyperfunc[otlp-http]  # HTTP/protobuf (port 4318)
```

```python
# Export to Jaeger
system.observability.export_otlp(endpoint="http://jaeger:4317")

# Export to Grafana Tempo via HTTP
system.observability.export_otlp(
    endpoint="http://tempo:4318",
    protocol="http",
)

# With custom service name and authentication
system.observability.export_otlp(
    endpoint="http://collector:4317",
    service_name="my-ai-service",
    headers={"Authorization": "Bearer token"},
)

# Or use OTLPExporter directly
from hyperfunc import OTLPExporter

exporter = OTLPExporter(
    endpoint="http://localhost:4317",
    service_name="hyperfunc-batch",
)
exporter.export(observations)
exporter.shutdown()
```

The OTLP exporter maps hyperfunc's `ObservationRecord` to OpenTelemetry spans with:
- **GenAI semantic conventions**: `gen_ai.system`, `gen_ai.request.model`, `gen_ai.usage.input_tokens`, etc.
- **Error status**: Failed observations set `otel.status_code=ERROR` with exception details
- **Custom metrics**: Extra metrics appear as `hyperfunc.*` attributes

**LangFuse integration:**

```python
from hyperfunc import LangFuseExporter

# Requires: pip install langfuse
exporter = LangFuseExporter(trace_name="my-trace")
system.observability.export(exporter)
```

**Key features:**

- **GenAI semantic conventions** following OpenTelemetry standards (model, tokens, cost)
- **Percentile latencies** (p50, p95, p99) per hyperfunction
- **Error tracking** with error rates and messages
- **Session management** with unique session IDs
- **Multiple export formats**: JSON, JSONL, OTLP, LangFuse
- **Markdown reports** via `summary.to_markdown()`


Evaluation Framework
--------------------

Hyperfunc includes an evaluation framework with scorers that provide both numeric scores and rich textual feedback. This is particularly useful for `PromptLearningOptimizer`, where detailed feedback drives prompt improvement.

**Built-in scorers:**

```python
from hyperfunc import ExactMatch, NumericDistance, ContainsMatch, RegexMatch, CompositeScorer

# Exact string match (case-insensitive)
scorer = ExactMatch(case_sensitive=False)
result = scorer.score(output="Hello", expected="hello")
# result.score = 1.0, result.feedback = ""

# Numeric comparison with tolerance
scorer = NumericDistance(tolerance=0.01)
result = scorer.score(output=3.14, expected=3.14159)
# result.score ≈ 0.99, result.feedback = "Off by 0.00159"

# Substring match
scorer = ContainsMatch()
result = scorer.score(output="The answer is 42", expected="42")
# result.score = 1.0

# Composite scorer with weights
scorer = CompositeScorer(scorers=[
    (ExactMatch(), 2.0),      # 2x weight
    (ContainsMatch(), 1.0),   # 1x weight
])
```

**LLM-as-judge for complex evaluations:**

```python
from hyperfunc import LLMJudge, SummarizationJudge

# Custom judge with criteria
judge = LLMJudge(
    model="gpt-4o-mini",
    criteria="Evaluate the response for accuracy, completeness, and clarity.",
    scale=(1, 5),
)
result = await judge.score(
    output="Paris is the capital.",
    expected="Paris is the capital of France.",
    inputs={"question": "What is the capital of France?"}
)
# result.score = 0.75 (normalized from 4/5)
# result.feedback = "Good answer but missing the country context..."

# Pre-configured judges for common tasks
summarization_judge = SummarizationJudge(model="gpt-4o-mini")
code_judge = CodeCorrectnessJudge(model="gpt-4o-mini")
```

**Integration with PromptLearningOptimizer:**

```python
from hyperfunc import PromptLearningOptimizer, ExactMatch, LLMJudge

# Use a Scorer for both metric AND feedback
scorer = ExactMatch(case_sensitive=False)
optimizer = PromptLearningOptimizer(
    model="gpt-4o-mini",
    scorer=scorer,  # Provides both score and feedback
    max_iterations=3,
)

# Or use LLMJudge for richer feedback
judge = LLMJudge(model="gpt-4o-mini", criteria="Evaluate classification accuracy")
optimizer = PromptLearningOptimizer(scorer=judge)

system = MySystem(prompt_optimizer=optimizer)
await system.optimize(train_data, metric_fn)  # Uses scorer for feedback
```


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


DSPy and Prompt Learning: tuning prompts in compound systems
------------------------------------------------------------

In parallel, there is a line of work that treats compound systems as programs whose main knobs are prompts:

- DSPy lets you describe a pipeline as Python functions and automatically tune prompts ("teleprompting") using supervised data and metrics.
- Prompt Learning (Arize-style) uses meta-prompting with rich textual feedback to iteratively refine prompts.

Hyperfunc supports DSPy-style signatures for defining semantic input/output schemas, combined with Prompt Learning for optimization. Unlike evolutionary approaches that require many iterations, Prompt Learning typically converges in 1-3 iterations by using detailed feedback about failures.


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


DSPy-style signatures
---------------------

Hyperfunc supports DSPy-style signatures for defining semantic LLM tasks with typed input/output fields:

```python
from hyperfunc import Signature, InputField, OutputField, Predict, HyperSystem

# Define a semantic signature
class QA(Signature):
    """Answer questions based on context."""
    context: str = InputField(desc="Background information")
    question: str = InputField(desc="Question to answer")
    answer: str = OutputField(desc="Concise answer")

# Create a hyperfunction from the signature
qa = Predict(QA, model="gpt-4o")

class QASystem(HyperSystem):
    async def run(self, context: str, question: str) -> str:
        result = await qa(context=context, question=question)
        return result["answer"]

# The signature's docstring becomes the optimizable prompt
# PromptLearningOptimizer can refine it during optimization
```

Key features:

- `Signature`: base class with docstring as task instruction
- `InputField(desc="...")`: defines input parameters with descriptions
- `OutputField(desc="...")`: defines expected outputs
- `Predict(signature, model)`: creates an ES-optimizable hyperfunction
- Both `optimize_prompt=True` and `optimize_hparams=True` are enabled by default


Prompt optimisation with Prompt Learning
----------------------------------------

Hyperfunc uses Arize-style Prompt Learning for prompt optimization. Unlike evolutionary approaches (like GEPA) that require many iterations, Prompt Learning uses meta-prompting with rich textual feedback to converge quickly (typically 1-3 iterations).

**How it works:**

1. Run the system on training data
2. Collect rich textual feedback explaining failures (not just scalar metrics)
3. Use an LLM to generate an improved prompt based on the feedback
4. Accept the new prompt if it improves the metric

**Default behavior:**

`HyperSystem` uses `PromptLearningOptimizer` by default for any hyperfunctions with `optimize_prompt=True`.

```python
from hyperfunc import HyperSystem, PromptLearningOptimizer

# Default: PromptLearningOptimizer is used automatically
system = MySystem()

# Or configure explicitly:
system = MySystem(
    prompt_optimizer=PromptLearningOptimizer(
        model="gpt-4o",       # Model for meta-prompting
        max_iterations=3,     # Usually converges in 1-3
        verbose=True,         # Print progress
    )
)

# Custom feedback function for domain-specific error explanations
def feedback_fn(inputs, output, expected):
    if len(output) > 100:
        return f"Output too long ({len(output)} chars). Keep under 100."
    if output != expected:
        return f"Expected '{expected}' but got '{output}'"
    return ""  # Empty = success

await system.optimize(train_data, metric_fn, feedback_fn=feedback_fn)
```

**Disabling prompt optimization:**

```python
from hyperfunc import NoOpPromptOptimizer

system = MySystem(prompt_optimizer=NoOpPromptOptimizer())
```


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
- `tests/test_signature.py` - DSPy-style signatures and Predict
- `tests/test_llm.py` - LiteLLM integration and LMParam
- `tests/test_eval.py` - Evaluation framework (Scorers, LLMJudge)
- `tests/test_prompt_learning.py` - Real LLM prompt optimization tests
- `tests/test_observability.py` - Tracing and export
- `tests/test_otlp_integration.py` - OTLP export with Jaeger testcontainer (run with `-m integration`)


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
- DSPy-style `Signature`, `InputField`, `OutputField`, and `Predict` for semantic task definitions
- `PromptLearningOptimizer` for meta-prompting-based prompt optimization

The higher-level tooling is intentionally thin so that you can adapt it to your own stack.
