# Hyperfunc Tasks

This file tracks the main areas of work for hyperfunc while the design is still baking. Status is approximate and can be refined as we iterate.

## 1. ES orchestration and population evaluation

- [x] Route system evaluation during ES through `HyperSystem.evaluate(...)` instead of calling hyperfunctions directly.
- [x] Expose the ES population structure (per-candidate θ) explicitly at the `HyperSystem` level, rather than only via `HyperModel.hp`, via `HyperSystem.evaluate_population(...)`.
- [x] Support population-aware evaluation helpers (e.g. evaluate a list of θ candidates over the same examples and return rewards).
- [x] Add tests that exercise ES behaviour end-to-end via `HyperSystem.optimize(...)` (including failure modes and metric edge cases).
  - `test_affine_composition.py`: Two composed 2x2 LoRA matrices learning rotation+scale
  - `test_xor_composition.py`: Two-layer network proving composition is mathematically required for XOR

## 2. Batching and staging of hyperfunction calls

- [x] Introduce a per-hyperfunction batching hook (`HyperSystem._run_batch(...)`) and use it from `evaluate(...)`.
- [x] Group `Example`s by target hyperfunction so that calls to the same `HyperFunction` can be executed together.
- [x] Extend `_run_batch(...)`/related hooks to be population-aware (batching calls across ES candidates as well as examples).
  - `evaluate_population()` with memory chunking via `max_batch_size`
- [x] Implement real batched execution for GPU-backed hyperfunctions (e.g. OCR, NER) using a single large forward per stage.
  - `torch.vmap` infrastructure: `_get_tensor_fn()`, `_is_vmappable()`, `_execute_hf_batched()`
  - Nested vmap for candidates × examples in a single kernel
  - Auto-fallback to `asyncio.gather` for non-vmappable functions
- [ ] Implement batched/asynchronous execution for HTTP-backed hyperfunctions (e.g. LLM APIs), with basic rate limiting.

## 3. Workflow/runtime semantics (`start`, `evaluate`, `optimize`)

- [x] Add `HyperSystem.evaluate(examples, metric_fn, environment_state=None, seed=None, collect_trace=False)`.
- [x] Keep `HyperSystem.optimize(...)` as the main tuning entrypoint and route its evaluations through `evaluate(...)`.
- [x] Auto-registration: hyperfunctions auto-register with Xavier-like init when first called in `run()`.
  - `get_hp_default_init()` generates initialization from `hp_type.shape()`
  - `register_hyperfunction()` still available for custom/pretrained weights
- [ ] Define the intended behaviour of `HyperSystem.start()` for agent-style runtimes and implement a minimal reference example.
- [ ] Clarify and document the semantics of `environment_state` and `seed` in `evaluate(...)` (how they affect randomness and control flow).

## 4. Tracing, history, and observability

- [x] Add lightweight per-call tracing via `CallRecord` and internal hooks (`_before_hf_call`, `_after_hf_call`).
- [ ] Expose a public API to retrieve call history (e.g. per-evaluation) in a stable format.
- [ ] Add basic summary utilities (per-hyperfunction counts, latency stats, error rates).
- [ ] Integrate optional OpenTelemetry spans/metrics in the internal hooks for richer observability.

## 5. HyperFunction runtime policy

- [x] Allow `@hyperfunction` to declare runtime policy (retries, timeout, max_calls) and enforce it best-effort in `__call__`.
- [ ] Add focused tests for retries, timeout, and max_calls (including interaction with evaluation and optimize).
- [ ] Document the current guarantees and limitations of these policies (per-process, not per-episode, etc.).

## 6. Examples and demos

- [x] Add at least one end-to-end example that wires multiple hyperfunctions into a simple workflow and uses `evaluate` + `optimize`.
  - `test_affine_composition.py` and `test_xor_composition.py` demonstrate composed hyperfunctions with ES optimization
  - `demos/shape_classifier/`: Full demo with color MLP + shape classifier, pre-extracted features, 100% accuracy
- [ ] Add a small "agent-style" demo that shows how `start()` might be used in a long-running loop.

## 7. Future / V2 scheduler work

- [ ] Use accumulated call history to drive a smarter scheduler (early exits, adaptive batching, focusing ES on impactful θ slices).
- [ ] Explore optional Temporal-style ideas (event histories per evaluation, limited replay) if they prove useful in practice.
