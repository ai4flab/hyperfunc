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
- [x] Implement batched/asynchronous execution for HTTP-backed hyperfunctions (e.g. LLM APIs), with basic rate limiting.
  - `llm.py`: `llm_completion()` hyperfunction wrapping litellm with ES-optimizable `LMParam`
  - `make_llm_completion()` factory for model-specific functions with system prompts
  - Automatic streaming disabled during evaluation/optimization via `_in_optimization` context
  - Rate limiting via `AdaptiveRateLimiter` (default on, disable-able)

## 3. Agent Type System (`FLOW`, `CHAT`, `GAME`)

- [x] Add `HyperSystem.evaluate(examples, metric_fn, environment_state=None, seed=None, collect_trace=False)`.
- [x] Keep `HyperSystem.optimize(...)` as the main tuning entrypoint and route its evaluations through `evaluate(...)`.
- [x] Auto-registration: hyperfunctions auto-register with Xavier-like init when first called in `run()`.
  - `get_hp_default_init()` generates initialization from `hp_type.shape()`
  - `register_hyperfunction()` still available for custom/pretrained weights
- [x] Implement `AgentType` enum with `FLOW`, `CHAT`, `GAME` variants.
  - `FLOW`: Single I/O (default, current behavior) - Example = one input/output pair
  - `CHAT`: Multi-turn conversation - Example = full conversation, `run()` returns `ChatResponse(message, done)`
  - `GAME`: RL-style episodes - Example = episode config, `run()` returns `GameResponse(action)`
- [x] Add `HyperSystem.agent_type` class attribute (default: `FLOW`).
- [x] Add `_run_example()` dispatcher that routes to appropriate runner based on `agent_type`.
- [x] Add `_run_conversation()` for CHAT: loops `run()` until `done=True`.
- [x] Add `_run_episode()` for GAME: loops `run()` with environment until episode ends.
- [x] Create response types: `FlowResponse`, `ChatResponse`, `GameResponse`.
  - Also: `ConversationTurn`, `ConversationResult`, `EpisodeResult`
- [x] Add `AdaptiveRateLimiter` for HTTP-backed hyperfunctions (LLM APIs).
- [x] Add tests for agent types (`tests/test_agent_types.py`).

## 4. Observability System

- [x] Add lightweight per-call tracing via `CallRecord` and internal hooks (`_before_hf_call`, `_after_hf_call`).
- [x] Expose public API: `get_call_history()`, `observability` property on HyperSystem.
- [x] Create `ObservabilityHub` with session management and trace collection.
  - `observability.trace()` context manager for enabling tracing
  - `observability.get_history()` returns `List[ObservationRecord]`
  - `observability.summary()` returns `TraceSummary` with stats
- [x] Add `ObservationRecord` with GenAI semantic conventions (model, temperature, tokens, cost).
  - Follows OpenTelemetry GenAI semantic conventions
  - Attributes: `gen_ai_system`, `gen_ai_request_model`, `gen_ai_usage_*_tokens`, `cost_usd`
- [x] Add `TraceSummary` with per-hyperfunction stats (counts, latency percentiles, error rates).
  - `HyperFunctionStats`: call_count, error_count, error_rate, p50/p95/p99 latencies
  - `to_json()`, `to_markdown()` for report generation
- [x] Implement exporters:
  - [x] `JSONExporter` for JSON/JSONL output
  - [x] `LangFuseExporter` for LangFuse integration
  - [x] `OTLPExporter` for Jaeger/Grafana via OpenTelemetry
    - Supports gRPC (port 4317) and HTTP/protobuf (port 4318) protocols
    - Maps `ObservationRecord` GenAI attributes to OpenTelemetry semantic conventions
    - Optional dependency: `pip install hyperfunc[otlp]` or `pip install hyperfunc[otlp-http]`
- [x] Add tests for observability (`tests/test_observability.py`).
- [x] Add OTLP integration tests with Jaeger testcontainer (`tests/test_otlp_integration.py`).
  - Run with `pytest -m integration`

## 5. HyperFunction runtime policy

- [x] Allow `@hyperfunction` to declare runtime policy (retries, timeout, max_calls) and enforce it best-effort in `__call__`.
- [ ] Add focused tests for retries, timeout, and max_calls (including interaction with evaluation and optimize).
- [ ] Document the current guarantees and limitations of these policies (per-process, not per-episode, etc.).

## 6. Evaluation Framework

- [x] Create `eval/` module with:
  - [x] `Scorer` protocol with `ScoreResult` (score + feedback)
  - [x] Built-in scorers: `ExactMatch`, `NumericDistance`, `ClassificationAccuracy`, `ContainsMatch`, `RegexMatch`, `CompositeScorer`
  - [x] `LLMJudge` for LLM-as-judge evaluation with rich feedback
  - [x] Specialized judges: `SummarizationJudge`, `CodeCorrectnessJudge`, `ConversationJudge`, `FactualAccuracyJudge`, `InstructionFollowingJudge`
  - [x] Integration with `PromptLearningOptimizer` via `scorer` parameter
- [x] Add tests for evaluation (`tests/test_eval.py`).
- [ ] Future: `Dataset` class with versioning, splits (train/val/test), filtering by tags
- [ ] Future: `Trajectory` from `ExecutionTrace` with metrics: exact_match, precision, recall, tool_efficiency
- [ ] Future: `Experiment` tracking with comparison to baselines
- [ ] Future: `EvalGate` for CI/CD pass/fail gates

## 7. Examples and demos

- [x] Add at least one end-to-end example that wires multiple hyperfunctions into a simple workflow and uses `evaluate` + `optimize`.
  - `test_affine_composition.py` and `test_xor_composition.py` demonstrate composed hyperfunctions with ES optimization
  - `demos/shape_classifier/`: Full demo with color MLP + shape classifier, pre-extracted features, 100% accuracy
- [ ] Add CHAT agent demo: multi-turn conversation with evaluation
- [ ] Add GAME agent demo: RL-style episode with environment

## 8. Memory System for CHAT Agents

- [x] Add persistent memory system for conversational agents.
  - `Memory`: SQLite/PostgreSQL-backed memory with full-text search
  - `MemoryEntry`: structured memory records with importance scoring
  - `MemoryType`: enum for FACT, PREFERENCE, CONTEXT, EVENT, ENTITY
- [x] SQLite FTS5 for local full-text search with BM25 ranking.
- [x] PostgreSQL TSVector support for shared database deployments.
- [x] Multi-tenant support via `user_id` field.
- [x] Auto-extraction of facts/preferences/entities from conversations.
- [x] Relevance scoring combining text match, importance, and recency.
- [x] Add tests for memory (`tests/test_memory.py`).

## 9. DSPy-style Signatures and Prompt Learning

- [x] Add DSPy-style signatures for semantic LLM task definitions.
  - `Signature`: base class with docstring as task instruction
  - `InputField(desc="...")`: defines input parameters with descriptions
  - `OutputField(desc="...")`: defines expected outputs
  - `Predict(signature, model)`: creates ES-optimizable hyperfunction from signature
- [x] Replace GEPA with Arize-style Prompt Learning optimizer.
  - `PromptLearningOptimizer`: meta-prompting with rich textual feedback (converges in 1-3 iterations)
  - `NoOpPromptOptimizer`: for disabling prompt optimization
  - `NoOpSystemOptimizer`: for disabling ES optimization (prompt-only mode)
  - `PromptLearningOptimizer` is now the default (not NoOp)
  - `scorer` parameter for Scorer/LLMJudge integration
- [x] Remove `gepa` dependency from pyproject.toml.
- [x] Add tests for signatures (`tests/test_signature.py`).
- [x] Add real LLM integration tests (`tests/test_prompt_learning.py`).
  - Sentiment, math, category classification with actual OpenAI calls
  - LLMJudge feedback verification
  - Prompt evolution tracking

## 10. Future / V2 scheduler work

- [ ] Use accumulated call history to drive a smarter scheduler (early exits, adaptive batching, focusing ES on impactful θ slices).
- [ ] Explore optional Temporal-style ideas (event histories per evaluation, limited replay) if they prove useful in practice.
