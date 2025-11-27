"""Prompt optimization using meta-prompting with rich textual feedback.

Inspired by Arize's Prompt Learning approach:
https://github.com/Arize-ai/prompt-learning
https://arize.com/blog/gepa-vs-prompt-learning-benchmarking-different-prompt-optimization-approaches/

Key differences from evolutionary approaches (like GEPA):
- Single-loop meta-prompting instead of population-based evolution
- Rich textual feedback instead of just scalar metrics
- User can provide custom feedback_fn or Scorer for domain-specific error explanations
- Converges faster (1-3 iterations vs 10+ generations)

Integration with eval framework:
- Pass a Scorer (e.g., LLMJudge) for rich feedback during optimization
- Scorer.score() returns both numeric score and textual feedback
- LLMJudge is ideal for complex tasks (summarization, code, conversation)
"""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, TYPE_CHECKING, Union

if TYPE_CHECKING:
    from .core import HyperSystem
    from .eval.scorer import Scorer

from .core import Example


@dataclass
class PromptLearningOptimizer:
    """Prompt optimizer using meta-prompting with rich textual feedback.

    Uses an LLM to iteratively refine prompts based on execution feedback.
    Unlike evolutionary approaches, this method:
    - Uses rich textual feedback explaining errors, not just scalar scores
    - Converges quickly (typically 1-3 iterations)
    - Allows custom feedback functions or Scorers for domain-specific feedback

    Args:
        model: LLM model for meta-prompting (default: "gpt-4o")
        max_iterations: Maximum optimization iterations (default: 3)
        context_size: Max tokens for feedback context (default: 8000)
        verbose: Print progress during optimization (default: False)
        scorer: Optional Scorer (e.g., LLMJudge) for rich feedback. When provided,
               the scorer is used both for generating feedback AND as the metric
               function (unless metric_fn is explicitly provided).

    Example with feedback_fn:
        optimizer = PromptLearningOptimizer(model="gpt-4o", max_iterations=3)

        def feedback_fn(inputs, output, expected):
            if output != expected:
                return f"Expected '{expected}' but got '{output}'"
            return ""  # Empty = success

        system = MySystem(prompt_optimizer=optimizer)
        await system.optimize(train_data, metric_fn, feedback_fn=feedback_fn)

    Example with Scorer (recommended):
        from hyperfunc.eval import SummarizationJudge

        optimizer = PromptLearningOptimizer(
            model="gpt-4o",
            scorer=SummarizationJudge(model="gpt-4o-mini"),  # Cheaper for judging
        )

        system = SummarySystem(prompt_optimizer=optimizer)
        await system.optimize(train_data)  # No metric_fn needed - scorer handles it
    """

    model: str = "gpt-4o"
    max_iterations: int = 3
    context_size: int = 8000
    verbose: bool = False
    scorer: Optional["Scorer"] = None
    _llm_completion: Optional[Callable] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        # Lazy import to avoid circular dependency
        if self._llm_completion is None:
            from .llm import llm_completion
            self._llm_completion = llm_completion

    async def optimize(
        self,
        system: "HyperSystem",
        train_data: Sequence[Example],
        metric_fn: Optional[Callable[[List[Any], List[Any]], float]] = None,
        feedback_fn: Optional[Callable[[Dict[str, Any], Any, Any], str]] = None,
    ) -> None:
        """Optimize prompts using meta-prompting with rich feedback.

        Args:
            system: The HyperSystem to optimize
            train_data: Training examples
            metric_fn: Scalar metric function for convergence check. If not provided
                      and a scorer is set, uses scorer-based metric.
            feedback_fn: Optional function(inputs, output, expected) -> str
                        Returns textual feedback explaining errors.
                        If not provided and scorer is set, uses scorer for feedback.
                        If neither provided, uses default feedback generator.
        """
        # If scorer is set but metric_fn is not, use scorer-based metric
        if metric_fn is None and self.scorer is not None:
            metric_fn = self._scorer_as_metric
        elif metric_fn is None:
            raise ValueError(
                "Either metric_fn must be provided or scorer must be set. "
                "Consider using a Scorer like ExactMatch or LLMJudge."
            )

        for hf in system.hyperfunctions:
            if not hf.optimize_prompt:
                continue

            if self.verbose:
                print(f"Optimizing prompt for: {hf.name}")

            current_prompt = hf.get_prompt()
            best_prompt = current_prompt
            best_score = await system.evaluate(train_data, metric_fn)

            if self.verbose:
                print(f"  Initial score: {best_score:.4f}")

            for iteration in range(self.max_iterations):
                # 1. Run system on training data
                outputs = await self._run_examples(system, train_data)

                # 2. Collect rich textual feedback
                feedback = await self._collect_feedback(
                    train_data, outputs, feedback_fn
                )

                if not feedback.strip():
                    # No failures = converged
                    if self.verbose:
                        print(f"  Iteration {iteration + 1}: No failures, converged")
                    break

                # 3. Generate improved prompt via meta-prompting
                new_prompt = await self._meta_prompt(
                    current_prompt, feedback, hf.__doc__
                )

                # 4. Evaluate improvement
                hf.set_prompt(new_prompt)
                new_score = await system.evaluate(train_data, metric_fn)

                if self.verbose:
                    print(f"  Iteration {iteration + 1}: score={new_score:.4f}")

                # 5. Accept if improved
                if new_score > best_score:
                    best_prompt = new_prompt
                    best_score = new_score
                    current_prompt = new_prompt
                    if self.verbose:
                        print(f"    -> Accepted (improvement)")
                else:
                    # Revert to best
                    hf.set_prompt(best_prompt)
                    if self.verbose:
                        print(f"    -> Rejected (no improvement)")
                    break

            # Ensure best prompt is set
            hf.set_prompt(best_prompt)

            if self.verbose:
                print(f"  Final score: {best_score:.4f}")

    async def _run_examples(
        self,
        system: "HyperSystem",
        train_data: Sequence[Example],
    ) -> List[Any]:
        """Run system on training examples and collect outputs."""
        outputs = []
        for ex in train_data:
            try:
                output = await system.run(**ex.inputs)
                outputs.append(output)
            except Exception as e:
                outputs.append(f"ERROR: {e}")
        return outputs

    async def _collect_feedback(
        self,
        examples: Sequence[Example],
        outputs: List[Any],
        feedback_fn: Optional[Callable[[Dict[str, Any], Any, Any], str]],
    ) -> str:
        """Collect rich textual feedback for failed examples."""
        feedback_items = []

        for ex, output in zip(examples, outputs):
            if self.scorer is not None and feedback_fn is None:
                # Use scorer for rich feedback
                fb = await self._get_scorer_feedback(output, ex.expected, ex.inputs)
            elif feedback_fn:
                # User-provided feedback function
                fb = feedback_fn(ex.inputs, output, ex.expected)
            else:
                # Auto-generate feedback
                fb = self._default_feedback(ex.inputs, output, ex.expected)

            if fb:  # Only include non-empty feedback (failures)
                feedback_items.append(fb)

        # Truncate if too long
        feedback = "\n---\n".join(feedback_items)
        if len(feedback) > self.context_size * 4:  # Rough token estimate
            feedback = feedback[: self.context_size * 4] + "\n... (truncated)"

        return feedback

    async def _get_scorer_feedback(
        self,
        output: Any,
        expected: Any,
        inputs: Dict[str, Any],
    ) -> str:
        """Get feedback from scorer (handles async and sync scorers)."""
        assert self.scorer is not None

        # Check if scorer.score is async
        score_method = self.scorer.score
        if inspect.iscoroutinefunction(score_method):
            result = await score_method(output, expected, inputs)
        else:
            result = score_method(output, expected, inputs)

        return result.feedback

    def _scorer_as_metric(
        self,
        outputs: List[Any],
        expected: List[Any],
    ) -> float:
        """Use scorer as a metric function (returns mean score).

        Note: This is synchronous for compatibility with existing metric_fn signature.
        For async scorers, we run them in a new event loop.
        """
        assert self.scorer is not None

        async def _compute_scores() -> float:
            scores = []
            for output, exp in zip(outputs, expected):
                score_method = self.scorer.score
                if inspect.iscoroutinefunction(score_method):
                    result = await score_method(output, exp, None)
                else:
                    result = score_method(output, exp, None)
                scores.append(result.score)
            return sum(scores) / len(scores) if scores else 0.0

        # Check if we're already in an async context
        try:
            loop = asyncio.get_running_loop()
            # We're in an async context, need to use nest_asyncio or similar
            # For simplicity, we'll create a task and return a placeholder
            # This shouldn't happen in practice since evaluate() is async
            import warnings
            warnings.warn(
                "Scorer metric called from async context - results may be inaccurate. "
                "Consider passing metric_fn explicitly.",
                RuntimeWarning,
            )
            return 0.0
        except RuntimeError:
            # No event loop running, safe to use asyncio.run
            return asyncio.run(_compute_scores())

    def _default_feedback(
        self,
        inputs: Dict[str, Any],
        output: Any,
        expected: Any,
    ) -> str:
        """Generate default feedback comparing output to expected."""
        # Convert to strings for comparison
        output_str = str(output) if output is not None else ""
        expected_str = str(expected) if expected is not None else ""

        if output_str == expected_str:
            return ""  # Success

        # Generate feedback
        feedback_lines = [
            f"Input: {inputs}",
            f"Output: {output_str[:500]}",  # Truncate long outputs
            f"Expected: {expected_str[:500]}",
        ]

        # Add specific observations if possible
        if isinstance(output, str) and isinstance(expected, str):
            if len(output) > len(expected) * 2:
                feedback_lines.append("Issue: Output is much longer than expected")
            elif len(output) < len(expected) // 2:
                feedback_lines.append("Issue: Output is much shorter than expected")

        return "\n".join(feedback_lines)

    async def _meta_prompt(
        self,
        current_prompt: str,
        feedback: str,
        task_description: Optional[str],
    ) -> str:
        """Use LLM to generate improved prompt based on feedback."""
        meta_prompt = f"""You are a prompt optimization expert. Your task is to improve a prompt based on feedback from failed examples.

## Current Prompt
{current_prompt}

## Task Description
{task_description or "Not specified"}

## Feedback from Failed Examples
{feedback}

## Instructions
Based on the feedback above, generate an improved version of the prompt that addresses the failure patterns.
Focus on:
1. Making instructions clearer and more specific
2. Adding constraints or requirements that were missing
3. Improving the output format if needed

Return ONLY the improved prompt text, with no additional explanation or commentary."""

        assert self._llm_completion is not None
        response = await self._llm_completion(
            model=self.model,
            messages=[{"role": "user", "content": meta_prompt}],
            temperature=0.7,
            max_tokens=2000,
        )
        return response.content.strip()


@dataclass
class NoOpPromptOptimizer:
    """A no-op prompt optimizer that does nothing.

    Used when no prompt optimization is desired.
    """

    async def optimize(
        self,
        system: "HyperSystem",
        train_data: Sequence[Example],
        metric_fn: Callable[[List[Any], List[Any]], float],
        feedback_fn: Optional[Callable[[Dict[str, Any], Any, Any], str]] = None,
    ) -> None:
        """Do nothing."""
        pass
