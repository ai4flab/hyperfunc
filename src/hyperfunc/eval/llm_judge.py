"""LLM-as-judge scorers for complex evaluations.

LLMJudge provides both numeric scores AND rich textual feedback,
making it ideal for use with PromptLearningOptimizer where detailed
feedback drives prompt improvement.

Usage:
    from hyperfunc.eval import LLMJudge, SummarizationJudge

    # Custom judge
    judge = LLMJudge(
        model="gpt-4o-mini",
        criteria="Evaluate the response for accuracy and helpfulness",
    )
    result = await judge.score(output="...", expected="...", inputs={"question": "..."})

    # Pre-configured judge
    judge = SummarizationJudge(model="gpt-4o-mini")
    result = await judge.score(output=summary, expected=reference)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

from .scorer import ScoreResult

if TYPE_CHECKING:
    from ..llm import LLMResponse


@dataclass
class LLMJudge:
    """LLM-as-judge scorer for complex evaluations.

    Uses an LLM to evaluate outputs against expected results, providing
    both a numeric score and rich textual feedback. The feedback is
    particularly valuable for PromptLearningOptimizer, where it drives
    iterative prompt improvement.

    Attributes:
        model: LLM model to use for judging (default: "gpt-4o")
        criteria: Evaluation criteria/rubric for the judge
        scale: Rating scale as (min, max) tuple (default: (1, 5))
        include_inputs: Whether to include task inputs in judge prompt
        temperature: LLM temperature for judging (default: 0.0 for consistency)

    Example:
        judge = LLMJudge(
            model="gpt-4o",
            criteria="Evaluate the summary for completeness, accuracy, and conciseness",
            scale=(1, 5),
        )
        result = await judge.score(
            output="AI is transforming industries.",
            expected="Artificial intelligence is revolutionizing multiple sectors...",
            inputs={"text": "Long article about AI..."}
        )
        # result.score = 0.6  (normalized from 3/5)
        # result.feedback = "Missing key details about specific industries..."
    """

    model: str = "gpt-4o"
    criteria: str = "Evaluate the quality of the output compared to expected."
    scale: Tuple[int, int] = (1, 5)
    include_inputs: bool = True
    temperature: float = 0.0
    _llm_completion: Optional[Callable] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Initialize LLM completion function."""
        if self._llm_completion is None:
            from ..llm import llm_completion
            self._llm_completion = llm_completion

    async def score(
        self,
        output: Any,
        expected: Any,
        inputs: Optional[Dict[str, Any]] = None,
    ) -> ScoreResult:
        """Score output using LLM judge.

        Args:
            output: The actual output from the system
            expected: The expected/reference output
            inputs: Optional dict of task inputs for context

        Returns:
            ScoreResult with normalized score (0-1) and detailed feedback
        """
        prompt = self._build_judge_prompt(output, expected, inputs)

        assert self._llm_completion is not None
        response = await self._llm_completion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=1000,
        )

        return self._parse_response(response.content)

    def _build_judge_prompt(
        self,
        output: Any,
        expected: Any,
        inputs: Optional[Dict[str, Any]],
    ) -> str:
        """Build the prompt for the LLM judge."""
        inputs_section = "N/A"
        if inputs and self.include_inputs:
            try:
                inputs_section = json.dumps(inputs, indent=2, default=str)
            except (TypeError, ValueError):
                inputs_section = str(inputs)

        expected_str = str(expected) if expected is not None else "N/A"
        output_str = str(output) if output is not None else "(empty)"

        return f"""You are an expert judge evaluating AI outputs.

## Evaluation Criteria
{self.criteria}

## Task Input
{inputs_section}

## Expected Output
{expected_str}

## Actual Output
{output_str}

## Instructions
1. Rate the output on a scale of {self.scale[0]} to {self.scale[1]}
2. Provide specific, actionable feedback explaining your rating
3. Focus on what could be improved and why

Respond in this exact format:
SCORE: <number between {self.scale[0]} and {self.scale[1]}>
FEEDBACK: <your detailed feedback>"""

    def _parse_response(self, content: str) -> ScoreResult:
        """Parse LLM response into ScoreResult."""
        # Extract SCORE
        score_match = re.search(r"SCORE:\s*(\d+(?:\.\d+)?)", content, re.IGNORECASE)
        if score_match:
            raw_score = float(score_match.group(1))
        else:
            # Fallback: look for any number at the start
            num_match = re.search(r"^(\d+(?:\.\d+)?)", content.strip())
            raw_score = float(num_match.group(1)) if num_match else float(self.scale[0])

        # Clamp to scale
        raw_score = max(self.scale[0], min(self.scale[1], raw_score))

        # Normalize to 0-1
        scale_range = self.scale[1] - self.scale[0]
        normalized = (raw_score - self.scale[0]) / scale_range if scale_range > 0 else 0.0

        # Extract FEEDBACK
        feedback_match = re.search(r"FEEDBACK:\s*(.+)", content, re.IGNORECASE | re.DOTALL)
        if feedback_match:
            feedback = feedback_match.group(1).strip()
        else:
            # Use everything after SCORE as feedback
            feedback = re.sub(r"^SCORE:\s*\d+(?:\.\d+)?\s*", "", content, flags=re.IGNORECASE).strip()
            if not feedback:
                feedback = content

        return ScoreResult(
            score=normalized,
            feedback=feedback,
            details={"raw_score": raw_score, "scale": self.scale},
        )

    def score_batch(
        self,
        outputs: List[Any],
        expected: List[Any],
        inputs: Optional[List[Dict[str, Any]]] = None,
    ) -> List[ScoreResult]:
        """Score multiple outputs (runs sequentially for now).

        Note: For async batch scoring, use asyncio.gather with score() directly.
        """
        raise NotImplementedError(
            "Use asyncio.gather([judge.score(o, e, i) for o, e, i in zip(outputs, expected, inputs)]) "
            "for batch scoring with LLMJudge"
        )


@dataclass
class SummarizationJudge(LLMJudge):
    """Pre-configured judge for summarization tasks.

    Evaluates summaries on completeness, accuracy, conciseness, and fluency.
    """

    criteria: str = field(default="""Evaluate the summary on these dimensions:
1. **Completeness** - Does it capture the key information from the source?
2. **Accuracy** - Is it factually correct with no hallucinations?
3. **Conciseness** - Is it appropriately brief without unnecessary content?
4. **Fluency** - Is it well-written, coherent, and easy to read?

Consider the expected output as a reference for what information should be included.""")


@dataclass
class CodeCorrectnessJudge(LLMJudge):
    """Pre-configured judge for code evaluation.

    Evaluates code on correctness, edge cases, style, and efficiency.
    """

    criteria: str = field(default="""Evaluate the code on these dimensions:
1. **Correctness** - Does it produce the expected output for the given inputs?
2. **Edge Cases** - Does it handle boundary conditions and edge cases?
3. **Style** - Does it follow best practices and coding conventions?
4. **Efficiency** - Is the time/space complexity reasonable?

Compare against the expected solution and note any differences in approach or output.""")


@dataclass
class ConversationJudge(LLMJudge):
    """Pre-configured judge for conversational AI responses.

    Evaluates responses on helpfulness, accuracy, tone, and completeness.
    """

    criteria: str = field(default="""Evaluate the conversational response on these dimensions:
1. **Helpfulness** - Does it address the user's need or question?
2. **Accuracy** - Is the information factually correct?
3. **Tone** - Is the tone appropriate, professional, and friendly?
4. **Completeness** - Does it provide sufficient information without being excessive?

The expected output shows the ideal response style and content.""")


@dataclass
class FactualAccuracyJudge(LLMJudge):
    """Pre-configured judge for factual accuracy.

    Focuses specifically on whether the output contains factual errors,
    hallucinations, or unsupported claims.
    """

    criteria: str = field(default="""Evaluate the factual accuracy of the output:
1. Are all stated facts correct and verifiable?
2. Are there any hallucinations or made-up information?
3. Are claims properly supported or hedged appropriately?
4. Does it avoid stating opinions as facts?

Score based primarily on factual correctness. Minor stylistic differences are acceptable.""")


@dataclass
class InstructionFollowingJudge(LLMJudge):
    """Pre-configured judge for instruction following.

    Evaluates whether the output correctly follows the given instructions/constraints.
    """

    criteria: str = field(default="""Evaluate how well the output follows the instructions:
1. Does it address all parts of the instruction/question?
2. Does it respect any constraints or requirements mentioned?
3. Is the format/structure as requested?
4. Does it stay on topic without unnecessary tangents?

The expected output demonstrates the ideal instruction-following behavior.""")
