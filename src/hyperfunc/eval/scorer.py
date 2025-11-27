"""Scorer protocol and built-in scorers for evaluation.

Scorers provide both numeric scores and textual feedback, making them
useful for both evaluation and prompt optimization via PromptLearningOptimizer.

Usage:
    from hyperfunc.eval import ExactMatch, NumericDistance, ScoreResult

    # Simple exact match
    scorer = ExactMatch()
    result = scorer.score(output="hello", expected="hello")
    # ScoreResult(score=1.0, feedback="")

    # Numeric with tolerance
    scorer = NumericDistance(tolerance=0.1)
    result = scorer.score(output=3.14, expected=3.0)
    # ScoreResult(score=0.95, feedback="Off by 0.1400")
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Union


@dataclass
class ScoreResult:
    """Result from a scorer - both numeric score and textual feedback.

    Attributes:
        score: Normalized score from 0.0 to 1.0
        feedback: Textual explanation (empty string for success).
                  Used by PromptLearningOptimizer for meta-prompting.
        details: Additional metadata (e.g., raw_score, diff, etc.)
    """

    score: float
    feedback: str
    details: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Clamp score to [0, 1]
        self.score = max(0.0, min(1.0, self.score))


class Scorer(Protocol):
    """Protocol for scoring system outputs.

    All scorers must implement at least `score()`. The `score_batch()`
    method has a default implementation that loops over `score()`.

    Scorers return both a numeric score AND textual feedback, making them
    ideal for use with PromptLearningOptimizer where rich feedback drives
    prompt improvement.
    """

    def score(
        self,
        output: Any,
        expected: Any,
        inputs: Optional[Dict[str, Any]] = None,
    ) -> ScoreResult:
        """Score a single output against expected.

        Args:
            output: The actual output from the system
            expected: The expected/reference output
            inputs: Optional dict of inputs (for context in feedback)

        Returns:
            ScoreResult with score and feedback
        """
        ...


@dataclass
class ExactMatch:
    """Binary exact match scorer with diff feedback.

    Returns 1.0 if output equals expected exactly, 0.0 otherwise.
    Feedback explains the mismatch for failed cases.

    Attributes:
        case_sensitive: If False, compare strings case-insensitively
        strip_whitespace: If True, strip leading/trailing whitespace before comparison
    """

    case_sensitive: bool = True
    strip_whitespace: bool = False

    def score(
        self,
        output: Any,
        expected: Any,
        inputs: Optional[Dict[str, Any]] = None,
    ) -> ScoreResult:
        """Score exact match."""
        out_str = str(output) if output is not None else ""
        exp_str = str(expected) if expected is not None else ""

        if self.strip_whitespace:
            out_str = out_str.strip()
            exp_str = exp_str.strip()

        if not self.case_sensitive:
            out_str = out_str.lower()
            exp_str = exp_str.lower()

        if out_str == exp_str:
            return ScoreResult(score=1.0, feedback="")

        # Generate helpful feedback
        feedback = f"Expected '{expected}' but got '{output}'"
        if len(str(output)) > 100 or len(str(expected)) > 100:
            feedback = f"Output does not match expected (lengths: {len(str(output))} vs {len(str(expected))})"

        return ScoreResult(score=0.0, feedback=feedback)

    def score_batch(
        self,
        outputs: List[Any],
        expected: List[Any],
        inputs: Optional[List[Dict[str, Any]]] = None,
    ) -> List[ScoreResult]:
        """Score multiple outputs."""
        inputs = inputs or [None] * len(outputs)
        return [
            self.score(o, e, i) for o, e, i in zip(outputs, expected, inputs)
        ]


@dataclass
class NumericDistance:
    """Numeric similarity scorer with tolerance.

    Scores based on how close the output is to expected. Score is 1.0
    if within tolerance, otherwise decreases linearly with distance.

    Attributes:
        tolerance: Absolute tolerance for perfect score
        max_distance: Distance at which score becomes 0 (default: uses expected magnitude)
    """

    tolerance: float = 0.01
    max_distance: Optional[float] = None

    def score(
        self,
        output: Any,
        expected: Any,
        inputs: Optional[Dict[str, Any]] = None,
    ) -> ScoreResult:
        """Score numeric distance."""
        try:
            out_val = float(output)
            exp_val = float(expected)
        except (TypeError, ValueError) as e:
            return ScoreResult(
                score=0.0,
                feedback=f"Cannot convert to numeric: {e}",
                details={"error": str(e)},
            )

        diff = abs(out_val - exp_val)

        # Perfect score if within tolerance
        if diff <= self.tolerance:
            return ScoreResult(score=1.0, feedback="", details={"diff": diff})

        # Calculate score based on distance
        max_dist = self.max_distance or (abs(exp_val) + 1.0)
        score = max(0.0, 1.0 - diff / max_dist)

        feedback = f"Off by {diff:.4f} (expected {exp_val}, got {out_val})"
        return ScoreResult(
            score=score,
            feedback=feedback,
            details={"diff": diff, "expected": exp_val, "actual": out_val},
        )

    def score_batch(
        self,
        outputs: List[Any],
        expected: List[Any],
        inputs: Optional[List[Dict[str, Any]]] = None,
    ) -> List[ScoreResult]:
        """Score multiple outputs."""
        inputs = inputs or [None] * len(outputs)
        return [
            self.score(o, e, i) for o, e, i in zip(outputs, expected, inputs)
        ]


@dataclass
class ClassificationAccuracy:
    """Classification scorer with confusion feedback.

    Returns 1.0 for correct classification, 0.0 otherwise.
    Feedback shows predicted vs expected class for errors.
    """

    def score(
        self,
        output: Any,
        expected: Any,
        inputs: Optional[Dict[str, Any]] = None,
    ) -> ScoreResult:
        """Score classification."""
        correct = output == expected

        if correct:
            return ScoreResult(score=1.0, feedback="")

        feedback = f"Predicted '{output}', expected '{expected}'"
        return ScoreResult(
            score=0.0,
            feedback=feedback,
            details={"predicted": output, "expected": expected},
        )

    def score_batch(
        self,
        outputs: List[Any],
        expected: List[Any],
        inputs: Optional[List[Dict[str, Any]]] = None,
    ) -> List[ScoreResult]:
        """Score multiple outputs."""
        inputs = inputs or [None] * len(outputs)
        return [
            self.score(o, e, i) for o, e, i in zip(outputs, expected, inputs)
        ]


@dataclass
class ContainsMatch:
    """Scorer that checks if expected is contained in output.

    Useful for checking if key information appears in a longer response.

    Attributes:
        case_sensitive: If False, compare case-insensitively
        partial_score: If True, score based on how many expected items are found
    """

    case_sensitive: bool = False

    def score(
        self,
        output: Any,
        expected: Any,
        inputs: Optional[Dict[str, Any]] = None,
    ) -> ScoreResult:
        """Score containment."""
        out_str = str(output) if output is not None else ""
        exp_str = str(expected) if expected is not None else ""

        if not self.case_sensitive:
            out_str = out_str.lower()
            exp_str = exp_str.lower()

        if exp_str in out_str:
            return ScoreResult(score=1.0, feedback="")

        feedback = f"Expected to find '{expected}' in output"
        return ScoreResult(score=0.0, feedback=feedback)

    def score_batch(
        self,
        outputs: List[Any],
        expected: List[Any],
        inputs: Optional[List[Dict[str, Any]]] = None,
    ) -> List[ScoreResult]:
        """Score multiple outputs."""
        inputs = inputs or [None] * len(outputs)
        return [
            self.score(o, e, i) for o, e, i in zip(outputs, expected, inputs)
        ]


@dataclass
class RegexMatch:
    """Scorer that checks if output matches a regex pattern.

    The pattern can be provided at construction or extracted from expected.

    Attributes:
        pattern: Regex pattern to match (if None, uses expected as pattern)
        flags: Regex flags (default: re.IGNORECASE)
        full_match: If True, require full string match; if False, search anywhere
    """

    pattern: Optional[str] = None
    flags: int = re.IGNORECASE
    full_match: bool = False

    def score(
        self,
        output: Any,
        expected: Any,
        inputs: Optional[Dict[str, Any]] = None,
    ) -> ScoreResult:
        """Score regex match."""
        out_str = str(output) if output is not None else ""
        pattern = self.pattern or str(expected)

        try:
            if self.full_match:
                match = re.fullmatch(pattern, out_str, self.flags)
            else:
                match = re.search(pattern, out_str, self.flags)

            if match:
                return ScoreResult(
                    score=1.0,
                    feedback="",
                    details={"match": match.group()},
                )

            feedback = f"Pattern '{pattern}' not found in output"
            return ScoreResult(score=0.0, feedback=feedback)

        except re.error as e:
            return ScoreResult(
                score=0.0,
                feedback=f"Invalid regex pattern: {e}",
                details={"error": str(e)},
            )

    def score_batch(
        self,
        outputs: List[Any],
        expected: List[Any],
        inputs: Optional[List[Dict[str, Any]]] = None,
    ) -> List[ScoreResult]:
        """Score multiple outputs."""
        inputs = inputs or [None] * len(outputs)
        return [
            self.score(o, e, i) for o, e, i in zip(outputs, expected, inputs)
        ]


@dataclass
class CompositeScorer:
    """Combine multiple scorers with weighted averaging.

    Useful for evaluating multiple aspects of an output.

    Attributes:
        scorers: List of (scorer, weight) tuples
        combine_feedback: If True, combine all feedback; if False, only show failures
    """

    scorers: List[tuple]  # List of (Scorer, weight) tuples
    combine_feedback: bool = True

    def score(
        self,
        output: Any,
        expected: Any,
        inputs: Optional[Dict[str, Any]] = None,
    ) -> ScoreResult:
        """Score using all scorers and combine results."""
        if not self.scorers:
            return ScoreResult(score=1.0, feedback="")

        total_weight = sum(w for _, w in self.scorers)
        weighted_score = 0.0
        feedback_parts = []
        details: Dict[str, Any] = {"individual_scores": []}

        for scorer, weight in self.scorers:
            result = scorer.score(output, expected, inputs)
            weighted_score += result.score * weight
            details["individual_scores"].append({
                "scorer": type(scorer).__name__,
                "score": result.score,
                "weight": weight,
            })

            if result.feedback and (self.combine_feedback or result.score < 1.0):
                feedback_parts.append(f"[{type(scorer).__name__}] {result.feedback}")

        final_score = weighted_score / total_weight if total_weight > 0 else 0.0
        feedback = " | ".join(feedback_parts) if feedback_parts else ""

        return ScoreResult(score=final_score, feedback=feedback, details=details)

    def score_batch(
        self,
        outputs: List[Any],
        expected: List[Any],
        inputs: Optional[List[Dict[str, Any]]] = None,
    ) -> List[ScoreResult]:
        """Score multiple outputs."""
        inputs = inputs or [None] * len(outputs)
        return [
            self.score(o, e, i) for o, e, i in zip(outputs, expected, inputs)
        ]
