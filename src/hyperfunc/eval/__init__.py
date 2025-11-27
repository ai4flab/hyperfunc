"""Evaluation framework for hyperfunc.

This module provides scorers for evaluating system outputs, including
LLM-as-judge scorers for complex evaluations. Scorers return both numeric
scores and textual feedback, making them useful for both evaluation and
prompt optimization via PromptLearningOptimizer.

Usage:
    from hyperfunc.eval import ExactMatch, LLMJudge, SummarizationJudge

    # Simple scorer
    scorer = ExactMatch()
    result = scorer.score(output="hello", expected="hello")

    # LLM judge
    judge = SummarizationJudge(model="gpt-4o-mini")
    result = await judge.score(output=summary, expected=reference)
"""

from .scorer import (
    ClassificationAccuracy,
    CompositeScorer,
    ContainsMatch,
    ExactMatch,
    NumericDistance,
    RegexMatch,
    ScoreResult,
    Scorer,
)
from .llm_judge import (
    CodeCorrectnessJudge,
    ConversationJudge,
    FactualAccuracyJudge,
    InstructionFollowingJudge,
    LLMJudge,
    SummarizationJudge,
)

__all__ = [
    # Core types
    "ScoreResult",
    "Scorer",
    # Built-in scorers
    "ExactMatch",
    "NumericDistance",
    "ClassificationAccuracy",
    "ContainsMatch",
    "RegexMatch",
    "CompositeScorer",
    # LLM judges
    "LLMJudge",
    "SummarizationJudge",
    "CodeCorrectnessJudge",
    "ConversationJudge",
    "FactualAccuracyJudge",
    "InstructionFollowingJudge",
]
