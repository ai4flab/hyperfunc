"""Tests for evaluation framework."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from hyperfunc import (
    ClassificationAccuracy,
    CompositeScorer,
    ContainsMatch,
    ExactMatch,
    NumericDistance,
    RegexMatch,
    ScoreResult,
)
from hyperfunc.eval import (
    LLMJudge,
    SummarizationJudge,
    CodeCorrectnessJudge,
    ConversationJudge,
    FactualAccuracyJudge,
    InstructionFollowingJudge,
)


class TestScoreResult:
    """Test ScoreResult dataclass."""

    def test_basic_creation(self):
        result = ScoreResult(score=0.8, feedback="Good job")
        assert result.score == 0.8
        assert result.feedback == "Good job"
        assert result.details == {}

    def test_with_details(self):
        result = ScoreResult(
            score=0.5,
            feedback="Partial match",
            details={"diff": 0.3, "reason": "missing info"},
        )
        assert result.details["diff"] == 0.3

    def test_score_clamping(self):
        result = ScoreResult(score=1.5, feedback="")
        assert result.score == 1.0

        result = ScoreResult(score=-0.5, feedback="")
        assert result.score == 0.0


class TestExactMatch:
    """Test ExactMatch scorer."""

    def test_exact_match(self):
        scorer = ExactMatch()
        result = scorer.score(output="hello", expected="hello")
        assert result.score == 1.0
        assert result.feedback == ""

    def test_mismatch(self):
        scorer = ExactMatch()
        result = scorer.score(output="hello", expected="world")
        assert result.score == 0.0
        assert "Expected 'world' but got 'hello'" in result.feedback

    def test_case_insensitive(self):
        scorer = ExactMatch(case_sensitive=False)
        result = scorer.score(output="Hello", expected="hello")
        assert result.score == 1.0

    def test_case_sensitive(self):
        scorer = ExactMatch(case_sensitive=True)
        result = scorer.score(output="Hello", expected="hello")
        assert result.score == 0.0

    def test_strip_whitespace(self):
        scorer = ExactMatch(strip_whitespace=True)
        result = scorer.score(output="  hello  ", expected="hello")
        assert result.score == 1.0

    def test_none_handling(self):
        scorer = ExactMatch()
        result = scorer.score(output=None, expected="hello")
        assert result.score == 0.0

    def test_score_batch(self):
        scorer = ExactMatch()
        results = scorer.score_batch(
            outputs=["a", "b", "c"],
            expected=["a", "x", "c"],
        )
        assert len(results) == 3
        assert results[0].score == 1.0
        assert results[1].score == 0.0
        assert results[2].score == 1.0


class TestNumericDistance:
    """Test NumericDistance scorer."""

    def test_exact_match(self):
        scorer = NumericDistance()
        result = scorer.score(output=5.0, expected=5.0)
        assert result.score == 1.0
        assert result.feedback == ""

    def test_within_tolerance(self):
        scorer = NumericDistance(tolerance=0.1)
        result = scorer.score(output=5.05, expected=5.0)
        assert result.score == 1.0

    def test_outside_tolerance(self):
        scorer = NumericDistance(tolerance=0.01)
        result = scorer.score(output=5.5, expected=5.0)
        assert result.score < 1.0
        assert "Off by" in result.feedback

    def test_score_decreases_with_distance(self):
        scorer = NumericDistance(tolerance=0, max_distance=10)
        result1 = scorer.score(output=5, expected=0)
        result2 = scorer.score(output=8, expected=0)
        assert result1.score > result2.score

    def test_invalid_input(self):
        scorer = NumericDistance()
        result = scorer.score(output="not a number", expected=5.0)
        assert result.score == 0.0
        assert "Cannot convert" in result.feedback

    def test_score_batch(self):
        scorer = NumericDistance(tolerance=0.1)
        results = scorer.score_batch(
            outputs=[1.0, 2.05, 3.0],
            expected=[1.0, 2.0, 5.0],
        )
        assert results[0].score == 1.0
        assert results[1].score == 1.0  # Within tolerance
        assert results[2].score < 1.0  # Outside tolerance


class TestClassificationAccuracy:
    """Test ClassificationAccuracy scorer."""

    def test_correct_classification(self):
        scorer = ClassificationAccuracy()
        result = scorer.score(output="cat", expected="cat")
        assert result.score == 1.0
        assert result.feedback == ""

    def test_incorrect_classification(self):
        scorer = ClassificationAccuracy()
        result = scorer.score(output="dog", expected="cat")
        assert result.score == 0.0
        assert "Predicted 'dog', expected 'cat'" in result.feedback

    def test_numeric_classes(self):
        scorer = ClassificationAccuracy()
        result = scorer.score(output=1, expected=1)
        assert result.score == 1.0


class TestContainsMatch:
    """Test ContainsMatch scorer."""

    def test_contains(self):
        scorer = ContainsMatch()
        result = scorer.score(
            output="The capital of France is Paris.",
            expected="Paris",
        )
        assert result.score == 1.0

    def test_not_contains(self):
        scorer = ContainsMatch()
        result = scorer.score(
            output="The capital of Germany is Berlin.",
            expected="Paris",
        )
        assert result.score == 0.0

    def test_case_insensitive(self):
        scorer = ContainsMatch(case_sensitive=False)
        result = scorer.score(output="PARIS is great", expected="paris")
        assert result.score == 1.0


class TestRegexMatch:
    """Test RegexMatch scorer."""

    def test_pattern_match(self):
        scorer = RegexMatch(pattern=r"\d{3}-\d{4}")
        result = scorer.score(output="Call me at 555-1234", expected=None)
        assert result.score == 1.0

    def test_no_match(self):
        scorer = RegexMatch(pattern=r"\d{3}-\d{4}")
        result = scorer.score(output="No phone number here", expected=None)
        assert result.score == 0.0

    def test_full_match(self):
        scorer = RegexMatch(pattern=r"\d+", full_match=True)
        result = scorer.score(output="123", expected=None)
        assert result.score == 1.0

        result = scorer.score(output="123abc", expected=None)
        assert result.score == 0.0

    def test_uses_expected_as_pattern(self):
        scorer = RegexMatch()
        result = scorer.score(output="hello world", expected="hello")
        assert result.score == 1.0

    def test_invalid_regex(self):
        scorer = RegexMatch(pattern=r"[invalid")
        result = scorer.score(output="test", expected=None)
        assert result.score == 0.0
        assert "Invalid regex" in result.feedback


class TestCompositeScorer:
    """Test CompositeScorer."""

    def test_weighted_average(self):
        scorer = CompositeScorer(
            scorers=[
                (ExactMatch(), 1.0),
                (ContainsMatch(), 1.0),
            ]
        )
        # Both pass
        result = scorer.score(output="hello", expected="hello")
        assert result.score == 1.0

    def test_partial_match(self):
        scorer = CompositeScorer(
            scorers=[
                (ExactMatch(), 1.0),
                (ContainsMatch(), 1.0),
            ]
        )
        # ContainsMatch passes, ExactMatch fails
        result = scorer.score(output="hello world", expected="hello")
        assert result.score == 0.5  # Average of 0 and 1

    def test_weighted_scoring(self):
        scorer = CompositeScorer(
            scorers=[
                (ExactMatch(), 3.0),  # 3x weight
                (ContainsMatch(), 1.0),  # 1x weight
            ]
        )
        # ContainsMatch passes (1.0), ExactMatch fails (0.0)
        # Weighted: (0*3 + 1*1) / (3+1) = 0.25
        result = scorer.score(output="hello world", expected="hello")
        assert result.score == 0.25


class TestLLMJudge:
    """Test LLMJudge scorer."""

    def test_build_prompt(self):
        judge = LLMJudge(
            criteria="Evaluate accuracy",
            scale=(1, 5),
        )
        prompt = judge._build_judge_prompt(
            output="Paris is the capital",
            expected="The capital of France is Paris",
            inputs={"question": "What is the capital of France?"},
        )
        assert "Evaluate accuracy" in prompt
        assert "Paris is the capital" in prompt
        assert "What is the capital of France?" in prompt
        assert "1" in prompt and "5" in prompt

    def test_parse_response_standard(self):
        judge = LLMJudge(scale=(1, 5))
        result = judge._parse_response(
            "SCORE: 4\nFEEDBACK: Good response with minor issues."
        )
        assert result.score == 0.75  # (4-1)/(5-1)
        assert "Good response" in result.feedback
        assert result.details["raw_score"] == 4

    def test_parse_response_no_format(self):
        judge = LLMJudge(scale=(1, 5))
        result = judge._parse_response("3\nThis is okay.")
        assert result.score == 0.5  # (3-1)/(5-1)

    def test_parse_response_clamp(self):
        judge = LLMJudge(scale=(1, 5))
        result = judge._parse_response("SCORE: 10\nFEEDBACK: Great!")
        assert result.score == 1.0  # Clamped to max

    @pytest.mark.asyncio
    async def test_score_async(self):
        """Test async scoring with mocked LLM."""
        mock_response = MagicMock()
        mock_response.content = "SCORE: 4\nFEEDBACK: Good answer."

        with patch.object(
            LLMJudge,
            "__post_init__",
            lambda self: setattr(self, "_llm_completion", AsyncMock(return_value=mock_response)),
        ):
            judge = LLMJudge(model="gpt-4o-mini")
            result = await judge.score(
                output="Paris",
                expected="Paris is the capital of France",
                inputs={"question": "What is the capital of France?"},
            )
            assert result.score == 0.75
            assert "Good answer" in result.feedback


class TestSpecializedJudges:
    """Test specialized LLM judges."""

    def test_summarization_judge_criteria(self):
        judge = SummarizationJudge()
        assert "Completeness" in judge.criteria
        assert "Accuracy" in judge.criteria
        assert "Conciseness" in judge.criteria
        assert "Fluency" in judge.criteria

    def test_code_correctness_judge_criteria(self):
        judge = CodeCorrectnessJudge()
        assert "Correctness" in judge.criteria
        assert "Edge Cases" in judge.criteria
        assert "Style" in judge.criteria

    def test_conversation_judge_criteria(self):
        judge = ConversationJudge()
        assert "Helpfulness" in judge.criteria
        assert "Accuracy" in judge.criteria
        assert "Tone" in judge.criteria

    def test_factual_accuracy_judge_criteria(self):
        judge = FactualAccuracyJudge()
        assert "factual" in judge.criteria.lower()
        assert "hallucination" in judge.criteria.lower()

    def test_instruction_following_judge_criteria(self):
        judge = InstructionFollowingJudge()
        assert "instruction" in judge.criteria.lower()
        assert "constraint" in judge.criteria.lower()


class TestIntegrationWithPromptOptimizer:
    """Test that scorers work with PromptLearningOptimizer."""

    def test_scorer_in_optimizer(self):
        from hyperfunc import PromptLearningOptimizer

        scorer = ExactMatch()
        optimizer = PromptLearningOptimizer(
            model="gpt-4o",
            scorer=scorer,
        )
        assert optimizer.scorer is scorer

    def test_scorer_as_metric(self):
        from hyperfunc import PromptLearningOptimizer

        scorer = ExactMatch()
        optimizer = PromptLearningOptimizer(scorer=scorer)

        # Test _scorer_as_metric (sync path)
        metric = optimizer._scorer_as_metric(
            outputs=["a", "b", "c"],
            expected=["a", "x", "c"],
        )
        assert metric == pytest.approx(2/3)  # 2 out of 3 correct
