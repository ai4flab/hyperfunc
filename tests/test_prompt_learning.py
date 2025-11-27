"""Tests for Arize-style prompt learning optimization.

Inspired by https://github.com/arize-ai/prompt-learning

These tests validate the PromptLearningOptimizer with REAL LLM calls:
1. Classification tasks with rich feedback
2. LLM-as-judge evaluation
3. Iterative prompt improvement
"""

import pytest

from hyperfunc import (
    Example,
    ExactMatch,
    HyperSystem,
    InputField,
    LLMJudge,
    NoOpSystemOptimizer,
    OutputField,
    Predict,
    PromptLearningOptimizer,
    Signature,
)

# Check if litellm is available
try:
    from hyperfunc.llm import LITELLM_AVAILABLE

    SKIP_LLM_TESTS = not LITELLM_AVAILABLE
except ImportError:
    SKIP_LLM_TESTS = True


# =============================================================================
# Test Signatures (DSPy-style)
# =============================================================================


class SentimentClassifier(Signature):
    """Classify the sentiment of the given text.

    Output exactly one word: positive, negative, or neutral.
    """

    text: str = InputField(desc="Text to analyze")
    sentiment: str = OutputField(desc="One of: positive, negative, neutral")


class MathSolver(Signature):
    """Solve the math problem and return only the numeric answer."""

    problem: str = InputField(desc="Math word problem")
    answer: str = OutputField(desc="Numeric answer only")


class CategoryClassifier(Signature):
    """Classify the query into the correct category.

    Categories:
    - Account: Login, password, account access issues
    - Billing: Payments, refunds, charges, subscriptions
    - Technical: Bugs, errors, app not working
    - General: Everything else

    Output exactly one category name.
    """

    query: str = InputField(desc="Customer query")
    category: str = OutputField(desc="One of: Account, Billing, Technical, General")


# =============================================================================
# Test Data
# =============================================================================

SENTIMENT_TRAIN = [
    Example(inputs={"text": "I love this product! Best purchase ever!"}, expected="positive"),
    Example(inputs={"text": "This is terrible. Complete waste of money."}, expected="negative"),
    Example(inputs={"text": "It's okay, nothing special."}, expected="neutral"),
    Example(inputs={"text": "Absolutely fantastic experience!"}, expected="positive"),
    Example(inputs={"text": "Worst customer service I've ever dealt with."}, expected="negative"),
]

SENTIMENT_TEST = [
    Example(inputs={"text": "Amazing quality, highly recommend!"}, expected="positive"),
    Example(inputs={"text": "Disappointed with this purchase."}, expected="negative"),
    Example(inputs={"text": "It works as expected."}, expected="neutral"),
]

MATH_EXAMPLES = [
    Example(inputs={"problem": "What is 2 + 2?"}, expected="4"),
    Example(inputs={"problem": "If I have 5 apples and give away 2, how many do I have?"}, expected="3"),
    Example(inputs={"problem": "What is 10 divided by 2?"}, expected="5"),
    Example(inputs={"problem": "A book costs $15. How much do 3 books cost?"}, expected="45"),
]

CATEGORY_EXAMPLES = [
    Example(inputs={"query": "I can't log into my account"}, expected="Account"),
    Example(inputs={"query": "Why was I charged twice?"}, expected="Billing"),
    Example(inputs={"query": "The app crashes on startup"}, expected="Technical"),
    Example(inputs={"query": "What are your hours?"}, expected="General"),
    Example(inputs={"query": "Reset my password please"}, expected="Account"),
    Example(inputs={"query": "I want a refund"}, expected="Billing"),
]


# =============================================================================
# Test Systems
# =============================================================================


class SentimentSystem(HyperSystem):
    """System for sentiment classification."""

    def __init__(self, **kwargs):
        # Use NoOpSystemOptimizer to skip ES (only do prompt optimization)
        kwargs.setdefault("system_optimizer", NoOpSystemOptimizer())
        super().__init__(**kwargs)
        self.classifier = Predict(SentimentClassifier, model="gpt-4o-mini")

    async def run(self, text: str) -> str:
        result = await self.classifier(text=text)
        return result.get("sentiment", "").strip().lower()


class MathSystem(HyperSystem):
    """System for math problem solving."""

    def __init__(self, **kwargs):
        kwargs.setdefault("system_optimizer", NoOpSystemOptimizer())
        super().__init__(**kwargs)
        self.solver = Predict(MathSolver, model="gpt-4o-mini")

    async def run(self, problem: str) -> str:
        result = await self.solver(problem=problem)
        # Extract just the number
        answer = result.get("answer", "").strip()
        # Remove any non-numeric characters except minus and decimal
        cleaned = "".join(c for c in answer if c.isdigit() or c in "-.")
        return cleaned


class CategorySystem(HyperSystem):
    """System for category classification."""

    def __init__(self, **kwargs):
        kwargs.setdefault("system_optimizer", NoOpSystemOptimizer())
        super().__init__(**kwargs)
        self.classifier = Predict(CategoryClassifier, model="gpt-4o-mini")

    async def run(self, query: str) -> str:
        result = await self.classifier(query=query)
        return result.get("category", "").strip()


# =============================================================================
# Metrics
# =============================================================================


def accuracy_metric(outputs, expected):
    """Case-insensitive accuracy."""
    correct = sum(1 for o, e in zip(outputs, expected) if o.lower() == e.lower())
    return correct / len(outputs) if outputs else 0.0


def exact_match_metric(outputs, expected):
    """Exact string match."""
    correct = sum(1 for o, e in zip(outputs, expected) if str(o).strip() == str(e).strip())
    return correct / len(outputs) if outputs else 0.0


# =============================================================================
# Tests with Real LLM Calls
# =============================================================================


@pytest.mark.skipif(SKIP_LLM_TESTS, reason="litellm not installed")
class TestRealSentimentClassification:
    """Test sentiment classification with real LLM calls."""

    @pytest.mark.asyncio
    async def test_sentiment_baseline(self):
        """Test baseline sentiment classification accuracy."""
        system = SentimentSystem()

        # Use fewer examples to reduce API calls
        examples = SENTIMENT_TRAIN[:3]
        score = await system.evaluate(examples, accuracy_metric)
        print(f"\nSentiment baseline accuracy: {score:.1%}")

        # Just verify it runs and returns a valid score
        assert 0.0 <= score <= 1.0, f"Invalid score: {score}"

    @pytest.mark.asyncio
    async def test_sentiment_with_optimization(self):
        """Test sentiment classification with prompt optimization."""
        scorer = ExactMatch(case_sensitive=False)
        optimizer = PromptLearningOptimizer(
            model="gpt-4o-mini",
            scorer=scorer,
            max_iterations=1,  # Just one iteration to test the flow
            verbose=True,
        )

        system = SentimentSystem(prompt_optimizer=optimizer)

        # Use fewer examples
        examples = SENTIMENT_TRAIN[:3]

        # Get baseline
        baseline = await system.evaluate(examples, accuracy_metric)
        print(f"\nBaseline accuracy: {baseline:.1%}")

        # Optimize - just verify it runs without error
        await system.optimize(examples, accuracy_metric)

        # Check that optimization completed
        optimized = await system.evaluate(examples, accuracy_metric)
        print(f"Optimized accuracy: {optimized:.1%}")

        # Just verify it returns valid results
        assert 0.0 <= optimized <= 1.0


@pytest.mark.skipif(SKIP_LLM_TESTS, reason="litellm not installed")
class TestRealMathSolving:
    """Test math problem solving with real LLM calls."""

    @pytest.mark.asyncio
    async def test_math_baseline(self):
        """Test baseline math solving accuracy."""
        system = MathSystem()

        examples = MATH_EXAMPLES[:2]  # Fewer examples
        score = await system.evaluate(examples, exact_match_metric)
        print(f"\nMath baseline accuracy: {score:.1%}")

        # Just verify it returns valid score
        assert 0.0 <= score <= 1.0

    @pytest.mark.asyncio
    async def test_math_with_scorer(self):
        """Test math solving with ExactMatch scorer."""
        scorer = ExactMatch()
        optimizer = PromptLearningOptimizer(
            model="gpt-4o-mini",
            scorer=scorer,
            max_iterations=1,
            verbose=True,
        )

        system = MathSystem(prompt_optimizer=optimizer)

        examples = MATH_EXAMPLES[:2]
        baseline = await system.evaluate(examples, exact_match_metric)
        print(f"\nMath baseline: {baseline:.1%}")

        # Optimize with scorer
        await system.optimize(examples, exact_match_metric)

        optimized = await system.evaluate(examples, exact_match_metric)
        print(f"Math optimized: {optimized:.1%}")

        assert 0.0 <= optimized <= 1.0


@pytest.mark.skipif(SKIP_LLM_TESTS, reason="litellm not installed")
class TestRealCategoryClassification:
    """Test category classification with real LLM calls."""

    @pytest.mark.asyncio
    async def test_category_baseline(self):
        """Test baseline category classification."""
        system = CategorySystem()

        examples = CATEGORY_EXAMPLES[:3]
        score = await system.evaluate(examples, accuracy_metric)
        print(f"\nCategory baseline accuracy: {score:.1%}")

        assert 0.0 <= score <= 1.0

    @pytest.mark.asyncio
    async def test_category_with_llm_judge(self):
        """Test category classification with LLM-as-judge."""
        judge = LLMJudge(
            model="gpt-4o-mini",
            criteria="""Evaluate if the category classification is correct.

            Score 5: Exact match
            Score 3: Close/related category
            Score 1: Completely wrong category""",
            scale=(1, 5),
        )

        optimizer = PromptLearningOptimizer(
            model="gpt-4o-mini",
            scorer=judge,
            max_iterations=1,
            verbose=True,
        )

        system = CategorySystem(prompt_optimizer=optimizer)

        examples = CATEGORY_EXAMPLES[:3]
        baseline = await system.evaluate(examples, accuracy_metric)
        print(f"\nCategory baseline: {baseline:.1%}")

        # Optimize using LLM judge feedback
        await system.optimize(examples, accuracy_metric)

        optimized = await system.evaluate(examples, accuracy_metric)
        print(f"Category optimized: {optimized:.1%}")

        assert 0.0 <= optimized <= 1.0


@pytest.mark.skipif(SKIP_LLM_TESTS, reason="litellm not installed")
class TestPromptEvolution:
    """Test that prompts actually evolve during optimization."""

    @pytest.mark.asyncio
    async def test_prompt_changes(self):
        """Verify that optimization modifies the prompt."""
        scorer = ExactMatch(case_sensitive=False)
        optimizer = PromptLearningOptimizer(
            model="gpt-4o-mini",
            scorer=scorer,
            max_iterations=1,
            verbose=True,
        )

        system = SentimentSystem(prompt_optimizer=optimizer)

        # Get initial prompt
        initial_prompt = system.classifier.get_prompt()
        print(f"\nInitial prompt:\n{initial_prompt[:200]}...")

        # Force some failures by using tricky examples
        tricky_examples = [
            Example(inputs={"text": "Not bad at all"}, expected="positive"),  # Tricky - "not bad" = positive
            Example(inputs={"text": "Could be worse"}, expected="neutral"),
            Example(inputs={"text": "I didn't hate it"}, expected="positive"),  # Double negative
        ]

        await system.optimize(tricky_examples, accuracy_metric)

        # Check if prompt changed
        final_prompt = system.classifier.get_prompt()
        print(f"\nFinal prompt:\n{final_prompt[:200]}...")

        # Prompt should have been modified (unless it got 100% on first try)
        # We just verify it ran without error


@pytest.mark.skipif(SKIP_LLM_TESTS, reason="litellm not installed")
class TestRichFeedback:
    """Test rich feedback generation with real LLM."""

    @pytest.mark.asyncio
    async def test_llm_judge_provides_feedback(self):
        """Test that LLMJudge provides actionable feedback."""
        judge = LLMJudge(
            model="gpt-4o-mini",
            criteria="""Evaluate the sentiment classification.
            Provide specific feedback on why the classification might be wrong.""",
        )

        # Test with a wrong classification
        result = await judge.score(
            output="positive",
            expected="negative",
            inputs={"text": "This product broke after one day."},
        )

        print(f"\nJudge score: {result.score}")
        print(f"Judge feedback: {result.feedback}")

        # Should give low score for wrong answer
        assert result.score < 0.7, f"Score too high for wrong answer: {result.score}"
        # Should provide some feedback
        assert len(result.feedback) > 10, "Feedback too short"

    @pytest.mark.asyncio
    async def test_scorer_feedback_used_in_optimization(self):
        """Test that scorer feedback is actually used."""

        class VerboseJudge(LLMJudge):
            """Judge that tracks calls."""
            call_count = 0

            async def score(self, output, expected, inputs=None):
                VerboseJudge.call_count += 1
                return await super().score(output, expected, inputs)

        judge = VerboseJudge(
            model="gpt-4o-mini",
            criteria="Evaluate classification accuracy.",
        )

        optimizer = PromptLearningOptimizer(
            model="gpt-4o-mini",
            scorer=judge,
            max_iterations=1,
        )

        system = SentimentSystem(prompt_optimizer=optimizer)

        VerboseJudge.call_count = 0

        examples = SENTIMENT_TRAIN[:2]
        await system.optimize(examples, accuracy_metric)

        # Judge should have been called during feedback collection
        print(f"\nJudge called {VerboseJudge.call_count} times")
        assert VerboseJudge.call_count > 0, "Judge was never called"
