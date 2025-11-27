"""Tests for DSPy-style signatures."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hyperfunc import InputField, OutputField, Predict, Signature

# Check if litellm is available
try:
    from hyperfunc.llm import LITELLM_AVAILABLE

    SKIP_LLM_TESTS = not LITELLM_AVAILABLE
except ImportError:
    SKIP_LLM_TESTS = True


class TestInputField:
    """Test InputField dataclass."""

    def test_default_desc(self):
        field = InputField()
        assert field.desc == ""

    def test_with_desc(self):
        field = InputField(desc="A text input")
        assert field.desc == "A text input"


class TestOutputField:
    """Test OutputField dataclass."""

    def test_default_desc(self):
        field = OutputField()
        assert field.desc == ""

    def test_with_desc(self):
        field = OutputField(desc="The output result")
        assert field.desc == "The output result"


class TestSignature:
    """Test Signature base class."""

    def test_input_fields(self):
        """Test extracting input fields from signature."""

        class QA(Signature):
            """Answer questions."""

            context: str = InputField(desc="Background info")
            question: str = InputField(desc="Question to answer")
            answer: str = OutputField(desc="The answer")

        inputs = QA.input_fields()
        assert len(inputs) == 2
        assert "context" in inputs
        assert "question" in inputs
        assert inputs["context"].desc == "Background info"
        assert inputs["question"].desc == "Question to answer"

    def test_output_fields(self):
        """Test extracting output fields from signature."""

        class QA(Signature):
            """Answer questions."""

            context: str = InputField(desc="Background info")
            question: str = InputField(desc="Question to answer")
            answer: str = OutputField(desc="The answer")

        outputs = QA.output_fields()
        assert len(outputs) == 1
        assert "answer" in outputs
        assert outputs["answer"].desc == "The answer"

    def test_to_prompt(self):
        """Test prompt generation from signature."""

        class Summarize(Signature):
            """Summarize the given text concisely."""

            text: str = InputField(desc="Text to summarize")
            summary: str = OutputField(desc="One-sentence summary")

        prompt = Summarize.to_prompt()
        assert "Summarize the given text concisely." in prompt
        assert "Inputs:" in prompt
        assert "text: Text to summarize" in prompt
        assert "Outputs:" in prompt
        assert "summary: One-sentence summary" in prompt

    def test_multiple_outputs(self):
        """Test signature with multiple output fields."""

        class Analysis(Signature):
            """Analyze sentiment and key points."""

            text: str = InputField(desc="Text to analyze")
            sentiment: str = OutputField(desc="Positive, negative, or neutral")
            key_points: str = OutputField(desc="Main points as bullet list")

        outputs = Analysis.output_fields()
        assert len(outputs) == 2
        assert "sentiment" in outputs
        assert "key_points" in outputs

    def test_empty_docstring(self):
        """Test signature without docstring."""

        class NoDoc(Signature):
            input_text: str = InputField()
            output_text: str = OutputField()

        prompt = NoDoc.to_prompt()
        assert "Inputs:" in prompt
        assert "Outputs:" in prompt


class TestParseOutput:
    """Test _parse_output function."""

    def test_single_field_colon(self):
        from hyperfunc.signature import _parse_output

        output_fields = {"answer": OutputField(desc="The answer")}
        result = _parse_output("answer: 42", output_fields)
        assert result["answer"] == "42"

    def test_single_field_equals(self):
        from hyperfunc.signature import _parse_output

        output_fields = {"answer": OutputField(desc="The answer")}
        result = _parse_output("answer = 42", output_fields)
        assert result["answer"] == "42"

    def test_multiple_fields(self):
        from hyperfunc.signature import _parse_output

        output_fields = {
            "name": OutputField(desc="Name"),
            "age": OutputField(desc="Age"),
        }
        result = _parse_output("name: John\nage: 30", output_fields)
        assert result["name"] == "John"
        assert result["age"] == "30"

    def test_fallback_single_field(self):
        """Single output field with no pattern uses full response."""
        from hyperfunc.signature import _parse_output

        output_fields = {"summary": OutputField(desc="Summary")}
        result = _parse_output("This is the full response text.", output_fields)
        assert result["summary"] == "This is the full response text."

    def test_case_insensitive(self):
        from hyperfunc.signature import _parse_output

        output_fields = {"Answer": OutputField(desc="The answer")}
        result = _parse_output("answer: yes", output_fields)
        assert result["Answer"] == "yes"


@pytest.mark.skipif(SKIP_LLM_TESTS, reason="litellm not installed")
@pytest.mark.asyncio
class TestPredict:
    """Test Predict hyperfunction factory."""

    async def test_predict_basic(self):
        """Test basic Predict usage with mocked LLM."""

        class QA(Signature):
            """Answer the question based on context."""

            context: str = InputField(desc="Background information")
            question: str = InputField(desc="Question to answer")
            answer: str = OutputField(desc="Concise answer")

        mock_response = MagicMock()
        mock_response.content = "answer: Python is a programming language"
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 50
        mock_response.usage.completion_tokens = 10
        mock_response.usage.total_tokens = 60
        mock_response.model = "gpt-4"
        mock_response.finish_reason = "stop"

        with patch("hyperfunc.signature.llm_completion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_response

            qa = Predict(QA, model="gpt-4")
            result = await qa(context="Python is a programming language.", question="What is Python?")

            assert "answer" in result
            assert result["answer"] == "Python is a programming language"
            mock_llm.assert_called_once()

    async def test_predict_sets_name(self):
        """Predict should set __name__ from signature."""

        class Summarize(Signature):
            """Summarize text."""

            text: str = InputField()
            summary: str = OutputField()

        summarizer = Predict(Summarize)
        assert summarizer.__name__ == "Summarize"

    async def test_predict_invalid_input_field(self):
        """Predict should reject unknown input fields."""

        class Simple(Signature):
            """A simple signature."""

            text: str = InputField()
            result: str = OutputField()

        predictor = Predict(Simple)

        with pytest.raises(ValueError, match="Unexpected input field 'unknown'"):
            await predictor(text="hello", unknown="bad")

    async def test_predict_get_set_prompt(self):
        """Predict hyperfunction should support get/set prompt."""

        class Task(Signature):
            """Initial instruction."""

            input_data: str = InputField()
            output_data: str = OutputField()

        predictor = Predict(Task)

        # Should have initial prompt from signature
        initial_prompt = predictor.get_prompt()
        assert "Initial instruction." in initial_prompt

        # Can update prompt (for optimization)
        predictor.set_prompt("Updated instruction.")
        assert predictor.get_prompt() == "Updated instruction."

    async def test_predict_with_hp(self):
        """Test Predict with LMParam hyperparameters."""
        from hyperfunc import LMParam

        class Question(Signature):
            """Answer the question."""

            query: str = InputField()
            response: str = OutputField()

        mock_response = MagicMock()
        mock_response.content = "response: Test answer"
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.usage.total_tokens = 15
        mock_response.model = "gpt-4"
        mock_response.finish_reason = "stop"

        with patch("hyperfunc.signature.llm_completion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_response

            qa = Predict(Question, model="gpt-4")
            hp = LMParam(temperature=0.5)
            result = await qa(query="What is 2+2?", hp=hp)

            # Verify hp was passed
            call_kwargs = mock_llm.call_args[1]
            assert call_kwargs["hp"] == hp


@pytest.mark.skipif(SKIP_LLM_TESTS, reason="litellm not installed")
class TestPromptOptimization:
    """Test that signatures work with prompt optimization."""

    def test_predict_is_hyperfunction(self):
        """Predict should return a hyperfunction."""
        from hyperfunc.core import HyperFunction

        class Simple(Signature):
            """Simple task."""

            input_text: str = InputField()
            output_text: str = OutputField()

        predictor = Predict(Simple)
        assert isinstance(predictor, HyperFunction)

    def test_predict_optimize_prompt_flag(self):
        """Predict hyperfunction should have optimize_prompt=True."""

        class Simple(Signature):
            """Simple task."""

            input_text: str = InputField()
            output_text: str = OutputField()

        predictor = Predict(Simple)
        assert predictor.optimize_prompt is True

    def test_predict_optimize_hparams_flag(self):
        """Predict hyperfunction should have optimize_hparams=True."""

        class Simple(Signature):
            """Simple task."""

            input_text: str = InputField()
            output_text: str = OutputField()

        predictor = Predict(Simple)
        assert predictor.optimize_hparams is True
