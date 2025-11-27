"""DSPy-style signatures for defining semantic LLM tasks.

Signatures define the input/output schema for LLM tasks with semantic field descriptions.
The signature's docstring becomes the optimizable prompt instruction.

Usage:
    class QA(Signature):
        '''Answer questions based on context.'''
        context: str = InputField(desc="Background information")
        question: str = InputField(desc="Question to answer")
        answer: str = OutputField(desc="Concise answer")

    qa = Predict(QA, model="gpt-4")
    result = await qa(context="...", question="What is X?")
    print(result["answer"])
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Type, TYPE_CHECKING

if TYPE_CHECKING:
    from .core import HyperFunction

from .core import LMParam, hyperfunction
from .llm import llm_completion


@dataclass
class InputField:
    """Marker for input fields in a Signature.

    Args:
        desc: Description of what this input represents.
    """

    desc: str = ""


@dataclass
class OutputField:
    """Marker for output fields in a Signature.

    Args:
        desc: Description of what this output should contain.
    """

    desc: str = ""


class Signature:
    """Base class for DSPy-style signatures.

    Define a signature by subclassing and adding InputField/OutputField attributes:

        class Summarize(Signature):
            '''Summarize text concisely.'''
            text: str = InputField(desc="Text to summarize")
            summary: str = OutputField(desc="One-sentence summary")

    The docstring becomes the task instruction that can be optimized.
    """

    @classmethod
    def input_fields(cls) -> Dict[str, InputField]:
        """Get all input fields defined on this signature."""
        fields = {}
        for name in dir(cls):
            if name.startswith("_"):
                continue
            value = getattr(cls, name, None)
            if isinstance(value, InputField):
                fields[name] = value
        return fields

    @classmethod
    def output_fields(cls) -> Dict[str, OutputField]:
        """Get all output fields defined on this signature."""
        fields = {}
        for name in dir(cls):
            if name.startswith("_"):
                continue
            value = getattr(cls, name, None)
            if isinstance(value, OutputField):
                fields[name] = value
        return fields

    @classmethod
    def to_prompt(cls) -> str:
        """Generate prompt from signature.

        Returns a prompt string with:
        - The signature's docstring as the task instruction
        - Input field descriptions
        - Output field descriptions
        """
        lines = [cls.__doc__.strip() if cls.__doc__ else ""]

        input_fields = cls.input_fields()
        if input_fields:
            lines.append("\nInputs:")
            for name, field in input_fields.items():
                desc = f": {field.desc}" if field.desc else ""
                lines.append(f"  {name}{desc}")

        output_fields = cls.output_fields()
        if output_fields:
            lines.append("\nOutputs:")
            for name, field in output_fields.items():
                desc = f": {field.desc}" if field.desc else ""
                lines.append(f"  {name}{desc}")

        return "\n".join(lines)


def _parse_output(response_text: str, output_fields: Dict[str, OutputField]) -> Dict[str, Any]:
    """Parse LLM response to extract output fields.

    Looks for patterns like:
        field_name: value
        field_name = value

    Args:
        response_text: Raw LLM response text
        output_fields: Expected output field names and descriptions

    Returns:
        Dictionary mapping field names to extracted values
    """
    result: Dict[str, Any] = {}

    for field_name in output_fields:
        # Try various patterns to find the field value
        patterns = [
            rf"{field_name}\s*[:=]\s*(.+?)(?:\n\n|\n[A-Za-z_]+\s*[:=]|$)",
            rf"^\s*{field_name}\s*[:=]\s*(.+)$",
        ]

        for pattern in patterns:
            match = re.search(pattern, response_text, re.MULTILINE | re.DOTALL | re.IGNORECASE)
            if match:
                value = match.group(1).strip()
                # Remove trailing punctuation or whitespace
                value = value.rstrip(".,;")
                result[field_name] = value
                break

        # If not found with patterns, try to extract from structured response
        if field_name not in result:
            # Fallback: if only one output field and no pattern found, use full response
            if len(output_fields) == 1:
                result[field_name] = response_text.strip()

    return result


def Predict(
    signature: Type[Signature],
    model: str = "gpt-4",
    *,
    rate_limit: bool = True,
    api_key: Optional[str] = None,
    **default_kwargs: Any,
) -> "HyperFunction":
    """Create a hyperfunction from a signature.

    The signature's docstring becomes the optimizable prompt instruction.
    PromptLearningOptimizer can refine this prompt during optimization.

    Args:
        signature: A Signature subclass defining inputs/outputs
        model: LLM model to use (default: "gpt-4")
        rate_limit: Enable rate limiting
        api_key: Optional API key override
        **default_kwargs: Additional kwargs passed to llm_completion

    Returns:
        A hyperfunction that accepts the signature's input fields as kwargs
        and returns a dict with the output field values.

    Example:
        class QA(Signature):
            '''Answer questions based on context.'''
            context: str = InputField(desc="Background information")
            question: str = InputField(desc="Question to answer")
            answer: str = OutputField(desc="Concise answer")

        qa = Predict(QA, model="gpt-4")
        result = await qa(context="Python is a programming language.", question="What is Python?")
        print(result["answer"])
    """
    initial_prompt = signature.to_prompt()
    output_fields = signature.output_fields()
    input_field_names = set(signature.input_fields().keys())

    @hyperfunction(hp_type=LMParam, optimize_hparams=True, optimize_prompt=True)
    async def predict(
        hp: Optional[LMParam] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        # Validate input fields
        for key in kwargs:
            if key not in input_field_names:
                raise ValueError(f"Unexpected input field '{key}'. Expected: {input_field_names}")

        # Format input
        input_lines = []
        for name in input_field_names:
            if name in kwargs:
                input_lines.append(f"{name}: {kwargs[name]}")

        input_text = "\n".join(input_lines)

        # Get current prompt (may have been optimized)
        current_prompt = predict.get_prompt()

        # Build output format instruction
        output_format = "Provide the following outputs:\n"
        for name, field in output_fields.items():
            output_format += f"  {name}: <{field.desc or 'your response'}>\n"

        # Construct full prompt
        full_prompt = f"{current_prompt}\n\n{input_text}\n\n{output_format}"

        # Build messages for completion API
        messages = [{"role": "user", "content": full_prompt}]

        # Call LLM
        merged_kwargs = {**default_kwargs}
        response = await llm_completion(
            model=model,
            messages=messages,
            hp=hp,
            rate_limit=rate_limit,
            api_key=api_key,
            **merged_kwargs,
        )

        # Parse output fields from response
        return _parse_output(response.content, output_fields)

    # Set initial prompt (GEPA/PromptLearning reads/writes via get_prompt/set_prompt)
    predict.set_prompt(initial_prompt)
    predict.__name__ = signature.__name__
    predict.__doc__ = signature.__doc__

    return predict
