#!/usr/bin/env python3
"""Demo: CHAT agent with multi-turn conversation and evaluation.

This demo shows how to build a conversational agent using the CHAT agent type.
The agent responds to user messages and the system evaluates conversation quality.

Key concepts demonstrated:
- AgentType.CHAT for multi-turn conversations
- ChatResponse with message and done flag
- Conversation evaluation with custom metrics
- History accumulation across turns

Run with:
    python demos/chat_agent/run.py
"""

import asyncio
from typing import Any, Dict, List

from hyperfunc import (
    AgentType,
    ChatResponse,
    ConversationResult,
    Example,
    HyperSystem,
    LMParam,
    hyperfunction,
)


# Define a hyperfunction for generating responses
@hyperfunction(hp_type=LMParam, optimize_hparams=True)
async def generate_response(message: str, history: List[Dict[str, str]], hp) -> str:
    """Generate a conversational response.

    In a real system, this would call an LLM. For this demo, we use a simple
    rule-based responder that demonstrates the conversation flow.

    The hp parameter contains ES-optimizable parameters like temperature,
    which could affect response generation style.
    """
    message_lower = message.lower()

    # Check for greeting
    if any(word in message_lower for word in ["hello", "hi", "hey"]):
        responses = [
            "Hello! How can I help you today?",
            "Hi there! What's on your mind?",
            "Hey! Nice to hear from you.",
        ]
        # Use temperature to select response (simulating LLM behavior)
        idx = int(hp.temperature * 10) % len(responses)
        return responses[idx]

    # Check for question about capabilities
    if "can you" in message_lower or "what can" in message_lower:
        return "I can have conversations, answer questions, and help with various tasks!"

    # Check for farewell
    if any(word in message_lower for word in ["bye", "goodbye", "quit", "exit"]):
        return "Goodbye! Have a great day!"

    # Check for thanks
    if "thank" in message_lower:
        return "You're welcome! Is there anything else I can help with?"

    # Context-aware responses based on history
    if history:
        last_assistant = next(
            (h["content"] for h in reversed(history) if h["role"] == "assistant"),
            None,
        )
        if last_assistant and "help you" in last_assistant.lower():
            return "I understand. Please tell me more about what you need."

    # Default response
    return f"I heard you say: '{message}'. How can I assist further?"


class SimpleChatBot(HyperSystem):
    """A simple chatbot demonstrating the CHAT agent type.

    The agent_type = CHAT means:
    - evaluate() processes each Example as a full conversation
    - run() is called repeatedly for each user message
    - run() returns ChatResponse with message and done flag
    - History accumulates across turns automatically
    """

    agent_type = AgentType.CHAT

    async def run(
        self,
        message: str,
        history: List[Dict[str, str]] | None = None,
        memory_context: str | None = None,
    ) -> ChatResponse:
        """Process one turn of conversation.

        Args:
            message: The user's message
            history: Previous turns as list of {"role": ..., "content": ...}
            memory_context: Retrieved memory (not used in this simple demo)

        Returns:
            ChatResponse with message and done flag
        """
        history = history or []

        # Generate response using our hyperfunction
        response_text = await generate_response(message, history)

        # Check if conversation should end
        message_lower = message.lower()
        done = any(
            word in message_lower for word in ["bye", "goodbye", "quit", "exit"]
        )

        return ChatResponse(message=response_text, done=done)


def conversation_quality_metric(
    results: List[ConversationResult],
    expected: List[Dict[str, Any]],
) -> float:
    """Evaluate conversation quality.

    Scoring criteria:
    - Completed all expected turns: 0.4 points
    - Proper termination (done flag when expected): 0.3 points
    - Response length appropriate (not too short): 0.3 points
    """
    if not results:
        return 0.0

    total_score = 0.0

    for result, exp in zip(results, expected):
        score = 0.0

        # Check turn count
        expected_turns = exp.get("min_turns", 1)
        if result.turns >= expected_turns:
            score += 0.4
        else:
            # Partial credit
            score += 0.4 * (result.turns / expected_turns)

        # Check termination
        if exp.get("should_end", False):
            if result.done:
                score += 0.3
        else:
            # No penalty if we expected ongoing conversation
            score += 0.3

        # Check response quality (not too short)
        avg_response_len = 0
        assistant_responses = [
            h.content for h in result.history if h.role == "assistant"
        ]
        if assistant_responses:
            avg_response_len = sum(len(r) for r in assistant_responses) / len(
                assistant_responses
            )

        if avg_response_len >= 10:  # Reasonable minimum length
            score += 0.3
        else:
            score += 0.3 * (avg_response_len / 10)

        total_score += score

    return total_score / len(results)


async def main():
    """Run the CHAT agent demo."""
    print("=" * 60)
    print("CHAT Agent Demo: Multi-turn Conversation")
    print("=" * 60)

    # Create the chatbot system
    bot = SimpleChatBot()

    # Define conversation examples for evaluation
    # Each example contains a list of user messages that form a conversation
    examples = [
        # Greeting conversation
        Example(
            inputs={
                "conversation": [
                    "Hello!",
                    "What can you do?",
                    "Thanks for the info!",
                ]
            },
            expected={"min_turns": 3, "should_end": False},
        ),
        # Short conversation that ends
        Example(
            inputs={"conversation": ["Hi there", "Goodbye!"]},
            expected={"min_turns": 2, "should_end": True},
        ),
        # Question-based conversation
        Example(
            inputs={
                "conversation": [
                    "Hey",
                    "Can you help me with something?",
                    "I need information about Python",
                    "Thank you, bye!",
                ]
            },
            expected={"min_turns": 4, "should_end": True},
        ),
    ]

    # Evaluate the chatbot
    print("\n1. Evaluating chatbot on test conversations...")
    score = await bot.evaluate(examples, conversation_quality_metric)
    print(f"   Conversation quality score: {score:.2%}")

    # Demo: Interactive conversation
    print("\n2. Demo conversation:")
    print("-" * 40)

    demo_messages = ["Hello!", "What can you do?", "That's interesting!", "Goodbye!"]

    history: List[Dict[str, str]] = []
    for msg in demo_messages:
        print(f"User: {msg}")
        response = await bot.run(msg, history=history)
        print(f"Bot:  {response.message}")
        if response.done:
            print("      [Conversation ended]")
            break
        # Accumulate history
        history.append({"role": "user", "content": msg})
        history.append({"role": "assistant", "content": response.message})
        print()

    # Show hyperparameter state
    print("\n3. Hyperparameter state:")
    hp_state = bot.get_hp_state()
    for name, tensor in hp_state.items():
        print(f"   {name}: shape={tuple(tensor.shape)}")

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
