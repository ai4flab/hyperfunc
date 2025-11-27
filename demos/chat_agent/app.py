#!/usr/bin/env python3
"""Gradio web UI for the CHAT agent demo.

This provides an interactive chat interface for the SimpleChatBot.

Run with:
    cd demos/chat_agent && uv run --extra demos python app.py
"""

import asyncio
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import gradio as gr
import torch

# Add the demo directory to path so we can import run.py
sys.path.insert(0, str(Path(__file__).parent))

from run import SimpleChatBot, generate_response, LMParam  # noqa: E402


# Create a single bot instance and register the hyperfunction
bot = SimpleChatBot()
bot.register_hyperfunction(generate_response, hp_init=torch.zeros(LMParam.dim()))


def chat(message: str, history: List) -> str:
    """Process a chat message and return the bot's response.

    Args:
        message: The user's message
        history: Gradio chat history

    Returns:
        The bot's response message
    """
    # Convert Gradio history format to our format
    # Gradio 4.x ChatInterface sends list of {"role": ..., "content": ...} dicts
    # But content can be a string or a list of content parts
    history_dicts = []
    for item in history:
        if isinstance(item, dict):
            role = item.get("role", "user")
            content = item.get("content", "")
            # Handle content being a list (multimodal) vs string
            if isinstance(content, list):
                # Extract text from content parts
                content = " ".join(
                    part.get("text", "") if isinstance(part, dict) else str(part)
                    for part in content
                )
            history_dicts.append({"role": role, "content": str(content)})

    # Run the async bot.run() in a sync context
    response = asyncio.run(bot.run(message, history=history_dicts))

    # Add indicator if conversation ended
    if response.done:
        return f"{response.message}\n\n[Conversation ended - type a new message to start over]"

    return response.message


# Create the Gradio interface
demo = gr.ChatInterface(
    fn=chat,
    title="Hyperfunc Chat Agent Demo",
    description="A simple chatbot demonstrating the CHAT agent type from hyperfunc. "
    "The bot uses ES-optimizable hyperparameters to control response generation.",
    examples=[
        "Hello!",
        "What can you do?",
        "Can you help me with something?",
        "Thank you!",
        "Goodbye!",
    ],
)

if __name__ == "__main__":
    demo.launch()
