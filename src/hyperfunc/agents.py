"""Agent type system for hyperfunc.

Defines three agent types that determine how HyperSystem.evaluate() works:
- FLOW: Single input → single output (default, current behavior)
- CHAT: Multi-turn conversation (Example = full conversation)
- GAME: RL-style episodes (Example = episode config)

The agent_type is declared at the class level and determines:
1. What an Example represents (single I/O, conversation, or episode)
2. How run() is called during evaluation (once, or in a loop)
3. What run() should return (FlowResponse, ChatResponse, or GameResponse)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class AgentType(Enum):
    """Type of agent behavior for a HyperSystem.

    Determines how evaluate() processes each Example:
    - FLOW: Calls run() once per example (default)
    - CHAT: Loops run() for each turn in a conversation until done
    - GAME: Loops run() with an environment until episode ends
    """

    FLOW = "flow"
    CHAT = "chat"
    GAME = "game"


@dataclass
class FlowResponse:
    """Response from a FLOW agent (single-turn).

    This is the simplest case - just wraps the output.
    FLOW agents can also return raw values (backwards compatible).
    """

    output: Any


@dataclass
class ChatResponse:
    """Response from a CHAT agent (multi-turn conversation).

    Attributes:
        message: The assistant's response message
        done: If True, conversation should end (e.g., user said goodbye)
        metadata: Optional extra data (e.g., confidence, tool calls)
    """

    message: str
    done: bool = False
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class GameResponse:
    """Response from a GAME agent (RL-style).

    Attributes:
        action: The action to take in the environment
        value: Optional value estimate (for actor-critic methods)
        metadata: Optional extra data (e.g., action probabilities)
    """

    action: Any
    value: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ConversationTurn:
    """A single turn in a conversation."""

    role: str  # "user" or "assistant"
    content: str
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ConversationResult:
    """Result of running a full conversation.

    Returned by _run_conversation() for evaluation.
    """

    history: List[ConversationTurn] = field(default_factory=list)
    turns: int = 0
    done: bool = False
    final_response: Optional[ChatResponse] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "history": [
                {"role": t.role, "content": t.content, "metadata": t.metadata}
                for t in self.history
            ],
            "turns": self.turns,
            "done": self.done,
        }


@dataclass
class EpisodeResult:
    """Result of running a full episode.

    Returned by _run_episode() for evaluation.
    """

    total_reward: float = 0.0
    steps: int = 0
    done: bool = False
    truncated: bool = False
    final_observation: Any = None
    info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_reward": self.total_reward,
            "steps": self.steps,
            "done": self.done,
            "truncated": self.truncated,
            "info": self.info,
        }


# Type alias for any response type
AgentResponse = FlowResponse | ChatResponse | GameResponse
