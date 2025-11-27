"""Tests for agent type system (FLOW, CHAT, GAME)."""

import asyncio

import pytest

from hyperfunc import (
    AgentType,
    ChatResponse,
    ConversationResult,
    EpisodeResult,
    Example,
    FlowResponse,
    GameResponse,
    HyperSystem,
)


class TestFlowAgent:
    """Tests for FLOW agent type (default single I/O)."""

    @pytest.mark.asyncio
    async def test_flow_agent_default(self):
        """FLOW is the default agent type."""

        class SimpleSystem(HyperSystem):
            async def run(self, x):
                return x * 2

        system = SimpleSystem()
        assert system.agent_type == AgentType.FLOW

    @pytest.mark.asyncio
    async def test_flow_agent_evaluate(self):
        """FLOW agent processes examples as single I/O pairs."""

        class DoubleSystem(HyperSystem):
            async def run(self, x):
                return x * 2

        system = DoubleSystem()
        examples = [
            Example(inputs={"x": 5}, expected=10),
            Example(inputs={"x": 3}, expected=6),
        ]

        def metric(result, expected):
            return 1.0 if result == expected else 0.0

        score = await system.evaluate(examples, metric)
        assert score == 1.0

    @pytest.mark.asyncio
    async def test_flow_agent_with_flow_response(self):
        """FLOW agent can return FlowResponse which gets unwrapped."""

        class ResponseSystem(HyperSystem):
            async def run(self, x):
                return FlowResponse(output=x * 2)

        system = ResponseSystem()
        examples = [Example(inputs={"x": 5}, expected=10)]

        def metric(result, expected):
            return 1.0 if result == expected else 0.0

        score = await system.evaluate(examples, metric)
        assert score == 1.0


class TestChatAgent:
    """Tests for CHAT agent type (multi-turn conversation)."""

    @pytest.mark.asyncio
    async def test_chat_agent_simple_conversation(self):
        """CHAT agent processes examples as full conversations."""

        class EchoBot(HyperSystem):
            agent_type = AgentType.CHAT

            async def run(self, message, history=None):
                # Echo bot that says done after 2 turns
                turn_count = len(history) // 2 if history else 0
                done = turn_count >= 1
                return ChatResponse(message=f"Echo: {message}", done=done)

        system = EchoBot()
        assert system.agent_type == AgentType.CHAT

        # Conversation example: two user messages
        examples = [
            Example(
                inputs={"conversation": ["Hello", "How are you?"]},
                expected={"turns": 2, "done": True},
            )
        ]

        def metric(preds, expected_list):
            # metric receives lists: preds is list of ConversationResult
            scores = []
            for result, exp in zip(preds, expected_list):
                if not isinstance(result, ConversationResult):
                    scores.append(0.0)
                    continue
                turns_match = result.turns == exp["turns"]
                done_match = result.done == exp["done"]
                scores.append(1.0 if turns_match and done_match else 0.0)
            return sum(scores) / len(scores) if scores else 0.0

        score = await system.evaluate(examples, metric)
        assert score == 1.0

    @pytest.mark.asyncio
    async def test_chat_agent_history_accumulates(self):
        """CHAT agent accumulates history across turns."""

        class HistoryBot(HyperSystem):
            agent_type = AgentType.CHAT

            async def run(self, message, history=None):
                history = history or []
                # Return the count of messages in history
                count = len(history)
                done = count >= 2  # Done after 2 turns (4 messages in history)
                return ChatResponse(message=f"History has {count} messages", done=done)

        system = HistoryBot()
        examples = [
            Example(
                inputs={"conversation": ["First", "Second"]},
                expected={"turns": 2},
            )
        ]

        def metric(preds, expected_list):
            scores = []
            for result, exp in zip(preds, expected_list):
                scores.append(1.0 if result.turns == exp["turns"] else 0.0)
            return sum(scores) / len(scores) if scores else 0.0

        score = await system.evaluate(examples, metric)
        assert score == 1.0

    @pytest.mark.asyncio
    async def test_chat_agent_early_done(self):
        """CHAT agent can terminate early with done=True."""

        class QuickBot(HyperSystem):
            agent_type = AgentType.CHAT

            async def run(self, message, history=None):
                # Always done immediately
                return ChatResponse(message="Goodbye!", done=True)

        system = QuickBot()
        examples = [
            Example(
                inputs={"conversation": ["Hello", "More", "Even more"]},
                expected={"turns": 1, "done": True},  # Only 1 turn despite 3 messages
            )
        ]

        def metric(preds, expected_list):
            scores = []
            for result, exp in zip(preds, expected_list):
                scores.append(1.0 if result.turns == exp["turns"] and result.done else 0.0)
            return sum(scores) / len(scores) if scores else 0.0

        score = await system.evaluate(examples, metric)
        assert score == 1.0


class TestGameAgent:
    """Tests for GAME agent type (RL-style episodes)."""

    @pytest.mark.asyncio
    async def test_game_agent_simple_episode(self):
        """GAME agent processes examples as episodes."""

        class SimpleGameAgent(HyperSystem):
            agent_type = AgentType.GAME

            async def run(self, observation):
                # Always take action 0
                return GameResponse(action=0)

        system = SimpleGameAgent()
        assert system.agent_type == AgentType.GAME

        # Episode example with mock environment
        examples = [
            Example(
                inputs={
                    "env": "mock",
                    "seed": 42,
                    "max_steps": 5,
                    "env_config": {
                        "reward_per_step": 1.0,
                        "done_after": 3,
                    },
                },
                expected={"min_reward": 2.0},
            )
        ]

        # Mock environment factory
        class MockEnv:
            def __init__(self, config):
                self.config = config
                self.steps = 0

            def reset(self, seed=None):
                self.steps = 0
                return {"obs": 0}

            def step(self, action):
                self.steps += 1
                reward = self.config.get("reward_per_step", 1.0)
                done = self.steps >= self.config.get("done_after", 10)
                return {"obs": self.steps}, reward, done, False, {}

        # Monkey-patch _make_env for testing
        original_make_env = system._make_env

        def mock_make_env(env_name, seed=None, config=None):
            return MockEnv(config or {})

        system._make_env = mock_make_env

        try:

            def metric(preds, expected_list):
                scores = []
                for result, exp in zip(preds, expected_list):
                    if not isinstance(result, EpisodeResult):
                        scores.append(0.0)
                        continue
                    scores.append(1.0 if result.total_reward >= exp["min_reward"] else 0.0)
                return sum(scores) / len(scores) if scores else 0.0

            score = await system.evaluate(examples, metric)
            assert score == 1.0
        finally:
            system._make_env = original_make_env

    @pytest.mark.asyncio
    async def test_game_agent_episode_result_fields(self):
        """GAME agent returns EpisodeResult with all fields populated."""

        class CountingAgent(HyperSystem):
            agent_type = AgentType.GAME

            async def run(self, observation):
                return GameResponse(action=1, value=0.5)

        system = CountingAgent()

        # Simple mock environment
        class CountEnv:
            def __init__(self):
                self.steps = 0

            def reset(self, seed=None):
                self.steps = 0
                return 0

            def step(self, action):
                self.steps += 1
                done = self.steps >= 2
                return self.steps, 10.0, done, False, {"custom": "data"}

        system._make_env = lambda env_name, seed=None, config=None: CountEnv()

        example = Example(
            inputs={"env": "count", "max_steps": 10},
            expected={},
        )

        result = await system._run_episode(example)

        assert isinstance(result, EpisodeResult)
        assert result.total_reward == 20.0  # 10 * 2 steps
        assert result.steps == 2
        assert result.done is True
        assert result.truncated is False


class TestAgentTypeIntegration:
    """Integration tests for agent types."""

    @pytest.mark.asyncio
    async def test_different_agent_types_evaluate_differently(self):
        """Different agent types process the same evaluation differently."""

        # FLOW: processes input directly
        class FlowSystem(HyperSystem):
            async def run(self, x):
                return x

        # CHAT: needs conversation structure
        class ChatSystem(HyperSystem):
            agent_type = AgentType.CHAT

            async def run(self, message, history=None):
                return ChatResponse(message=message, done=True)

        flow = FlowSystem()
        chat = ChatSystem()

        # FLOW metric (receives lists)
        def flow_metric(preds, expected_list):
            scores = []
            for result, exp in zip(preds, expected_list):
                scores.append(1.0 if result == exp else 0.5)
            return sum(scores) / len(scores) if scores else 0.0

        # FLOW example
        flow_examples = [Example(inputs={"x": 42}, expected=42)]
        flow_score = await flow.evaluate(flow_examples, flow_metric)
        assert flow_score == 1.0

        # CHAT metric (receives lists)
        def chat_metric(preds, expected_list):
            scores = []
            for result, exp in zip(preds, expected_list):
                if isinstance(result, ConversationResult):
                    scores.append(1.0 if result.turns == exp["turns"] else 0.0)
                else:
                    scores.append(0.0)
            return sum(scores) / len(scores) if scores else 0.0

        chat_examples = [
            Example(inputs={"conversation": ["hi"]}, expected={"turns": 1})
        ]
        chat_score = await chat.evaluate(chat_examples, chat_metric)
        assert chat_score == 1.0

    @pytest.mark.asyncio
    async def test_response_types_serialization(self):
        """Response types can be converted to dicts for serialization."""

        flow = FlowResponse(output={"result": 123})
        assert flow.output == {"result": 123}

        chat = ChatResponse(message="Hello", done=False, metadata={"tokens": 10})
        assert chat.message == "Hello"
        assert chat.done is False
        assert chat.metadata == {"tokens": 10}

        game = GameResponse(action=[0, 1], value=0.95, metadata={"probs": [0.5, 0.5]})
        assert game.action == [0, 1]
        assert game.value == 0.95

    @pytest.mark.asyncio
    async def test_conversation_result_to_dict(self):
        """ConversationResult can be serialized to dict."""
        from hyperfunc import ConversationTurn

        result = ConversationResult(
            history=[
                ConversationTurn(role="user", content="Hello"),
                ConversationTurn(role="assistant", content="Hi there!"),
            ],
            turns=1,
            done=True,
        )

        d = result.to_dict()
        assert d["turns"] == 1
        assert d["done"] is True
        assert len(d["history"]) == 2
        assert d["history"][0]["role"] == "user"

    @pytest.mark.asyncio
    async def test_episode_result_to_dict(self):
        """EpisodeResult can be serialized to dict."""
        result = EpisodeResult(
            total_reward=100.0,
            steps=50,
            done=True,
            truncated=False,
            info={"level": 2},
        )

        d = result.to_dict()
        assert d["total_reward"] == 100.0
        assert d["steps"] == 50
        assert d["done"] is True
        assert d["info"] == {"level": 2}
