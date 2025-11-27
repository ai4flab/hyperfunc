#!/usr/bin/env python3
"""Demo: GAME agent with RL-style episode execution and evaluation.

This demo shows how to build an RL-style agent using the GAME agent type.
The agent interacts with an environment, taking actions based on observations
and accumulating rewards over an episode.

Key concepts demonstrated:
- AgentType.GAME for RL-style episodes
- GameResponse with action (and optional value estimate)
- Custom environment with reset() and step()
- Episode evaluation with reward-based metrics
- Using LoRAWeight for learnable policy parameters

Run with:
    python demos/game_agent/run.py
"""

import asyncio
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch

from hyperfunc import (
    AgentType,
    EpisodeResult,
    Example,
    GameResponse,
    HyperSystem,
    LoRAWeight,
    hyperfunction,
)


# Custom environment: GridWorld
# The agent must navigate from start to goal in a 5x5 grid
@dataclass
class GridWorldEnv:
    """Simple grid world environment.

    The agent starts at (0, 0) and must reach the goal at (4, 4).
    Actions: 0=up, 1=right, 2=down, 3=left

    Rewards:
    - +10 for reaching the goal
    - -0.1 per step (encourages efficiency)
    - -1 for hitting a wall (invalid move)
    """

    size: int = 5
    agent_pos: Tuple[int, int] = (0, 0)
    goal_pos: Tuple[int, int] = (4, 4)
    max_steps: int = 50

    def __post_init__(self):
        self.steps = 0

    def reset(self, seed: int | None = None) -> Dict[str, Any]:
        """Reset environment to initial state."""
        if seed is not None:
            random.seed(seed)
        self.agent_pos = (0, 0)
        self.steps = 0
        return self._get_observation()

    def _get_observation(self) -> Dict[str, Any]:
        """Get current observation."""
        ax, ay = self.agent_pos
        gx, gy = self.goal_pos
        return {
            "agent_x": ax,
            "agent_y": ay,
            "goal_x": gx,
            "goal_y": gy,
            "distance": abs(ax - gx) + abs(ay - gy),  # Manhattan distance
        }

    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, bool, Dict]:
        """Take an action and return (obs, reward, done, truncated, info)."""
        self.steps += 1
        ax, ay = self.agent_pos

        # Action mapping: 0=up, 1=right, 2=down, 3=left
        dx, dy = [(0, -1), (1, 0), (0, 1), (-1, 0)][action]
        new_x, new_y = ax + dx, ay + dy

        # Check bounds
        if 0 <= new_x < self.size and 0 <= new_y < self.size:
            self.agent_pos = (new_x, new_y)
            reward = -0.1  # Step cost
        else:
            reward = -1.0  # Wall penalty

        # Check goal
        done = self.agent_pos == self.goal_pos
        if done:
            reward = 10.0

        # Check truncation
        truncated = self.steps >= self.max_steps

        obs = self._get_observation()
        info = {"steps": self.steps}

        return obs, reward, done, truncated, info


# Define MLP policy with two layers:
# - Hidden layer: 5 inputs -> 8 hidden units (5*8 = 40 params + 8 bias = 48)
# - Output layer: 8 hidden -> 4 actions (8*4 = 32 params + 4 bias = 36)
# Total: 84 parameters packed into a single weight matrix for ES optimization

# We pack all MLP weights into one LoRAWeight matrix
# Layout: row 0-7 = hidden weights (8x5) + bias (8x1) = 8x6
#         row 8-11 = output weights (4x8) + bias (4x1) = 4x9
# Use a 12x9 matrix to hold everything
PolicyWeights = LoRAWeight.create(out_dim=12, in_dim=9, noise_rank=4)


@hyperfunction(hp_type=PolicyWeights, optimize_hparams=True)
async def select_action(observation: Dict[str, Any], hp) -> int:
    """Select an action based on observation using a 2-layer MLP.

    Architecture:
    - Input: 5 features (agent_x, agent_y, goal_x, goal_y, distance)
    - Hidden: 8 units with ReLU activation
    - Output: 4 action logits (up, right, down, left)

    All weights are packed into hp.weight and unpacked here.
    ES optimizes the full weight matrix.
    """
    # Convert observation to feature tensor
    features = torch.tensor(
        [
            observation["agent_x"] / 4.0,  # Normalize to [0, 1]
            observation["agent_y"] / 4.0,
            observation["goal_x"] / 4.0,
            observation["goal_y"] / 4.0,
            observation["distance"] / 8.0,  # Max Manhattan distance is 8
        ],
        dtype=torch.float32,
    )

    # Unpack weights from hp.weight (12x9 matrix)
    w = hp.weight

    # Hidden layer weights: first 8 rows, first 5 cols = W1 (8x5)
    # Hidden layer bias: first 8 rows, col 5 = b1 (8,)
    W1 = w[:8, :5]  # (8, 5)
    b1 = w[:8, 5]   # (8,)

    # Output layer weights: rows 8-11, first 8 cols = W2 (4x8)
    # Output layer bias: rows 8-11, col 8 = b2 (4,)
    W2 = w[8:12, :8]  # (4, 8)
    b2 = w[8:12, 8]   # (4,)

    # Forward pass
    hidden = torch.relu(features @ W1.T + b1)  # (8,)
    logits = hidden @ W2.T + b2  # (4,)

    # Select action with highest logit (greedy policy)
    action = int(logits.argmax().item())

    return action


class GridWorldAgent(HyperSystem):
    """Agent for navigating GridWorld.

    The agent_type = GAME means:
    - evaluate() processes each Example as an episode
    - run() is called repeatedly until episode ends
    - run() receives observation and returns GameResponse
    - Episode ends when done=True or truncated=True
    """

    agent_type = AgentType.GAME

    def _make_env(self, env_spec: Any, seed: int | None = None) -> Any:
        """Create GridWorld environment."""
        if isinstance(env_spec, str) and env_spec == "gridworld":
            return GridWorldEnv()
        if isinstance(env_spec, GridWorldEnv):
            return env_spec
        raise ValueError(f"Unknown environment: {env_spec}")

    async def run(self, observation: Dict[str, Any]) -> GameResponse:
        """Select action for current observation.

        Args:
            observation: Dict with agent_x, agent_y, goal_x, goal_y, distance

        Returns:
            GameResponse with selected action
        """
        action = await select_action(observation)

        # Optional: compute value estimate (for actor-critic methods)
        # For this simple demo, we just return the action
        return GameResponse(action=action)


def episode_reward_metric(
    results: List[EpisodeResult],
    expected: List[Dict[str, Any]],
) -> float:
    """Evaluate agent performance based on episode rewards.

    Scoring based on:
    - Reaching the goal (done=True): major component
    - Efficiency: fewer steps = higher score
    - Total reward as tiebreaker
    """
    if not results:
        return 0.0

    total_score = 0.0

    for result, exp in zip(results, expected):
        if result.done:
            # Reached goal! Score based on efficiency
            # Optimal is 8 steps, max is 50
            # Score: 1.0 for 8 steps, 0.5 for ~30 steps, 0.0 for 50+ steps
            optimal_steps = 8
            efficiency = max(0.0, 1.0 - (result.steps - optimal_steps) / 42.0)
            total_score += 0.5 + 0.5 * efficiency  # 0.5 to 1.0 range
        else:
            # Didn't reach goal - score based on how close we got
            # Use reward as proxy (less negative = got closer)
            # Normalize from [-50, 0] to [0, 0.5]
            normalized = (result.total_reward + 50.0) / 100.0
            total_score += max(0.0, min(0.5, normalized))

    return total_score / len(results)


async def main():
    """Run the GAME agent demo."""
    print("=" * 60)
    print("GAME Agent Demo: GridWorld Navigation")
    print("=" * 60)

    # Create the agent
    agent = GridWorldAgent()

    # Define episode examples for evaluation
    examples = [
        Example(
            inputs={"env": "gridworld", "seed": 42, "max_steps": 50},
            expected={"min_reward": 5.0},
        ),
        Example(
            inputs={"env": "gridworld", "seed": 123, "max_steps": 50},
            expected={"min_reward": 5.0},
        ),
        Example(
            inputs={"env": "gridworld", "seed": 456, "max_steps": 50},
            expected={"min_reward": 5.0},
        ),
    ]

    # Evaluate before training
    print("\n1. Initial evaluation (random policy)...")
    initial_score = await agent.evaluate(examples, episode_reward_metric)
    print(f"   Initial score: {initial_score:.2%}")

    # Demo: Run a single episode and show trajectory
    print("\n2. Demo episode trajectory:")
    print("-" * 40)

    env = GridWorldEnv()
    obs = env.reset(seed=42)
    print(f"   Start: ({obs['agent_x']}, {obs['agent_y']})")
    print(f"   Goal:  ({obs['goal_x']}, {obs['goal_y']})")
    print()

    action_names = ["up", "right", "down", "left"]
    trajectory = [(obs["agent_x"], obs["agent_y"])]
    total_reward = 0.0

    for step in range(15):  # Limit steps for demo
        response = await agent.run(observation=obs)
        action = response.action

        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        trajectory.append((obs["agent_x"], obs["agent_y"]))

        print(
            f"   Step {step + 1}: action={action_names[action]:5s} -> "
            f"pos=({obs['agent_x']}, {obs['agent_y']}) reward={reward:+.1f}"
        )

        if done:
            print("   GOAL REACHED!")
            break
        if truncated:
            print("   (truncated)")
            break

    print(f"\n   Total reward: {total_reward:.1f}")

    # Visualize trajectory on grid
    print("\n3. Grid visualization:")
    print("-" * 40)
    grid = [["." for _ in range(5)] for _ in range(5)]
    grid[0][0] = "S"  # Start
    grid[4][4] = "G"  # Goal

    # Mark trajectory
    for i, (x, y) in enumerate(trajectory[1:-1], 1):
        if grid[y][x] == ".":
            grid[y][x] = str(i % 10)

    # Mark final position
    if trajectory:
        fx, fy = trajectory[-1]
        if (fx, fy) == (4, 4):
            grid[fy][fx] = "*"  # Reached goal
        else:
            grid[fy][fx] = "X"  # Final position

    for row in grid:
        print("   " + " ".join(row))

    print("   Legend: S=start, G=goal, numbers=path, *=reached goal, X=final pos")

    # Show hyperparameter state
    print("\n4. Policy weights (shape):")
    hp_state = agent.get_hp_state()
    for name, tensor in hp_state.items():
        print(f"   {name}: {tuple(tensor.shape)}")
        print(f"   Sample weights:\n{tensor[:2, :3]}")

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)
    print("\nNote: To improve the agent's performance, you would run:")
    print("  await agent.optimize(examples, episode_reward_metric)")
    print("This uses ES to evolve the policy weights over many episodes.")


if __name__ == "__main__":
    asyncio.run(main())
