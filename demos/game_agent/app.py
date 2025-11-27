#!/usr/bin/env python3
"""Gradio web UI for the GAME agent demo.

This provides an interactive GridWorld visualization for the GridWorldAgent.

Run with:
    uv run --extra demos python demos/game_agent/app.py
"""

import asyncio
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import gradio as gr
import torch

# Add the demo directory to path so we can import run.py
sys.path.insert(0, str(Path(__file__).parent))

from run import GridWorldAgent, GridWorldEnv, select_action, PolicyWeights, Example, episode_reward_metric  # noqa: E402
from hyperfunc import ESHybridSystemOptimizer, NoOpPromptOptimizer  # noqa: E402


def create_agent(steps: int = 10, pop_size: int = 16, sigma: float = 0.3, lr: float = 0.2):
    """Create a GridWorldAgent with ES optimizer configured."""
    return GridWorldAgent(
        system_optimizer=ESHybridSystemOptimizer(
            steps=steps,
            pop_size=pop_size,
            sigma=sigma,
            lr=lr,
            antithetic=True,
        ),
        # Disable prompt optimization - we only optimize policy weights
        prompt_optimizer=NoOpPromptOptimizer(),
    )


# Global agent instance with registered hyperfunction
# Start with default ES settings
agent = create_agent()

# Start with RANDOM weights - the agent will be bad initially
# Training with ES will improve these weights
init_weights = torch.randn(PolicyWeights.shape()) * 0.1
agent.register_hyperfunction(select_action, hp_init=init_weights)

# Track training status
training_log = []


@dataclass
class GameState:
    """Holds the current game state."""

    env: GridWorldEnv = field(default_factory=GridWorldEnv)
    trajectory: List[Tuple[int, int]] = field(default_factory=list)
    total_reward: float = 0.0
    last_action: str = "-"
    last_reward: float = 0.0
    done: bool = False
    truncated: bool = False
    steps: int = 0


# Global game state
state = GameState()


def render_grid() -> str:
    """Render the grid as an HTML table with emojis."""
    grid_size = state.env.size
    ax, ay = state.env.agent_pos
    gx, gy = state.env.goal_pos

    # CSS for the grid
    css = """
    <style>
        .grid-table {
            border-collapse: collapse;
            margin: 20px auto;
        }
        .grid-table td {
            width: 60px;
            height: 60px;
            text-align: center;
            font-size: 32px;
            border: 2px solid #ccc;
            background-color: #f5f5f5;
        }
        .grid-table .agent {
            background-color: #e3f2fd;
        }
        .grid-table .goal {
            background-color: #e8f5e9;
        }
        .grid-table .path {
            background-color: #fff3e0;
        }
        .grid-table .goal-reached {
            background-color: #c8e6c9;
        }
    </style>
    """

    # Build the grid
    html = css + '<table class="grid-table">'

    for y in range(grid_size):
        html += "<tr>"
        for x in range(grid_size):
            cell_class = ""
            if (x, y) == (ax, ay) and (x, y) == (gx, gy):
                # Agent reached the goal
                emoji = "🏆"
                cell_class = "goal-reached"
            elif (x, y) == (ax, ay):
                emoji = "🤖"
                cell_class = "agent"
            elif (x, y) == (gx, gy):
                emoji = "🎯"
                cell_class = "goal"
            elif (x, y) in state.trajectory[:-1]:  # Path (excluding current pos)
                emoji = "👣"
                cell_class = "path"
            else:
                emoji = "⬜"

            html += f'<td class="{cell_class}">{emoji}</td>'
        html += "</tr>"

    html += "</table>"
    return html


def get_status() -> str:
    """Get the current game status as formatted text."""
    action_map = {0: "⬆️ Up", 1: "➡️ Right", 2: "⬇️ Down", 3: "⬅️ Left", "-": "-"}
    action_str = action_map.get(state.last_action, str(state.last_action))

    status = f"""
**Episode Status**
- Steps: {state.steps}
- Total Reward: {state.total_reward:.1f}
- Last Action: {action_str}
- Last Reward: {state.last_reward:+.1f}

**Agent Position**: ({state.env.agent_pos[0]}, {state.env.agent_pos[1]})
**Goal Position**: ({state.env.goal_pos[0]}, {state.env.goal_pos[1]})
**Distance to Goal**: {abs(state.env.agent_pos[0] - state.env.goal_pos[0]) + abs(state.env.agent_pos[1] - state.env.goal_pos[1])}
"""

    if state.done:
        status += "\n**🎉 GOAL REACHED!**"
    elif state.truncated:
        status += "\n**⏰ Episode truncated (max steps)**"

    return status


def reset_game() -> Tuple[str, str]:
    """Reset the game to initial state."""
    global state
    state = GameState()
    state.env.reset(seed=42)
    state.trajectory = [(0, 0)]
    return render_grid(), get_status()


def take_step() -> Tuple[str, str]:
    """Take one step in the environment."""
    if state.done or state.truncated:
        return render_grid(), get_status()

    # Get observation
    obs = state.env._get_observation()

    # Get action from agent
    response = asyncio.run(agent.run(observation=obs))
    action = response.action

    # Take step in environment
    _, reward, done, truncated, _ = state.env.step(action)

    # Update state
    state.last_action = action
    state.last_reward = reward
    state.total_reward += reward
    state.done = done
    state.truncated = truncated
    state.steps += 1
    state.trajectory.append(state.env.agent_pos)

    return render_grid(), get_status()


def auto_play(speed: float) -> Tuple[str, str]:
    """Auto-play until episode ends."""
    while not state.done and not state.truncated:
        take_step()
        time.sleep(1.0 / speed)  # Control speed

    return render_grid(), get_status()


def run_episode_for_reward() -> float:
    """Run one episode and return total reward (for training evaluation).

    Uses shaped rewards: gives bonus for getting closer to goal, penalty for
    moving away. This makes the reward signal denser and easier to learn.
    """
    env = GridWorldEnv()
    env.reset(seed=None)  # Random seed for variety
    total_reward = 0.0

    # Track distance for reward shaping
    prev_distance = env._get_observation()["distance"]

    for _ in range(50):  # max steps
        obs = env._get_observation()
        response = asyncio.run(agent.run(observation=obs))
        _, reward, done, truncated, _ = env.step(response.action)

        # Add shaped reward: +1 for getting closer, -1 for moving away
        curr_distance = env._get_observation()["distance"]
        shaped_reward = (prev_distance - curr_distance)  # +1 if closer, -1 if farther
        prev_distance = curr_distance

        total_reward += reward + shaped_reward
        if done or truncated:
            break

    return total_reward


def train_agent(num_generations: int, pop_size: int) -> str:
    """Train the agent using hyperfunc's system.optimize()."""
    global agent, training_log
    training_log = []

    num_generations = int(num_generations)
    pop_size = int(pop_size)
    # Ensure even pop_size for antithetic sampling
    if pop_size % 2 != 0:
        pop_size += 1

    log_text = "**Training Log:**\n\n"

    # Evaluate before training
    before_rewards = [run_episode_for_reward() for _ in range(5)]
    before_avg = sum(before_rewards) / len(before_rewards)
    log_text += f"Before training: Avg={before_avg:.1f}\n\n"

    # Save current weights so we can restore them if needed
    current_weights = agent.get_hp_state()["select_action"].clone()

    # Create a new agent with the ES optimizer configured
    agent = create_agent(steps=num_generations, pop_size=pop_size, sigma=0.3, lr=0.2)
    agent.register_hyperfunction(select_action, hp_init=current_weights)

    # Create training examples for the ES optimizer
    # Each example is an episode configuration
    examples = [
        Example(
            inputs={"env": "gridworld", "seed": i, "max_steps": 50},
            expected={"min_reward": 5.0},
        )
        for i in range(10)  # Multiple episode seeds for better evaluation
    ]

    # Run optimization using hyperfunc's system.optimize()
    log_text += f"Running ES optimization: {num_generations} steps, pop_size={pop_size}\n"
    asyncio.run(agent.optimize(examples, episode_reward_metric))

    # Evaluate after training
    after_rewards = [run_episode_for_reward() for _ in range(5)]
    after_avg = sum(after_rewards) / len(after_rewards)

    log_text += f"\nAfter training:\n"
    log_text += f"Avg reward: {before_avg:.1f} → {after_avg:.1f}\n\n"

    improvement = after_avg - before_avg
    if after_avg > 5:
        log_text += f"✅ Agent learned to reach the goal! (improved by {improvement:+.1f})"
    elif improvement > 5:
        log_text += f"📈 Good improvement (+{improvement:.1f})! Try more generations."
    elif improvement > 0:
        log_text += f"📈 Some improvement (+{improvement:.1f}). Try more generations."
    else:
        log_text += "❌ No improvement. Try different settings or more generations."

    return log_text


def reset_agent_weights() -> str:
    """Reset agent to random weights."""
    new_weights = torch.randn(PolicyWeights.shape()) * 0.1
    agent.set_hp_state({"select_action": new_weights})
    return "Agent reset to random weights. It will be bad again!"


# Manual control functions
def manual_move(direction: int):
    """Take a manual step in the given direction."""
    if state.done or state.truncated:
        return render_grid(), get_status()

    # Take step in environment
    _, reward, done, truncated, _ = state.env.step(direction)

    # Update state
    action_names = {0: "Up", 1: "Right", 2: "Down", 3: "Left"}
    state.last_action = direction
    state.last_reward = reward
    state.total_reward += reward
    state.done = done
    state.truncated = truncated
    state.steps += 1
    state.trajectory.append(state.env.agent_pos)

    return render_grid(), get_status()

def move_up():
    return manual_move(0)

def move_right():
    return manual_move(1)

def move_down():
    return manual_move(2)

def move_left():
    return manual_move(3)


# Create the Gradio interface
with gr.Blocks(title="GridWorld Agent Demo") as demo:
    gr.Markdown(
        """
    # Hyperfunc Game Agent Demo: GridWorld Navigation

    Navigate from **Start** (top-left) to **Goal** (bottom-right) in a 5x5 grid.

    **Two modes:**
    1. **Manual**: Use the arrow buttons below to control the agent yourself
    2. **AI Agent**: Let the trained AI agent play (uses ES-optimized policy weights)

    **Rewards**: +10 for reaching goal, -0.1 per step, -1 for hitting walls
    """
    )

    with gr.Row():
        with gr.Column(scale=2):
            grid_display = gr.HTML(value=render_grid(), label="Grid")

            # Manual control buttons
            gr.Markdown("### Manual Control")
            with gr.Row():
                gr.Column(scale=1)
                up_btn = gr.Button("⬆️ Up", scale=1)
                gr.Column(scale=1)
            with gr.Row():
                left_btn = gr.Button("⬅️ Left", scale=1)
                down_btn = gr.Button("⬇️ Down", scale=1)
                right_btn = gr.Button("➡️ Right", scale=1)

        with gr.Column(scale=1):
            status_display = gr.Markdown(value=get_status(), label="Status")

            gr.Markdown("### Controls")
            reset_btn = gr.Button("🔄 Reset Game", variant="secondary")

            gr.Markdown("### AI Agent")
            step_btn = gr.Button("🤖 AI Step (one move)", variant="primary")
            auto_btn = gr.Button("▶️ AI Auto-play", variant="primary")

            speed_slider = gr.Slider(
                minimum=1,
                maximum=10,
                value=5,
                step=1,
                label="Auto-play Speed",
            )

    # Training section
    gr.Markdown("---")
    gr.Markdown(
        """
    ## Training the AI Agent

    The AI agent starts with **random weights** and performs poorly. Use Evolution Strategies (ES)
    to train it! ES works by:
    1. Creating a population of slightly different weight variations
    2. Testing each one on the game (an "episode" = one full game from start to goal/timeout)
    3. Keeping changes that led to higher rewards
    4. Repeating for multiple generations
    """
    )

    with gr.Row():
        with gr.Column(scale=1):
            gen_slider = gr.Slider(
                minimum=1,
                maximum=20,
                value=5,
                step=1,
                label="Generations",
            )
            pop_slider = gr.Slider(
                minimum=4,
                maximum=32,
                value=8,
                step=4,
                label="Population Size",
            )
            with gr.Row():
                train_btn = gr.Button("🎓 Train Agent", variant="primary")
                reset_weights_btn = gr.Button("🔀 Reset to Random", variant="secondary")

        with gr.Column(scale=2):
            training_log = gr.Markdown(
                value="*Training log will appear here after you click 'Train Agent'*",
                label="Training Log"
            )

    # Connect training buttons
    train_btn.click(
        fn=train_agent,
        inputs=[gen_slider, pop_slider],
        outputs=[training_log]
    )
    reset_weights_btn.click(
        fn=reset_agent_weights,
        outputs=[training_log]
    )

    # Connect manual control buttons
    up_btn.click(fn=move_up, outputs=[grid_display, status_display])
    down_btn.click(fn=move_down, outputs=[grid_display, status_display])
    left_btn.click(fn=move_left, outputs=[grid_display, status_display])
    right_btn.click(fn=move_right, outputs=[grid_display, status_display])

    # Connect AI control buttons
    reset_btn.click(fn=reset_game, outputs=[grid_display, status_display])
    step_btn.click(fn=take_step, outputs=[grid_display, status_display])
    auto_btn.click(
        fn=auto_play, inputs=[speed_slider], outputs=[grid_display, status_display]
    )

    gr.Markdown(
        """
    ---
    **Legend**: 🤖 Agent | 🎯 Goal | 👣 Path | ⬜ Empty | 🏆 Goal Reached

    **Purpose**: This demo shows how hyperfunc's GAME agent type works with RL environments.
    The AI agent's policy is a small neural network whose weights can be optimized using
    Evolution Strategies (ES) - no backpropagation needed!
    """
    )

# Initialize the game
reset_game()

if __name__ == "__main__":
    demo.launch()
