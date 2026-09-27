#!/usr/bin/env python3
"""A2: isolate discounting, exploration, and potential-based reward shaping."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path


ROWS, COLS = 5, 5
START, GOAL, TRAP = (0, 0), (4, 4), (0, 4)
ACTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))
SEEDS = (7, 19, 31, 43, 59, 71, 89, 101)
EPISODES, MAX_STEPS, ALPHA = 1_200, 40, 0.20
CHECKPOINTS = (25, 50, 100, 200, 400, 800, 1_200)
OUT = Path("results/a2_gridworld_controls")


@dataclass(frozen=True)
class Condition:
    name: str
    gamma: float
    epsilon_decay_episodes: int
    potential_shaping: bool
    note: str


CONDITIONS = (
    Condition("baseline", 0.95, 1_000, False, "long-horizon objective, gradual exploration"),
    Condition("fast_epsilon_decay", 0.95, 25, False, "only exploration schedule changes"),
    Condition("low_gamma", 0.50, 1_000, False, "only discount factor changes"),
    Condition("potential_shaping", 0.95, 1_000, True, "only potential-based shaping changes"),
)


def all_states():
    return [(row, col) for row in range(ROWS) for col in range(COLS)]


def base_transition(state, action):
    row = min(ROWS - 1, max(0, state[0] + action[0]))
    col = min(COLS - 1, max(0, state[1] + action[1]))
    nxt = (row, col)
    if nxt == GOAL:
        return nxt, 10.0, True, "goal"
    if nxt == TRAP:
        return nxt, 1.0, True, "trap"
    return nxt, -1.0, False, "ongoing"


def potential(state):
    """Potential is zero in terminal states so shaping only redistributes return."""
    if state in (GOAL, TRAP):
        return 0.0
    return -float(abs(state[0] - GOAL[0]) + abs(state[1] - GOAL[1]))


def training_reward(state, nxt, base_reward, gamma, shaped):
    if not shaped:
        return base_reward
    return base_reward + gamma * potential(nxt) - potential(state)


def greedy_action(q, state):
    return max(range(len(ACTIONS)), key=lambda index: (q[state][index], -index))


def epsilon_for_episode(episode, decay_episodes):
    progress = min(1.0, episode / max(1, decay_episodes - 1))
    return 1.0 - 0.98 * progress


def evaluate(q):
    state, total = START, 0.0
    for step in range(1, MAX_STEPS + 1):
        action_index = greedy_action(q, state)
        state, reward, done, outcome = base_transition(state, ACTIONS[action_index])
        total += reward
        if done:
            return {"outcome": outcome, "return": total, "steps": step}
    return {"outcome": "timeout", "return": total, "steps": MAX_STEPS}


def run_seed(condition, seed):
    rng = random.Random(seed)
    q = {state: [0.0] * len(ACTIONS) for state in all_states()}
    checkpoints = []
    for episode in range(EPISODES):
        state = START
        epsilon = epsilon_for_episode(episode, condition.epsilon_decay_episodes)
        for _ in range(MAX_STEPS):
            action_index = rng.randrange(len(ACTIONS)) if rng.random() < epsilon else greedy_action(q, state)
            nxt, base_reward, done, _ = base_transition(state, ACTIONS[action_index])
            reward = training_reward(state, nxt, base_reward, condition.gamma, condition.potential_shaping)
            target = reward if done else reward + condition.gamma * max(q[nxt])
            q[state][action_index] += ALPHA * (target - q[state][action_index])
            state = nxt
            if done:
                break
        if episode + 1 in CHECKPOINTS:
            checkpoints.append({"episode": episode + 1, **evaluate(q)})
    return {"seed": seed, "final": evaluate(q), "checkpoints": checkpoints}


def checkpoint_aggregate(runs):
    result = []
    for episode in CHECKPOINTS:
        outcomes = [next(item for item in run["checkpoints"] if item["episode"] == episode)["outcome"] for run in runs]
        result.append(
            {
                "episode": episode,
                "goal_rate": sum(outcome == "goal" for outcome in outcomes) / len(outcomes),
                "trap_rate": sum(outcome == "trap" for outcome in outcomes) / len(outcomes),
                "timeout_rate": sum(outcome == "timeout" for outcome in outcomes) / len(outcomes),
            }
        )
    return result


def summarize(condition, runs):
    final_outcomes = [run["final"]["outcome"] for run in runs]
    curves = checkpoint_aggregate(runs)
    first_full_goal = next((point["episode"] for point in curves if point["goal_rate"] == 1.0), None)
    return {
        "condition": asdict(condition),
        "seeds": list(SEEDS),
        "final": {
            "goal_rate": sum(outcome == "goal" for outcome in final_outcomes) / len(runs),
            "trap_rate": sum(outcome == "trap" for outcome in final_outcomes) / len(runs),
            "timeout_rate": sum(outcome == "timeout" for outcome in final_outcomes) / len(runs),
            "mean_return": sum(run["final"]["return"] for run in runs) / len(runs),
            "mean_steps": sum(run["final"]["steps"] for run in runs) / len(runs),
            "first_full_goal_checkpoint": first_full_goal,
        },
        "checkpoints": curves,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    raw_runs = {}
    summaries = []
    for condition in CONDITIONS:
        runs = [run_seed(condition, seed) for seed in SEEDS]
        raw_runs[condition.name] = runs
        summaries.append(summarize(condition, runs))
    aggregate = {
        "schema_version": "RL_LAB_A2_GRIDWORLD_CONTROLS_V1",
        "environment": {
            "rows": ROWS,
            "cols": COLS,
            "start": START,
            "goal": {"state": GOAL, "terminal_reward": 10.0},
            "nearby_terminal": {"state": TRAP, "terminal_reward": 1.0},
            "step_reward": -1.0,
            "deterministic": True,
        },
        "protocol": {"episodes": EPISODES, "max_steps": MAX_STEPS, "alpha": ALPHA, "evaluation": "greedy policy; base rewards; no exploration"},
        "summaries": summaries,
        "authorization": {"production_action": False, "training_label": False},
    }
    (OUT / "aggregate_metrics.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    (OUT / "seed_summaries.json").write_text(json.dumps(raw_runs, indent=2), encoding="utf-8")
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
