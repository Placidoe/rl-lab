#!/usr/bin/env python3
"""A1: exact planning baseline versus sampled Q-learning in a deterministic GridWorld."""

from __future__ import annotations

import json
import random
from pathlib import Path


ROWS, COLS = 5, 5
START, GOAL = (0, 0), (4, 4)
ACTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))
GAMMA, ALPHA = 0.95, 0.20
EPISODES, MAX_STEPS = 6000, 40
SEEDS = (7, 19, 31)
OUT = Path("results/a1_gridworld")


def states():
    return [(r, c) for r in range(ROWS) for c in range(COLS)]


def transition(state, action):
    r = min(ROWS - 1, max(0, state[0] + action[0]))
    c = min(COLS - 1, max(0, state[1] + action[1]))
    nxt = (r, c)
    return nxt, (10.0 if nxt == GOAL else -1.0), nxt == GOAL


def value_iteration():
    value = {s: 0.0 for s in states()}
    for iteration in range(10_000):
        updated = {}
        delta = 0.0
        for state in states():
            if state == GOAL:
                updated[state] = 0.0
                continue
            candidates = [reward + (0.0 if done else GAMMA * value[nxt]) for action in ACTIONS for nxt, reward, done in [transition(state, action)]]
            updated[state] = max(candidates)
            delta = max(delta, abs(updated[state] - value[state]))
        value = updated
        if delta < 1e-10:
            return value, iteration + 1
    raise RuntimeError("value iteration did not converge")


def greedy_action(q, state):
    values = q[state]
    return max(range(len(ACTIONS)), key=lambda idx: (values[idx], -idx))


def evaluate(q):
    state, total = START, 0.0
    actions = []
    for _ in range(MAX_STEPS):
        idx = greedy_action(q, state)
        actions.append(idx)
        state, reward, done = transition(state, ACTIONS[idx])
        total += reward
        if done:
            return {"success": True, "return": total, "steps": len(actions), "action_indices": actions}
    return {"success": False, "return": total, "steps": MAX_STEPS, "action_indices": actions}


def q_learning(seed):
    rng = random.Random(seed)
    q = {state: [0.0] * len(ACTIONS) for state in states()}
    checkpoints = []
    for episode in range(EPISODES):
        state = START
        epsilon = max(0.02, 1.0 - 0.98 * episode / (EPISODES - 1))
        for _ in range(MAX_STEPS):
            idx = rng.randrange(len(ACTIONS)) if rng.random() < epsilon else greedy_action(q, state)
            nxt, reward, done = transition(state, ACTIONS[idx])
            target = reward if done else reward + GAMMA * max(q[nxt])
            q[state][idx] += ALPHA * (target - q[state][idx])
            state = nxt
            if done:
                break
        if (episode + 1) % 500 == 0:
            checkpoints.append({"episode": episode + 1, **evaluate(q)})
    return {"seed": seed, "evaluation": evaluate(q), "checkpoints": checkpoints}


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    optimal_value, iterations = value_iteration()
    runs = [q_learning(seed) for seed in SEEDS]
    aggregate = {
        "schema_version": "RL_LAB_A1_GRIDWORLD_V1",
        "environment": {"rows": ROWS, "cols": COLS, "start": START, "goal": GOAL, "deterministic": True},
        "algorithm": {"gamma": GAMMA, "alpha": ALPHA, "episodes": EPISODES, "epsilon": "linear 1.0 -> 0.02"},
        "planning_reference": {"value_iteration_iterations": iterations, "optimal_start_value": optimal_value[START]},
        "q_learning": {
            "seeds": list(SEEDS),
            "successes": sum(run["evaluation"]["success"] for run in runs),
            "mean_return": sum(run["evaluation"]["return"] for run in runs) / len(runs),
            "mean_steps": sum(run["evaluation"]["steps"] for run in runs) / len(runs),
        },
        "authorization": {"production_action": False, "training_label": False},
    }
    (OUT / "aggregate_metrics.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    (OUT / "seed_summaries.json").write_text(json.dumps(runs, indent=2), encoding="utf-8")
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
