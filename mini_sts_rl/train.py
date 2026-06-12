"""Training, evaluation, and plotting utilities."""
from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .agents import DQNAgent, QLearningAgent, RandomAgent, Transition
from .env import ACTION_NAMES, MiniSlayTheSpireEnv


def run_episode_random(env: MiniSlayTheSpireEnv, agent: RandomAgent, seed: int | None = None) -> Dict[str, float]:
    env.reset(seed=seed)
    total_reward = 0.0
    steps = 0
    while not env.done:
        action = agent.act(env)
        _, reward, done, info = env.step(action)
        total_reward += reward
        steps += 1
        if done:
            break
    return {
        "reward": total_reward,
        "win": 1.0 if env.enemy_hp <= 0 else 0.0,
        "remaining_hp": float(env.player_hp),
        "turns": float(env.turn),
        "steps": float(steps),
    }




def run_random_reward_curve(
    env: MiniSlayTheSpireEnv,
    agent: RandomAgent,
    episodes: int = 200,
    seed: int = 10000,
) -> List[float]:
    """Return total reward per episode for the Random baseline.

    Random does not learn, so this is an evaluation reward curve rather than a
    training curve. It is useful as a visual baseline in the report.
    """
    rewards: List[float] = []
    for ep in range(episodes):
        result = run_episode_random(env, agent, seed=seed + ep)
        rewards.append(float(result["reward"]))
    return rewards

def train_qlearning(env: MiniSlayTheSpireEnv, episodes: int, seed: int = 42) -> Tuple[QLearningAgent, List[float]]:
    agent = QLearningAgent(n_actions=env.n_actions, seed=seed)
    rewards: List[float] = []
    for ep in range(episodes):
        env.reset(seed=seed + ep)
        total_reward = 0.0
        while not env.done:
            state = env.get_discrete_state()
            valid_actions = env.get_valid_actions()
            action = agent.act(state, valid_actions, training=True)
            _, reward, done, _ = env.step(action)
            next_state = env.get_discrete_state()
            next_valid_actions = env.get_valid_actions() if not done else list(range(env.n_actions))
            agent.update(state, action, reward, next_state, done, next_valid_actions)
            total_reward += reward
        agent.decay_epsilon()
        rewards.append(total_reward)
    return agent, rewards


def train_dqn(env: MiniSlayTheSpireEnv, episodes: int, seed: int = 42) -> Tuple[DQNAgent, List[float], List[float]]:
    agent = DQNAgent(state_dim=env.state_dim, n_actions=env.n_actions, seed=seed)
    rewards: List[float] = []
    losses: List[float] = []
    for ep in range(episodes):
        env.reset(seed=seed + ep)
        total_reward = 0.0
        state = env.get_state(normalize=True)
        while not env.done:
            valid_actions = env.get_valid_actions()
            action = agent.act(state, valid_actions, training=True)
            _, reward, done, _ = env.step(action)
            next_state = env.get_state(normalize=True)
            next_valid_actions = env.get_valid_actions() if not done else list(range(env.n_actions))
            agent.remember(Transition(state, action, reward, next_state, done, next_valid_actions))
            loss = agent.update()
            if loss is not None:
                losses.append(loss)
            state = next_state
            total_reward += reward
        agent.decay_epsilon()
        rewards.append(total_reward)
    return agent, rewards, losses


def evaluate_agent(env: MiniSlayTheSpireEnv, agent, episodes: int = 200, seed: int = 10000, agent_type: str = "random") -> Dict[str, float]:
    episode_results: List[Dict[str, float]] = []
    for ep in range(episodes):
        env.reset(seed=seed + ep)
        total_reward = 0.0
        steps = 0
        while not env.done:
            if agent_type == "random":
                action = agent.act(env)
            elif agent_type == "qlearning":
                action = agent.act(env.get_discrete_state(), env.get_valid_actions(), training=False)
            elif agent_type == "dqn":
                action = agent.act(env.get_state(normalize=True), env.get_valid_actions(), training=False)
            else:
                raise ValueError(f"Unknown agent_type: {agent_type}")
            _, reward, done, _ = env.step(action)
            total_reward += reward
            steps += 1
            if done:
                break
        episode_results.append(
            {
                "reward": total_reward,
                "win": 1.0 if env.enemy_hp <= 0 else 0.0,
                "remaining_hp": float(env.player_hp),
                "turns": float(env.turn),
                "steps": float(steps),
            }
        )

    summary = {}
    for key in ["reward", "win", "remaining_hp", "turns", "steps"]:
        values = np.array([r[key] for r in episode_results], dtype=np.float32)
        summary[f"mean_{key}"] = float(values.mean())
        summary[f"std_{key}"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    return summary




def trace_random_policy(
    env: MiniSlayTheSpireEnv,
    agent: RandomAgent | None = None,
    seed: int = 42,
    max_steps: int = 100,
) -> List[Dict[str, object]]:
    """Roll out one Random-agent episode and return detailed step rows.

    This gives a direct visual comparison to the learned DQN policy timeline.
    """
    if agent is None:
        agent = RandomAgent()
    random.seed(seed)
    env.reset(seed=seed)
    rows: List[Dict[str, object]] = []
    total_reward = 0.0
    step_idx = 0

    while not env.done and step_idx < max_steps:
        before = env.state_as_dict()
        before_turn = env.turn
        before_player_block = env.block
        before_enemy_move = env.enemy_move_name()

        action = agent.act(env)
        next_state, reward, done, info = env.step(action)
        total_reward += reward

        row: Dict[str, object] = {
            "step": step_idx,
            "turn": before_turn,
            "player_hp_before": before["player_hp"],
            "player_block_before": before_player_block,
            "enemy_hp_before": before["enemy_hp"],
            "enemy_block_before": before["enemy_block"],
            "enemy_strength_before": before["enemy_strength"],
            "enemy_attack_intent": before["enemy_attack_intent"],
            "enemy_block_intent": before["enemy_block_intent"],
            "enemy_strength_intent": before["enemy_strength_intent"],
            "enemy_vulnerable_before": before["enemy_vulnerable_turns"],
            "energy_before": before["energy"],
            "hand_strike": before["strike_in_hand"],
            "hand_defend": before["defend_in_hand"],
            "hand_bash": before["bash_in_hand"],
            "enemy_move": before_enemy_move,
            "action": action,
            "action_name": ACTION_NAMES[action],
            "damage_before_block": info.get("damage_before_block", 0),
            "blocked_damage": info.get("blocked_damage", 0),
            "damage_dealt": info.get("damage_dealt", 0),
            "damage_taken": info.get("damage_taken", 0),
            "enemy_block_gained": info.get("enemy_block_gained", 0),
            "enemy_strength_gained": info.get("enemy_strength_gained", 0),
            "enemy_block_after": info.get("enemy_block", env.enemy_block),
            "enemy_strength_after": info.get("enemy_strength", env.enemy_strength),
            "enemy_vulnerable_after": info.get("enemy_vulnerable_turns", env.enemy_vulnerable_turns),
            "player_hp_after": env.player_hp,
            "enemy_hp_after": env.enemy_hp,
            "reward": round(float(reward), 4),
            "total_reward_after": round(float(total_reward), 4),
            "done": done,
            "win": bool(env.enemy_hp <= 0),
            "next_state": [int(x) for x in next_state.tolist()],
        }
        rows.append(row)
        step_idx += 1
    return rows

def trace_greedy_policy(env: MiniSlayTheSpireEnv, agent, agent_type: str = "dqn", seed: int = 42, max_steps: int = 100) -> List[Dict[str, object]]:
    """Roll out a trained policy greedily and return detailed step rows.

    This is not a mathematical proof of the globally optimal sequence. It is the
    learned greedy policy produced by the trained agent, which is what we inspect
    in the project report.
    """
    env.reset(seed=seed)
    rows: List[Dict[str, object]] = []
    total_reward = 0.0
    step_idx = 0

    while not env.done and step_idx < max_steps:
        before = env.state_as_dict()
        before_turn = env.turn
        before_player_block = env.block
        before_enemy_move = env.enemy_move_name()

        if agent_type == "dqn":
            action = agent.act(env.get_state(normalize=True), env.get_valid_actions(), training=False)
        elif agent_type == "qlearning":
            action = agent.act(env.get_discrete_state(), env.get_valid_actions(), training=False)
        else:
            raise ValueError("trace_greedy_policy supports qlearning and dqn agents")

        next_state, reward, done, info = env.step(action)
        total_reward += reward

        row: Dict[str, object] = {
            "step": step_idx,
            "turn": before_turn,
            "player_hp_before": before["player_hp"],
            "player_block_before": before_player_block,
            "enemy_hp_before": before["enemy_hp"],
            "enemy_block_before": before["enemy_block"],
            "enemy_strength_before": before["enemy_strength"],
            "enemy_attack_intent": before["enemy_attack_intent"],
            "enemy_block_intent": before["enemy_block_intent"],
            "enemy_strength_intent": before["enemy_strength_intent"],
            "enemy_vulnerable_before": before["enemy_vulnerable_turns"],
            "energy_before": before["energy"],
            "hand_strike": before["strike_in_hand"],
            "hand_defend": before["defend_in_hand"],
            "hand_bash": before["bash_in_hand"],
            "enemy_move": before_enemy_move,
            "action": action,
            "action_name": ACTION_NAMES[action],
            "damage_before_block": info.get("damage_before_block", 0),
            "blocked_damage": info.get("blocked_damage", 0),
            "damage_dealt": info.get("damage_dealt", 0),
            "damage_taken": info.get("damage_taken", 0),
            "enemy_block_gained": info.get("enemy_block_gained", 0),
            "enemy_strength_gained": info.get("enemy_strength_gained", 0),
            "enemy_block_after": info.get("enemy_block", env.enemy_block),
            "enemy_strength_after": info.get("enemy_strength", env.enemy_strength),
            "enemy_vulnerable_after": info.get("enemy_vulnerable_turns", env.enemy_vulnerable_turns),
            "player_hp_after": env.player_hp,
            "enemy_hp_after": env.enemy_hp,
            "reward": round(float(reward), 4),
            "total_reward_after": round(float(total_reward), 4),
            "done": done,
            "win": bool(env.enemy_hp <= 0),
            "next_state": [int(x) for x in next_state.tolist()],
        }
        rows.append(row)
        step_idx += 1
    return rows


def save_training_curve(path: Path, rewards: List[float], label: str, window: int = 50) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.arange(1, len(rewards) + 1)
    plt.figure(figsize=(9, 5))
    plt.plot(x, rewards, alpha=0.35, label="Episode reward")
    if len(rewards) >= window:
        moving = np.convolve(rewards, np.ones(window) / window, mode="valid")
        plt.plot(np.arange(window, len(rewards) + 1), moving, label=f"Moving average ({window})")
    plt.xlabel("Episode")
    plt.ylabel("Total reward")
    plt.title(f"Training reward curve: {label}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()




def save_random_reward_curve(path: Path, rewards: List[float], label: str, window: int = 20) -> None:
    """Save the Random baseline reward curve over evaluation episodes."""
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.arange(1, len(rewards) + 1)
    plt.figure(figsize=(9, 5))
    plt.plot(x, rewards, alpha=0.45, label="Random episode reward")
    if len(rewards) >= window:
        moving = np.convolve(rewards, np.ones(window) / window, mode="valid")
        plt.plot(np.arange(window, len(rewards) + 1), moving, label=f"Moving average ({window})")
    plt.xlabel("Episode")
    plt.ylabel("Total reward")
    plt.title(f"Random agent reward curve: {label}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_random_reward_curves_by_mode(path: Path, rewards_by_mode: Dict[str, List[float]]) -> None:
    """Save one chart comparing Random reward curves across modes."""
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 5))
    for mode, rewards in rewards_by_mode.items():
        if not rewards:
            continue
        x = np.arange(1, len(rewards) + 1)
        plt.plot(x, rewards, alpha=0.35, label=f"{mode} episode reward")
        if len(rewards) >= 20:
            moving = np.convolve(rewards, np.ones(20) / 20, mode="valid")
            plt.plot(np.arange(20, len(rewards) + 1), moving, linewidth=2, label=f"{mode} moving avg")
    plt.xlabel("Episode")
    plt.ylabel("Total reward")
    plt.title("Random agent reward curves by difficulty mode")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()

def save_comparison_chart(path: Path, summaries: Dict[str, Dict[str, float]]) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    labels = list(summaries.keys())
    metrics = ["mean_win", "mean_reward", "mean_remaining_hp", "mean_turns"]
    titles = {
        "mean_win": "Win rate",
        "mean_reward": "Average reward",
        "mean_remaining_hp": "Average remaining HP",
        "mean_turns": "Average turns",
    }
    for metric in metrics:
        plt.figure(figsize=(7, 4))
        values = [summaries[label][metric] for label in labels]
        plt.bar(labels, values)
        plt.ylabel(titles[metric])
        plt.title(titles[metric] + " by agent")
        if metric == "mean_win":
            plt.ylim(0, 1.05)
        plt.tight_layout()
        metric_path = path.with_name(f"comparison_{metric}.png")
        plt.savefig(metric_path, dpi=160)
        plt.close()


def save_mode_metric_chart(path: Path, all_summaries: Dict[str, Dict[str, Dict[str, float]]], metric: str, title: str, ylabel: str) -> None:
    """Save grouped bar chart by mode and agent. Random is included."""
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    modes = list(all_summaries.keys())
    agents = []
    for summaries in all_summaries.values():
        for agent in summaries.keys():
            if agent not in agents:
                agents.append(agent)

    x = np.arange(len(modes))
    width = 0.8 / max(len(agents), 1)
    plt.figure(figsize=(9, 5))
    for idx, agent in enumerate(agents):
        offsets = x - 0.4 + width / 2 + idx * width
        values = [all_summaries[mode].get(agent, {}).get(metric, 0.0) for mode in modes]
        plt.bar(offsets, values, width, label=agent)
    plt.xticks(x, modes)
    plt.ylabel(ylabel)
    plt.title(title)
    if metric == "mean_win":
        plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_random_win_chart(path: Path, all_summaries: Dict[str, Dict[str, Dict[str, float]]]) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    modes = list(all_summaries.keys())
    values = [all_summaries[mode]["Random"]["mean_win"] for mode in modes if "Random" in all_summaries[mode]]
    labels = [mode for mode in modes if "Random" in all_summaries[mode]]
    plt.figure(figsize=(7, 4))
    plt.bar(labels, values)
    plt.ylim(0, 1.05)
    plt.ylabel("Random agent win rate")
    plt.title("Random agent win rate by difficulty mode")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_policy_trace_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_policy_rows_by_turn(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    """Aggregate step-level policy trace rows into turn-level summaries.

    One turn may contain several agent actions, for example
    ``Bash -> Strike -> End Turn``. This summary is intended for report figures:
    it shows the action sequence and the player/Nibbit HP at the start and end
    of each turn.
    """
    if not rows:
        return []

    grouped: Dict[int, List[Dict[str, object]]] = {}
    for row in rows:
        turn = int(row["turn"])
        grouped.setdefault(turn, []).append(row)

    summaries: List[Dict[str, object]] = []
    for turn in sorted(grouped):
        turn_rows = grouped[turn]
        first = turn_rows[0]
        last = turn_rows[-1]
        actions = " -> ".join(str(r["action_name"]) for r in turn_rows)
        summaries.append(
            {
                "turn": turn,
                "steps": " -> ".join(str(r["step"]) for r in turn_rows),
                "actions": actions,
                "enemy_move": first.get("enemy_move", ""),
                "enemy_attack_intent": first.get("enemy_attack_intent", 0),
                "enemy_block_intent": first.get("enemy_block_intent", 0),
                "enemy_strength_intent": first.get("enemy_strength_intent", 0),
                "player_hp_start": first.get("player_hp_before", 0),
                "player_hp_end": last.get("player_hp_after", 0),
                "nibbit_hp_start": first.get("enemy_hp_before", 0),
                "nibbit_hp_end": last.get("enemy_hp_after", 0),
                "player_block_start": first.get("player_block_before", 0),
                "enemy_block_start": first.get("enemy_block_before", 0),
                "enemy_block_end": last.get("enemy_block_after", 0),
                "enemy_strength_start": first.get("enemy_strength_before", 0),
                "enemy_strength_end": last.get("enemy_strength_after", 0),
                "enemy_vulnerable_start": first.get("enemy_vulnerable_before", 0),
                "enemy_vulnerable_end": last.get("enemy_vulnerable_after", 0),
                "damage_dealt_total": sum(float(r.get("damage_dealt", 0)) for r in turn_rows),
                "damage_taken_total": sum(float(r.get("damage_taken", 0)) for r in turn_rows),
                "reward_total": round(sum(float(r.get("reward", 0.0)) for r in turn_rows), 4),
                "done": bool(last.get("done", False)),
                "win": bool(last.get("win", False)),
            }
        )
    return summaries


def save_policy_turn_summary_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    """Save a turn-level policy summary CSV for easier report tables."""
    path.parent.mkdir(parents=True, exist_ok=True)
    summaries = summarize_policy_rows_by_turn(rows)
    if not summaries:
        return
    fieldnames = list(summaries[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)


def save_policy_action_timeline(path: Path, rows: List[Dict[str, object]], title: str) -> None:
    """Save a turn-level action timeline with Player/Nibbit HP.

    Earlier versions plotted one point per step. In this version, the x-axis is
    the game turn. Each turn label contains the agent's action sequence, and
    the two line plots show Player HP and Nibbit HP at the end of that turn.
    This makes it easier to compare the learned policy with the Nibbit pattern.
    """
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    summaries = summarize_policy_rows_by_turn(rows)
    if not summaries:
        return

    turns = [int(r["turn"]) for r in summaries]
    player_hp_end = [float(r["player_hp_end"]) for r in summaries]
    enemy_hp_end = [float(r["nibbit_hp_end"]) for r in summaries]
    player_hp_start = float(summaries[0]["player_hp_start"])
    enemy_hp_start = float(summaries[0]["nibbit_hp_start"])

    # Include the initial state as turn 0 so the HP drop per turn is visible.
    x_hp = [0] + turns
    player_hp = [player_hp_start] + player_hp_end
    enemy_hp = [enemy_hp_start] + enemy_hp_end

    fig_width = max(12.0, len(turns) * 1.9)
    fig, ax = plt.subplots(figsize=(fig_width, 6.4))
    ax.plot(x_hp, player_hp, marker="o", linewidth=2, label="Player HP")
    ax.plot(x_hp, enemy_hp, marker="o", linewidth=2, label="Nibbit HP")

    max_hp = max(max(player_hp), max(enemy_hp), 1)
    bottom_margin = max(18.0, max_hp * 0.28)
    ax.set_ylim(-bottom_margin, max_hp + max(8.0, max_hp * 0.12))
    ax.set_xlim(-0.35, max(turns) + 0.45)
    ax.set_xticks([0] + turns)
    ax.set_xticklabels(["Start"] + [str(t) for t in turns])
    ax.set_xlabel("Turn")
    ax.set_ylabel("HP")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper right")

    # Annotate HP values at the end of each turn.
    for x, p_hp, e_hp in zip(turns, player_hp_end, enemy_hp_end):
        ax.annotate(f"P {int(p_hp)}", (x, p_hp), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8)
        ax.annotate(f"N {int(e_hp)}", (x, e_hp), textcoords="offset points", xytext=(0, -13), ha="center", fontsize=8)

    # Put compact action sequences and Nibbit intent below the x-axis area.
    # Abbreviations keep the report figure readable even when an episode lasts many turns.
    action_code = {"Strike": "S", "Defend": "D", "Bash": "B", "End Turn": "E"}
    for summary in summaries:
        x = int(summary["turn"])
        actions = "-".join(action_code.get(a.strip(), a.strip()) for a in str(summary["actions"]).split("->"))
        intent_parts = []
        if float(summary.get("enemy_attack_intent", 0)) > 0:
            intent_parts.append(f"A{int(float(summary['enemy_attack_intent']))}")
        if float(summary.get("enemy_block_intent", 0)) > 0:
            intent_parts.append(f"B{int(float(summary['enemy_block_intent']))}")
        if float(summary.get("enemy_strength_intent", 0)) > 0:
            intent_parts.append(f"S{int(float(summary['enemy_strength_intent']))}")
        intent = "+".join(intent_parts) if intent_parts else "NoAtk"
        hp_text = f"P{int(float(summary['player_hp_start']))}>{int(float(summary['player_hp_end']))} N{int(float(summary['nibbit_hp_start']))}>{int(float(summary['nibbit_hp_end']))}"
        label = f"{actions}\n{intent}\n{hp_text}"
        ax.text(x, -bottom_margin * 0.12, label, ha="center", va="top", fontsize=8, rotation=0)

    # Abbreviation legend for the action labels.
    fig.text(0.01, 0.01, "Action codes: S=Strike, D=Defend, B=Bash, E=End Turn | Intent codes: A=attack, B=block gain, S=strength gain", fontsize=8)
    fig.subplots_adjust(bottom=0.34)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_policy_hp_trace(path: Path, rows: List[Dict[str, object]], title: str) -> None:
    """Save a turn-level HP trace for Player and Nibbit.

    Earlier versions plotted HP after every agent step. For report use, this
    function now aggregates the step-level trace by game turn and plots HP at
    the beginning of combat and at the end of each turn. This aligns the HP
    curve with the Nibbit pattern, which is defined per turn rather than per
    action step.
    """
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    summaries = summarize_policy_rows_by_turn(rows)
    if not summaries:
        return

    turns = [int(r["turn"]) for r in summaries]
    player_hp_start = float(summaries[0]["player_hp_start"])
    enemy_hp_start = float(summaries[0]["nibbit_hp_start"])
    player_hp_end = [float(r["player_hp_end"]) for r in summaries]
    enemy_hp_end = [float(r["nibbit_hp_end"]) for r in summaries]

    x = [0] + turns
    player_hp = [player_hp_start] + player_hp_end
    enemy_hp = [enemy_hp_start] + enemy_hp_end

    plt.figure(figsize=(9, 5))
    plt.plot(x, player_hp, marker="o", linewidth=2, label="Player HP")
    plt.plot(x, enemy_hp, marker="o", linewidth=2, label="Nibbit HP")
    plt.xticks(x, ["Start"] + [str(t) for t in turns])
    plt.xlabel("Turn")
    plt.ylabel("HP")
    plt.title(title + " (turn-level)")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def save_summary_csv(path: Path, summaries: Dict[str, Dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["agent"] + sorted(next(iter(summaries.values())).keys())
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for agent, summary in summaries.items():
            row = {"agent": agent}
            row.update(summary)
            writer.writerow(row)


def save_all_modes_summary_csv(path: Path, all_summaries: Dict[str, Dict[str, Dict[str, float]]], mode_meta: Dict[str, Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metric_keys = sorted(next(iter(next(iter(all_summaries.values())).values())).keys())
    fieldnames = ["mode", "mode_label", "player_hp", "enemy_hp", "agent"] + metric_keys
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for mode, summaries in all_summaries.items():
            for agent, summary in summaries.items():
                row = {
                    "mode": mode,
                    "mode_label": mode_meta[mode]["label"],
                    "player_hp": mode_meta[mode]["player_hp"],
                    "enemy_hp": mode_meta[mode]["enemy_hp"],
                    "agent": agent,
                }
                row.update(summary)
                writer.writerow(row)
