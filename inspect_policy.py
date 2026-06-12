"""Inspect a trained DQN policy and print the learned action pattern.

Examples
--------
After running all mode experiments, inspect hard mode:
    python inspect_policy.py --mode hard --show-q

Inspect from a manually entered state vector:
    python inspect_policy.py --mode easy --manual-state "80 46 12 0 0 0 0 0 3 2 2 1 2 2 0 0 0 0" --show-q

The script prints a step-by-step trace and saves a CSV file in outputs/<mode>/.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import numpy as np

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise ImportError("PyTorch is required. Install dependencies with: pip install -r requirements.txt") from exc

from mini_sts_rl.agents import DQNAgent
from mini_sts_rl.config import get_mode
from mini_sts_rl.env import ACTION_NAMES, MiniSlayTheSpireEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect trained DQN policy behavior against Nibbit.")
    parser.add_argument("--mode", choices=["easy", "normal", "hard", "custom"], default="easy")
    parser.add_argument("--model", type=str, default=None, help="Path to trained DQN model. Defaults to outputs/<mode>/dqn_model.pt.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for a normal rollout.")
    parser.add_argument("--episodes", type=int, default=1, help="Number of normal episodes to inspect.")
    parser.add_argument("--max-steps", type=int, default=100, help="Safety limit for printed steps.")
    parser.add_argument(
        "--manual-state",
        type=str,
        default=None,
        help="18 raw state values, e.g. '80 46 12 0 0 0 0 0 3 2 2 1 2 2 0 0 0 0'.",
    )
    parser.add_argument("--block", type=int, default=0, help="Player block value for manual-state mode.")
    parser.add_argument("--turn", type=int, default=1, help="Turn value for manual-state mode.")
    parser.add_argument("--player-hp", type=int, default=None, help="Player max HP override.")
    parser.add_argument("--enemy-hp", type=int, default=None, help="Nibbit max HP override.")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Directory to save trace CSV.")
    parser.add_argument("--show-q", action="store_true", help="Print DQN Q-values for each action.")
    return parser.parse_args()


def resolve_settings(args: argparse.Namespace) -> tuple[int, int, Path, Path]:
    if args.mode == "custom":
        player_hp = args.player_hp if args.player_hp is not None else 80
        enemy_hp = args.enemy_hp if args.enemy_hp is not None else 46
        mode_dir = Path(args.output_dir)
    else:
        cfg = get_mode(args.mode)
        player_hp = args.player_hp if args.player_hp is not None else cfg.player_hp
        enemy_hp = args.enemy_hp if args.enemy_hp is not None else cfg.enemy_hp
        mode_dir = Path(args.output_dir) / args.mode
    model_path = Path(args.model) if args.model is not None else mode_dir / "dqn_model.pt"
    return int(player_hp), int(enemy_hp), mode_dir, model_path


def parse_state_vector(text: str, expected_dim: int) -> List[int]:
    values = [int(x) for x in text.replace(",", " ").split()]
    if len(values) != expected_dim:
        raise ValueError(f"Expected {expected_dim} values, got {len(values)}: {values}")
    return values


def get_q_values(agent: DQNAgent, env: MiniSlayTheSpireEnv) -> np.ndarray:
    state = env.get_state(normalize=True)
    with torch.no_grad():
        x = torch.tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(0)
        q_values = agent.policy_net(x).squeeze(0).detach().cpu().numpy()
    invalid = set(range(env.n_actions)) - set(env.get_valid_actions())
    for action in invalid:
        q_values[action] = -1e9
    return q_values


def trace_one_episode(
    env: MiniSlayTheSpireEnv,
    agent: DQNAgent,
    episode_id: int,
    max_steps: int,
    show_q: bool,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    total_reward = 0.0
    step_idx = 0

    print("\n" + "=" * 125)
    print(f"Policy trace episode={episode_id} | enemy={env.enemy_name}")
    print("=" * 125)
    print(
        "step | turn | HP | eHP | eBlk | eStr | atkI | blkI | strI | vuln | energy | hand(S/D/B) | action | "
        "hp_dmg | blk_dmg | dmg_taken | reward"
    )
    print("-" * 125)

    while not env.done and step_idx < max_steps:
        state_dict = env.state_as_dict()
        q_values = get_q_values(agent, env)
        action = int(np.argmax(q_values))
        action_name = ACTION_NAMES[action]

        before = dict(state_dict)
        before_block = env.block
        before_turn = env.turn
        before_enemy_move = env.enemy_move_name()
        next_state, reward, done, info = env.step(action)
        total_reward += reward

        q_value_text = "; ".join(
            f"{ACTION_NAMES[i]}={q_values[i]:.2f}" if q_values[i] > -1e8 else f"{ACTION_NAMES[i]}=INVALID"
            for i in range(env.n_actions)
        )

        row: Dict[str, object] = {
            "episode": episode_id,
            "step": step_idx,
            "turn_before": before_turn,
            "player_block_before": before_block,
            "enemy_move_before": before_enemy_move,
            "action": action,
            "action_name": action_name,
            "reward": round(float(reward), 4),
            "done": done,
            "win": bool(info.get("win", False)),
            "damage_before_block": info.get("damage_before_block", 0),
            "blocked_damage": info.get("blocked_damage", 0),
            "damage_dealt": info.get("damage_dealt", 0),
            "damage_taken": info.get("damage_taken", 0),
            "enemy_block_gained": info.get("enemy_block_gained", 0),
            "enemy_strength_gained": info.get("enemy_strength_gained", 0),
            "enemy_block_after": info.get("enemy_block", env.enemy_block),
            "enemy_strength_after": info.get("enemy_strength", env.enemy_strength),
            "enemy_vulnerable_after": info.get("enemy_vulnerable_turns", env.enemy_vulnerable_turns),
            "q_values": q_value_text,
            "next_state": [int(x) for x in next_state.tolist()],
        }
        for key, value in before.items():
            row[f"state_{key}"] = value
        rows.append(row)

        print(
            f"{step_idx:>4} | {before_turn:>4} | {before['player_hp']:>2} | "
            f"{before['enemy_hp']:>3} | {before['enemy_block']:>4} | {before['enemy_strength']:>4} | "
            f"{before['enemy_attack_intent']:>4} | {before['enemy_block_intent']:>4} | "
            f"{before['enemy_strength_intent']:>4} | {before['enemy_vulnerable_turns']:>4} | "
            f"{before['energy']:>6} | "
            f"{before['strike_in_hand']}/{before['defend_in_hand']}/{before['bash_in_hand']}           | "
            f"{action_name:<8} | {info.get('damage_dealt', 0):>6} | "
            f"{info.get('blocked_damage', 0):>7} | {info.get('damage_taken', 0):>9} | {reward:>6.2f}"
        )
        if show_q:
            print(f"     move={before_enemy_move}")
            print(f"     Q-values: {q_value_text}")

        step_idx += 1

    print("-" * 125)
    result = "WIN" if env.enemy_hp <= 0 else "LOSE/TIMEOUT"
    print(f"Result: {result}, total_reward={total_reward:.2f}, remaining_hp={env.player_hp}, turns={env.turn}")
    return rows


def save_trace_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    player_hp, enemy_hp, mode_dir, model_path = resolve_settings(args)
    if not model_path.exists():
        raise FileNotFoundError(
            f"DQN model not found: {model_path}\n"
            "First train the model, for example:\n"
            f"  python main.py --mode {args.mode} --agent dqn\n"
            "or run all requested modes:\n"
            "  python run_experiments.py --quick"
        )

    env = MiniSlayTheSpireEnv(player_max_hp=player_hp, enemy_max_hp=enemy_hp, seed=args.seed)
    agent = DQNAgent(state_dim=env.state_dim, n_actions=env.n_actions, seed=args.seed)
    state_dict = torch.load(model_path, map_location=agent.device)
    agent.policy_net.load_state_dict(state_dict)
    agent.target_net.load_state_dict(state_dict)
    agent.epsilon = 0.0

    all_rows: List[Dict[str, object]] = []

    if args.manual_state is not None:
        values = parse_state_vector(args.manual_state, env.state_dim)
        env.set_manual_state_from_vector(values, block=args.block, turn=args.turn)
        rows = trace_one_episode(env, agent, episode_id=0, max_steps=args.max_steps, show_q=args.show_q)
        all_rows.extend(rows)
        output_path = mode_dir / "policy_trace_manual.csv"
    else:
        for ep in range(args.episodes):
            env.reset(seed=args.seed + ep)
            rows = trace_one_episode(env, agent, episode_id=ep, max_steps=args.max_steps, show_q=args.show_q)
            all_rows.extend(rows)
        output_path = mode_dir / f"policy_trace_seed_{args.seed}.csv"

    save_trace_csv(output_path, all_rows)
    print(f"\nSaved policy trace CSV: {output_path.resolve()}")


if __name__ == "__main__":
    main()
