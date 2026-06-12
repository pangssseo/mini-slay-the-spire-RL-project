"""Command line entry point for a single Mini Slay the Spire RL experiment.

Default enemy is Nibbit(깨작이), whose 3-turn pattern repeats:
    1) Attack 12
    2) Attack 6 + gain 5 block
    3) Gain 2 strength

Difficulty presets:
    easy   : player HP 80, Nibbit HP 46
    normal : player HP 64, Nibbit HP 46
    hard   : player HP 64, Nibbit HP 48

Examples
--------
Run easy mode with all agents:
    python main.py --mode easy --agent all

Run a quick smoke test:
    python main.py --mode easy --agent all --q-episodes 200 --dqn-episodes 200 --eval-episodes 50

Override HP manually if needed:
    python main.py --mode custom --player-hp 70 --enemy-hp 60 --agent all
"""
from __future__ import annotations

import argparse
from pathlib import Path
import random

import numpy as np

from mini_sts_rl.agents import RandomAgent
from mini_sts_rl.config import MODES, get_mode
from mini_sts_rl.env import MiniSlayTheSpireEnv
from mini_sts_rl.train import (
    evaluate_agent,
    run_random_reward_curve,
    save_comparison_chart,
    save_policy_action_timeline,
    save_policy_hp_trace,
    save_policy_trace_csv,
    save_policy_turn_summary_csv,
    save_random_reward_curve,
    save_summary_csv,
    save_training_curve,
    trace_greedy_policy,
    trace_random_policy,
    train_dqn,
    train_qlearning,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", choices=["random", "qlearning", "dqn", "all"], default="all")
    parser.add_argument("--mode", choices=["easy", "normal", "hard", "custom"], default="easy")
    parser.add_argument("--q-episodes", type=int, default=3000)
    parser.add_argument("--dqn-episodes", type=int, default=1500)
    parser.add_argument("--eval-episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--player-hp", type=int, default=None)
    parser.add_argument("--enemy-hp", type=int, default=None)
    parser.add_argument("--max-turns", type=int, default=40)
    parser.add_argument("--trace-policy", action="store_true", help="Deprecated: policy traces are saved automatically for Random, Q-Learning, and DQN when available.")
    return parser.parse_args()


def resolve_hp(args: argparse.Namespace) -> tuple[int, int, str]:
    if args.mode == "custom":
        player_hp = args.player_hp if args.player_hp is not None else 80
        enemy_hp = args.enemy_hp if args.enemy_hp is not None else 46
        label = f"Custom mode (Player HP {player_hp}, Nibbit HP {enemy_hp})"
        return int(player_hp), int(enemy_hp), label

    cfg = get_mode(args.mode)
    player_hp = args.player_hp if args.player_hp is not None else cfg.player_hp
    enemy_hp = args.enemy_hp if args.enemy_hp is not None else cfg.enemy_hp
    label = cfg.label
    if args.player_hp is not None or args.enemy_hp is not None:
        label += f" with manual HP override (Player HP {player_hp}, Nibbit HP {enemy_hp})"
    return int(player_hp), int(enemy_hp), label


def make_env(player_hp: int, enemy_hp: int, args: argparse.Namespace, seed: int) -> MiniSlayTheSpireEnv:
    return MiniSlayTheSpireEnv(
        player_max_hp=player_hp,
        enemy_max_hp=enemy_hp,
        max_turns=args.max_turns,
        seed=seed,
    )


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    player_hp, enemy_hp, mode_label = resolve_hp(args)
    output_dir = Path(args.output_dir)
    if args.mode != "custom":
        output_dir = output_dir / args.mode
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = {}

    print(f"Mode: {mode_label}")
    print(f"Player HP={player_hp}, Nibbit HP={enemy_hp}")
    print("Enemy setting: Nibbit(깨작이)")
    print("Pattern: Turn1 Attack 12 -> Turn2 Attack 6 + Block 5 -> Turn3 Gain Strength 2 -> repeat")

    if args.agent in {"random", "all"}:
        print("[1/3] Evaluating Random agent...")
        env = make_env(player_hp, enemy_hp, args, args.seed)
        random_agent = RandomAgent()
        summaries["Random"] = evaluate_agent(
            env, random_agent, episodes=args.eval_episodes, seed=args.seed + 1000, agent_type="random"
        )
        print("Random:", summaries["Random"])

        random_rewards = run_random_reward_curve(
            make_env(player_hp, enemy_hp, args, args.seed),
            random_agent,
            episodes=args.eval_episodes,
            seed=args.seed + 1100,
        )
        save_random_reward_curve(
            output_dir / "random_reward_curve.png",
            random_rewards,
            mode_label,
            window=min(20, max(2, args.eval_episodes // 5)),
        )

        random_trace_rows = trace_random_policy(
            make_env(player_hp, enemy_hp, args, args.seed),
            random_agent,
            seed=args.seed + 1200,
        )
        save_policy_trace_csv(output_dir / "random_policy_trace.csv", random_trace_rows)
        save_policy_turn_summary_csv(output_dir / "random_policy_turn_summary.csv", random_trace_rows)
        save_policy_action_timeline(
            output_dir / "random_policy_action_timeline.png",
            random_trace_rows,
            f"Random action timeline - {mode_label}",
        )
        save_policy_hp_trace(
            output_dir / "random_policy_hp_trace.png",
            random_trace_rows,
            f"HP trace under Random policy - {mode_label}",
        )

    if args.agent in {"qlearning", "all"}:
        print("[2/3] Training Q-Learning agent...")
        env = make_env(player_hp, enemy_hp, args, args.seed)
        q_agent, q_rewards = train_qlearning(env, episodes=args.q_episodes, seed=args.seed)
        save_training_curve(output_dir / "qlearning_training_curve.png", q_rewards, f"Q-Learning - {mode_label}")
        summaries["Q-Learning"] = evaluate_agent(
            env, q_agent, episodes=args.eval_episodes, seed=args.seed + 2000, agent_type="qlearning"
        )
        print("Q-Learning:", summaries["Q-Learning"])

        # Save Q-Learning greedy policy visualizations as well.
        q_trace_rows = trace_greedy_policy(
            make_env(player_hp, enemy_hp, args, args.seed), q_agent, agent_type="qlearning", seed=args.seed
        )
        save_policy_trace_csv(output_dir / "qlearning_policy_trace.csv", q_trace_rows)
        save_policy_turn_summary_csv(output_dir / "qlearning_policy_turn_summary.csv", q_trace_rows)
        save_policy_action_timeline(
            output_dir / "qlearning_policy_action_timeline.png",
            q_trace_rows,
            f"Learned Q-Learning action timeline - {mode_label}",
        )
        save_policy_hp_trace(
            output_dir / "qlearning_policy_hp_trace.png",
            q_trace_rows,
            f"HP trace under learned Q-Learning policy - {mode_label}",
        )

    if args.agent in {"dqn", "all"}:
        print("[3/3] Training DQN agent...")
        env = make_env(player_hp, enemy_hp, args, args.seed)
        dqn_agent, dqn_rewards, dqn_losses = train_dqn(env, episodes=args.dqn_episodes, seed=args.seed)
        save_training_curve(output_dir / "dqn_training_curve.png", dqn_rewards, f"DQN - {mode_label}")
        dqn_agent.save(str(output_dir / "dqn_model.pt"))
        summaries["DQN"] = evaluate_agent(
            env, dqn_agent, episodes=args.eval_episodes, seed=args.seed + 3000, agent_type="dqn"
        )
        print("DQN:", summaries["DQN"])

        # Save DQN greedy policy visualizations.
        trace_env = make_env(player_hp, enemy_hp, args, args.seed)
        trace_rows = trace_greedy_policy(trace_env, dqn_agent, agent_type="dqn", seed=args.seed)
        save_policy_trace_csv(output_dir / "dqn_policy_trace.csv", trace_rows)
        save_policy_turn_summary_csv(output_dir / "dqn_policy_turn_summary.csv", trace_rows)
        save_policy_action_timeline(
            output_dir / "dqn_policy_action_timeline.png",
            trace_rows,
            f"Learned DQN action timeline - {mode_label}",
        )
        save_policy_hp_trace(
            output_dir / "dqn_policy_hp_trace.png",
            trace_rows,
            f"HP trace under learned DQN policy - {mode_label}",
        )

    save_summary_csv(output_dir / "summary_results.csv", summaries)
    save_comparison_chart(output_dir / "comparison.png", summaries)
    print(f"\nSaved results to: {output_dir.resolve()}")
    print("Generated files include training curves, Random/Q-Learning/DQN action timelines, turn-level HP traces, comparison charts, summary CSV, and DQN model.")


if __name__ == "__main__":
    main()
