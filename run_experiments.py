"""Run the three requested difficulty-mode experiments at once.

Requested modes
---------------
1. easy   : player HP 80, Nibbit HP 46
2. normal : player HP 64, Nibbit HP 46
3. hard   : player HP 64, Nibbit HP 48

This script trains/evaluates Random, Q-Learning, and DQN for every mode, then
saves cross-mode comparison graphs and Random, Q-Learning, and DQN turn-level policy visualizations.

Quick check:
    python run_experiments.py --quick

Submission-quality run:
    python run_experiments.py
"""
from __future__ import annotations

import argparse
from pathlib import Path
import random
from typing import Dict, Tuple

import numpy as np

from mini_sts_rl.agents import RandomAgent
from mini_sts_rl.config import ModeConfig, iter_modes
from mini_sts_rl.env import MiniSlayTheSpireEnv
from mini_sts_rl.train import (
    evaluate_agent,
    save_all_modes_summary_csv,
    save_comparison_chart,
    save_mode_metric_chart,
    save_policy_action_timeline,
    save_policy_hp_trace,
    save_policy_trace_csv,
    save_policy_turn_summary_csv,
    run_random_reward_curve,
    save_random_reward_curve,
    save_random_reward_curves_by_mode,
    save_random_win_chart,
    save_summary_csv,
    save_training_curve,
    trace_greedy_policy,
    trace_random_policy,
    train_dqn,
    train_qlearning,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all Mini Slay the Spire Nibbit mode experiments.")
    parser.add_argument("--modes", type=str, default="all", help="all, easy, normal, hard, or comma-separated list.")
    parser.add_argument("--q-episodes", type=int, default=3000)
    parser.add_argument("--dqn-episodes", type=int, default=1500)
    parser.add_argument("--eval-episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--max-turns", type=int, default=40)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use short episode counts for checking code execution quickly.",
    )
    return parser.parse_args()


def selected_modes(text: str) -> list[ModeConfig]:
    if text.strip().lower() == "all":
        return iter_modes("all")
    return iter_modes([part.strip() for part in text.split(",") if part.strip()])


def make_env(mode: ModeConfig, seed: int, max_turns: int) -> MiniSlayTheSpireEnv:
    return MiniSlayTheSpireEnv(
        player_max_hp=mode.player_hp,
        enemy_max_hp=mode.enemy_hp,
        max_turns=max_turns,
        seed=seed,
    )


def run_one_mode(mode: ModeConfig, args: argparse.Namespace, output_dir: Path) -> Tuple[Dict[str, Dict[str, float]], list[float]]:
    print("\n" + "=" * 90)
    print(f"Running {mode.label}: player HP={mode.player_hp}, Nibbit HP={mode.enemy_hp}")
    print("Nibbit pattern: T1 attack 12 -> T2 attack 6 + block 5 -> T3 gain strength 2 -> repeat")
    print("=" * 90)

    mode_dir = output_dir / mode.key
    mode_dir.mkdir(parents=True, exist_ok=True)

    summaries: Dict[str, Dict[str, float]] = {}

    print("[1/3] Evaluating Random agent...")
    random_agent = RandomAgent()
    random_env = make_env(mode, args.seed, args.max_turns)
    summaries["Random"] = evaluate_agent(
        random_env,
        random_agent,
        episodes=args.eval_episodes,
        seed=args.seed + 1000,
        agent_type="random",
    )
    print("Random:", summaries["Random"])

    # Random baseline visualizations for the report.
    random_curve_env = make_env(mode, args.seed, args.max_turns)
    random_rewards = run_random_reward_curve(
        random_curve_env,
        random_agent,
        episodes=args.eval_episodes,
        seed=args.seed + 1100,
    )
    save_random_reward_curve(
        mode_dir / "random_reward_curve.png",
        random_rewards,
        mode.label,
        window=min(20, max(2, args.eval_episodes // 5)),
    )

    random_trace_env = make_env(mode, args.seed, args.max_turns)
    random_trace_rows = trace_random_policy(random_trace_env, random_agent, seed=args.seed + 1200)
    save_policy_trace_csv(mode_dir / "random_policy_trace.csv", random_trace_rows)
    save_policy_turn_summary_csv(mode_dir / "random_policy_turn_summary.csv", random_trace_rows)
    save_policy_action_timeline(
        mode_dir / "random_policy_action_timeline.png",
        random_trace_rows,
        f"Random action timeline - {mode.label}",
    )
    save_policy_hp_trace(
        mode_dir / "random_policy_hp_trace.png",
        random_trace_rows,
        f"HP trace under Random policy - {mode.label}",
    )

    print("[2/3] Training Q-Learning agent...")
    q_env = make_env(mode, args.seed, args.max_turns)
    q_agent, q_rewards = train_qlearning(q_env, episodes=args.q_episodes, seed=args.seed)
    save_training_curve(mode_dir / "qlearning_training_curve.png", q_rewards, f"Q-Learning - {mode.label}")
    summaries["Q-Learning"] = evaluate_agent(
        q_env,
        q_agent,
        episodes=args.eval_episodes,
        seed=args.seed + 2000,
        agent_type="qlearning",
    )
    print("Q-Learning:", summaries["Q-Learning"])

    # Learned greedy policy visualization for Q-Learning as well.
    q_trace_env = make_env(mode, args.seed, args.max_turns)
    q_trace_rows = trace_greedy_policy(q_trace_env, q_agent, agent_type="qlearning", seed=args.seed)
    save_policy_trace_csv(mode_dir / "qlearning_policy_trace.csv", q_trace_rows)
    save_policy_turn_summary_csv(mode_dir / "qlearning_policy_turn_summary.csv", q_trace_rows)
    save_policy_action_timeline(
        mode_dir / "qlearning_policy_action_timeline.png",
        q_trace_rows,
        f"Learned Q-Learning action timeline - {mode.label}",
    )
    save_policy_hp_trace(
        mode_dir / "qlearning_policy_hp_trace.png",
        q_trace_rows,
        f"HP trace under learned Q-Learning policy - {mode.label}",
    )

    print("[3/3] Training DQN agent...")
    dqn_env = make_env(mode, args.seed, args.max_turns)
    dqn_agent, dqn_rewards, dqn_losses = train_dqn(dqn_env, episodes=args.dqn_episodes, seed=args.seed)
    save_training_curve(mode_dir / "dqn_training_curve.png", dqn_rewards, f"DQN - {mode.label}")
    dqn_agent.save(str(mode_dir / "dqn_model.pt"))
    summaries["DQN"] = evaluate_agent(
        dqn_env,
        dqn_agent,
        episodes=args.eval_episodes,
        seed=args.seed + 3000,
        agent_type="dqn",
    )
    print("DQN:", summaries["DQN"])

    # Learned greedy policy step visualization for the report.
    trace_env = make_env(mode, args.seed, args.max_turns)
    trace_rows = trace_greedy_policy(trace_env, dqn_agent, agent_type="dqn", seed=args.seed)
    save_policy_trace_csv(mode_dir / "dqn_policy_trace.csv", trace_rows)
    save_policy_turn_summary_csv(mode_dir / "dqn_policy_turn_summary.csv", trace_rows)
    save_policy_action_timeline(
        mode_dir / "dqn_policy_action_timeline.png",
        trace_rows,
        f"Learned DQN action timeline - {mode.label}",
    )
    save_policy_hp_trace(
        mode_dir / "dqn_policy_hp_trace.png",
        trace_rows,
        f"HP trace under learned DQN policy - {mode.label}",
    )

    save_summary_csv(mode_dir / "summary_results.csv", summaries)
    save_comparison_chart(mode_dir / "comparison.png", summaries)
    return summaries, random_rewards


def main() -> None:
    args = parse_args()
    if args.quick:
        args.q_episodes = min(args.q_episodes, 200)
        args.dqn_episodes = min(args.dqn_episodes, 200)
        args.eval_episodes = min(args.eval_episodes, 50)

    random.seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    modes = selected_modes(args.modes)
    all_summaries: Dict[str, Dict[str, Dict[str, float]]] = {}
    mode_meta: Dict[str, Dict[str, object]] = {}
    random_rewards_by_mode: Dict[str, list[float]] = {}

    for mode in modes:
        summaries, random_rewards = run_one_mode(mode, args, output_dir)
        all_summaries[mode.key] = summaries
        random_rewards_by_mode[mode.key] = random_rewards
        mode_meta[mode.key] = {
            "label": mode.label,
            "player_hp": mode.player_hp,
            "enemy_hp": mode.enemy_hp,
        }

    save_all_modes_summary_csv(output_dir / "summary_all_modes.csv", all_summaries, mode_meta)

    # Cross-mode graphs. The win-rate graph includes Random, Q-Learning, and DQN.
    save_mode_metric_chart(
        output_dir / "win_rate_by_mode.png",
        all_summaries,
        metric="mean_win",
        title="Win rate by difficulty mode and agent",
        ylabel="Win rate",
    )
    save_mode_metric_chart(
        output_dir / "average_reward_by_mode.png",
        all_summaries,
        metric="mean_reward",
        title="Average reward by difficulty mode and agent",
        ylabel="Average reward",
    )
    save_mode_metric_chart(
        output_dir / "remaining_hp_by_mode.png",
        all_summaries,
        metric="mean_remaining_hp",
        title="Average remaining HP by difficulty mode and agent",
        ylabel="Average remaining HP",
    )
    save_random_win_chart(output_dir / "random_win_rate_by_mode.png", all_summaries)
    save_random_reward_curves_by_mode(output_dir / "random_reward_curves_by_mode.png", random_rewards_by_mode)

    print("\nAll experiments finished.")
    print(f"Results saved to: {output_dir.resolve()}")
    print("Important report files:")
    print("  outputs/summary_all_modes.csv")
    print("  outputs/win_rate_by_mode.png")
    print("  outputs/random_win_rate_by_mode.png")
    print("  outputs/random_reward_curves_by_mode.png")
    print("  outputs/<mode>/random_reward_curve.png")
    print("  outputs/<mode>/random_policy_action_timeline.png")
    print("  outputs/<mode>/qlearning_policy_action_timeline.png")
    print("  outputs/<mode>/dqn_policy_action_timeline.png")
    print("  outputs/<mode>/random_policy_hp_trace.png")
    print("  outputs/<mode>/qlearning_policy_hp_trace.png")
    print("  outputs/<mode>/dqn_policy_hp_trace.png")
    print("  outputs/<mode>/random_policy_trace.csv")
    print("  outputs/<mode>/qlearning_policy_trace.csv")
    print("  outputs/<mode>/dqn_policy_trace.csv")


if __name__ == "__main__":
    main()
