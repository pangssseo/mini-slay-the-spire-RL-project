"""Interactive manual-state tester for the Mini Slay the Spire RL environment.

Run:
    python manual_state.py

This script lets you type the 18 state values directly and then choose actions
(Strike, Defend, Bash, End Turn) to observe the reward and next state.
The default enemy setting is Nibbit (깨작이), with a repeating 3-turn pattern:
    1) Attack 12
    2) Attack 6 + Block 5
    3) Gain Strength 2
"""
from __future__ import annotations

from typing import Dict, List

from mini_sts_rl.env import ACTION_NAMES, CARD_NAMES, STATE_NAMES, MiniSlayTheSpireEnv


def ask_int(prompt: str, default: int | None = None, min_value: int = 0) -> int:
    while True:
        suffix = f" [{default}]" if default is not None else ""
        raw = input(f"{prompt}{suffix}: ").strip()
        if raw == "" and default is not None:
            return default
        try:
            value = int(raw)
            if value < min_value:
                print(f"  -> {min_value} 이상의 정수를 입력하세요.")
                continue
            return value
        except ValueError:
            print("  -> 정수를 입력하세요.")


def ask_yes_no(prompt: str, default: bool = True) -> bool:
    default_text = "Y/n" if default else "y/N"
    while True:
        raw = input(f"{prompt} [{default_text}]: ").strip().lower()
        if raw == "":
            return default
        if raw in {"y", "yes", "예", "ㅇ"}:
            return True
        if raw in {"n", "no", "아니오", "ㄴ"}:
            return False
        print("  -> y 또는 n을 입력하세요.")


def ask_card_counts(title: str, defaults: Dict[str, int]) -> Dict[str, int]:
    print(f"\n[{title} 카드 개수 입력]")
    return {card: ask_int(f"{card} 개수", defaults.get(card, 0)) for card in CARD_NAMES}


def print_nibbit_pattern() -> None:
    print("\n기본 적: Nibbit(깨작이)")
    print("  Turn 1: 12 피해 공격")
    print("  Turn 2: 6 피해 공격 + 방어도 5 획득")
    print("  Turn 3: 힘 2 획득")
    print("  이후 위 패턴 반복")
    print("  힘은 이후 공격 피해량에 더해집니다. 예: 힘 2 상태의 Turn 1 공격 = 14")


def print_state(env: MiniSlayTheSpireEnv) -> None:
    print("\n현재 State Vector")
    print("-" * 70)
    for idx, (name, value) in enumerate(env.state_as_dict().items()):
        print(f"{idx:02d}. {name:30s} = {value}")
    print("-" * 70)
    print(f"내부 변수: player_block={env.block}, turn={env.turn}, done={env.done}")
    print(f"적 패턴: {env.enemy_move_name()}")


def print_actions(env: MiniSlayTheSpireEnv) -> None:
    valid = env.get_valid_actions()
    print("\nAction 목록")
    for idx, name in enumerate(ACTION_NAMES):
        marker = "사용 가능" if idx in valid else "현재 불가"
        print(f"  {idx}: {name:8s} ({marker})")


def read_vector_state(env: MiniSlayTheSpireEnv) -> None:
    print("\n18개 state 값을 한 줄에 입력하세요.")
    print("순서:")
    print(", ".join(STATE_NAMES))
    print("\n예시 - Nibbit 1턴 기본 상태:")
    print("80 46 12 0 0 0 0 0 3 2 2 1 2 2 0 0 0 0")
    while True:
        raw = input("state vector > ").strip().replace(",", " ")
        try:
            values = [int(x) for x in raw.split()]
            env.set_manual_state_from_vector(
                values,
                block=ask_int("현재 플레이어 block", 0),
                turn=ask_int("현재 turn", 1),
            )
            return
        except Exception as exc:  # noqa: BLE001 - interactive script
            print(f"  -> 입력 오류: {exc}")


def read_step_by_step_state(env: MiniSlayTheSpireEnv) -> None:
    print_nibbit_pattern()
    print("\n[기본 전투 정보]")
    turn = ask_int("현재 turn", 1)
    player_hp = ask_int("player_hp", env.player_max_hp)
    enemy_hp = ask_int("enemy_hp", env.enemy_max_hp)
    enemy_strength = ask_int("현재 enemy_strength", 0)
    enemy_block = ask_int("현재 enemy_block", 0)
    enemy_vulnerable_turns = ask_int("enemy_vulnerable_turns", 0)
    energy = ask_int("energy", env.energy_per_turn)
    player_block = ask_int("현재 플레이어 block", 0)

    use_nibbit_intent = ask_yes_no("turn과 enemy_strength를 기준으로 Nibbit intent를 자동 설정할까요?", True)
    if use_nibbit_intent:
        env.turn = turn
        env.enemy_strength = enemy_strength
        env._update_enemy_intents_from_pattern()  # internal helper for manual tester
        enemy_attack_intent = env.enemy_attack_intent
        enemy_block_intent = env.enemy_block_intent
        enemy_strength_intent = env.enemy_strength_intent
        print(
            f"  -> 자동 설정 intent: attack={enemy_attack_intent}, "
            f"block={enemy_block_intent}, strength={enemy_strength_intent}"
        )
    else:
        enemy_attack_intent = ask_int("enemy_attack_intent", 12)
        enemy_block_intent = ask_int("enemy_block_intent", 0)
        enemy_strength_intent = ask_int("enemy_strength_intent", 0)

    hand_counts = ask_card_counts("현재 손패", {"Strike": 2, "Defend": 2, "Bash": 1})
    draw_counts = ask_card_counts("남은 드로우 덱", {"Strike": 2, "Defend": 2, "Bash": 0})
    discard_counts = ask_card_counts("버린 카드 더미", {"Strike": 0, "Defend": 0, "Bash": 0})

    env.set_manual_state(
        player_hp=player_hp,
        enemy_hp=enemy_hp,
        enemy_attack_intent=enemy_attack_intent,
        enemy_block_intent=enemy_block_intent,
        enemy_strength_intent=enemy_strength_intent,
        enemy_block=enemy_block,
        enemy_strength=enemy_strength,
        enemy_vulnerable_turns=enemy_vulnerable_turns,
        energy=energy,
        hand_counts=hand_counts,
        draw_counts=draw_counts,
        discard_counts=discard_counts,
        block=player_block,
        turn=turn,
    )


def input_state(env: MiniSlayTheSpireEnv) -> None:
    print("\n입력 방식을 선택하세요.")
    print("  1: 항목별 입력, Nibbit 패턴 자동 계산 가능")
    print("  2: 18개 state vector 한 줄 입력")
    while True:
        choice = input("선택 [1]: ").strip() or "1"
        if choice == "1":
            read_step_by_step_state(env)
            return
        if choice == "2":
            read_vector_state(env)
            return
        print("  -> 1 또는 2를 입력하세요.")


def main() -> None:
    env = MiniSlayTheSpireEnv()
    print("Mini Slay the Spire RL - Manual State Tester")
    print("기본 적은 Nibbit(깨작이)입니다. 원하는 state를 직접 입력한 뒤 action 결과를 확인할 수 있습니다.")

    input_state(env)

    while True:
        print_state(env)
        print_actions(env)
        raw = input("\n행동 번호 입력 / r=state 재입력 / q=종료 > ").strip().lower()
        if raw == "q":
            print("종료합니다.")
            break
        if raw == "r":
            input_state(env)
            continue
        try:
            action = int(raw)
            if not (0 <= action < env.n_actions):
                print(f"  -> action은 0~{env.n_actions - 1} 범위여야 합니다.")
                continue
            next_state, reward, done, info = env.step(action)
            print("\n실행 결과")
            print("-" * 70)
            print(f"선택 action: {action} ({ACTION_NAMES[action]})")
            print(f"reward: {reward:.2f}")
            print(f"done: {done}")
            print("info:")
            for key, value in info.items():
                print(f"  {key}: {value}")
            print(f"next_state: {[int(x) for x in next_state.tolist()]}")
            print("-" * 70)
            if done:
                print("에피소드가 종료되었습니다. r을 눌러 새 state를 입력하거나 q로 종료하세요.")
        except RuntimeError as exc:
            print(f"  -> 실행 오류: {exc}")
            print("     에피소드가 이미 종료된 경우 r을 눌러 새 state를 입력하세요.")
        except ValueError:
            print("  -> 숫자 action, r, q 중 하나를 입력하세요.")


if __name__ == "__main__":
    main()
