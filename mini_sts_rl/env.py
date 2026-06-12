"""Mini Slay the Spire style battle environment with a default Nibbit enemy.

This environment is intentionally small and self-contained so it can be used in
an undergraduate reinforcement learning project without requiring Gymnasium.

Default enemy: Nibbit (깨작이)
    The enemy repeats a fixed 3-turn pattern:
        1. Attack for 12 damage
        2. Attack for 6 damage and gain 5 block
        3. Gain 2 strength

State vector contains 18 raw values:
    [player_hp, enemy_hp,
     enemy_attack_intent, enemy_block_intent, enemy_strength_intent,
     enemy_block, enemy_strength, enemy_vulnerable_turns, energy,
     strike_in_hand, defend_in_hand, bash_in_hand,
     strike_in_draw_pile, defend_in_draw_pile, bash_in_draw_pile,
     strike_in_discard_pile, defend_in_discard_pile, bash_in_discard_pile]

Actions:
    0: Strike   cost 1, damage 6
    1: Defend   cost 1, block 5
    2: Bash     cost 2, damage 8, applies 2 turns of Vulnerable
    3: End Turn

Simplifications:
    * Strength increases enemy attack damage by +strength.
    * Enemy block reduces incoming player damage and remains until depleted.
    * Vulnerable makes enemy take 1.5x attack damage before block is applied.
"""
from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Dict, List, Tuple, Optional

import numpy as np


CARD_NAMES = ("Strike", "Defend", "Bash")
ACTION_NAMES = ("Strike", "Defend", "Bash", "End Turn")
STATE_NAMES = (
    "player_hp",
    "enemy_hp",
    "enemy_attack_intent",
    "enemy_block_intent",
    "enemy_strength_intent",
    "enemy_block",
    "enemy_strength",
    "enemy_vulnerable_turns",
    "energy",
    "strike_in_hand",
    "defend_in_hand",
    "bash_in_hand",
    "strike_in_draw_pile",
    "defend_in_draw_pile",
    "bash_in_draw_pile",
    "strike_in_discard_pile",
    "defend_in_discard_pile",
    "bash_in_discard_pile",
)


@dataclass(frozen=True)
class Card:
    name: str
    cost: int
    damage: int = 0
    block: int = 0
    vulnerable: int = 0


@dataclass(frozen=True)
class EnemyMove:
    name: str
    attack: int = 0
    block: int = 0
    strength: int = 0


CARDS: Dict[str, Card] = {
    "Strike": Card("Strike", cost=1, damage=6, block=0, vulnerable=0),
    "Defend": Card("Defend", cost=1, damage=0, block=5, vulnerable=0),
    "Bash": Card("Bash", cost=2, damage=8, block=0, vulnerable=2),
}

# User-provided Slay the Spire 2 inspired Nibbit pattern.
NIBBIT_PATTERN: Tuple[EnemyMove, ...] = (
    EnemyMove("Nibbit Turn 1: Attack 12", attack=12, block=0, strength=0),
    EnemyMove("Nibbit Turn 2: Attack 6 + Block 5", attack=6, block=5, strength=0),
    EnemyMove("Nibbit Turn 3: Gain Strength 2", attack=0, block=0, strength=2),
)


class MiniSlayTheSpireEnv:
    """A small turn-based card combat environment.

    The goal is to defeat one enemy before the player's HP reaches 0.
    The default enemy is Nibbit, whose action pattern repeats every three turns.
    The agent observes enemy attack/block/strength intent and chooses which card
    to play or when to end the turn.
    """

    def __init__(
        self,
        player_max_hp: int = 80,
        enemy_max_hp: int = 46,
        energy_per_turn: int = 3,
        hand_size: int = 5,
        deck_counts: Optional[Dict[str, int]] = None,
        enemy_name: str = "Nibbit",
        enemy_pattern: Tuple[EnemyMove, ...] = NIBBIT_PATTERN,
        max_turns: int = 40,
        invalid_action_penalty: float = -5.0,
        step_penalty: float = -0.05,
        vulnerable_multiplier: float = 1.5,
        seed: Optional[int] = None,
    ) -> None:
        self.player_max_hp = int(player_max_hp)
        self.enemy_max_hp = int(enemy_max_hp)
        self.energy_per_turn = int(energy_per_turn)
        self.hand_size = int(hand_size)
        self.deck_counts = deck_counts or {"Strike": 4, "Defend": 4, "Bash": 1}
        self.enemy_name = enemy_name
        self.enemy_pattern = enemy_pattern
        self.max_turns = int(max_turns)
        self.invalid_action_penalty = invalid_action_penalty
        self.step_penalty = step_penalty
        self.vulnerable_multiplier = vulnerable_multiplier
        self._rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)

        self.player_hp = self.player_max_hp
        self.enemy_hp = self.enemy_max_hp
        self.enemy_attack_intent = 0
        self.enemy_block_intent = 0
        self.enemy_strength_intent = 0
        self.enemy_block = 0
        self.enemy_strength = 0
        self.enemy_vulnerable_turns = 0
        self.energy = self.energy_per_turn
        self.block = 0
        self.turn = 1
        self.done = False
        self.draw_pile: List[str] = []
        self.hand: List[str] = []
        self.discard_pile: List[str] = []

    @property
    def n_actions(self) -> int:
        return 4

    @property
    def state_dim(self) -> int:
        return len(STATE_NAMES)

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            self._rng.seed(seed)
            self._np_rng = np.random.default_rng(seed)

        self.player_hp = self.player_max_hp
        self.enemy_hp = self.enemy_max_hp
        self.enemy_block = 0
        self.enemy_strength = 0
        self.enemy_vulnerable_turns = 0
        self.energy = self.energy_per_turn
        self.block = 0
        self.turn = 1
        self.done = False
        self.discard_pile = []
        self.hand = []
        self.draw_pile = []

        for card_name, count in self.deck_counts.items():
            self.draw_pile.extend([card_name] * int(count))
        self._rng.shuffle(self.draw_pile)
        self._update_enemy_intents_from_pattern()
        self._draw_cards(self.hand_size)
        return self.get_state(normalize=False)

    def _current_enemy_move(self) -> EnemyMove:
        idx = (max(self.turn, 1) - 1) % len(self.enemy_pattern)
        return self.enemy_pattern[idx]

    def _update_enemy_intents_from_pattern(self) -> None:
        move = self._current_enemy_move()
        self.enemy_attack_intent = int(move.attack + self.enemy_strength) if move.attack > 0 else 0
        self.enemy_block_intent = int(move.block)
        self.enemy_strength_intent = int(move.strength)

    def enemy_move_name(self) -> str:
        return self._current_enemy_move().name

    def _count_cards(self, pile: List[str]) -> List[int]:
        return [pile.count(name) for name in CARD_NAMES]

    def _draw_cards(self, n: int) -> None:
        for _ in range(n):
            if not self.draw_pile:
                if not self.discard_pile:
                    return
                self.draw_pile = self.discard_pile
                self.discard_pile = []
                self._rng.shuffle(self.draw_pile)
            self.hand.append(self.draw_pile.pop())

    def _make_pile_from_counts(self, counts: Dict[str, int]) -> List[str]:
        """Create a card pile list from card-count dictionary."""
        pile: List[str] = []
        for card_name in CARD_NAMES:
            count = int(counts.get(card_name, 0))
            if count < 0:
                raise ValueError(f"Card count cannot be negative: {card_name}={count}")
            pile.extend([card_name] * count)
        return pile

    def set_manual_state(
        self,
        player_hp: int,
        enemy_hp: int,
        enemy_attack_intent: int,
        enemy_block_intent: int,
        enemy_strength_intent: int,
        enemy_block: int,
        enemy_strength: int,
        enemy_vulnerable_turns: int,
        energy: int,
        hand_counts: Dict[str, int],
        draw_counts: Dict[str, int],
        discard_counts: Dict[str, int],
        block: int = 0,
        turn: int = 1,
    ) -> np.ndarray:
        """Manually set the environment state for debugging and demonstration.

        ``block`` and ``turn`` are internal combat variables. ``block`` is the
        player's current block, while ``enemy_block`` is included in the state
        because it directly changes how much HP damage attacks deal.
        """
        numeric_values = {
            "player_hp": player_hp,
            "enemy_hp": enemy_hp,
            "enemy_attack_intent": enemy_attack_intent,
            "enemy_block_intent": enemy_block_intent,
            "enemy_strength_intent": enemy_strength_intent,
            "enemy_block": enemy_block,
            "enemy_strength": enemy_strength,
            "enemy_vulnerable_turns": enemy_vulnerable_turns,
            "energy": energy,
            "block": block,
            "turn": turn,
        }
        for name, value in numeric_values.items():
            if int(value) < 0:
                raise ValueError(f"{name} cannot be negative: {value}")

        self.player_hp = int(player_hp)
        self.enemy_hp = int(enemy_hp)
        self.enemy_attack_intent = int(enemy_attack_intent)
        self.enemy_block_intent = int(enemy_block_intent)
        self.enemy_strength_intent = int(enemy_strength_intent)
        self.enemy_block = int(enemy_block)
        self.enemy_strength = int(enemy_strength)
        self.enemy_vulnerable_turns = int(enemy_vulnerable_turns)
        self.energy = int(energy)
        self.block = int(block)
        self.turn = int(turn)
        self.done = self.player_hp <= 0 or self.enemy_hp <= 0 or self.turn > self.max_turns

        self.hand = self._make_pile_from_counts(hand_counts)
        self.draw_pile = self._make_pile_from_counts(draw_counts)
        self.discard_pile = self._make_pile_from_counts(discard_counts)

        self.deck_counts = {
            card_name: (
                int(hand_counts.get(card_name, 0))
                + int(draw_counts.get(card_name, 0))
                + int(discard_counts.get(card_name, 0))
            )
            for card_name in CARD_NAMES
        }
        return self.get_state(normalize=False)

    def set_manual_state_from_vector(
        self,
        state_vector: List[int],
        block: int = 0,
        turn: int = 1,
    ) -> np.ndarray:
        """Set state by passing the 18-dimensional raw state vector directly."""
        if len(state_vector) != self.state_dim:
            raise ValueError(f"Expected {self.state_dim} state values, got {len(state_vector)}")
        values = [int(x) for x in state_vector]
        return self.set_manual_state(
            player_hp=values[0],
            enemy_hp=values[1],
            enemy_attack_intent=values[2],
            enemy_block_intent=values[3],
            enemy_strength_intent=values[4],
            enemy_block=values[5],
            enemy_strength=values[6],
            enemy_vulnerable_turns=values[7],
            energy=values[8],
            hand_counts={"Strike": values[9], "Defend": values[10], "Bash": values[11]},
            draw_counts={"Strike": values[12], "Defend": values[13], "Bash": values[14]},
            discard_counts={"Strike": values[15], "Defend": values[16], "Bash": values[17]},
            block=block,
            turn=turn,
        )

    def state_as_dict(self) -> Dict[str, int]:
        """Return the current raw state as a dictionary for readable printing."""
        raw = self.get_state(normalize=False)
        return {name: int(value) for name, value in zip(STATE_NAMES, raw)}

    def get_state(self, normalize: bool = False) -> np.ndarray:
        raw = np.array(
            [
                self.player_hp,
                self.enemy_hp,
                self.enemy_attack_intent,
                self.enemy_block_intent,
                self.enemy_strength_intent,
                self.enemy_block,
                self.enemy_strength,
                self.enemy_vulnerable_turns,
                self.energy,
                *self._count_cards(self.hand),
                *self._count_cards(self.draw_pile),
                *self._count_cards(self.discard_pile),
            ],
            dtype=np.float32,
        )
        if not normalize:
            return raw
        deck_size = max(sum(self.deck_counts.values()), 1)
        max_vulnerable = max(card.vulnerable for card in CARDS.values())
        max_base_attack = max((move.attack for move in self.enemy_pattern), default=1)
        max_block_intent = max((move.block for move in self.enemy_pattern), default=1)
        max_strength_intent = max((move.strength for move in self.enemy_pattern), default=1)
        # Strength and block can accumulate, so use conservative scales.
        enemy_strength_scale = max(1, self.max_turns * max(1, max_strength_intent))
        enemy_block_scale = max(10, self.max_turns * max(1, max_block_intent))
        max_attack_with_strength = max_base_attack + enemy_strength_scale
        scale = np.array(
            [
                self.player_max_hp,
                self.enemy_max_hp,
                max_attack_with_strength,
                max_block_intent,
                max_strength_intent,
                enemy_block_scale,
                enemy_strength_scale,
                max_vulnerable,
                self.energy_per_turn,
                self.hand_size,
                self.hand_size,
                self.hand_size,
                deck_size,
                deck_size,
                deck_size,
                deck_size,
                deck_size,
                deck_size,
            ],
            dtype=np.float32,
        )
        return raw / np.maximum(scale, 1.0)

    def get_discrete_state(self) -> Tuple[int, ...]:
        return tuple(int(x) for x in self.get_state(normalize=False))

    def get_valid_actions(self) -> List[int]:
        valid = [3]  # End Turn is always valid.
        for action, card_name in enumerate(CARD_NAMES):
            card = CARDS[card_name]
            if self.hand.count(card_name) > 0 and self.energy >= card.cost:
                valid.append(action)
        return sorted(valid)

    def _calculate_attack_damage_before_block(self, base_damage: int) -> int:
        if base_damage <= 0:
            return 0
        if self.enemy_vulnerable_turns > 0:
            return int(base_damage * self.vulnerable_multiplier)
        return base_damage

    def _apply_damage_to_enemy(self, damage_before_block: int) -> Tuple[int, int]:
        """Apply damage to enemy block/HP. Return (hp_damage, blocked_damage)."""
        blocked_damage = min(self.enemy_block, damage_before_block)
        self.enemy_block -= blocked_damage
        hp_damage = max(0, damage_before_block - blocked_damage)
        self.enemy_hp -= hp_damage
        return hp_damage, blocked_damage

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, object]]:
        if self.done:
            raise RuntimeError("Episode is already done. Call reset() before step().")

        info: Dict[str, object] = {
            "action_name": ACTION_NAMES[action] if 0 <= action < len(ACTION_NAMES) else "Invalid",
            "enemy_name": self.enemy_name,
            "enemy_move": self.enemy_move_name(),
            "damage_before_block": 0,
            "blocked_damage": 0,
            "damage_dealt": 0,
            "damage_taken": 0,
            "enemy_block_gained": 0,
            "enemy_strength_gained": 0,
            "invalid_action": False,
            "win": False,
            "enemy_block": self.enemy_block,
            "enemy_strength": self.enemy_strength,
            "enemy_vulnerable_turns": self.enemy_vulnerable_turns,
        }
        reward = self.step_penalty

        if action not in range(self.n_actions):
            reward += self.invalid_action_penalty
            info["invalid_action"] = True
            return self.get_state(False), reward, self.done, info

        if action == 3:
            reward += self._end_turn(info)
            return self.get_state(False), reward, self.done, info

        card_name = CARD_NAMES[action]
        card = CARDS[card_name]

        if self.hand.count(card_name) <= 0 or self.energy < card.cost:
            reward += self.invalid_action_penalty
            info["invalid_action"] = True
            return self.get_state(False), reward, self.done, info

        self.hand.remove(card_name)
        self.discard_pile.append(card_name)
        self.energy -= card.cost

        self.block += card.block

        damage_before_block = self._calculate_attack_damage_before_block(card.damage)
        hp_damage, blocked_damage = self._apply_damage_to_enemy(damage_before_block)
        info["damage_before_block"] = damage_before_block
        info["blocked_damage"] = blocked_damage
        info["damage_dealt"] = hp_damage
        info["enemy_block"] = self.enemy_block
        reward += float(hp_damage)

        if card.vulnerable > 0:
            self.enemy_vulnerable_turns = max(self.enemy_vulnerable_turns, card.vulnerable)
            info["enemy_vulnerable_turns"] = self.enemy_vulnerable_turns

        if self.enemy_hp <= 0:
            self.enemy_hp = 0
            self.done = True
            info["win"] = True
            reward += 100.0

        return self.get_state(False), reward, self.done, info

    def _end_turn(self, info: Dict[str, object]) -> float:
        reward = 0.0

        damage_taken = max(0, self.enemy_attack_intent - self.block)
        self.player_hp -= damage_taken
        info["damage_taken"] = damage_taken
        reward -= float(damage_taken)

        if self.enemy_block_intent > 0:
            self.enemy_block += self.enemy_block_intent
            info["enemy_block_gained"] = self.enemy_block_intent

        if self.enemy_strength_intent > 0:
            self.enemy_strength += self.enemy_strength_intent
            info["enemy_strength_gained"] = self.enemy_strength_intent

        if self.enemy_vulnerable_turns > 0:
            self.enemy_vulnerable_turns -= 1

        info["enemy_block"] = self.enemy_block
        info["enemy_strength"] = self.enemy_strength
        info["enemy_vulnerable_turns"] = self.enemy_vulnerable_turns

        if self.player_hp <= 0:
            self.player_hp = 0
            self.done = True
            reward -= 100.0
            return reward

        self.discard_pile.extend(self.hand)
        self.hand = []
        self.energy = self.energy_per_turn
        self.block = 0
        self.turn += 1

        if self.turn > self.max_turns:
            self.done = True
            reward -= 100.0
            return reward

        self._update_enemy_intents_from_pattern()
        self._draw_cards(self.hand_size)
        return reward

    def render(self) -> None:
        print(
            f"Turn {self.turn} | Player HP {self.player_hp}/{self.player_max_hp} "
            f"Block {self.block} | Energy {self.energy}"
        )
        print(
            f"{self.enemy_name} HP {self.enemy_hp}/{self.enemy_max_hp} "
            f"Block {self.enemy_block} Strength {self.enemy_strength} "
            f"Vulnerable {self.enemy_vulnerable_turns}"
        )
        print(
            f"Enemy intent: attack={self.enemy_attack_intent}, "
            f"block={self.enemy_block_intent}, strength={self.enemy_strength_intent} "
            f"({self.enemy_move_name()})"
        )
        print(f"Hand: {self.hand} | Draw: {len(self.draw_pile)} | Discard: {len(self.discard_pile)}")
