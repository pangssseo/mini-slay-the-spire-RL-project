"""Experiment mode configuration for the Nibbit project."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable


@dataclass(frozen=True)
class ModeConfig:
    key: str
    label: str
    player_hp: int
    enemy_hp: int
    description: str


MODES: Dict[str, ModeConfig] = {
    "easy": ModeConfig(
        key="easy",
        label="Easy mode",
        player_hp=80,
        enemy_hp=46,
        description="Player HP 80, Nibbit HP 46",
    ),
    "normal": ModeConfig(
        key="normal",
        label="Normal mode",
        player_hp=64,
        enemy_hp=46,
        description="Player HP 64, Nibbit HP 46",
    ),
    "hard": ModeConfig(
        key="hard",
        label="Hard mode",
        player_hp=64,
        enemy_hp=48,
        description="Player HP 64, Nibbit HP 48",
    ),
}


def get_mode(mode: str) -> ModeConfig:
    key = mode.lower().strip()
    if key not in MODES:
        valid = ", ".join(MODES)
        raise ValueError(f"Unknown mode: {mode}. Valid modes: {valid}")
    return MODES[key]


def iter_modes(selected: str | Iterable[str] = "all") -> list[ModeConfig]:
    if isinstance(selected, str):
        if selected == "all":
            return [MODES["easy"], MODES["normal"], MODES["hard"]]
        return [get_mode(selected)]
    return [get_mode(item) for item in selected]
