"""Agents for Mini Slay the Spire RL project."""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
import random
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    torch.set_num_threads(1)
    import torch.nn as nn
    import torch.optim as optim
except ImportError:  # DQN remains optional until used.
    torch = None
    nn = None
    optim = None

from .env import MiniSlayTheSpireEnv


class RandomAgent:
    def act(self, env: MiniSlayTheSpireEnv) -> int:
        return random.choice(env.get_valid_actions())


class QLearningAgent:
    """Tabular Q-learning agent.

    The environment state is small enough for this simplified project. The state
    is represented as a tuple of integer values from the environment.
    """

    def __init__(
        self,
        n_actions: int,
        alpha: float = 0.12,
        gamma: float = 0.98,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        seed: int = 42,
    ) -> None:
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.q: Dict[Tuple[int, ...], np.ndarray] = defaultdict(lambda: np.zeros(n_actions, dtype=np.float32))
        self.rng = random.Random(seed)

    def act(self, state: Tuple[int, ...], valid_actions: List[int], training: bool = True) -> int:
        if training and self.rng.random() < self.epsilon:
            return self.rng.choice(valid_actions)
        q_values = self.q[state].copy()
        invalid = set(range(self.n_actions)) - set(valid_actions)
        for a in invalid:
            q_values[a] = -1e9
        return int(np.argmax(q_values))

    def update(
        self,
        state: Tuple[int, ...],
        action: int,
        reward: float,
        next_state: Tuple[int, ...],
        done: bool,
        next_valid_actions: List[int],
    ) -> None:
        current = self.q[state][action]
        if done:
            target = reward
        else:
            next_q = self.q[next_state].copy()
            invalid = set(range(self.n_actions)) - set(next_valid_actions)
            for a in invalid:
                next_q[a] = -1e9
            target = reward + self.gamma * float(np.max(next_q))
        self.q[state][action] = current + self.alpha * (target - current)

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    next_valid_actions: List[int]


class ReplayBuffer:
    def __init__(self, capacity: int = 50_000, seed: int = 42) -> None:
        self.buffer: Deque[Transition] = deque(maxlen=capacity)
        self.rng = random.Random(seed)

    def push(self, transition: Transition) -> None:
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> List[Transition]:
        return self.rng.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)


class QNetwork(nn.Module):  # type: ignore[misc]
    def __init__(self, state_dim: int, n_actions: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    """DQN agent with valid-action masking."""

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        lr: float = 1e-3,
        gamma: float = 0.98,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        batch_size: int = 32,
        replay_capacity: int = 50_000,
        target_update_interval: int = 100,
        seed: int = 42,
        device: Optional[str] = None,
    ) -> None:
        if torch is None:
            raise ImportError("PyTorch is required for DQN. Install it with: pip install torch")
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_interval = target_update_interval
        self.rng = random.Random(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.policy_net = QNetwork(state_dim, n_actions).to(self.device)
        self.target_net = QNetwork(state_dim, n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()
        self.replay = ReplayBuffer(replay_capacity, seed=seed)
        self.train_steps = 0

    def act(self, state: np.ndarray, valid_actions: List[int], training: bool = True) -> int:
        if training and self.rng.random() < self.epsilon:
            return self.rng.choice(valid_actions)
        with torch.no_grad():
            x = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.policy_net(x).squeeze(0).detach().cpu().numpy()
        invalid = set(range(self.n_actions)) - set(valid_actions)
        for a in invalid:
            q_values[a] = -1e9
        return int(np.argmax(q_values))

    def remember(self, transition: Transition) -> None:
        self.replay.push(transition)

    def update(self) -> Optional[float]:
        if len(self.replay) < self.batch_size:
            return None

        batch = self.replay.sample(self.batch_size)
        states = torch.tensor(np.stack([t.state for t in batch]), dtype=torch.float32, device=self.device)
        actions = torch.tensor([t.action for t in batch], dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.tensor(np.stack([t.next_state for t in batch]), dtype=torch.float32, device=self.device)
        dones = torch.tensor([t.done for t in batch], dtype=torch.float32, device=self.device).unsqueeze(1)

        q_values = self.policy_net(states).gather(1, actions)

        with torch.no_grad():
            next_q_values = self.target_net(next_states)
            # Mask invalid next actions before max operation.
            for i, t in enumerate(batch):
                invalid = set(range(self.n_actions)) - set(t.next_valid_actions)
                for a in invalid:
                    next_q_values[i, a] = -1e9
            max_next_q = next_q_values.max(dim=1, keepdim=True).values
            target = rewards + (1.0 - dones) * self.gamma * max_next_q

        loss = self.loss_fn(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=5.0)
        self.optimizer.step()

        self.train_steps += 1
        if self.train_steps % self.target_update_interval == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        return float(loss.item())

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save(self, path: str) -> None:
        torch.save(self.policy_net.state_dict(), path)
