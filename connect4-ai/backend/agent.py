import random
from collections import deque
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from .game import get_legal_moves, ROWS, COLS


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SIZE = ROWS * COLS


# ---------------------------------------------------------------------------
# Network: 3-channel CNN  (own | opp | empty)  →  Q-value per column
# ---------------------------------------------------------------------------

class _ConvNet(nn.Module):
    """
    Input: (batch, 3, ROWS, COLS) — binary channels for own/opp/empty.
    Convolutional layers capture spatial patterns (threats, blocks, forks).
    """
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3,   64,  kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64,  128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * ROWS * COLS, 256),
            nn.ReLU(),
            nn.Linear(256, COLS),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.conv(x))


# ---------------------------------------------------------------------------
# Board → tensor helpers
# ---------------------------------------------------------------------------

def _flat_to_3ch_batch(flat_t: torch.Tensor) -> torch.Tensor:
    """Convert (batch, SIZE) flat board tensor to (batch, 3, ROWS, COLS)."""
    boards = flat_t.reshape(-1, ROWS, COLS)
    own   = (boards ==  1).float().unsqueeze(1)
    opp   = (boards == -1).float().unsqueeze(1)
    empty = (boards ==  0).float().unsqueeze(1)
    return torch.cat([own, opp, empty], dim=1)


def _board_to_tensor(board: List[int]) -> torch.Tensor:
    """Single board list → (1, 3, ROWS, COLS) tensor on DEVICE."""
    t = torch.tensor(board, dtype=torch.float32, device=DEVICE).reshape(ROWS, COLS)
    own   = (t ==  1).float()
    opp   = (t == -1).float()
    empty = (t ==  0).float()
    return torch.stack([own, opp, empty], dim=0).unsqueeze(0)  # (1,3,R,C)


# ---------------------------------------------------------------------------
# DQN Agent
# ---------------------------------------------------------------------------

class DQNAgent:
    def __init__(
        self,
        lr: float = 5e-4,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.999997,
        epsilon_min: float = 0.02,
        buffer_size: int = 200_000,
        batch_size: int = 256,
        target_update_every: int = 1_000,
    ) -> None:
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update_every = target_update_every
        self._steps = 0

        self.policy_net = _ConvNet().to(DEVICE)
        self.target_net = _ConvNet().to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.buffer: deque = deque(maxlen=buffer_size)

    # ------------------------------------------------------------------
    # Illegal-column mask (flat board list → additive logit mask)
    # ------------------------------------------------------------------

    def _illegal_mask(self, board: List[int]) -> torch.Tensor:
        """Returns (COLS,) tensor: 0 for legal columns, -1e9 for full ones."""
        mask = torch.zeros(COLS, device=DEVICE)
        for c in range(COLS):
            if board[c] != 0:          # top row occupied → column full
                mask[c] = -1e9
        return mask

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def get_action(self, board: List[int]) -> int:
        legal = get_legal_moves(board)
        if not legal:
            raise ValueError("No legal moves")
        if random.random() < self.epsilon:
            return random.choice(legal)
        return self._greedy(board)

    def get_greedy_action(self, board: List[int]) -> int:
        legal = get_legal_moves(board)
        if not legal:
            raise ValueError("No legal moves")
        return self._greedy(board)

    def _greedy(self, board: List[int]) -> int:
        with torch.no_grad():
            q = self.policy_net(_board_to_tensor(board)).squeeze(0)
            q = q + self._illegal_mask(board)
        return int(q.argmax().item())

    # ------------------------------------------------------------------
    # Experience replay
    # ------------------------------------------------------------------

    def push(
        self,
        state: List[int],
        action: int,
        reward: float,
        next_state: Optional[List[int]],
        done: bool,
    ) -> None:
        self.buffer.append((state[:], action, reward,
                            next_state[:] if next_state is not None else None,
                            done))

    def learn(self) -> Optional[float]:
        if len(self.buffer) < self.batch_size:
            return None

        batch = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # ── build tensors ────────────────────────────────────────────────
        flat_s = torch.tensor(states, dtype=torch.float32, device=DEVICE)
        states_t  = _flat_to_3ch_batch(flat_s)                          # (B,3,R,C)
        actions_t = torch.tensor(actions, dtype=torch.long,    device=DEVICE)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=DEVICE)
        dones_t   = torch.tensor(dones,   dtype=torch.float32, device=DEVICE)

        # ── Q(s, a) from policy net ──────────────────────────────────────
        q_pred = self.policy_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # ── target: r  (terminal)  or  r − γ·Q_target(s', a*)  (non-term)
        # Minus sign = zero-sum: opponent's gain is our loss.
        # Double DQN: policy net picks action, target net evaluates it.
        with torch.no_grad():
            future_q = torch.zeros(self.batch_size, device=DEVICE)
            non_term = [i for i, d in enumerate(dones) if not d]

            if non_term:
                ns_list = [next_states[i] for i in non_term]
                flat_ns = torch.tensor(ns_list, dtype=torch.float32, device=DEVICE)
                ns_t    = _flat_to_3ch_batch(flat_ns)                   # (k,3,R,C)

                # Illegal mask from "empty" channel, top row
                # ns_t[:,2,0,:] = 1 where top row empty; 0 where full
                top_empty = ns_t[:, 2, 0, :]                            # (k, COLS)
                illegal   = (top_empty < 0.5).float() * (-1e9)

                # Double DQN: policy selects, target evaluates
                policy_q  = self.policy_net(ns_t) + illegal             # (k, COLS)
                best_a    = policy_q.argmax(dim=1, keepdim=True)        # (k, 1)
                target_q  = self.target_net(ns_t).gather(1, best_a).squeeze(1)  # (k,)

                for rank, idx in enumerate(non_term):
                    future_q[idx] = -target_q[rank]   # zero-sum negation

            target = rewards_t + self.gamma * (1.0 - dones_t) * future_q

        # ── update ──────────────────────────────────────────────────────
        loss = nn.functional.smooth_l1_loss(q_pred, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        self._steps += 1
        if self._steps % self.target_update_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    # ------------------------------------------------------------------
    # Epsilon decay
    # ------------------------------------------------------------------

    def decay_epsilon(self) -> None:
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        torch.save({
            "arch":                 "cnn_v1",
            "policy_state_dict":    self.policy_net.state_dict(),
            "target_state_dict":    self.target_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epsilon":              self.epsilon,
            "steps":                self._steps,
            "gamma":                self.gamma,
            "epsilon_decay":        self.epsilon_decay,
            "epsilon_min":          self.epsilon_min,
            "batch_size":           self.batch_size,
            "target_update_every":  self.target_update_every,
        }, path)

    @classmethod
    def load(cls, path: str, map_location=None) -> "DQNAgent":
        data = torch.load(path, map_location=map_location or DEVICE, weights_only=False)
        if data.get("arch") != "cnn_v1":
            raise ValueError("Incompatible model: expected cnn_v1 architecture")
        agent = cls(
            gamma=               data.get("gamma",               0.95),
            epsilon=             data.get("epsilon",             0.0),
            epsilon_decay=       data.get("epsilon_decay",       0.999997),
            epsilon_min=         data.get("epsilon_min",         0.02),
            batch_size=          data.get("batch_size",          256),
            target_update_every= data.get("target_update_every", 1_000),
        )
        agent.policy_net.load_state_dict(data["policy_state_dict"])
        agent.target_net.load_state_dict(data["target_state_dict"])
        agent.optimizer.load_state_dict(data["optimizer_state_dict"])
        agent._steps = data.get("steps", 0)
        agent.policy_net.eval()
        agent.target_net.eval()
        return agent
