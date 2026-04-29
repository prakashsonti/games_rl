import pickle
import random
from typing import Dict, List, Tuple, Optional

from .game import get_legal_moves, ROWS, COLS


State = Tuple[int, ...]
SIZE = ROWS * COLS


def _horizontal_flip() -> List[int]:
    """Permutation that mirrors the board left-right (only valid symmetry given gravity)."""
    perm = [0] * SIZE
    for r in range(ROWS):
        for c in range(COLS):
            perm[r * COLS + c] = r * COLS + (COLS - 1 - c)
    return perm


IDENTITY: List[int] = list(range(SIZE))
FLIP_H: List[int] = _horizontal_flip()
PERMS: List[List[int]] = [IDENTITY, FLIP_H]


def _invert_perm(perm: List[int]) -> List[int]:
    inv = [0] * len(perm)
    for i, p in enumerate(perm):
        inv[p] = i
    return inv


def _apply_perm(board: List[int], perm: List[int]) -> List[int]:
    newb = [0] * SIZE
    for i in range(SIZE):
        newb[perm[i]] = board[i]
    return newb


def _canonicalize(board: List[int]) -> Tuple[State, List[int], List[int]]:
    best: Optional[State] = None
    best_perm = PERMS[0]
    for perm in PERMS:
        tb = tuple(_apply_perm(board, perm))
        if best is None or tb < best:
            best = tb
            best_perm = perm
    inv = _invert_perm(best_perm)
    assert best is not None
    return best, best_perm, inv


def _col_through_perm(col: int, perm: List[int]) -> int:
    """Map a column index through a permutation using the row-0 cell index as proxy."""
    return perm[col] % COLS


class QLearningAgent:
    def __init__(
        self,
        alpha: float = 0.3,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.999997,
        epsilon_min: float = 0.02,
    ) -> None:
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_table: Dict[State, Dict[int, float]] = {}

    def _ensure_state(self, state: State, legal_cols: List[int]) -> None:
        if state not in self.q_table:
            self.q_table[state] = {a: 0.0 for a in legal_cols}
        else:
            for a in legal_cols:
                if a not in self.q_table[state]:
                    self.q_table[state][a] = 0.0

    def _best_action(self, state: State, legal_cols: List[int]) -> Tuple[int, float]:
        self._ensure_state(state, legal_cols)
        action_values = self.q_table[state]
        max_q = max(action_values[a] for a in legal_cols)
        best = [a for a in legal_cols if action_values[a] == max_q]
        return random.choice(best), max_q

    def get_action(self, board: List[int]) -> int:
        legal = get_legal_moves(board)
        if not legal:
            raise ValueError("No legal moves")
        if random.random() < self.epsilon:
            return random.choice(legal)
        c_state, perm, inv = _canonicalize(board)
        c_legal = [_col_through_perm(c, perm) for c in legal]
        action_c, _ = self._best_action(c_state, c_legal)
        return inv[action_c] % COLS

    def get_greedy_action(self, board: List[int]) -> int:
        legal = get_legal_moves(board)
        if not legal:
            raise ValueError("No legal moves")
        c_state, perm, inv = _canonicalize(board)
        c_legal = [_col_through_perm(c, perm) for c in legal]
        action_c, _ = self._best_action(c_state, c_legal)
        return inv[action_c] % COLS

    def update(
        self,
        state: List[int],
        action: int,
        reward: float,
        next_state: Optional[List[int]],
        done: bool,
        zero_sum: bool = False,
    ) -> None:
        c_state, perm, _ = _canonicalize(state)
        action_c = _col_through_perm(action, perm)
        c_legal = [_col_through_perm(c, perm) for c in get_legal_moves(state)]
        self._ensure_state(c_state, c_legal)
        current_q = self.q_table[c_state][action_c]

        target = reward
        if not done and next_state is not None:
            c_next, perm_next, _ = _canonicalize(next_state)
            next_legal = get_legal_moves(next_state)
            if next_legal:
                c_next_legal = [_col_through_perm(c, perm_next) for c in next_legal]
                _, max_next_q = self._best_action(c_next, c_next_legal)
                target = reward - self.gamma * max_next_q if zero_sum else reward + self.gamma * max_next_q

        self.q_table[c_state][action_c] = current_q + self.alpha * (target - current_q)

    def decay_epsilon(self) -> None:
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "alpha": self.alpha,
                    "gamma": self.gamma,
                    "epsilon": self.epsilon,
                    "epsilon_decay": self.epsilon_decay,
                    "epsilon_min": self.epsilon_min,
                    "q_table": self.q_table,
                },
                f,
            )

    @classmethod
    def load(cls, path: str) -> "QLearningAgent":
        with open(path, "rb") as f:
            data = pickle.load(f)
        agent = cls(
            alpha=data.get("alpha", 0.3),
            gamma=data.get("gamma", 0.95),
            epsilon=data.get("epsilon", 0.0),
            epsilon_decay=data.get("epsilon_decay", 0.999997),
            epsilon_min=data.get("epsilon_min", 0.02),
        )
        agent.q_table = data.get("q_table", {})
        return agent
