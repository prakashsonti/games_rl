import pickle
import random
from typing import Dict, List, Tuple, Optional

from .game import get_legal_moves


State = Tuple[int, ...]  # perspective state (current player's view)


def _permutations() -> List[List[int]]:
    # Index map for 3x3 grid (0..8)
    # Coordinates r,c with idx = r*3+c
    def rc(i):
        return divmod(i, 3)

    def idx(r, c):
        return r * 3 + c

    perms = []
    # identity
    perms.append([i for i in range(9)])
    # rot90
    perms.append([idx(c, 2 - r) for i in range(9) for r, c in [rc(i)]])
    # rot180
    perms.append([idx(2 - r, 2 - c) for i in range(9) for r, c in [rc(i)]])
    # rot270
    perms.append([idx(2 - c, r) for i in range(9) for r, c in [rc(i)]])
    # flipH (horizontal mirror over vertical axis)
    perms.append([idx(r, 2 - c) for i in range(9) for r, c in [rc(i)]])
    # flipV (vertical mirror over horizontal axis)
    perms.append([idx(2 - r, c) for i in range(9) for r, c in [rc(i)]])
    # diag (main diagonal)
    perms.append([idx(c, r) for i in range(9) for r, c in [rc(i)]])
    # anti-diagonal
    perms.append([idx(2 - c, 2 - r) for i in range(9) for r, c in [rc(i)]])
    return perms


PERMS: List[List[int]] = _permutations()


def _invert_perm(perm: List[int]) -> List[int]:
    inv = [0] * len(perm)
    for i, p in enumerate(perm):
        inv[p] = i
    return inv


def _apply_perm_board(board: List[int], perm: List[int]) -> List[int]:
    newb = [0] * 9
    for i in range(9):
        newb[perm[i]] = board[i]
    return newb


def _canonicalize(board: List[int]) -> Tuple[Tuple[int, ...], List[int], List[int]]:
    # Returns (canonical_state_tuple, perm_used, inv_perm)
    best = None
    best_perm = PERMS[0]
    for perm in PERMS:
        tb = tuple(_apply_perm_board(board, perm))
        if best is None or tb < best:
            best = tb
            best_perm = perm
    inv = _invert_perm(best_perm)
    assert best is not None
    return best, best_perm, inv


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

    def _ensure_state(self, state: State, legal_moves: List[int]) -> None:
        if state not in self.q_table:
            self.q_table[state] = {a: 0.0 for a in legal_moves}
        else:
            # Ensure new legal moves exist (in case created later)
            for a in legal_moves:
                if a not in self.q_table[state]:
                    self.q_table[state][a] = 0.0

    def _best_action(self, state: State, legal_moves: List[int]) -> Tuple[int, float]:
        self._ensure_state(state, legal_moves)
        action_values = self.q_table[state]
        # Choose the action with max Q, tie-break randomly among best
        max_q = max(action_values[a] for a in legal_moves)
        best_actions = [a for a in legal_moves if action_values[a] == max_q]
        action = random.choice(best_actions)
        return action, max_q

    def get_action(self, board: List[int]) -> int:
        # Epsilon-greedy using current epsilon with symmetry canonicalization
        legal = get_legal_moves(board)
        if not legal:
            raise ValueError("No legal moves available")
        # Explore directly in original board space
        if random.random() < self.epsilon:
            return random.choice(legal)
        # Greedy via canonical state
        c_state, perm, inv = _canonicalize(board)
        c_legal = [perm[i] for i in legal]
        action_c, _ = self._best_action(c_state, c_legal)
        action_orig = inv[action_c]
        return action_orig

    def get_greedy_action(self, board: List[int]) -> int:
        legal = get_legal_moves(board)
        if not legal:
            raise ValueError("No legal moves available")
        c_state, perm, inv = _canonicalize(board)
        c_legal = [perm[i] for i in legal]
        action_c, _ = self._best_action(c_state, c_legal)
        return inv[action_c]

    def update(
        self,
        state: List[int],
        action: int,
        reward: float,
        next_state: Optional[List[int]],
        done: bool,
        zero_sum: bool = False,
    ) -> None:
        # Canonicalize current state/action
        c_state, perm, _ = _canonicalize(state)
        action_c = perm[action]
        c_legal = [perm[i] for i in get_legal_moves(state)]
        self._ensure_state(c_state, c_legal)
        current_q = self.q_table[c_state][action_c]

        target = reward
        if not done and next_state is not None:
            c_next, perm_next, _inv_next = _canonicalize(next_state)
            next_legal = get_legal_moves(next_state)
            if next_legal:
                c_next_legal = [perm_next[i] for i in next_legal]
                _, max_next_q = self._best_action(c_next, c_next_legal)
                # zero_sum: next_state is opponent's perspective, so opponent's gain = our loss
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
            alpha=data.get("alpha", 0.5),
            gamma=data.get("gamma", 0.9),
            epsilon=data.get("epsilon", 0.05),
            epsilon_decay=data.get("epsilon_decay", 0.9999),
            epsilon_min=data.get("epsilon_min", 0.05),
        )
        agent.q_table = data.get("q_table", {})
        return agent
