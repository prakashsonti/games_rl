import pickle
import random
from typing import Dict, List, Tuple, Optional

from .game import get_legal_moves


State = Tuple[int, ...]  # perspective state (current player's view)


class QLearningAgent:
    def __init__(
        self,
        alpha: float = 0.5,
        gamma: float = 0.9,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.9999,
        epsilon_min: float = 0.05,
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
        # Epsilon-greedy using current epsilon
        legal = get_legal_moves(board)
        if not legal:
            raise ValueError("No legal moves available")
        state: State = tuple(board)
        self._ensure_state(state, legal)
        if random.random() < self.epsilon:
            return random.choice(legal)
        action, _ = self._best_action(state, legal)
        return action

    def get_greedy_action(self, board: List[int]) -> int:
        legal = get_legal_moves(board)
        if not legal:
            raise ValueError("No legal moves available")
        state: State = tuple(board)
        action, _ = self._best_action(state, legal)
        return action

    def update(
        self,
        state: List[int],
        action: int,
        reward: float,
        next_state: Optional[List[int]],
        done: bool,
    ) -> None:
        legal = get_legal_moves(state)
        self._ensure_state(tuple(state), legal)
        current_q = self.q_table[tuple(state)][action]

        target = reward
        if not done and next_state is not None:
            next_legal = get_legal_moves(next_state)
            if next_legal:
                _, max_next_q = self._best_action(tuple(next_state), next_legal)
                target = reward + self.gamma * max_next_q

        self.q_table[tuple(state)][action] = current_q + self.alpha * (target - current_q)

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
