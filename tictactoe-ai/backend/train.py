import os
import random
import argparse
from collections import defaultdict
from typing import List, Tuple, Optional

from .agent import QLearningAgent
from .game import get_legal_moves, make_move, check_winner, is_draw


def perspective(board: List[int], player: int) -> List[int]:
    return [v * player for v in board]


# ---------------------------------------------------------------------------
# Minimax oracle (memoised — ~5k unique tic-tac-toe positions)
# ---------------------------------------------------------------------------

_MINIMAX_CACHE: dict = {}


def _minimax(board_key: tuple, player: int) -> int:
    """Return game value from X's (player=1) perspective: +1 X wins, -1 O wins, 0 draw."""
    key = (board_key, player)
    if key in _MINIMAX_CACHE:
        return _MINIMAX_CACHE[key]

    board = list(board_key)
    w = check_winner(board)
    if w != 0:
        _MINIMAX_CACHE[key] = w
        return w

    legal = get_legal_moves(board)
    if not legal:
        _MINIMAX_CACHE[key] = 0
        return 0

    if player == 1:  # X maximises
        best = -2
        for move in legal:
            val = _minimax(tuple(make_move(board, move, player)), -player)
            if val > best:
                best = val
            if best == 1:
                break
    else:  # O minimises
        best = 2
        for move in legal:
            val = _minimax(tuple(make_move(board, move, player)), -player)
            if val < best:
                best = val
            if best == -1:
                break

    _MINIMAX_CACHE[key] = best
    return best


def get_minimax_action(board: List[int], player: int) -> int:
    legal = get_legal_moves(board)
    best_val = -2 if player == 1 else 2
    best_moves: List[int] = []

    for move in legal:
        val = _minimax(tuple(make_move(board, move, player)), -player)
        if player == 1:
            if val > best_val:
                best_val, best_moves = val, [move]
            elif val == best_val:
                best_moves.append(move)
        else:
            if val < best_val:
                best_val, best_moves = val, [move]
            elif val == best_val:
                best_moves.append(move)

    return random.choice(best_moves)


def _precompute_minimax() -> None:
    """Warm up the minimax cache from the empty board (covers all ~5k reachable states)."""
    _minimax(tuple([0] * 9), 1)


_precompute_minimax()


# ---------------------------------------------------------------------------
# Training episodes
# ---------------------------------------------------------------------------

def play_self_play_episode(
    agent: QLearningAgent,
    draw_reward: float = 0.5,
    loss_reward: float = -1.0,
) -> Tuple[int, bool]:
    board = [0] * 9
    player = 1
    done = False
    winner = 0
    prev_state_p: Optional[List[int]] = None
    prev_action: Optional[int] = None

    while not done:
        state_p = perspective(board, player)
        action = agent.get_action(state_p)
        next_board = make_move(board, action, player)

        w = check_winner(next_board)
        d = is_draw(next_board)

        if w != 0:
            winner = w
            done = True
            # Reward the winning move
            agent.update(state_p, action, 1.0, None, True)
            # Penalise the loser's last move explicitly
            if prev_state_p is not None and prev_action is not None:
                agent.update(prev_state_p, prev_action, loss_reward, None, True)
        elif d:
            done = True
            agent.update(state_p, action, draw_reward, None, True)
            if prev_state_p is not None and prev_action is not None:
                agent.update(prev_state_p, prev_action, draw_reward, None, True)
        else:
            # Zero-sum bootstrapping: opponent's future value is our loss
            next_state_p = perspective(next_board, -player)
            agent.update(state_p, action, 0.0, next_state_p, False, zero_sum=True)

        prev_state_p = state_p
        prev_action = action
        board = next_board
        player = -player
        agent.decay_epsilon()

    return winner, is_draw(board)


def play_vs_minimax_episode(
    agent: QLearningAgent,
    draw_reward: float = 0.5,
    loss_reward: float = -1.0,
) -> Tuple[int, bool]:
    """Train the Q-agent against a perfect minimax opponent (AI randomly assigned X or O)."""
    ai_player = random.choice([1, -1])
    board = [0] * 9
    current_player = 1
    done = False
    winner = 0
    prev_state_p: Optional[List[int]] = None
    prev_action: Optional[int] = None

    while not done:
        if current_player == ai_player:
            state_p = perspective(board, current_player)
            action = agent.get_action(state_p)
        else:
            # Minimax picks the optimal move; no Q-update for the oracle
            action = get_minimax_action(board, current_player)

        next_board = make_move(board, action, current_player)
        w = check_winner(next_board)
        d = is_draw(next_board)

        if current_player == ai_player:
            if w != 0:
                winner = w
                done = True
                agent.update(state_p, action, 1.0, None, True)
            elif d:
                done = True
                agent.update(state_p, action, draw_reward, None, True)
                if prev_state_p is not None and prev_action is not None:
                    agent.update(prev_state_p, prev_action, draw_reward, None, True)
            else:
                next_state_p = perspective(next_board, -current_player)
                agent.update(state_p, action, 0.0, next_state_p, False, zero_sum=True)
            prev_state_p = state_p
            prev_action = action
        else:
            # Minimax just moved
            if w != 0:
                winner = w
                done = True
                # Minimax won = AI lost; penalise AI's last move
                if prev_state_p is not None and prev_action is not None:
                    agent.update(prev_state_p, prev_action, loss_reward, None, True)
            elif d:
                done = True
                if prev_state_p is not None and prev_action is not None:
                    agent.update(prev_state_p, prev_action, draw_reward, None, True)

        board = next_board
        current_player = -current_player
        agent.decay_epsilon()

    return winner, is_draw(board)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(
    games: int = 200_000,
    save_path: str = os.path.join(os.path.dirname(__file__), "q_table.pkl"),
    resume: bool = True,
    epsilon: Optional[float] = None,
    checkpoint_every: int = 10_000,
    draw_reward: float = 0.5,
    loss_reward: float = -1.0,
    minimax_fraction: float = 0.3,
    epsilon_min: Optional[float] = None,
    epsilon_decay: Optional[float] = None,
) -> None:
    if resume and os.path.exists(save_path):
        print(f"Resuming training from existing Q-table: {save_path}")
        agent = QLearningAgent.load(save_path)
        if epsilon is not None:
            agent.epsilon = epsilon
        if epsilon_min is not None:
            agent.epsilon_min = epsilon_min
        if epsilon_decay is not None:
            agent.epsilon_decay = epsilon_decay
    else:
        agent = QLearningAgent()
        if epsilon is not None:
            agent.epsilon = epsilon
        if epsilon_min is not None:
            agent.epsilon_min = epsilon_min
        if epsilon_decay is not None:
            agent.epsilon_decay = epsilon_decay

    results = defaultdict(int)
    minimax_results = defaultdict(int)

    for i in range(1, games + 1):
        if random.random() < minimax_fraction:
            w, d = play_vs_minimax_episode(agent, draw_reward=draw_reward, loss_reward=loss_reward)
            if d:
                minimax_results["draw"] += 1
            elif w == 1:
                minimax_results["x_win"] += 1
            else:
                minimax_results["o_win"] += 1
        else:
            w, d = play_self_play_episode(agent, draw_reward=draw_reward, loss_reward=loss_reward)

        if d:
            results["draw"] += 1
        elif w == 1:
            results["x_win"] += 1
        else:
            results["o_win"] += 1

        if i % 10_000 == 0:
            total = i
            x = results["x_win"]
            o = results["o_win"]
            dr = results["draw"]
            mm = sum(minimax_results.values())
            mm_draw = minimax_results["draw"]
            mm_str = f" | vs-minimax draws: {mm_draw/mm:.2%}" if mm else ""
            print(
                f"Games: {total:6d} | X wins: {x/total:.2%} | O wins: {o/total:.2%}"
                f" | Draws: {dr/total:.2%}{mm_str} | epsilon: {agent.epsilon:.4f}"
            )
        if checkpoint_every and (i % checkpoint_every == 0):
            agent.save(save_path)
            print(f"Checkpoint saved at {i} games -> {save_path}")

    agent.save(save_path)
    print(f"Saved Q-table to: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Tic-Tac-Toe Q-learning agent")
    parser.add_argument("--games", type=int, default=500_000, help="Number of training games")
    parser.add_argument(
        "--save-path",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "q_table.pkl"),
        help="Where to save the Q-table",
    )
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="Do not load existing table")
    parser.add_argument("--resume", dest="resume", action="store_true", help="Load existing table if present")
    parser.set_defaults(resume=True)
    parser.add_argument("--epsilon", type=float, default=None, help="Override starting epsilon (e.g., 1.0)")
    parser.add_argument("--checkpoint-every", type=int, default=10_000, help="Save every N games (0 to disable)")
    parser.add_argument("--draw-reward", type=float, default=0.5, help="Reward for a draw (default: 0.5)")
    parser.add_argument("--loss-reward", type=float, default=-1.0, help="Reward for losing (default: -1.0)")
    parser.add_argument("--minimax-fraction", type=float, default=0.3, help="Fraction of episodes vs minimax (0.0–1.0)")
    parser.add_argument("--epsilon-min", type=float, default=None, help="Minimum epsilon floor (override)")
    parser.add_argument("--epsilon-decay", type=float, default=None, help="Epsilon multiplicative decay per step")
    args = parser.parse_args()

    train(
        games=args.games,
        save_path=args.save_path,
        resume=args.resume,
        epsilon=args.epsilon,
        checkpoint_every=args.checkpoint_every,
        draw_reward=args.draw_reward,
        loss_reward=args.loss_reward,
        minimax_fraction=args.minimax_fraction,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
    )
