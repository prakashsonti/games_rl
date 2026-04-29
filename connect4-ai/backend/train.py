import os
import random
import argparse
from collections import defaultdict
from typing import List, Tuple, Optional

from .agent import QLearningAgent
from .game import get_legal_moves, make_move, check_winner, is_draw, WIN_LINES, ROWS, COLS


def perspective(board: List[int], player: int) -> List[int]:
    return [v * player for v in board]


# ---------------------------------------------------------------------------
# Alpha-beta oracle (depth-limited — full minimax is intractable for Connect 4)
# ---------------------------------------------------------------------------

def _score_window(window: List[int], player: int) -> float:
    opp = -player
    pc = window.count(player)
    oc = window.count(opp)
    if pc > 0 and oc > 0:
        return 0.0
    if pc == 4:
        return 1000.0
    if pc == 3:
        return 5.0
    if pc == 2:
        return 2.0
    if oc == 4:
        return -1000.0
    if oc == 3:
        return -4.0
    return 0.0


def _heuristic(board: List[int], player: int) -> float:
    score = 0.0
    for line in WIN_LINES:
        score += _score_window([board[i] for i in line], player)
    center = COLS // 2
    for r in range(ROWS):
        if board[r * COLS + center] == player:
            score += 0.3
        elif board[r * COLS + center] == -player:
            score -= 0.3
    return score


def _order_moves(legal: List[int]) -> List[int]:
    center = COLS // 2
    return sorted(legal, key=lambda c: abs(c - center))


def _alphabeta(board: List[int], depth: int, alpha: float, beta: float, maximizing: bool, ai_player: int) -> float:
    w = check_winner(board)
    if w == ai_player:
        return 100.0 + depth
    if w == -ai_player:
        return -100.0 - depth
    legal = get_legal_moves(board)
    if not legal:
        return 0.0
    if depth == 0:
        return _heuristic(board, ai_player)

    current = ai_player if maximizing else -ai_player
    if maximizing:
        value = -float("inf")
        for col in _order_moves(legal):
            nb = make_move(board, col, current)
            value = max(value, _alphabeta(nb, depth - 1, alpha, beta, False, ai_player))
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return value
    else:
        value = float("inf")
        for col in _order_moves(legal):
            nb = make_move(board, col, current)
            value = min(value, _alphabeta(nb, depth - 1, alpha, beta, True, ai_player))
            beta = min(beta, value)
            if alpha >= beta:
                break
        return value


def get_alphabeta_action(board: List[int], ai_player: int, depth: int = 4) -> int:
    legal = get_legal_moves(board)
    best_val = -float("inf")
    best_moves: List[int] = []
    alpha = -float("inf")
    for col in _order_moves(legal):
        nb = make_move(board, col, ai_player)
        val = _alphabeta(nb, depth - 1, alpha, float("inf"), False, ai_player)
        if val > best_val:
            best_val = val
            best_moves = [col]
            alpha = max(alpha, val)
        elif val == best_val:
            best_moves.append(col)
    return random.choice(best_moves)


# ---------------------------------------------------------------------------
# Training episodes
# ---------------------------------------------------------------------------

def play_self_play_episode(
    agent: QLearningAgent,
    draw_reward: float = 0.3,
    loss_reward: float = -1.0,
) -> Tuple[int, bool]:
    board = [0] * (ROWS * COLS)
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
            agent.update(state_p, action, 1.0, None, True)
            if prev_state_p is not None and prev_action is not None:
                agent.update(prev_state_p, prev_action, loss_reward, None, True)
        elif d:
            done = True
            agent.update(state_p, action, draw_reward, None, True)
            if prev_state_p is not None and prev_action is not None:
                agent.update(prev_state_p, prev_action, draw_reward, None, True)
        else:
            next_state_p = perspective(next_board, -player)
            agent.update(state_p, action, 0.0, next_state_p, False, zero_sum=True)

        prev_state_p = state_p
        prev_action = action
        board = next_board
        player = -player
        agent.decay_epsilon()

    return winner, is_draw(board)


def play_vs_alphabeta_episode(
    agent: QLearningAgent,
    ab_depth: int = 3,
    draw_reward: float = 0.3,
    loss_reward: float = -1.0,
) -> Tuple[int, bool]:
    """Train the Q-agent against a depth-limited alpha-beta opponent."""
    ai_player = random.choice([1, -1])
    board = [0] * (ROWS * COLS)
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
            action = get_alphabeta_action(board, current_player, depth=ab_depth)

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
            if w != 0:
                winner = w
                done = True
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
    games: int = 500_000,
    save_path: str = os.path.join(os.path.dirname(__file__), "q_table.pkl"),
    resume: bool = True,
    epsilon: Optional[float] = None,
    checkpoint_every: int = 10_000,
    draw_reward: float = 0.3,
    loss_reward: float = -1.0,
    alphabeta_fraction: float = 0.3,
    ab_depth: int = 3,
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
    ab_results = defaultdict(int)

    for i in range(1, games + 1):
        if random.random() < alphabeta_fraction:
            w, d = play_vs_alphabeta_episode(agent, ab_depth=ab_depth, draw_reward=draw_reward, loss_reward=loss_reward)
            if d:
                ab_results["draw"] += 1
            elif w == 1:
                ab_results["x_win"] += 1
            else:
                ab_results["o_win"] += 1
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
            ab = sum(ab_results.values())
            ab_draw = ab_results["draw"]
            ab_str = f" | vs-ab draws: {ab_draw/ab:.2%}" if ab else ""
            print(
                f"Games: {total:6d} | X wins: {x/total:.2%} | O wins: {o/total:.2%}"
                f" | Draws: {dr/total:.2%}{ab_str} | epsilon: {agent.epsilon:.4f}"
                f" | Q-states: {len(agent.q_table)}"
            )
        if checkpoint_every and (i % checkpoint_every == 0):
            agent.save(save_path)
            print(f"Checkpoint saved at {i} games -> {save_path}")

    agent.save(save_path)
    print(f"Saved Q-table to: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Connect 4 Q-learning agent")
    parser.add_argument("--games", type=int, default=500_000)
    parser.add_argument("--save-path", type=str, default=os.path.join(os.path.dirname(__file__), "q_table.pkl"))
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--resume", dest="resume", action="store_true")
    parser.set_defaults(resume=True)
    parser.add_argument("--epsilon", type=float, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=10_000)
    parser.add_argument("--draw-reward", type=float, default=0.3)
    parser.add_argument("--loss-reward", type=float, default=-1.0)
    parser.add_argument("--alphabeta-fraction", type=float, default=0.3)
    parser.add_argument("--ab-depth", type=int, default=3, help="Alpha-beta search depth for oracle opponent")
    parser.add_argument("--epsilon-min", type=float, default=None)
    parser.add_argument("--epsilon-decay", type=float, default=None)
    args = parser.parse_args()

    train(
        games=args.games,
        save_path=args.save_path,
        resume=args.resume,
        epsilon=args.epsilon,
        checkpoint_every=args.checkpoint_every,
        draw_reward=args.draw_reward,
        loss_reward=args.loss_reward,
        alphabeta_fraction=args.alphabeta_fraction,
        ab_depth=args.ab_depth,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
    )
