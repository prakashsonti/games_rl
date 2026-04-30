import os
import random
import argparse
from collections import defaultdict
from typing import List, Tuple, Optional

from .agent import DQNAgent
from .game import get_legal_moves, make_move, check_winner, is_draw, WIN_LINES, ROWS, COLS


SIZE = ROWS * COLS


def perspective(board: List[int], player: int) -> List[int]:
    return [v * player for v in board]


# ---------------------------------------------------------------------------
# Alpha-beta oracle (depth-limited — used as training opponent)
# ---------------------------------------------------------------------------

def _score_window(window: List[int], player: int) -> float:
    opp = -player
    pc = window.count(player)
    oc = window.count(opp)
    if pc > 0 and oc > 0:
        return 0.0
    if pc == 4: return  1000.0
    if pc == 3: return     5.0
    if pc == 2: return     2.0
    if oc == 4: return -1000.0
    if oc == 3: return    -4.0
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


def _order(legal: List[int]) -> List[int]:
    center = COLS // 2
    return sorted(legal, key=lambda c: abs(c - center))


def _alphabeta(board: List[int], depth: int, alpha: float, beta: float,
               maximizing: bool, ai_player: int) -> float:
    w = check_winner(board)
    if w == ai_player:  return  100.0 + depth
    if w == -ai_player: return -100.0 - depth
    legal = get_legal_moves(board)
    if not legal:       return 0.0
    if depth == 0:      return _heuristic(board, ai_player)

    current = ai_player if maximizing else -ai_player
    if maximizing:
        value = -float("inf")
        for col in _order(legal):
            nb = make_move(board, col, current)
            value = max(value, _alphabeta(nb, depth - 1, alpha, beta, False, ai_player))
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return value
    else:
        value = float("inf")
        for col in _order(legal):
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
    for col in _order(legal):
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
    agent: DQNAgent,
    draw_reward: float = 0.3,
    loss_reward: float = -1.0,
    learn_every: int = 4,
) -> Tuple[int, bool]:
    board = [0] * SIZE
    player = 1
    prev_state_p: Optional[List[int]] = None
    prev_action: Optional[int] = None
    winner = 0
    step = 0

    while True:
        state_p = perspective(board, player)
        action  = agent.get_action(state_p)
        next_board = make_move(board, action, player)
        step += 1

        w = check_winner(next_board)
        d = is_draw(next_board)

        if w != 0:
            winner = w
            agent.push(state_p, action, 1.0, None, True)
            if prev_state_p is not None:
                agent.push(prev_state_p, prev_action, loss_reward, None, True)
            agent.learn()
            agent.decay_epsilon()
            break
        elif d:
            agent.push(state_p, action, draw_reward, None, True)
            if prev_state_p is not None:
                agent.push(prev_state_p, prev_action, draw_reward, None, True)
            agent.learn()
            agent.decay_epsilon()
            break
        else:
            # Zero-sum: store opponent's perspective as next_state so the
            # target net predicts opponent's value, which we negate in learn().
            next_state_p = perspective(next_board, -player)
            agent.push(state_p, action, 0.0, next_state_p, False)
            if step % learn_every == 0:
                agent.learn()
            agent.decay_epsilon()

        prev_state_p = state_p
        prev_action  = action
        board  = next_board
        player = -player

    return winner, is_draw(board)


def play_vs_alphabeta_episode(
    agent: DQNAgent,
    ab_depth: int = 3,
    draw_reward: float = 0.3,
    loss_reward: float = -1.0,
    learn_every: int = 4,
) -> Tuple[int, bool]:
    """Train the DQN agent against a depth-limited alpha-beta opponent."""
    ai_player = random.choice([1, -1])
    board = [0] * SIZE
    current_player = 1
    prev_state_p: Optional[List[int]] = None
    prev_action: Optional[int] = None
    winner = 0
    step = 0

    while True:
        if current_player == ai_player:
            state_p = perspective(board, current_player)
            action  = agent.get_action(state_p)
        else:
            action = get_alphabeta_action(board, current_player, depth=ab_depth)

        next_board = make_move(board, action, current_player)
        w = check_winner(next_board)
        d = is_draw(next_board)
        step += 1

        if current_player == ai_player:
            if w != 0:
                winner = w
                agent.push(state_p, action, 1.0, None, True)
                agent.learn()
                agent.decay_epsilon()
                break
            elif d:
                agent.push(state_p, action, draw_reward, None, True)
                if prev_state_p is not None:
                    agent.push(prev_state_p, prev_action, draw_reward, None, True)
                agent.learn()
                agent.decay_epsilon()
                break
            else:
                next_state_p = perspective(next_board, -current_player)
                agent.push(state_p, action, 0.0, next_state_p, False)
                if step % learn_every == 0:
                    agent.learn()
                agent.decay_epsilon()
            prev_state_p = state_p
            prev_action  = action
        else:
            # Alpha-beta just moved
            if w != 0:
                winner = w
                if prev_state_p is not None:
                    agent.push(prev_state_p, prev_action, loss_reward, None, True)
                agent.learn()
                agent.decay_epsilon()
                break
            elif d:
                if prev_state_p is not None:
                    agent.push(prev_state_p, prev_action, draw_reward, None, True)
                agent.learn()
                agent.decay_epsilon()
                break

        board  = next_board
        current_player = -current_player

    return winner, is_draw(board)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(
    games: int = 500_000,
    save_path: str = os.path.join(os.path.dirname(__file__), "dqn_model.pt"),
    resume: bool = True,
    epsilon: Optional[float] = None,
    checkpoint_every: int = 10_000,
    print_every: int = 1_000,
    draw_reward: float = 0.3,
    loss_reward: float = -1.0,
    alphabeta_fraction: float = 0.3,
    ab_depth: int = 3,
    epsilon_min: Optional[float] = None,
    epsilon_decay: Optional[float] = None,
    lr: float = 1e-3,
    batch_size: int = 128,
    buffer_size: int = 100_000,
    target_update_every: int = 500,
    learn_every: int = 4,
) -> None:
    if resume and os.path.exists(save_path):
        print(f"Resuming from {save_path}")
        agent = DQNAgent.load(save_path)
        agent.policy_net.train()
        if epsilon       is not None: agent.epsilon       = epsilon
        if epsilon_min   is not None: agent.epsilon_min   = epsilon_min
        if epsilon_decay is not None: agent.epsilon_decay = epsilon_decay
    else:
        agent = DQNAgent(
            lr=lr,
            gamma=0.95,
            epsilon=epsilon if epsilon is not None else 1.0,
            epsilon_decay=epsilon_decay if epsilon_decay is not None else 0.999997,
            epsilon_min=epsilon_min if epsilon_min is not None else 0.02,
            buffer_size=buffer_size,
            batch_size=batch_size,
            target_update_every=target_update_every,
        )

    results     = defaultdict(int)
    ab_results  = defaultdict(int)
    _game_steps = 0   # total env steps across all episodes

    print(f"Training on: {next(agent.policy_net.parameters()).device}")
    print(f"Print every {print_every} games | checkpoint every {checkpoint_every} | learn every {learn_every} steps")

    for i in range(1, games + 1):
        if random.random() < alphabeta_fraction:
            w, d = play_vs_alphabeta_episode(agent, ab_depth=ab_depth,
                                             draw_reward=draw_reward, loss_reward=loss_reward,
                                             learn_every=learn_every)
            if d:       ab_results["draw"]  += 1
            elif w == 1:ab_results["x_win"] += 1
            else:       ab_results["o_win"] += 1
        else:
            w, d = play_self_play_episode(agent, draw_reward=draw_reward, loss_reward=loss_reward,
                                          learn_every=learn_every)

        if d:        results["draw"]  += 1
        elif w == 1: results["x_win"] += 1
        else:        results["o_win"] += 1

        if i % print_every == 0:
            total = i
            x, o, dr = results["x_win"], results["o_win"], results["draw"]
            ab  = sum(ab_results.values())
            ab_draw = ab_results["draw"]
            ab_str  = f" | vs-ab draws: {ab_draw/ab:.2%}" if ab else ""
            print(
                f"Games: {total:6d} | X: {x/total:.2%} | O: {o/total:.2%}"
                f" | Draw: {dr/total:.2%}{ab_str}"
                f" | ε: {agent.epsilon:.4f} | buf: {len(agent.buffer)}"
            )
        if checkpoint_every and (i % checkpoint_every == 0):
            agent.save(save_path)
            print(f"  checkpoint → {save_path}")

    agent.save(save_path)
    print(f"Saved model to: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Connect 4 DQN agent")
    parser.add_argument("--games",              type=int,   default=500_000)
    parser.add_argument("--save-path",          type=str,
                        default=os.path.join(os.path.dirname(__file__), "dqn_model.pt"))
    parser.add_argument("--no-resume",  dest="resume", action="store_false")
    parser.add_argument("--resume",     dest="resume", action="store_true")
    parser.set_defaults(resume=True)
    parser.add_argument("--epsilon",            type=float, default=None)
    parser.add_argument("--checkpoint-every",   type=int,   default=10_000)
    parser.add_argument("--print-every",        type=int,   default=1_000,
                        help="Print stats every N games (default 1000)")
    parser.add_argument("--learn-every",        type=int,   default=4,
                        help="Run a gradient step every N env steps (default 4)")
    parser.add_argument("--draw-reward",        type=float, default=0.3)
    parser.add_argument("--loss-reward",        type=float, default=-1.0)
    parser.add_argument("--alphabeta-fraction", type=float, default=0.3)
    parser.add_argument("--ab-depth",           type=int,   default=3,
                        help="Alpha-beta depth for training opponent")
    parser.add_argument("--epsilon-min",        type=float, default=None)
    parser.add_argument("--epsilon-decay",      type=float, default=None)
    parser.add_argument("--lr",                 type=float, default=1e-3)
    parser.add_argument("--batch-size",         type=int,   default=128)
    parser.add_argument("--buffer-size",        type=int,   default=100_000)
    parser.add_argument("--target-update-every",type=int,   default=500)
    args = parser.parse_args()

    train(
        games=args.games,
        save_path=args.save_path,
        resume=args.resume,
        epsilon=args.epsilon,
        checkpoint_every=args.checkpoint_every,
        print_every=args.print_every,
        draw_reward=args.draw_reward,
        loss_reward=args.loss_reward,
        alphabeta_fraction=args.alphabeta_fraction,
        ab_depth=args.ab_depth,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
        lr=args.lr,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        target_update_every=args.target_update_every,
        learn_every=args.learn_every,
    )
