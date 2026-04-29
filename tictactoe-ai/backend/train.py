import os
import argparse
from collections import defaultdict
from typing import List, Tuple, Optional

from .agent import QLearningAgent
from .game import get_legal_moves, make_move, check_winner, is_draw


def perspective(board: List[int], player: int) -> List[int]:
    return [v * player for v in board]


def play_self_play_episode(agent: QLearningAgent) -> Tuple[int, bool]:
    board = [0] * 9
    player = 1  # X starts
    done = False

    # Stats: 1 if X wins, -1 if O wins, 0 if draw/ongoing
    winner = 0

    while not done:
        state_p = perspective(board, player)
        action = agent.get_action(state_p)
        next_board = make_move(board, action, player)

        w = check_winner(next_board)
        d = is_draw(next_board)

        reward = 0.0
        if w != 0:
            winner = w
            reward = 1.0  # current player won
            done = True
            agent.update(state_p, action, reward, None, True)
        elif d:
            reward = 0.5
            done = True
            agent.update(state_p, action, reward, None, True)
        else:
            # Non-terminal: next state's perspective is for the next player
            next_state_p = perspective(next_board, -player)
            agent.update(state_p, action, reward, next_state_p, False)

        board = next_board
        player = -player
        agent.decay_epsilon()

    return winner, is_draw(board)


def train(
    games: int = 200_000,
    save_path: str = os.path.join(os.path.dirname(__file__), "q_table.pkl"),
    resume: bool = True,
    epsilon: Optional[float] = None,
    checkpoint_every: int = 10_000,
) -> None:
    if resume and os.path.exists(save_path):
        print(f"Resuming training from existing Q-table: {save_path}")
        agent = QLearningAgent.load(save_path)
        if epsilon is not None:
            agent.epsilon = epsilon
    else:
        agent = QLearningAgent()
        if epsilon is not None:
            agent.epsilon = epsilon
    results = defaultdict(int)

    for i in range(1, games + 1):
        w, d = play_self_play_episode(agent)
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
            print(
                f"Games: {total:6d} | X wins: {x/total:.2%} | O wins: {o/total:.2%} | Draws: {dr/total:.2%} | epsilon: {agent.epsilon:.4f}"
            )
        if checkpoint_every and (i % checkpoint_every == 0):
            agent.save(save_path)
            # lightweight feedback to ensure persistence on long runs
            print(f"Checkpoint saved at {i} games -> {save_path}")

    agent.save(save_path)
    print(f"Saved Q-table to: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Tic-Tac-Toe Q-learning agent")
    parser.add_argument("--games", type=int, default=200_000, help="Number of self-play games")
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
    args = parser.parse_args()

    train(
        games=args.games,
        save_path=args.save_path,
        resume=args.resume,
        epsilon=args.epsilon,
        checkpoint_every=args.checkpoint_every,
    )
