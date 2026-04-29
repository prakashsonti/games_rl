import os
import random
from typing import List

from flask import Flask, jsonify, request
from flask_cors import CORS

from .agent import QLearningAgent
from .game import get_legal_moves, make_move, check_winner, is_draw, ROWS, COLS
from .train import get_alphabeta_action


app = Flask(__name__)
CORS(app, resources={r"*": {"origins": ["http://localhost:5174", "http://localhost:5175"]}})


Q_TABLE_PATH = os.path.join(os.path.dirname(__file__), "q_table.pkl")

_Q_TABLE_TRAINED_THRESHOLD = 500


def perspective(board: List[int], player: int) -> List[int]:
    return [v * player for v in board]


def load_agent() -> QLearningAgent:
    if os.path.exists(Q_TABLE_PATH):
        try:
            agent = QLearningAgent.load(Q_TABLE_PATH)
            agent.epsilon = 0.0
            return agent
        except Exception:
            pass
    return QLearningAgent(epsilon=0.0)


agent = load_agent()


def _is_trained() -> bool:
    return len(agent.q_table) >= _Q_TABLE_TRAINED_THRESHOLD


@app.get("/health")
def health():
    return jsonify({"status": "ok", "q_states": len(agent.q_table), "trained": _is_trained()})


@app.post("/reload")
def reload_qtable():
    global agent
    try:
        agent = load_agent()
        return jsonify({"status": "reloaded", "q_states": len(agent.q_table)})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.post("/move")
def move():
    data = request.get_json(force=True)
    board = data.get("board")
    player = data.get("player")
    difficulty = (data.get("difficulty") or "hard").lower()

    if not isinstance(board, list) or len(board) != ROWS * COLS:
        return jsonify({"error": "Invalid board"}), 400
    if player not in (1, -1):
        return jsonify({"error": "Invalid player"}), 400

    legal = get_legal_moves(board)
    if not legal:
        return jsonify({"error": "No legal moves"}), 400

    col: int
    if difficulty == "easy":
        col = random.choice(legal)
    elif difficulty == "medium":
        if _is_trained() and random.random() < 0.5:
            col = agent.get_greedy_action(perspective(board, player))
        else:
            col = get_alphabeta_action(board, player, depth=3)
    else:  # hard
        if _is_trained():
            col = agent.get_greedy_action(perspective(board, player))
        else:
            col = get_alphabeta_action(board, player, depth=6)

    new_board = make_move(board, col, player)

    # find the board index where the piece landed
    position = next(
        r * COLS + col
        for r in range(ROWS)
        if new_board[r * COLS + col] == player and board[r * COLS + col] == 0
    )

    w = check_winner(new_board)
    d = is_draw(new_board)

    return jsonify({
        "position": position,
        "winner": int(w) if w != 0 else None,
        "is_draw": bool(d),
    })


def run():
    app.run(host="0.0.0.0", port=5001)


if __name__ == "__main__":
    run()
