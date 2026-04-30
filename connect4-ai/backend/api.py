import os
import random
from typing import List

from flask import Flask, jsonify, request
from flask_cors import CORS

from .agent import DQNAgent, DEVICE
from .game import get_legal_moves, make_move, check_winner, is_draw, ROWS, COLS
from .train import get_alphabeta_action


app = Flask(__name__)
CORS(app, resources={r"*": {"origins": ["http://localhost:5174", "http://localhost:5175"]}})


MODEL_PATH = os.path.join(os.path.dirname(__file__), "dqn_model.pt")


def perspective(board: List[int], player: int) -> List[int]:
    return [v * player for v in board]


def _load_agent() -> DQNAgent:
    if os.path.exists(MODEL_PATH):
        try:
            import torch
            agent = DQNAgent.load(MODEL_PATH, map_location=torch.device("cpu"))
            agent.epsilon = 0.0
            agent.policy_net.eval()
            return agent
        except Exception:
            pass
    return DQNAgent(epsilon=0.0)


agent = _load_agent()
_model_loaded = os.path.exists(MODEL_PATH)


@app.get("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": _model_loaded})


@app.post("/reload")
def reload_model():
    global agent, _model_loaded
    try:
        agent = _load_agent()
        _model_loaded = os.path.exists(MODEL_PATH)
        return jsonify({"status": "reloaded", "model_loaded": _model_loaded})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.post("/move")
def move():
    data = request.get_json(force=True)
    board      = data.get("board")
    player     = data.get("player")
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
        if _model_loaded and random.random() < 0.5:
            col = agent.get_greedy_action(perspective(board, player))
        else:
            col = get_alphabeta_action(board, player, depth=3)
    else:  # hard
        if _model_loaded:
            col = agent.get_greedy_action(perspective(board, player))
        else:
            col = get_alphabeta_action(board, player, depth=6)

    new_board = make_move(board, col, player)

    # Determine the board index where the piece landed
    position = next(
        r * COLS + col
        for r in range(ROWS)
        if new_board[r * COLS + col] == player and board[r * COLS + col] == 0
    )

    w = check_winner(new_board)
    d = is_draw(new_board)

    return jsonify({
        "position": position,
        "winner":   int(w) if w != 0 else None,
        "is_draw":  bool(d),
    })


def run():
    app.run(host="0.0.0.0", port=5001)


if __name__ == "__main__":
    run()
