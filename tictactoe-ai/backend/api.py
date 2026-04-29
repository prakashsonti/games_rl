import os
import random
from typing import List

from flask import Flask, jsonify, request
from flask_cors import CORS

from .agent import QLearningAgent
from .game import get_legal_moves, make_move, check_winner, is_draw


app = Flask(__name__)
CORS(app, resources={r"*": {"origins": ["http://localhost:5173", "http://localhost:5174"]}})


Q_TABLE_PATH = os.path.join(os.path.dirname(__file__), "q_table.pkl")


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
    # fallback empty agent
    agent = QLearningAgent(epsilon=0.0)
    return agent


agent = load_agent()


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/reload")
def reload_qtable():
    global agent
    try:
        agent = load_agent()
        return jsonify({"status": "reloaded"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.post("/move")
def move():
    data = request.get_json(force=True)
    board = data.get("board")
    player = data.get("player")  # expected -1 for AI in this app
    difficulty = (data.get("difficulty") or "hard").lower()

    if not isinstance(board, list) or len(board) != 9:
        return jsonify({"error": "Invalid board"}), 400
    if player not in (1, -1):
        return jsonify({"error": "Invalid player"}), 400

    legal = get_legal_moves(board)
    if not legal:
        return jsonify({"error": "No legal moves"}), 400

    pos: int
    if difficulty == "easy":
        pos = random.choice(legal)
    elif difficulty == "medium":
        if random.random() < 0.5:
            pos = random.choice(legal)
        else:
            pos = agent.get_greedy_action(perspective(board, player))
    else:  # hard
        pos = agent.get_greedy_action(perspective(board, player))

    new_board = make_move(board, pos, player)
    w = check_winner(new_board)
    d = is_draw(new_board)

    return jsonify({
        "position": pos,
        "winner": int(w) if w != 0 else None,
        "is_draw": bool(d),
    })


def run():
    app.run(host="0.0.0.0", port=5000)


if __name__ == "__main__":
    run()
