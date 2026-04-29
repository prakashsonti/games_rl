# Tic-Tac-Toe RL

Reinforcement Learning Tic-Tac-Toe agent (Q-learning) with a Flask API and a React (Vite) frontend. Humans play as X; the AI plays as O.

## Tech Stack
- Python 3.11, Flask, Flask-CORS
- React 18 + Vite
- Tailwind (Play CDN) for styling
- Framer Motion for animations

## Backend
Files live in `backend/`: `game.py`, `agent.py`, `train.py`, `api.py`. Trained Q-table is saved as `backend/q_table.pkl`.

## Frontend
React app in `frontend/` using Vite. Tailwind via CDN loaded in `public/index.html`.

## Quickstart

1) Install Python requirements
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r tictactoe-ai/requirements.txt
```

2) Run training (saves Q-table to `backend/q_table.pkl`)
```bash
cd tictactoe-ai
python -m backend.train
```

3) Start Flask server (port 5000)
```bash
cd tictactoe-ai
python -m backend.api
```

4) Frontend: install deps and start Vite (port 5173)
```bash
cd tictactoe-ai/frontend
npm install
npm run dev
```

Open http://localhost:5173 and play.
