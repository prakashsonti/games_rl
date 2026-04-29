# Backend

Python Q-learning agent for Tic-Tac-Toe, served over a Flask API.

## Files

### `game.py`
Pure game logic — no ML, no state. Everything else imports from here.

- `Board` — type alias for a 9-element list of ints (`0` = empty, `1` = X, `-1` = O). Positions are indexed 0–8 left-to-right, top-to-bottom.
- `get_legal_moves(board)` — returns a list of empty position indices.
- `make_move(board, position, player)` — returns a new board with the player's piece placed; raises if the position is already taken.
- `check_winner(board)` — returns `1` if X wins, `-1` if O wins, `0` if no winner yet.
- `is_draw(board)` — returns `True` when the board is full and there is no winner.
- `print_board(board)` — ASCII pretty-print for debugging.

---

### `agent.py`
The Q-learning agent. Stores and queries a Q-table that maps board states to per-action values.

**Key design choices:**

- **Perspective normalisation** — the agent always sees the board from the *current player's* point of view: its own pieces are `+1`, the opponent's are `-1`. This lets a single Q-table serve both sides.
- **8-fold symmetry canonicalisation** — before every Q-table lookup the board is transformed into its lexicographically smallest equivalent across all rotations and reflections. This collapses the ~5 000 reachable positions down to ~765 canonical states, making the table much smaller and training much faster.

**`QLearningAgent` constructor parameters:**

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `alpha` | `0.3` | Learning rate |
| `gamma` | `0.95` | Discount factor |
| `epsilon` | `1.0` | Starting exploration rate |
| `epsilon_decay` | `0.999997` | Multiplicative decay applied after every step |
| `epsilon_min` | `0.02` | Floor for epsilon |

**Key methods:**

| Method | Purpose |
|--------|---------|
| `get_action(board)` | Epsilon-greedy action (used during training) |
| `get_greedy_action(board)` | Pure greedy action with no exploration (used at inference) |
| `update(state, action, reward, next_state, done, zero_sum)` | Bellman Q-update; pass `zero_sum=True` for two-player updates so the opponent's future value is *subtracted* rather than added |
| `decay_epsilon()` | Multiplies epsilon by `epsilon_decay`, floored at `epsilon_min` |
| `save(path)` / `load(path)` | Pickle the Q-table and hyperparameters to/from disk |

---

### `train.py`
Trains the agent via self-play and games against a minimax oracle.

**Minimax oracle** (`_minimax`, `get_minimax_action`) — a fully memoised minimax search over all ~5 000 tic-tac-toe positions. The cache is pre-warmed at import time so there is no runtime recursion cost during training. Games against the oracle force the agent to handle optimal play and prevent it from converging on a "mutual exploit" equilibrium that self-play alone can produce.

**`play_self_play_episode`** — one game where the agent plays both sides.  
Key fixes vs. naïve self-play:
1. **Zero-sum bootstrapping** — non-terminal updates use `zero_sum=True` so that a strong opponent position penalises the current player's Q-value instead of boosting it.
2. **Explicit loss penalty** — when a player wins, the *loser's* last move is updated with `loss_reward = -1.0`. Without this, the losing side receives no gradient from its fatal mistake.

**`play_vs_minimax_episode`** — one game where the agent is randomly assigned X or O and the other side is the minimax oracle.

**`train()`** — main loop. Mixes self-play and minimax episodes according to `minimax_fraction`, prints stats every 10 000 games, and saves checkpoints.

#### Training flags

Run from the `tictactoe-ai` directory:

```bash
../.venv/bin/python -m backend.train [flags]
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--games` | `500000` | Number of training episodes |
| `--no-resume` | — | Start from scratch, ignoring any existing `q_table.pkl` |
| `--resume` | *(default)* | Load existing Q-table and continue training |
| `--epsilon` | `1.0` | Override the starting exploration rate |
| `--epsilon-decay` | *(from agent)* | Multiplicative decay per step; `0.999997` keeps exploration meaningful for ~150 k games out of 500 k |
| `--epsilon-min` | *(from agent)* | Floor for epsilon |
| `--minimax-fraction` | `0.3` | Fraction of episodes played against the perfect minimax oracle; `0.0` = pure self-play |
| `--draw-reward` | `0.5` | Reward assigned to both sides on a draw |
| `--loss-reward` | `-1.0` | Reward assigned to the losing player's last move |
| `--checkpoint-every` | `10000` | Save `q_table.pkl` every N games (`0` to disable) |
| `--save-path` | `backend/q_table.pkl` | Output path for the Q-table |

**Recommended fresh-training command:**

```bash
../.venv/bin/python -m backend.train \
  --no-resume \
  --games 500000 \
  --epsilon 1.0 \
  --epsilon-decay 0.999997 \
  --epsilon-min 0.02 \
  --minimax-fraction 0.3
```

---

### `api.py`
Flask REST API that the React frontend calls.

The agent is loaded at startup with `epsilon = 0.0` (pure greedy, no exploration). The `POST /reload` endpoint hot-swaps the in-memory agent from disk without restarting the server — useful after a training run.

**Endpoints:**

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Returns `{"status": "ok"}` |
| `POST` | `/reload` | Reloads `q_table.pkl` from disk into the running server |
| `POST` | `/move` | Returns the AI's chosen move given the current board |

**`POST /move` request body:**

```json
{
  "board": [0, 0, 0, 0, 0, 0, 0, 0, 0],
  "player": -1,
  "difficulty": "hard"
}
```

- `board` — 9-element array; `0` empty, `1` X, `-1` O.
- `player` — which side the AI is playing (`-1` in the standard frontend setup, where the human is always X).
- `difficulty` — `"easy"` (random), `"medium"` (50 % random / 50 % greedy), or `"hard"` (fully greedy).

**`POST /move` response:**

```json
{
  "position": 4,
  "winner": null,
  "is_draw": false
}
```

After retraining, reload without restarting:

```bash
curl -X POST http://localhost:5000/reload
```

---

### `q_table.pkl`
Pickle file produced by `train.py`. Contains the serialised Q-table dictionary and the agent's hyperparameters (`alpha`, `gamma`, `epsilon`, `epsilon_decay`, `epsilon_min`). Not checked into version control — regenerate by running `train.py`.
