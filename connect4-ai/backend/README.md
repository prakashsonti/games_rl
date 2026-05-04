# Backend

Python Deep Q-Network (DQN) agent for Connect 4, served over a Flask API.

---

## Approach

### Why not a Q-table?

Tic-tac-toe has roughly 5,000 reachable board positions, so a plain lookup table (Q-table) can be fully populated during training. Connect 4 has approximately **4.5 trillion** reachable positions. A table-based agent would stay 99.99 % empty no matter how long you train — it would never generalise to positions it hasn't seen before.

### Deep Q-Network (DQN)

Instead of storing one Q-value per `(state, action)` pair in a table, a DQN uses a **neural network** to *approximate* the Q-function:

```
Q(s, a ; θ) ≈ expected discounted reward of taking action a in state s
```

The network takes a board position as input and outputs one Q-value for each of the 7 columns. The agent picks the column with the highest Q-value (masked to ignore full columns).

### Board representation — 3-channel binary encoding

A flat list of 42 integers with values `{-1, 0, 1}` is a poor input for a neural network because adjacent cells have no structural relationship in a fully-connected layer. Instead the board is encoded as **three binary channels of shape (3 × 6 × 7)**:

| Channel | Meaning | Value |
|---------|---------|-------|
| 0 | Current player's pieces | 1 where occupied, else 0 |
| 1 | Opponent's pieces | 1 where occupied, else 0 |
| 2 | Empty cells | 1 where empty, else 0 |

This lets convolutional filters learn spatial patterns like "three in a row with a gap" or "diagonal threat" — things a flat layer would need orders of magnitude more data to discover.

**Perspective normalisation** is applied before encoding: the board is always shown from the *current player's* point of view (their pieces are always channel 0). A single network therefore serves both players during self-play.

### Convolutional neural network

```
Input (3, 6, 7)
  │
  ├─ Conv2d(3 → 64,  3×3, padding=1) → BatchNorm2d → ReLU
  ├─ Conv2d(64 → 128, 3×3, padding=1) → BatchNorm2d → ReLU
  ├─ Conv2d(128 → 128, 3×3, padding=1) → BatchNorm2d → ReLU
  │
  └─ Flatten  →  Linear(5376, 256)  →  ReLU  →  Linear(256, 7)

Output: Q-values for columns 0–6
```

Three `3×3` convolutional layers with `same` padding scan every possible 3×3 window on the board, building increasingly abstract representations of threats and opportunities. `BatchNorm2d` stabilises training by normalising activations after each conv layer. The final two fully-connected layers map the spatial features to per-column Q-values.

### Training algorithm

#### Experience replay

Rather than updating the network immediately after each move, every `(state, action, reward, next_state, done)` transition is stored in a **replay buffer** (capacity 200,000). At each gradient step a random mini-batch of 256 transitions is sampled. This breaks the temporal correlation between consecutive transitions and stabilises training.

#### Target network

A second network — the **target network** — is used to compute the Bellman backup target. Its weights are frozen and only copied from the policy network every 1,000 gradient steps. This prevents the target from shifting under the gradient and avoids the "chasing a moving target" instability common in vanilla DQN.

#### Double DQN

Standard DQN tends to *overestimate* Q-values because the same network selects and evaluates the best next action. Double DQN decouples these two steps:

1. **Policy network** selects the best next action: `a* = argmax_a Q_policy(s', a)`
2. **Target network** evaluates it: `Q_target(s', a*)`

This reduces overestimation bias and leads to better-calibrated Q-values.

#### Zero-sum Bellman update

Connect 4 is a two-player zero-sum game. After the agent plays, it is the *opponent's* turn. The value of the resulting position for the opponent is the *negative* of its value for the agent. The Bellman target therefore subtracts the opponent's future value rather than adding it:

```
target = r + γ · (−max_a Q_target(s_opponent, a))     (non-terminal)
target = r                                              (terminal)
```

`next_state` stored in the buffer is always from the opponent's perspective, so this negation is applied uniformly during the `learn()` step.

#### Self-play and alpha-beta oracle

Training alternates between two episode types:

- **Self-play (70 %)** — the agent plays both sides. Early in training this produces random games; as Q-values improve, both sides gradually play stronger. A zero-sum loss penalty is applied to the loser's last move so both sides receive a gradient signal.

- **vs alpha-beta (30 %)** — the agent is randomly assigned yellow or red, and the other side is a depth-limited alpha-beta searcher. This exposes the agent to principled play it would not encounter in pure self-play, preventing it from converging on mutual-exploit equilibria.

#### Reward scheme

| Outcome | Reward |
|---------|--------|
| Win | +1.0 |
| Loss (applied to loser's last move) | −1.0 |
| Draw | +0.3 |
| Non-terminal step | 0.0 |

#### Epsilon-greedy exploration

During training the agent selects a random legal column with probability ε. ε starts at 1.0 (fully random) and decays multiplicatively by `0.999997` per environment step, reaching the floor of `0.02` after roughly 3–4 million steps.

---

## Core libraries

| Library | Version | Role |
|---------|---------|------|
| **PyTorch** (`torch`) | ≥ 2.0 | Neural network, GPU compute, automatic differentiation |
| **Flask** | 3.0.2 | REST API server |
| **Flask-CORS** | 4.0.1 | Cross-origin headers so the Vite frontend can call the API |

PyTorch is the only heavy dependency. It provides `nn.Module` for the network definition, `optim.Adam` for gradient descent, and transparent CUDA support so the same code runs on CPU or GPU without changes.

---

## Files

### `game.py`

Pure game logic — no ML, no state. All other modules import from here.

| Symbol | Type | Description |
|--------|------|-------------|
| `ROWS`, `COLS` | `int` | Board dimensions: 6 rows × 7 columns |
| `Board` | type alias | `List[int]` — 42 integers, row-major. `0` = empty, `1` = Yellow/Human, `-1` = Red/AI |
| `WIN_LINES` | tuple of 4-tuples | All 69 winning combinations (24 horizontal + 21 vertical + 12 diagonal-right + 12 diagonal-left), built once at import time |
| `get_legal_moves(board)` | function | Returns a list of column indices (0–6) where the top cell is empty. A column is legal iff `board[c] == 0`. |
| `make_move(board, col, player)` | function | Returns a new board with `player`'s piece dropped in `col`. Iterates from the bottom row upward to find the first empty cell — this models gravity. Raises `ValueError` if the column is full. |
| `check_winner(board)` | function | Scans every entry in `WIN_LINES`. Returns `1` if Yellow wins, `-1` if Red wins, `0` if no winner yet. Uses integer summation: a line sums to `4` iff all four cells are Yellow, `−4` iff all are Red. |
| `is_draw(board)` | function | Returns `True` when the board is full and `check_winner` returns `0`. |

---

### `agent.py`

Defines the neural network and the DQN agent that wraps it.

#### `_ConvNet` (PyTorch module)

The policy and target networks are both instances of this class.

| Layer | Parameters | Output shape | Purpose |
|-------|-----------|--------------|---------|
| `Conv2d(3, 64, 3×3, padding=1)` + BN + ReLU | 1,792 | (64, 6, 7) | Detect low-level features across every 3×3 window |
| `Conv2d(64, 128, 3×3, padding=1)` + BN + ReLU | 73,856 | (128, 6, 7) | Combine low-level features into threats / blocks |
| `Conv2d(128, 128, 3×3, padding=1)` + BN + ReLU | 147,584 | (128, 6, 7) | Higher-order spatial patterns |
| `Flatten` | — | 5,376 | Linearise spatial feature maps |
| `Linear(5376, 256)` + ReLU | 1,376,512 | 256 | Abstract strategic representation |
| `Linear(256, 7)` | 1,799 | 7 | One Q-value per column |

Total trainable parameters: ~1.6 million.

#### Board tensor helpers

| Function | Signature | Description |
|----------|-----------|-------------|
| `_board_to_tensor(board)` | `List[int] → Tensor (1,3,6,7)` | Converts a single flat board (already in current-player perspective) to a 3-channel tensor on `DEVICE`. Used during inference. |
| `_flat_to_3ch_batch(flat_t)` | `Tensor (B,42) → Tensor (B,3,6,7)` | Vectorised batch conversion. Used inside `learn()` to avoid a Python loop over the batch. |

#### `DQNAgent`

| Method | Description |
|--------|-------------|
| `__init__(lr, gamma, epsilon, …)` | Instantiates policy net, target net (frozen copy), Adam optimiser, and replay buffer deque. Both nets are moved to `DEVICE` (CUDA if available). |
| `get_action(board)` | Epsilon-greedy action. Returns a random legal column with probability ε, otherwise calls `_greedy`. Used during training. |
| `get_greedy_action(board)` | Pure greedy action with no exploration. Used at inference (API calls). |
| `_greedy(board)` | Runs a forward pass through the policy net, applies the illegal-column mask (`-1e9` on full columns), and returns `argmax`. |
| `_illegal_mask(board)` | Builds a `(7,)` additive mask. Full columns (top cell ≠ 0) receive `−1e9` so they are never chosen by `argmax`. |
| `push(state, action, reward, next_state, done)` | Appends a transition to the replay buffer. `state` and `next_state` are stored as flat lists; tensor conversion happens lazily in `learn()`. |
| `learn()` | Samples a mini-batch of 256, converts boards to 3-channel tensors, computes the Double DQN zero-sum Bellman target, applies Huber loss (`smooth_l1_loss`), clips gradients at norm 1.0, and steps the optimiser. Copies weights to the target net every `target_update_every` steps. Returns `None` if the buffer is smaller than `batch_size`. |
| `decay_epsilon()` | Multiplies ε by `epsilon_decay`, floored at `epsilon_min`. Called once per environment step. |
| `save(path)` | Saves both network state dicts, optimiser state, and all hyperparameters to a `.pt` file. Includes an `arch: "cnn_v1"` tag so `load` can reject incompatible checkpoints. |
| `load(path)` | Class method. Raises `ValueError` if the checkpoint was saved by an older architecture. Restores all weights and sets both nets to `eval()` mode. |

---

### `train.py`

Trains the agent via self-play and games against an alpha-beta oracle.

#### Alpha-beta oracle

Connect 4 has too many positions for a full minimax search, so the oracle uses **depth-limited alpha-beta** with a heuristic evaluation function.

| Function | Description |
|----------|-------------|
| `_score_window(window, player)` | Scores a 4-cell window. Returns large positive for windows the player dominates and large negative for windows the opponent dominates. Returns `0` if both players have pieces in the window (blocked). |
| `_heuristic(board, player)` | Sums scores across all 69 `WIN_LINES` and adds a small bonus for pieces in the centre column (column 3 controls the most win lines). |
| `_order(legal)` | Sorts legal columns by distance from the centre. Centre-first move ordering dramatically improves alpha-beta pruning efficiency. |
| `_alphabeta(board, depth, alpha, beta, maximizing, ai_player)` | Recursive alpha-beta search. Returns a score from `ai_player`'s perspective. Terminal wins return `100 + remaining_depth` (rewards faster wins). Terminal losses return `−100 − remaining_depth` (penalises faster losses, incentivising the agent to delay). |
| `get_alphabeta_action(board, ai_player, depth)` | Entry point for the oracle. Iterates over legal moves, calls `_alphabeta` for each, and returns the best column (random tie-break). |

#### Training episodes

| Function | Description |
|----------|-------------|
| `perspective(board, player)` | Returns `[v * player for v in board]` — flips piece ownership so the current player's pieces are always `+1`. Applied before passing any board to the agent. |
| `play_self_play_episode(agent, …)` | One game where the agent plays both sides. Non-terminal transitions are pushed with `next_state` in the opponent's perspective (for the zero-sum update). On a win, the winner gets `+1` and the loser's most recent transition is overwritten with `−1`. `learn()` and `decay_epsilon()` are called every `learn_every` steps. |
| `play_vs_alphabeta_episode(agent, ab_depth, …)` | One game where the agent is randomly assigned a colour and the other side is the alpha-beta oracle. Only the DQN player's transitions are pushed into the buffer; the oracle's moves provide structure to the environment without receiving gradient updates. |
| `train(games, save_path, resume, …)` | Main training loop. Mixes self-play and vs-alphabeta episodes according to `alphabeta_fraction`. Prints a stats line every `print_every` games and saves a checkpoint every `checkpoint_every` games. |

#### Key training flags (CLI)

| Flag | Default | Meaning |
|------|---------|---------|
| `--games` | `500000` | Total training episodes |
| `--no-resume` | — | Start from scratch, ignoring any existing `dqn_model.pt` |
| `--resume` | *(default)* | Load the existing checkpoint and continue |
| `--learn-every` | `4` | Run a gradient step every N environment steps. Higher values speed up the Python loop at the cost of fewer updates per game. `8` is a good balance on CPU. |
| `--print-every` | `1000` | Print a stats line every N games |
| `--ab-depth` | `3` | Alpha-beta search depth for the oracle opponent |
| `--alphabeta-fraction` | `0.3` | Fraction of episodes played against the oracle |
| `--epsilon` | `None` | Override the starting ε (e.g. `1.0` for a fresh run) |
| `--lr` | `5e-4` | Adam learning rate |
| `--batch-size` | `256` | Replay buffer sample size per gradient step |
| `--buffer-size` | `200000` | Maximum transitions stored in the replay buffer |
| `--target-update-every` | `1000` | Copy policy weights to target net every N gradient steps |

**Recommended fresh-training command:**

```bash
../.venv/bin/python -m backend.train \
  --no-resume \
  --games 500000 \
  --learn-every 8 \
  --print-every 1000 \
  --ab-depth 3
```

---

### `api.py`

Flask REST API that the React frontend calls.

The agent is loaded at startup with `epsilon = 0.0` (pure greedy, no exploration). If `dqn_model.pt` does not exist or has an incompatible architecture, the API automatically falls back to alpha-beta so the game is always playable. The `POST /reload` endpoint hot-swaps the in-memory agent from disk without restarting the server.

**Endpoints:**

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Returns `{"status": "ok", "model_loaded": bool}` |
| `POST` | `/reload` | Reloads `dqn_model.pt` from disk into the running server |
| `POST` | `/move` | Returns the AI's chosen move given the current board |

**`POST /move` request body:**

```json
{
  "board": [0, 0, 0, …],
  "player": -1,
  "difficulty": "hard"
}
```

- `board` — 42-element array, row-major. `0` empty, `1` Yellow/Human, `−1` Red/AI.
- `player` — which side the AI is playing (`−1` in the standard frontend setup).
- `difficulty` — controls the AI strategy:

| Difficulty | Strategy |
|------------|---------|
| `easy` | Random legal column |
| `medium` | 50 % DQN (if trained) / 50 % alpha-beta depth 3 |
| `hard` | DQN if `dqn_model.pt` exists, otherwise alpha-beta depth 6 |

**`POST /move` response:**

```json
{
  "position": 38,
  "winner": null,
  "is_draw": false
}
```

- `position` — board index (0–41) where the piece landed, computed by `make_move` + scanning for the changed cell.

After retraining, reload without restarting:

```bash
curl -X POST http://localhost:5001/reload
```

---

### `dqn_model.pt`

PyTorch checkpoint produced by `train.py`. Contains both network state dicts (`policy_state_dict`, `target_state_dict`), the optimiser state, all hyperparameters, total gradient steps taken, and the current epsilon. Not checked into version control — regenerate by running `train.py`.
