You are helping me build a Reinforcement Learning tic-tac-toe AI 
that humans can play against in a browser.

## Project Structure
Create the following structure:
tictactoe-ai/
├── backend/
│   ├── game.py
│   ├── agent.py
│   ├── train.py
│   ├── api.py
│   └── q_table.pkl        # generated after training
├── frontend/
│   ├── public/
│   │   └── index.html
│   ├── src/
│   │   ├── components/
│   │   │   ├── Board.jsx
│   │   │   ├── Cell.jsx
│   │   │   ├── StatusBar.jsx
│   │   │   ├── ScoreTracker.jsx
│   │   │   └── DifficultySelector.jsx
│   │   ├── hooks/
│   │   │   └── useGameState.js
│   │   ├── App.jsx
│   │   └── main.jsx
│   ├── package.json
│   └── tailwind.config.js
├── requirements.txt
└── README.md

## Tech Stack
- Python 3.11
- Flask + Flask-CORS for the API
- React 18 + Vite for the frontend
- Tailwind CSS for styling
- Framer Motion for animations
- Q-Learning with epsilon-greedy exploration
- Q-table stored as a pickle file

## Phase 1 - Build game.py
Create game.py with:
- Board as a list of 9 ints (0=empty, 1=X, -1=O)
- get_legal_moves(board)
- make_move(board, position, player)
- check_winner(board) → returns 1, -1, or 0
- is_draw(board)
- print_board(board) for debugging

## Phase 2 - Build agent.py
Create agent.py with:
- QLearningAgent class
- __init__(alpha=0.5, gamma=0.9, epsilon=1.0, 
           epsilon_decay=0.9999, epsilon_min=0.05)
- get_action(board) → epsilon-greedy move selection
- update(state, action, reward, next_state, done)
- save(path) and load(path) using pickle

## Phase 3 - Build train.py
Create train.py with:
- Train via self-play for 200,000 games
- Agent plays both X and O (flips board perspective each turn)
- Rewards: +1 win, -1 loss, +0.5 draw, 0 ongoing
- Print progress every 10,000 games with win/draw/loss rates
- Save q_table.pkl when done

## Phase 4 - Build api.py
Create api.py with:
- Flask app with CORS enabled for localhost:5173
- POST /move endpoint
  - receives { board: [...], player: -1, difficulty: "easy|medium|hard" }
  - easy = random legal move
  - medium = 50% Q-table, 50% random
  - hard = full Q-table (greedy)
  - returns { position: int, winner: int|null, is_draw: bool }
- GET /health endpoint
- Run on port 5000

## Phase 5 - Build the React frontend

### useGameState.js hook
- Manage board state, current player, winner, scores, difficulty
- Human is X (1) and always goes first
- AI is O (-1)
- handleCellClick(index) → validates move, updates board, calls API
- resetGame() → clears board, keeps scores
- All API calls to http://localhost:5000

### Cell.jsx
- Framer Motion animation on X and O appearing
- Highlight cell green if part of winning line
- Disabled state when cell is taken or game over

### Board.jsx
- 3x3 grid of Cell components
- Pass winning line indices down to cells

### StatusBar.jsx
- Shows: "Your turn", "AI is thinking...", "You win!", 
  "AI wins!", "It's a draw!"
- Smooth text transition using Framer Motion

### ScoreTracker.jsx
- Tracks Human wins, AI wins, Draws across games
- Clean pill-style score badges

### DifficultySelector.jsx
- Toggle between Easy / Medium / Hard
- Resets game on difficulty change

### App.jsx
- Dark themed layout (slate-900 background)
- Centered card with board, status, scores, difficulty
- Clean modern design with rounded corners and shadows
- Fully mobile responsive

## Instructions
- Build all files completely, no placeholders
- Use Tailwind utility classes only, no custom CSS files
- Use Framer Motion for all animations
- After creating all files, show me the exact terminal 
  commands to:
  1. Install Python requirements
  2. Run training
  3. Start Flask server
  4. Install frontend dependencies and start Vite dev server
- Do not ask clarifying questions, make sensible defaults