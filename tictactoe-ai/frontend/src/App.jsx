import Board from './components/Board.jsx'
import StatusBar from './components/StatusBar.jsx'
import ScoreTracker from './components/ScoreTracker.jsx'
import DifficultySelector from './components/DifficultySelector.jsx'
import StartRuleToggle from './components/StartRuleToggle.jsx'
import useGameState from './hooks/useGameState.js'
import { AnimatePresence, motion } from 'framer-motion'

export default function App() {
  const {
    board,
    status,
    winningLine,
    aiThinking,
    scores,
    difficulty,
    handleCellClick,
    resetGame,
    changeDifficulty,
    loserStarts,
    setLoserStartsFlag,
  } = useGameState()

  return (
    <div className="min-h-screen flex items-center justify-center px-4">
      <div className="max-w-md w-full bg-slate-800/60 backdrop-blur rounded-2xl p-4 sm:p-6 shadow-2xl border border-slate-700">
        <div className="flex items-center justify-between mb-4">
          <h1 className="text-xl font-bold text-slate-100">Tic-Tac-Toe RL</h1>
          <DifficultySelector difficulty={difficulty} onChange={changeDifficulty} />
        </div>

        <div className="flex items-center justify-end mb-2">
          <StartRuleToggle enabled={loserStarts} onChange={setLoserStartsFlag} />
        </div>

        <StatusBar status={status} />

        <div className="flex items-center justify-center my-4">
          <Board
            board={board}
            onCellClick={handleCellClick}
            winningLine={winningLine}
            disabled={aiThinking}
          />
        </div>

        <div className="flex items-center justify-between mt-4 gap-2">
          <ScoreTracker scores={scores} />
          <div className="flex items-center gap-2">
            <AnimatePresence initial={false}>
              {(winningLine || status.includes('win') || status.includes('draw')) && (
                <motion.button
                  key="play-again"
                  initial={{ opacity: 0, y: 8, scale: 0.95 }}
                  animate={{ opacity: 1, y: 0, scale: 1 }}
                  exit={{ opacity: 0, y: -8, scale: 0.95 }}
                  transition={{ duration: 0.2 }}
                  onClick={resetGame}
                  className="px-3 py-1 text-sm rounded-lg bg-emerald-600 hover:bg-emerald-500 transition-colors border border-emerald-500 text-white shadow"
                  title="Start the next game"
                >
                  Play again
                </motion.button>
              )}
            </AnimatePresence>
            <button onClick={resetGame} className="px-3 py-1 text-sm rounded-lg bg-slate-700 hover:bg-slate-600 transition-colors border border-slate-600">Reset</button>
          </div>
        </div>
      </div>
    </div>
  )
}
