import Cell from './Cell.jsx'

const COLS = 7

export default function Board({ board, onColumnClick, winningLine, disabled, hoverCol, setHoverCol }) {
  const isWinning = (i) => winningLine ? winningLine.includes(i) : false

  return (
    <div>
      {/* Column drop indicators */}
      <div className="grid grid-cols-7 gap-0 mb-1">
        {Array.from({ length: COLS }, (_, c) => {
          const full = board[c] !== 0
          return (
            <button
              key={c}
              onClick={() => onColumnClick(c)}
              disabled={disabled || full}
              onMouseEnter={() => setHoverCol(c)}
              onMouseLeave={() => setHoverCol(null)}
              className="h-7 flex items-center justify-center text-yellow-400/60 hover:text-yellow-400 disabled:opacity-20 transition-colors focus:outline-none"
              aria-label={`Drop in column ${c + 1}`}
            >
              ▾
            </button>
          )
        })}
      </div>

      {/* 6×7 board */}
      <div
        className="grid grid-cols-7 gap-0 rounded-xl overflow-hidden bg-blue-900 p-1.5 shadow-2xl shadow-slate-900/70"
        onMouseLeave={() => setHoverCol(null)}
      >
        {board.map((v, i) => (
          <Cell
            key={i}
            value={v}
            isWinning={isWinning(i)}
            isHovered={hoverCol === i % COLS && v === 0}
            onClick={() => onColumnClick(i % COLS)}
            disabled={disabled}
          />
        ))}
      </div>
    </div>
  )
}
