import Cell from './Cell.jsx'

export default function Board({ board, onCellClick, winningLine, disabled }) {
  const isWin = (i) => winningLine ? winningLine.includes(i) : false
  return (
    <div className="grid grid-cols-3 gap-0 rounded-lg overflow-hidden shadow-lg shadow-slate-900/50">
      {board.map((v, i) => (
        <Cell
          key={i}
          value={v}
          isWinning={isWin(i)}
          disabled={disabled || v !== 0}
          onClick={() => onCellClick(i)}
        />
      ))}
    </div>
  )
}
