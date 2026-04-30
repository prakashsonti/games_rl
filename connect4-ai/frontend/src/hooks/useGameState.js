import { useCallback, useMemo, useState } from 'react'

const API_BASE = 'http://localhost:5001'
const ROWS = 6
const COLS = 7

const WIN_LINES = (() => {
  const lines = []
  for (let r = 0; r < ROWS; r++)
    for (let c = 0; c <= COLS - 4; c++)
      lines.push([r*COLS+c, r*COLS+c+1, r*COLS+c+2, r*COLS+c+3])
  for (let c = 0; c < COLS; c++)
    for (let r = 0; r <= ROWS - 4; r++)
      lines.push([r*COLS+c, (r+1)*COLS+c, (r+2)*COLS+c, (r+3)*COLS+c])
  for (let r = 0; r <= ROWS - 4; r++)
    for (let c = 0; c <= COLS - 4; c++)
      lines.push([r*COLS+c, (r+1)*COLS+c+1, (r+2)*COLS+c+2, (r+3)*COLS+c+3])
  for (let r = 0; r <= ROWS - 4; r++)
    for (let c = 3; c < COLS; c++)
      lines.push([r*COLS+c, (r+1)*COLS+c-1, (r+2)*COLS+c-2, (r+3)*COLS+c-3])
  return lines
})()

function dropRow(board, col) {
  for (let r = ROWS - 1; r >= 0; r--) {
    if (board[r * COLS + col] === 0) return r
  }
  return -1
}

export default function useGameState() {
  const [board, setBoard] = useState(Array(ROWS * COLS).fill(0))
  const [currentPlayer, setCurrentPlayer] = useState(1)
  const [winner, setWinner] = useState(null)
  const [isDraw, setIsDraw] = useState(false)
  const [difficulty, setDifficulty] = useState('hard')
  const [scores, setScores] = useState({ human: 0, ai: 0, draw: 0 })
  const [aiThinking, setAiThinking] = useState(false)
  const [loserStarts, setLoserStarts] = useState(false)
  const [nextStarter, setNextStarter] = useState(1)

  const winningLine = useMemo(() => {
    for (const line of WIN_LINES) {
      const s = line.reduce((acc, i) => acc + board[i], 0)
      if (s === 4 || s === -4) return line
    }
    return null
  }, [board])

  const status = useMemo(() => {
    if (winner === 1) return 'You win!'
    if (winner === -1) return 'AI wins!'
    if (isDraw) return "It's a draw!"
    if (aiThinking) return 'AI is thinking...'
    return 'Your turn'
  }, [winner, isDraw, aiThinking])

  const applyAIMove = useCallback(async (b) => {
    setAiThinking(true)
    try {
      const res = await fetch(`${API_BASE}/move`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ board: b, player: -1, difficulty }),
      })
      const data = await res.json()
      if (data && typeof data.position === 'number') {
        const nb = b.slice()
        nb[data.position] = -1
        setBoard(nb)
        if (data.winner === -1) {
          setWinner(-1)
          setScores(s => ({ ...s, ai: s.ai + 1 }))
          setNextStarter(1)
        } else if (data.is_draw) {
          setIsDraw(true)
          setScores(s => ({ ...s, draw: s.draw + 1 }))
        } else {
          setCurrentPlayer(1)
        }
      }
    } finally {
      setAiThinking(false)
    }
  }, [difficulty])

  const resetGame = useCallback(() => {
    const empty = Array(ROWS * COLS).fill(0)
    setBoard(empty)
    setWinner(null)
    setIsDraw(false)
    setAiThinking(false)
    setCurrentPlayer(nextStarter)
    if (nextStarter === -1) {
      setTimeout(() => applyAIMove(empty), 0)
    }
  }, [nextStarter, applyAIMove])

  const handleColumnClick = useCallback((col) => {
    if (winner || isDraw) return
    if (aiThinking) return
    if (currentPlayer !== 1) return

    const row = dropRow(board, col)
    if (row === -1) return

    const pos = row * COLS + col
    const nb = board.slice()
    nb[pos] = 1

    let humanWon = false
    for (const line of WIN_LINES) {
      if (line.reduce((s, i) => s + nb[i], 0) === 4) { humanWon = true; break }
    }
    if (humanWon) {
      setBoard(nb)
      setWinner(1)
      setScores(s => ({ ...s, human: s.human + 1 }))
      setNextStarter(loserStarts ? -1 : 1)
      return
    }

    if (nb.every(v => v !== 0)) {
      setBoard(nb)
      setIsDraw(true)
      setScores(s => ({ ...s, draw: s.draw + 1 }))
      return
    }

    setBoard(nb)
    setCurrentPlayer(-1)
    applyAIMove(nb)
  }, [board, currentPlayer, winner, isDraw, aiThinking, loserStarts, applyAIMove])

  const changeDifficulty = useCallback((d) => {
    setDifficulty(d)
    resetGame()
  }, [resetGame])

  const setLoserStartsFlag = useCallback((v) => {
    setLoserStarts(v)
    if (winner === 1) setNextStarter(v ? -1 : 1)
    else if (winner === -1) setNextStarter(1)
    else setNextStarter(1)
    resetGame()
  }, [winner, resetGame])

  return {
    board,
    currentPlayer,
    winner,
    isDraw,
    difficulty,
    scores,
    aiThinking,
    winningLine,
    status,
    handleColumnClick,
    resetGame,
    changeDifficulty,
    loserStarts,
    setLoserStartsFlag,
  }
}
