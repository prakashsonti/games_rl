import { useCallback, useMemo, useState } from 'react'

const API_BASE = 'http://localhost:5000'

const WIN_LINES = [
  [0,1,2],[3,4,5],[6,7,8],
  [0,3,6],[1,4,7],[2,5,8],
  [0,4,8],[2,4,6]
]

export default function useGameState() {
  const [board, setBoard] = useState(Array(9).fill(0))
  const [currentPlayer, setCurrentPlayer] = useState(1) // human X first
  const [winner, setWinner] = useState(null)
  const [isDraw, setIsDraw] = useState(false)
  const [difficulty, setDifficulty] = useState('hard')
  const [scores, setScores] = useState({ human: 0, ai: 0, draw: 0 })
  const [aiThinking, setAiThinking] = useState(false)
  const [loserStarts, setLoserStarts] = useState(false)
  const [nextStarter, setNextStarter] = useState(1)

  const winningLine = useMemo(() => {
    for (const [a,b,c] of WIN_LINES) {
      const s = board[a] + board[b] + board[c]
      if (s === 3 || s === -3) return [a,b,c]
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
  
  // Define applyAIMove before resetGame to avoid TDZ in dependency arrays
  const applyAIMove = useCallback(async (b) => {
    setAiThinking(true)
    try {
      const res = await fetch(`${API_BASE}/move`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ board: b, player: -1, difficulty })
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
    const empty = Array(9).fill(0)
    setBoard(empty)
    setWinner(null)
    setIsDraw(false)
    setAiThinking(false)
    setCurrentPlayer(nextStarter)
    if (nextStarter === -1) {
      // AI starts immediately using the fresh empty board
      setTimeout(() => applyAIMove(empty), 0)
    }
  }, [nextStarter, applyAIMove])

  const handleCellClick = useCallback((index) => {
    if (winner || isDraw) return
    if (aiThinking) return
    if (board[index] !== 0) return

    if (currentPlayer !== 1) return

    const nb = board.slice()
    nb[index] = 1
    // Check if human wins immediately
    let humanWon = false
    for (const [a,b,c] of WIN_LINES) {
      const s = nb[a] + nb[b] + nb[c]
      if (s === 3) { humanWon = true; break }
    }
    if (humanWon) {
      setBoard(nb)
      setWinner(1)
      setScores(s => ({ ...s, human: s.human + 1 }))
      setNextStarter(loserStarts ? -1 : 1)
      return
    }

    // Check draw before AI moves
    if (nb.every(v => v !== 0)) {
      setBoard(nb)
      setIsDraw(true)
      setScores(s => ({ ...s, draw: s.draw + 1 }))
      return
    }

    setBoard(nb)
    setCurrentPlayer(-1)
    applyAIMove(nb)
  }, [board, currentPlayer, winner, isDraw, aiThinking, applyAIMove])

  const changeDifficulty = useCallback((d) => {
    setDifficulty(d)
    resetGame()
  }, [resetGame])

  const setLoserStartsFlag = useCallback((v) => {
    setLoserStarts(v)
    // Recompute next starter based on current last result
    // If there is a winner, loser starts next when enabled
    if (winner === 1) setNextStarter(v ? -1 : 1)
    else if (winner === -1) setNextStarter(1) // loser would be human -> 1
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
    handleCellClick,
    resetGame,
    changeDifficulty,
    loserStarts,
    setLoserStartsFlag,
  }
}
