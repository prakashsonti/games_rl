import { motion } from 'framer-motion'

export default function Cell({ value, onClick, disabled, isWinning }) {
  const content = value === 1 ? 'X' : value === -1 ? 'O' : ''
  const color = value === 1 ? 'text-emerald-400' : value === -1 ? 'text-rose-400' : ''
  const bg = isWinning ? 'bg-emerald-900/30' : ''

  return (
    <button
      disabled={disabled}
      onClick={onClick}
      className={`relative flex items-center justify-center w-24 h-24 sm:w-28 sm:h-28 md:w-32 md:h-32 border border-slate-700 ${bg} disabled:opacity-50`}
    >
      {content && (
        <motion.span
          initial={{ scale: 0, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ type: 'spring', stiffness: 260, damping: 12 }}
          className={`text-4xl sm:text-5xl md:text-6xl font-bold ${color}`}
        >
          {content}
        </motion.span>
      )}
    </button>
  )
}
