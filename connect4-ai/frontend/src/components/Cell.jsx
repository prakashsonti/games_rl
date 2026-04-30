import { motion } from 'framer-motion'

export default function Cell({ value, onClick, disabled, isWinning, isHovered }) {
  const circleColor =
    value === 1  ? 'bg-yellow-400' :
    value === -1 ? 'bg-red-500' :
    isHovered    ? 'bg-yellow-400/25' :
                   'bg-slate-800/80'

  const ring = isWinning ? 'ring-2 ring-white ring-offset-1 ring-offset-blue-900' : ''

  return (
    <button
      onClick={onClick}
      disabled={disabled || value !== 0}
      className="flex items-center justify-center w-11 h-11 sm:w-12 sm:h-12 md:w-14 md:h-14 p-1 bg-blue-900 disabled:cursor-default focus:outline-none"
    >
      {value !== 0 ? (
        <motion.div
          initial={{ scale: 0, y: -16 }}
          animate={{ scale: 1, y: 0 }}
          transition={{ type: 'spring', stiffness: 300, damping: 20 }}
          className={`w-full h-full rounded-full ${circleColor} ${ring}`}
        />
      ) : (
        <div className={`w-full h-full rounded-full ${circleColor} ${ring} transition-colors`} />
      )}
    </button>
  )
}
