import { motion, AnimatePresence } from 'framer-motion'

export default function StatusBar({ status }) {
  return (
    <div className="h-8 flex items-center justify-center">
      <AnimatePresence mode="wait">
        <motion.div
          key={status}
          initial={{ opacity: 0, y: 6 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -6 }}
          transition={{ duration: 0.2 }}
          className="text-sm sm:text-base text-slate-300"
        >
          {status}
        </motion.div>
      </AnimatePresence>
    </div>
  )
}
