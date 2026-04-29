export default function ScoreTracker({ scores }) {
  const Pill = ({ label, value, color }) => (
    <div className={`px-3 py-1 rounded-full text-sm font-medium ${color} bg-slate-800 border border-slate-700`}>
      {label}: <span className="text-slate-100 ml-1">{value}</span>
    </div>
  )
  return (
    <div className="flex gap-2 flex-wrap items-center justify-center">
      <Pill label="You" value={scores.human} color="text-yellow-300" />
      <Pill label="AI" value={scores.ai} color="text-red-400" />
      <Pill label="Draws" value={scores.draw} color="text-slate-300" />
    </div>
  )
}
