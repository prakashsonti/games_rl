export default function DifficultySelector({ difficulty, onChange }) {
  const opts = ['easy', 'medium', 'hard']
  return (
    <div className="inline-flex rounded-lg overflow-hidden border border-slate-700">
      {opts.map((d) => (
        <button
          key={d}
          onClick={() => onChange(d)}
          className={`px-3 py-1 text-sm capitalize ${
            difficulty === d ? 'bg-slate-700 text-slate-100' : 'bg-slate-800 text-slate-300'
          } hover:bg-slate-700 transition-colors`}
        >
          {d}
        </button>
      ))}
    </div>
  )
}
