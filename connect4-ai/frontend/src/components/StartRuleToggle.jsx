export default function StartRuleToggle({ enabled, onChange }) {
  return (
    <label className="flex items-center gap-2 text-xs sm:text-sm text-slate-300 select-none">
      <input
        type="checkbox"
        checked={enabled}
        onChange={(e) => onChange(e.target.checked)}
        className="accent-yellow-400"
      />
      <span>Loser starts next game</span>
    </label>
  )
}
