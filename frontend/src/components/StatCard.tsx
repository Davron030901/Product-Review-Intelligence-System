export function StatCard({
  label,
  value,
  detail,
  accent,
}: {
  label: string
  value: string | number
  detail?: string
  accent?: string
}) {
  return (
    <div className="card p-5 relative overflow-hidden">
      <div
        className="absolute left-0 top-0 bottom-0 w-1"
        style={{ backgroundColor: accent ?? '#23483C' }}
        aria-hidden="true"
      />
      <p className="eyebrow">{label}</p>
      <p className="font-display font-extrabold text-3xl sm:text-4xl mt-2 tabular-nums leading-none">
        {value}
      </p>
      {detail && <p className="text-sm text-slate mt-2">{detail}</p>}
    </div>
  )
}
