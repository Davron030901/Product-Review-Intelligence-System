import { pct } from '../lib/format'

/** A confidence reading with the decision threshold drawn on it, so the number
 *  is legible as a judgement rather than as a score. */
export function ConfidenceBar({
  value,
  threshold = 0.55,
  color = '#23483C',
  label,
}: {
  value: number
  threshold?: number
  color?: string
  label?: string
}) {
  const below = value < threshold
  return (
    <div>
      {label && (
        <div className="flex justify-between items-baseline mb-1.5">
          <span className="eyebrow">{label}</span>
          <span className="font-mono text-sm tabular-nums text-ink">{pct(value)}</span>
        </div>
      )}
      <div
        className="relative h-2 rounded-full bg-ink/8 overflow-hidden"
        role="meter"
        aria-valuenow={Math.round(value * 100)}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-label={label ?? 'Confidence'}
      >
        <div
          className="h-full rounded-full transition-[width] duration-500"
          style={{ width: `${Math.max(value * 100, 2)}%`, backgroundColor: below ? '#B87611' : color }}
        />
        <div
          className="absolute top-0 bottom-0 w-px bg-ink/35"
          style={{ left: `${threshold * 100}%` }}
          aria-hidden="true"
        />
      </div>
      {below && (
        <p className="mt-1.5 text-xs text-signal-ink">Below the {pct(threshold)} threshold.</p>
      )}
    </div>
  )
}
