import { SENTIMENT_STYLE } from '../lib/categories'
import { pct } from '../lib/format'
import type { SentimentLabel } from '../api/types'

/** Colour + glyph + word. Never colour alone. */
export function SentimentBadge({
  label,
  confidence,
  size = 'md',
}: {
  label: SentimentLabel
  confidence?: number
  size?: 'sm' | 'md' | 'lg'
}) {
  const s = SENTIMENT_STYLE[label]
  const sizing =
    size === 'lg' ? 'text-base px-4 py-2' : size === 'sm' ? 'text-xs px-2 py-1' : 'text-sm px-3 py-1.5'
  return (
    <span className={`inline-flex items-center gap-2 border rounded-md font-medium ${s.className} ${sizing}`}>
      <span aria-hidden="true" className="font-mono leading-none">{s.glyph}</span>
      {s.label}
      {confidence !== undefined && confidence > 0 && (
        <span className="font-mono text-[11px] tabular-nums opacity-70">{pct(confidence)}</span>
      )}
    </span>
  )
}
