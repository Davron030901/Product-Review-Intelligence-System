import { CATEGORY_HEX, CATEGORY_MEANING } from '../lib/categories'
import { pct } from '../lib/format'
import type { IssueCategory } from '../api/types'

/** A category rendered as a physical label tag, notched like a parcel ticket. */
export function IssueTag({
  category,
  confidence,
  onClick,
  selected,
}: {
  category: IssueCategory
  confidence?: number
  onClick?: () => void
  selected?: boolean
}) {
  const hex = CATEGORY_HEX[category]
  const inner = (
    <span
      className="tag-notch inline-flex items-center gap-2 pl-3.5 pr-3 py-1.5 text-sm font-medium"
      style={{
        backgroundColor: `${hex}14`,
        color: hex,
        boxShadow: `inset 3px 0 0 ${hex}`,
      }}
    >
      {category}
      {confidence !== undefined && (
        <span className="font-mono text-[11px] tabular-nums opacity-70">{pct(confidence)}</span>
      )}
    </span>
  )

  if (!onClick) {
    return <span title={CATEGORY_MEANING[category]}>{inner}</span>
  }
  return (
    <button
      type="button"
      onClick={onClick}
      aria-pressed={selected}
      title={CATEGORY_MEANING[category]}
      className={`rounded-sm transition-opacity ${selected ? '' : 'opacity-45 hover:opacity-80'}`}
    >
      {inner}
    </button>
  )
}
