import type { IssueCategory, SentimentLabel } from '../api/types'

export const ISSUE_ORDER: IssueCategory[] = [
  'delivery', 'packaging', 'quality', 'defect', 'price', 'service', 'fit', 'other',
]

/** One hue per category, matching tailwind.config.js. Recharts needs the hex. */
export const CATEGORY_HEX: Record<IssueCategory, string> = {
  delivery: '#2F6F8F',
  packaging: '#7A5C9E',
  quality: '#2E7D5B',
  defect: '#B03A3A',
  price: '#8A6414',
  service: '#B04A7D',
  fit: '#5B7A2E',
  other: '#5F655D',
}

/** Plain-language description shown on hover, so the taxonomy is self-teaching. */
export const CATEGORY_MEANING: Record<IssueCategory, string> = {
  delivery: 'Shipping speed, courier handling, tracking',
  packaging: 'The box, wrapping, and how it arrived',
  quality: 'Materials and construction as designed',
  defect: 'A single unit that arrived broken or incomplete',
  price: 'Cost, value for money, refunds',
  service: 'Support responsiveness, returns, warranty',
  fit: 'Sizing and dimensions',
  other: 'A complaint that matches no other category',
}

/**
 * Sentiment never relies on colour alone -- each carries a glyph and a word,
 * so the state survives greyscale and colour-vision differences.
 */
export const SENTIMENT_STYLE: Record<
  SentimentLabel,
  { glyph: string; label: string; className: string }
> = {
  negative: { glyph: '▼', label: 'Negative', className: 'bg-cat-defect/10 text-cat-defect border-cat-defect/35' },
  neutral: { glyph: '■', label: 'Neutral', className: 'bg-slate/10 text-slate border-slate/35' },
  positive: { glyph: '▲', label: 'Positive', className: 'bg-cat-quality/10 text-cat-quality border-cat-quality/35' },
  unknown: { glyph: '?', label: 'Unknown', className: 'bg-ink/5 text-ink/60 border-ink/20' },
}
