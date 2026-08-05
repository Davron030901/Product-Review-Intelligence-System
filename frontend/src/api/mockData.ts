/**
 * Mock backend. Same contract, same abstention rules, no network.
 * Lets the UI be built and demoed before the model is trained -- and lets
 * empty/error/low-confidence states be exercised on demand.
 */
import type { AnalysisResult, AnalyzedReview, IssueCategory, SentimentLabel } from './types'

const LEXICON: Record<IssueCategory, string[]> = {
  delivery: ['late', 'arrived', 'shipping', 'delivery', 'courier', 'delayed', 'dispatch', 'tracking'],
  packaging: ['box', 'packaging', 'wrapped', 'crushed', 'torn', 'sealed', 'envelope'],
  quality: ['quality', 'fabric', 'material', 'cheap', 'flimsy', 'thin', 'sturdy', 'well made'],
  defect: ['broken', 'defect', 'faulty', 'crack', 'damaged', 'missing', 'stopped working', 'leak'],
  price: ['price', 'expensive', 'overpriced', 'cost', 'worth', 'refund', 'value'],
  service: ['customer service', 'support', 'rude', 'unhelpful', 'no reply', 'warranty', 'complaint'],
  fit: ['fit', 'size', 'sizing', 'too small', 'too big', 'tight', 'loose', 'true to size'],
  other: [],
}

const NEGATIVE = ['late', 'never', 'crushed', 'torn', 'cheap', 'flimsy', 'thin', 'broken', 'faulty',
  'crack', 'damaged', 'missing', 'expensive', 'overpriced', 'rude', 'unhelpful', 'awful', 'terrible',
  'disappointed', 'refund', 'returning', 'stopped working', 'too small', 'too tight', 'worst']
const POSITIVE = ['love', 'great', 'excellent', 'perfect', 'beautiful', 'happy', 'recommend',
  'sturdy', 'well made', 'true to size', 'fast', 'quick', 'worth', 'lovely', 'comfortable']

const wait = (ms: number) => new Promise((r) => setTimeout(r, ms))

export async function mockAnalyze(
  text: string,
  category?: string,
  latencyMs = 260 + Math.random() * 320,
): Promise<AnalysisResult> {
  if (latencyMs > 0) await wait(latencyMs)

  const clean = text.trim()
  const lower = clean.toLowerCase()
  const words = clean ? clean.split(/\s+/).length : 0
  const now = new Date().toISOString()
  const base = {
    input_category: category ?? null,
    word_count: words,
    truncated: clean.length > 20000,
    model_version: 'mock-v1',
    model_backend: 'mock',
    processed_at: now,
  }

  if (!clean) {
    return { ...base, sentiment: { label: 'unknown', confidence: 0 }, issues: [],
      low_confidence: true, reasons: ['Review text is empty.'] }
  }

  const issues = (Object.keys(LEXICON) as IssueCategory[])
    .filter((c) => LEXICON[c].some((k) => lower.includes(k)))
    .map((category) => ({
      category,
      confidence: Number((0.42 + Math.random() * 0.52).toFixed(3)),
    }))
    .sort((a, b) => b.confidence - a.confidence)

  const neg = NEGATIVE.filter((w) => lower.includes(w)).length
  const pos = POSITIVE.filter((w) => lower.includes(w)).length
  let label: SentimentLabel = 'neutral'
  let confidence = 0.44
  if (neg > pos) { label = 'negative'; confidence = Math.min(0.55 + neg * 0.12, 0.97) }
  else if (pos > neg) { label = 'positive'; confidence = Math.min(0.55 + pos * 0.12, 0.97) }
  else if (neg > 0) { label = 'neutral'; confidence = 0.51 }

  const reasons: string[] = []
  if (words < 3) reasons.push(`Review is only ${words} word(s) long.`)
  if (confidence < 0.55) reasons.push('Sentiment prediction is below the confidence threshold.')
  if (issues.length === 0) reasons.push('No issue category scored above the threshold.')

  return { ...base, sentiment: { label, confidence: Number(confidence.toFixed(3)) },
    issues, low_confidence: reasons.length > 0, reasons }
}

/** Seed history so the dashboard has something to show on first load. */
const SEED_TEXTS = [
  'Arrived two weeks late and the box was completely crushed. Returning it.',
  'Absolutely love this — true to size and the fabric feels lovely.',
  'The zipper broke after three days. Customer service never replied to my complaint.',
  'Great value for money, shipping was quick too.',
  'Overpriced for what it is. Material is thin and cheap.',
  'meh',
  'Packaging was beautiful but the item inside was damaged.',
  'Runs very small, had to return it. The return process was painless though.',
  'Perfect. Would recommend to anyone.',
  'Delivery took a month and support was unhelpful the whole time.',
  'Good quality, well made and sturdy. Happy with it.',
  'ok',
  'The colour is beautiful but it is too tight across the shoulders.',
  'Item never arrived and tracking stopped updating a week ago.',
  'Comfortable and worth the price. Fast delivery.',
]
const SEED_CATEGORIES = ['Tops', 'Dresses', 'Home', 'Electronics', 'Beauty']

export async function seedHistory(): Promise<AnalyzedReview[]> {
  // No artificial latency here: this is fixture data, not a simulated request.
  // Awaiting 44 delayed calls in sequence would stall the first paint for ~18s.
  const out = await Promise.all(
    Array.from({ length: 44 }, async (_, i) => {
      const text = SEED_TEXTS[i % SEED_TEXTS.length]
      const category = SEED_CATEGORIES[i % SEED_CATEGORIES.length]
      const result = await mockAnalyze(text, category, 0)
      const daysAgo = Math.floor(i / 3)
      const at = new Date(Date.now() - daysAgo * 864e5 - i * 37e5)
      return { ...result, id: `seed-${i}`, text, processed_at: at.toISOString() }
    }),
  )
  return out.sort((a, b) => +new Date(b.processed_at) - +new Date(a.processed_at))
}
