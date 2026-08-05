import { useState } from 'react'
import { api } from '../api/client'
import { ApiError } from '../api/types'
import type { AnalysisResult, AnalyzedReview } from '../api/types'
import { ConfidenceBar } from '../components/ConfidenceBar'
import { IssueTag } from '../components/IssueTag'
import { SentimentBadge } from '../components/SentimentBadge'
import { Spinner, ErrorState } from '../components/States'
import { Stamp } from '../components/Stamp'
import { CATEGORY_MEANING } from '../lib/categories'

const SAMPLES = [
  'Arrived two weeks late and the box was completely crushed. Returning it.',
  'Love it — true to size and the fabric feels lovely. Shipping was quick too.',
  'meh',
]

export function Analyzer({ onAnalyzed }: { onAnalyzed: (r: AnalyzedReview) => void }) {
  const [text, setText] = useState('')
  const [category, setCategory] = useState('')
  const [result, setResult] = useState<AnalysisResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function analyze() {
    setLoading(true)
    setError(null)
    try {
      const res = await api.analyze(text, category || undefined)
      setResult(res)
      onAnalyzed({ ...res, id: crypto.randomUUID(), text })
    } catch (e) {
      setError(e instanceof ApiError ? e.message : 'Something went wrong analyzing this review.')
      setResult(null)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div>
      <header className="mb-7">
        <p className="eyebrow">One review</p>
        <h1 className="font-display font-extrabold text-3xl sm:text-4xl tracking-tight mt-1.5">
          What is this customer actually telling you?
        </h1>
        <p className="text-slate mt-2.5 max-w-2xl">
          Paste a review. You get the sentiment, the issues it raises, and — when the model isn't
          sure — a clear flag instead of a confident guess.
        </p>
      </header>

      <div className="grid lg:grid-cols-2 gap-6 items-start">
        {/* Input */}
        <section className="card p-5 sm:p-6">
          <label htmlFor="review-text" className="eyebrow block mb-2">
            Review text
          </label>
          <textarea
            id="review-text"
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={(e) => {
              if ((e.metaKey || e.ctrlKey) && e.key === 'Enter' && text.trim()) analyze()
            }}
            rows={7}
            placeholder="Paste what the customer wrote…"
            className="w-full rounded-md border border-line bg-paper px-3.5 py-3 text-[15px]
                       leading-relaxed resize-y placeholder:text-slate/60"
          />

          <div className="flex flex-col sm:flex-row gap-3 mt-4">
            <div className="flex-1">
              <label htmlFor="category" className="eyebrow block mb-2">
                Product category <span className="normal-case tracking-normal">(optional)</span>
              </label>
              <input
                id="category"
                value={category}
                onChange={(e) => setCategory(e.target.value)}
                placeholder="e.g. Dresses"
                className="w-full min-h-[44px] rounded-md border border-line bg-paper px-3.5
                           text-sm placeholder:text-slate/60"
              />
            </div>
            <button
              type="button"
              onClick={analyze}
              disabled={loading || !text.trim()}
              className="btn-primary sm:self-end sm:w-auto w-full"
            >
              {loading && <Spinner className="w-4 h-4" />}
              {loading ? 'Analyzing' : 'Analyze review'}
            </button>
          </div>

          <div className="mt-5 pt-4 border-t border-line">
            <p className="eyebrow mb-2.5">Or try one</p>
            <div className="flex flex-wrap gap-2">
              {SAMPLES.map((s, i) => (
                <button
                  key={i}
                  type="button"
                  onClick={() => setText(s)}
                  className="text-xs text-slate border border-line rounded-full px-3 py-2
                             hover:bg-ink/5 hover:text-ink transition-colors text-left max-w-full"
                >
                  <span className="line-clamp-1">{s}</span>
                </button>
              ))}
            </div>
          </div>
        </section>

        {/* Result */}
        <section aria-live="polite" className="lg:sticky lg:top-8">
          {error && <ErrorState message={error} onRetry={analyze} />}

          {!error && !result && !loading && (
            <div className="card p-8 border-dashed text-center">
              <p className="font-display font-bold text-lg">Nothing analyzed yet</p>
              <p className="text-slate text-sm mt-2 max-w-xs mx-auto">
                The result appears here: sentiment, issue tags, and the confidence behind each one.
              </p>
            </div>
          )}

          {loading && !result && (
            <div className="card p-8 flex items-center justify-center gap-3 text-slate">
              <Spinner />
              <span className="text-sm">Reading the review…</span>
            </div>
          )}

          {result && !error && <ResultCard result={result} />}
        </section>
      </div>
    </div>
  )
}

function ResultCard({ result }: { result: AnalysisResult }) {
  return (
    <article className="card p-5 sm:p-6 animate-rise">
      <div className="flex items-start justify-between gap-4 flex-wrap">
        <div>
          <p className="eyebrow">Result</p>
          <div className="mt-2.5">
            <SentimentBadge label={result.sentiment.label} confidence={result.sentiment.confidence} size="lg" />
          </div>
        </div>
        {result.low_confidence && <Stamp />}
      </div>

      <div className="mt-6">
        <ConfidenceBar value={result.sentiment.confidence} label="Sentiment confidence" />
      </div>

      <div className="mt-6">
        <p className="eyebrow mb-3">
          Issues raised {result.issues.length > 0 && `(${result.issues.length})`}
        </p>
        {result.issues.length > 0 ? (
          <>
            <div className="flex flex-wrap gap-2">
              {result.issues.map((i) => (
                <IssueTag key={i.category} category={i.category} confidence={i.confidence} />
              ))}
            </div>
            <dl className="mt-4 space-y-1.5">
              {result.issues.map((i) => (
                <div key={i.category} className="flex gap-2 text-xs text-slate">
                  <dt className="font-medium text-ink w-20 shrink-0">{i.category}</dt>
                  <dd>{CATEGORY_MEANING[i.category]}</dd>
                </div>
              ))}
            </dl>
          </>
        ) : (
          <p className="text-sm text-slate">
            No issue category cleared the threshold. Either this review raises nothing specific, or it
            says something the taxonomy doesn't cover yet.
          </p>
        )}
      </div>

      {result.low_confidence && result.reasons.length > 0 && (
        <div className="mt-6 border border-signal/40 bg-signal-wash/50 rounded-md p-4">
          <p className="font-medium text-sm text-signal-ink">Why this was flagged</p>
          <ul className="mt-2 space-y-1.5">
            {result.reasons.map((r, i) => (
              <li key={i} className="text-sm text-ink/80 flex gap-2">
                <span aria-hidden="true" className="text-signal">—</span>
                {r}
              </li>
            ))}
          </ul>
          <p className="text-xs text-slate mt-3">Sent to the review queue for a person to check.</p>
        </div>
      )}

      <footer className="mt-6 pt-4 border-t border-line flex flex-wrap gap-x-5 gap-y-1.5
                         font-mono text-[11px] text-slate">
        <span>{result.word_count} words</span>
        <span>model {result.model_version}</span>
        {result.truncated && <span className="text-signal-ink">input truncated</span>}
      </footer>
    </article>
  )
}
