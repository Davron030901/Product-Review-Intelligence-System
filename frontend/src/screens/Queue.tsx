import { useMemo, useState } from 'react'
import type { AnalyzedReview, SentimentLabel } from '../api/types'
import { ConfidenceBar } from '../components/ConfidenceBar'
import { IssueTag } from '../components/IssueTag'
import { SentimentBadge } from '../components/SentimentBadge'
import { EmptyState } from '../components/States'
import { Stamp } from '../components/Stamp'
import { timeAgo } from '../lib/format'

const CHOICES: SentimentLabel[] = ['negative', 'neutral', 'positive']

/**
 * The working queue. Everything the model wasn't sure about lands here, with
 * the reason attached, so a person can confirm or correct it in one click.
 * These corrections are the raw material for the next training round.
 */
export function Queue({
  reviews,
  onResolve,
  onGoToAnalyzer,
}: {
  reviews: AnalyzedReview[]
  onResolve: (id: string, sentiment: SentimentLabel) => void
  onGoToAnalyzer: () => void
}) {
  const [showResolved, setShowResolved] = useState(false)

  const flagged = useMemo(() => reviews.filter((r) => r.low_confidence), [reviews])
  const pending = flagged.filter((r) => !r.resolution)
  const resolved = flagged.filter((r) => r.resolution)
  const list = showResolved ? resolved : pending

  return (
    <div>
      <header className="mb-6">
        <p className="eyebrow">Flagged by the model</p>
        <h1 className="font-display font-extrabold text-3xl sm:text-4xl tracking-tight mt-1.5">
          Review queue
        </h1>
        <p className="text-slate mt-2.5 max-w-2xl">
          The model says "I'm not sure" rather than guessing. Each one below tells you why. Confirm
          it or correct it — your answer is what the next training round learns from.
        </p>
      </header>

      <div role="tablist" aria-label="Queue filter" className="flex gap-2 mb-6">
        <Tab active={!showResolved} onClick={() => setShowResolved(false)}
          label="Waiting" count={pending.length} />
        <Tab active={showResolved} onClick={() => setShowResolved(true)}
          label="Checked" count={resolved.length} />
      </div>

      {list.length === 0 ? (
        <div className="card">
          {showResolved ? (
            <EmptyState title="Nothing checked yet"
              body="Confirm or correct a flagged review and it moves here." />
          ) : flagged.length === 0 ? (
            <EmptyState
              title="Queue is clear"
              body="Every result so far cleared the confidence threshold. Anything the model is unsure about will appear here automatically."
              action={<button type="button" onClick={onGoToAnalyzer} className="btn-primary">Analyze a review</button>}
            />
          ) : (
            <EmptyState title="All caught up"
              body="Every flagged review has been checked. Nice work." />
          )}
        </div>
      ) : (
        <ul className="space-y-4">
          {list.map((r) => (
            <li key={r.id}>
              <QueueCard review={r} onResolve={onResolve} />
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}

function Tab({ active, onClick, label, count }: {
  active: boolean; onClick: () => void; label: string; count: number
}) {
  return (
    <button type="button" role="tab" aria-selected={active} onClick={onClick}
      className={`min-h-[44px] px-4 rounded-md border text-sm font-medium transition-colors
        ${active ? 'bg-pine text-paper border-pine' : 'border-line text-slate hover:bg-ink/5'}`}>
      {label}
      <span className={`ml-2 font-mono text-xs tabular-nums ${active ? 'opacity-75' : 'opacity-60'}`}>
        {count}
      </span>
    </button>
  )
}

function QueueCard({
  review, onResolve,
}: {
  review: AnalyzedReview
  onResolve: (id: string, sentiment: SentimentLabel) => void
}) {
  const done = !!review.resolution

  return (
    <article className={`card p-5 ${done ? 'opacity-70' : ''}`}>
      <div className="flex items-start justify-between gap-4 flex-wrap-reverse">
        <p className="text-[15px] leading-relaxed flex-1 min-w-[200px]">{review.text}</p>
        {!done && <Stamp small />}
      </div>

      <div className="flex flex-wrap items-center gap-2 mt-4">
        <SentimentBadge label={review.sentiment.label} confidence={review.sentiment.confidence} size="sm" />
        {review.issues.map((i) => <IssueTag key={i.category} category={i.category} confidence={i.confidence} />)}
        <span className="font-mono text-[11px] text-slate ml-auto">{timeAgo(review.processed_at)}</span>
      </div>

      <div className="mt-4">
        <ConfidenceBar value={review.sentiment.confidence} label="Model confidence" />
      </div>

      {review.reasons.length > 0 && (
        <ul className="mt-4 space-y-1">
          {review.reasons.map((reason, i) => (
            <li key={i} className="text-sm text-slate flex gap-2">
              <span aria-hidden="true" className="text-signal">—</span>{reason}
            </li>
          ))}
        </ul>
      )}

      <div className="mt-5 pt-4 border-t border-line">
        {done ? (
          <p className="text-sm text-slate">
            Marked{' '}
            <span className="font-medium text-ink">{review.resolution!.sentiment}</span>{' '}
            by a person {timeAgo(review.resolution!.at)}.
          </p>
        ) : (
          <>
            <p className="eyebrow mb-2.5">What is it really?</p>
            <div className="flex flex-wrap gap-2">
              {CHOICES.map((s) => (
                <button key={s} type="button" onClick={() => onResolve(review.id, s)}
                  className="btn-ghost !min-h-[44px] text-sm capitalize">
                  {s === review.sentiment.label ? `Confirm ${s}` : s}
                </button>
              ))}
            </div>
          </>
        )}
      </div>
    </article>
  )
}
