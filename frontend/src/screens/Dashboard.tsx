import { useMemo, useState } from 'react'
import {
  Bar, BarChart, CartesianGrid, Cell, Legend, Line, LineChart, ResponsiveContainer,
  Tooltip, XAxis, YAxis,
} from 'recharts'
import type { AnalyzedReview, IssueCategory } from '../api/types'
import { IssueTag } from '../components/IssueTag'
import { SentimentBadge } from '../components/SentimentBadge'
import { StatCard } from '../components/StatCard'
import { EmptyState } from '../components/States'
import { Stamp } from '../components/Stamp'
import { CATEGORY_HEX, ISSUE_ORDER } from '../lib/categories'
import { dayKey, pct, shortDate, timeAgo, truncate } from '../lib/format'

type SentimentFilter = 'all' | 'negative' | 'neutral' | 'positive'
type RangeFilter = 7 | 30 | 0 // 0 = all time

export function Dashboard({
  reviews,
  onGoToAnalyzer,
}: {
  reviews: AnalyzedReview[]
  onGoToAnalyzer: () => void
}) {
  const [sentiment, setSentiment] = useState<SentimentFilter>('all')
  const [range, setRange] = useState<RangeFilter>(30)
  const [issues, setIssues] = useState<IssueCategory[]>([])
  const [category, setCategory] = useState('all')

  const categories = useMemo(
    () => ['all', ...Array.from(new Set(reviews.map((r) => r.input_category).filter(Boolean) as string[]))],
    [reviews],
  )

  const filtered = useMemo(() => {
    const cutoff = range === 0 ? 0 : Date.now() - range * 864e5
    return reviews.filter((r) => {
      if (+new Date(r.processed_at) < cutoff) return false
      if (sentiment !== 'all' && r.sentiment.label !== sentiment) return false
      if (category !== 'all' && r.input_category !== category) return false
      if (issues.length && !issues.some((i) => r.issues.some((x) => x.category === i))) return false
      return true
    })
  }, [reviews, sentiment, range, issues, category])

  const stats = useMemo(() => {
    const total = filtered.length
    const neg = filtered.filter((r) => r.sentiment.label === 'negative').length
    const flagged = filtered.filter((r) => r.low_confidence).length
    const counts = new Map<IssueCategory, number>()
    filtered.forEach((r) => r.issues.forEach((i) => counts.set(i.category, (counts.get(i.category) ?? 0) + 1)))
    const top = [...counts.entries()].sort((a, b) => b[1] - a[1])[0]
    return {
      total,
      negShare: total ? neg / total : 0,
      flagged,
      top: top ? { category: top[0], count: top[1] } : null,
    }
  }, [filtered])

  const issueData = useMemo(() => {
    const counts = new Map<IssueCategory, number>()
    filtered.forEach((r) => r.issues.forEach((i) => counts.set(i.category, (counts.get(i.category) ?? 0) + 1)))
    return ISSUE_ORDER.map((c) => ({ category: c, count: counts.get(c) ?? 0 }))
      .filter((d) => d.count > 0)
      .sort((a, b) => b.count - a.count)
  }, [filtered])

  const trendData = useMemo(() => {
    const byDay = new Map<string, { negative: number; neutral: number; positive: number }>()
    filtered.forEach((r) => {
      const k = dayKey(r.processed_at)
      const row = byDay.get(k) ?? { negative: 0, neutral: 0, positive: 0 }
      if (r.sentiment.label !== 'unknown') row[r.sentiment.label] += 1
      byDay.set(k, row)
    })
    return [...byDay.entries()]
      .sort((a, b) => a[0].localeCompare(b[0]))
      .map(([day, v]) => ({ day: shortDate(day), ...v }))
  }, [filtered])

  if (reviews.length === 0) {
    return (
      <div className="card">
        <EmptyState
          title="No reviews analyzed yet"
          body="Analyze a review and this page fills in: what customers are complaining about, how it's trending, and which results need a second look."
          action={
            <button type="button" onClick={onGoToAnalyzer} className="btn-primary">
              Analyze your first review
            </button>
          }
        />
      </div>
    )
  }

  const toggleIssue = (c: IssueCategory) =>
    setIssues((prev) => (prev.includes(c) ? prev.filter((x) => x !== c) : [...prev, c]))

  return (
    <div>
      <header className="mb-7">
        <p className="eyebrow">All reviews</p>
        <h1 className="font-display font-extrabold text-3xl sm:text-4xl tracking-tight mt-1.5">
          What keeps coming up
        </h1>
      </header>

      {/* Filters */}
      <section aria-label="Filters" className="card p-4 mb-6">
        <div className="flex flex-wrap gap-4">
          <Select label="Period" value={String(range)} onChange={(v) => setRange(Number(v) as RangeFilter)}
            options={[['7', 'Last 7 days'], ['30', 'Last 30 days'], ['0', 'All time']]} />
          <Select label="Sentiment" value={sentiment} onChange={(v) => setSentiment(v as SentimentFilter)}
            options={[['all', 'All'], ['negative', 'Negative'], ['neutral', 'Neutral'], ['positive', 'Positive']]} />
          <Select label="Category" value={category} onChange={setCategory}
            options={categories.map((c) => [c, c === 'all' ? 'All categories' : c])} />
        </div>
        <div className="mt-4 pt-4 border-t border-line">
          <p className="eyebrow mb-2.5">
            Issue {issues.length > 0 && <span className="text-pine normal-case">· {issues.length} selected</span>}
          </p>
          <div className="flex flex-wrap gap-2">
            {ISSUE_ORDER.map((c) => (
              <IssueTag key={c} category={c} onClick={() => toggleIssue(c)}
                selected={issues.length === 0 || issues.includes(c)} />
            ))}
            {issues.length > 0 && (
              <button type="button" onClick={() => setIssues([])}
                className="text-xs text-slate underline underline-offset-4 px-2 min-h-[44px]">
                Clear
              </button>
            )}
          </div>
        </div>
      </section>

      {/* KPIs */}
      <section className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
        <StatCard label="Reviews analyzed" value={stats.total} />
        <StatCard label="Negative" value={pct(stats.negShare)} accent="#B03A3A"
          detail={`${Math.round(stats.negShare * stats.total)} of ${stats.total}`} />
        <StatCard label="Top issue" value={stats.top?.category ?? '—'} accent={stats.top ? CATEGORY_HEX[stats.top.category] : undefined}
          detail={stats.top ? `mentioned ${stats.top.count} times` : 'nothing recurring yet'} />
        <StatCard label="Needs review" value={stats.flagged} accent="#B87611"
          detail={stats.flagged ? 'the model was unsure' : 'all results were confident'} />
      </section>

      {filtered.length === 0 ? (
        <div className="card">
          <EmptyState title="No reviews match these filters"
            body="Widen the period or clear the issue selection to see results again." />
        </div>
      ) : (
        <>
          <section className="grid lg:grid-cols-2 gap-6 mb-6">
            <div className="card p-5">
              <h2 className="font-display font-bold text-lg">Issues by volume</h2>
              <p className="text-sm text-slate mt-1 mb-4">How often each category is mentioned.</p>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={issueData} layout="vertical" margin={{ left: 8, right: 16 }}>
                    <CartesianGrid horizontal={false} stroke="#D2D7CF" />
                    <XAxis type="number" tick={{ fontSize: 11, fill: '#4A5A52' }} allowDecimals={false} />
                    <YAxis type="category" dataKey="category" width={78}
                      tick={{ fontSize: 12, fill: '#161A17' }} />
                    <Tooltip cursor={{ fill: 'rgba(22,26,23,0.05)' }} contentStyle={TOOLTIP} />
                    <Bar dataKey="count" name="Mentions" radius={[0, 3, 3, 0]}>
                      {issueData.map((d) => (
                        <Cell key={d.category} fill={CATEGORY_HEX[d.category]} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            <div className="card p-5">
              <h2 className="font-display font-bold text-lg">Sentiment over time</h2>
              <p className="text-sm text-slate mt-1 mb-4">Daily counts by sentiment.</p>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={trendData} margin={{ left: -18, right: 8 }}>
                    <CartesianGrid stroke="#D2D7CF" />
                    <XAxis dataKey="day" tick={{ fontSize: 11, fill: '#4A5A52' }} minTickGap={18} />
                    <YAxis tick={{ fontSize: 11, fill: '#4A5A52' }} allowDecimals={false} />
                    <Tooltip contentStyle={TOOLTIP} />
                    <Legend wrapperStyle={{ fontSize: 12 }} />
                    <Line type="monotone" dataKey="negative" stroke="#B03A3A" strokeWidth={2} dot={false} />
                    <Line type="monotone" dataKey="neutral" stroke="#5F655D" strokeWidth={2} dot={false} />
                    <Line type="monotone" dataKey="positive" stroke="#2E7D5B" strokeWidth={2} dot={false} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          </section>

          <ReviewTable reviews={filtered} />
        </>
      )}
    </div>
  )
}

const TOOLTIP = {
  background: '#F6F7F4',
  border: '1px solid #D2D7CF',
  borderRadius: 8,
  fontSize: 12,
} as const

function Select({
  label, value, onChange, options,
}: {
  label: string
  value: string
  onChange: (v: string) => void
  options: [string, string][]
}) {
  const id = `filter-${label.toLowerCase()}`
  return (
    <div className="flex-1 min-w-[140px]">
      <label htmlFor={id} className="eyebrow block mb-2">{label}</label>
      <select id={id} value={value} onChange={(e) => onChange(e.target.value)}
        className="w-full min-h-[44px] rounded-md border border-line bg-paper px-3 text-sm">
        {options.map(([v, l]) => <option key={v} value={v}>{l}</option>)}
      </select>
    </div>
  )
}

function ReviewTable({ reviews }: { reviews: AnalyzedReview[] }) {
  const [limit, setLimit] = useState(12)
  const shown = reviews.slice(0, limit)

  return (
    <section className="card overflow-hidden">
      <div className="p-5 border-b border-line">
        <h2 className="font-display font-bold text-lg">Recent reviews</h2>
        <p className="text-sm text-slate mt-1">{reviews.length} matching this filter.</p>
      </div>

      {/* Desktop table */}
      <div className="hidden md:block overflow-x-auto">
        <table className="w-full text-sm">
          <caption className="sr-only">Analyzed reviews with sentiment and issue categories</caption>
          <thead>
            <tr className="text-left border-b border-line">
              <th scope="col" className="eyebrow font-normal px-5 py-3">Review</th>
              <th scope="col" className="eyebrow font-normal px-5 py-3 w-40">Sentiment</th>
              <th scope="col" className="eyebrow font-normal px-5 py-3 w-64">Issues</th>
              <th scope="col" className="eyebrow font-normal px-5 py-3 w-28">When</th>
            </tr>
          </thead>
          <tbody>
            {shown.map((r) => (
              <tr key={r.id} className="border-b border-line/60 last:border-0 align-top">
                <td className="px-5 py-4 max-w-md">
                  <p className="text-ink">{truncate(r.text, 110)}</p>
                  {r.low_confidence && <span className="inline-block mt-2"><Stamp small /></span>}
                </td>
                <td className="px-5 py-4">
                  <SentimentBadge label={r.sentiment.label} confidence={r.sentiment.confidence} size="sm" />
                </td>
                <td className="px-5 py-4">
                  <div className="flex flex-wrap gap-1.5">
                    {r.issues.length === 0
                      ? <span className="text-slate text-xs">none detected</span>
                      : r.issues.map((i) => <IssueTag key={i.category} category={i.category} />)}
                  </div>
                </td>
                <td className="px-5 py-4 font-mono text-xs text-slate whitespace-nowrap">
                  {timeAgo(r.processed_at)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Mobile cards -- a table would force horizontal scrolling here. */}
      <ul className="md:hidden divide-y divide-line">
        {shown.map((r) => (
          <li key={r.id} className="p-4">
            <p className="text-sm text-ink">{truncate(r.text, 140)}</p>
            <div className="flex flex-wrap items-center gap-2 mt-3">
              <SentimentBadge label={r.sentiment.label} confidence={r.sentiment.confidence} size="sm" />
              {r.issues.map((i) => <IssueTag key={i.category} category={i.category} />)}
            </div>
            <div className="flex items-center gap-3 mt-3">
              <span className="font-mono text-[11px] text-slate">{timeAgo(r.processed_at)}</span>
              {r.low_confidence && <Stamp small />}
            </div>
          </li>
        ))}
      </ul>

      {limit < reviews.length && (
        <div className="p-4 border-t border-line">
          <button type="button" onClick={() => setLimit((l) => l + 12)} className="btn-ghost w-full text-sm">
            Show more ({reviews.length - limit} left)
          </button>
        </div>
      )}
    </section>
  )
}
