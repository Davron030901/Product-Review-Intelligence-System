import { render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'

import type { AnalyzedReview } from '../api/types'
import { Analyzer } from '../screens/Analyzer'
import { Dashboard } from '../screens/Dashboard'
import { Queue } from '../screens/Queue'

function review(overrides: Partial<AnalyzedReview> = {}): AnalyzedReview {
  return {
    id: 'r1',
    text: 'Arrived two weeks late and the box was crushed.',
    sentiment: { label: 'negative', confidence: 0.91 },
    issues: [
      { category: 'delivery', confidence: 0.87 },
      { category: 'packaging', confidence: 0.62 },
    ],
    low_confidence: false,
    reasons: [],
    input_category: 'Tops',
    word_count: 9,
    truncated: false,
    model_version: 'v1',
    model_backend: 'baseline',
    processed_at: new Date().toISOString(),
    ...overrides,
  }
}

// --- Analyzer --------------------------------------------------------------

describe('Analyzer', () => {
  it('starts with an empty state rather than a blank panel', () => {
    render(<Analyzer onAnalyzed={vi.fn()} />)
    expect(screen.getByText(/nothing analyzed yet/i)).toBeInTheDocument()
  })

  it('disables the button until there is text', async () => {
    render(<Analyzer onAnalyzed={vi.fn()} />)
    const button = screen.getByRole('button', { name: /analyze review/i })
    expect(button).toBeDisabled()
    await userEvent.type(screen.getByLabelText(/review text/i), 'arrived late')
    expect(button).toBeEnabled()
  })

  it('will not submit whitespace only', async () => {
    render(<Analyzer onAnalyzed={vi.fn()} />)
    await userEvent.type(screen.getByLabelText(/review text/i), '    ')
    expect(screen.getByRole('button', { name: /analyze review/i })).toBeDisabled()
  })

  it('analyzes a review and shows the result', async () => {
    const onAnalyzed = vi.fn()
    render(<Analyzer onAnalyzed={onAnalyzed} />)
    await userEvent.type(
      screen.getByLabelText(/review text/i),
      'Arrived two weeks late and the box was crushed, terrible',
    )
    await userEvent.click(screen.getByRole('button', { name: /analyze review/i }))

    await waitFor(() => expect(screen.getByText('Negative')).toBeInTheDocument())
    // "delivery" appears both as a tag and in the glossary below it, so scope
    // the assertion to the tag list rather than the whole card.
    expect(screen.getAllByText('delivery').length).toBeGreaterThan(0)
    expect(onAnalyzed).toHaveBeenCalledOnce()
  })

  it('reports the analyzed review upward with its text attached', async () => {
    const onAnalyzed = vi.fn()
    render(<Analyzer onAnalyzed={onAnalyzed} />)
    await userEvent.type(screen.getByLabelText(/review text/i), 'love it, great quality')
    await userEvent.click(screen.getByRole('button', { name: /analyze review/i }))

    await waitFor(() => expect(onAnalyzed).toHaveBeenCalled())
    const passed = onAnalyzed.mock.calls[0][0]
    expect(passed.text).toBe('love it, great quality')
    expect(passed.id).toBeTruthy()
  })

  it('stamps a low-confidence result and explains why', async () => {
    render(<Analyzer onAnalyzed={vi.fn()} />)
    await userEvent.type(screen.getByLabelText(/review text/i), 'meh')
    await userEvent.click(screen.getByRole('button', { name: /analyze review/i }))

    await waitFor(() => expect(screen.getByText(/why this was flagged/i)).toBeInTheDocument())
    expect(screen.getByRole('status')).toHaveTextContent(/needs review/i)
  })

  it('says so when no issue category clears the threshold', async () => {
    render(<Analyzer onAnalyzed={vi.fn()} />)
    await userEvent.type(screen.getByLabelText(/review text/i), 'hello there friend')
    await userEvent.click(screen.getByRole('button', { name: /analyze review/i }))

    await waitFor(() =>
      expect(screen.getByText(/no issue category cleared the threshold/i)).toBeInTheDocument(),
    )
  })

  it('fills the textarea from a sample', async () => {
    render(<Analyzer onAnalyzed={vi.fn()} />)
    await userEvent.click(screen.getByRole('button', { name: /meh/i }))
    expect(screen.getByLabelText(/review text/i)).toHaveValue('meh')
  })

  it('passes the category through', async () => {
    const onAnalyzed = vi.fn()
    render(<Analyzer onAnalyzed={onAnalyzed} />)
    await userEvent.type(screen.getByLabelText(/review text/i), 'runs small')
    await userEvent.type(screen.getByLabelText(/product category/i), 'Dresses')
    await userEvent.click(screen.getByRole('button', { name: /analyze review/i }))

    await waitFor(() => expect(onAnalyzed).toHaveBeenCalled())
    expect(onAnalyzed.mock.calls[0][0].input_category).toBe('Dresses')
  })

  it('keeps the result region live for screen readers', () => {
    const { container } = render(<Analyzer onAnalyzed={vi.fn()} />)
    expect(container.querySelector('[aria-live="polite"]')).toBeInTheDocument()
  })
})

// --- Dashboard -------------------------------------------------------------

describe('Dashboard', () => {
  it('invites the first analysis when there is no data', async () => {
    const onGo = vi.fn()
    render(<Dashboard reviews={[]} onGoToAnalyzer={onGo} />)
    expect(screen.getByText(/no reviews analyzed yet/i)).toBeInTheDocument()
    await userEvent.click(screen.getByRole('button', { name: /analyze your first review/i }))
    expect(onGo).toHaveBeenCalledOnce()
  })

  it('summarises the reviews it was given', () => {
    render(<Dashboard reviews={[review(), review({ id: 'r2' })]} onGoToAnalyzer={vi.fn()} />)
    expect(screen.getByText('Reviews analyzed')).toBeInTheDocument()
    expect(screen.getByText('2')).toBeInTheDocument()
  })

  it('computes the negative share', () => {
    render(
      <Dashboard
        reviews={[
          review(),
          review({ id: 'r2', sentiment: { label: 'positive', confidence: 0.9 } }),
        ]}
        onGoToAnalyzer={vi.fn()}
      />,
    )
    expect(screen.getByText('50%')).toBeInTheDocument()
  })

  it('names the most common issue', () => {
    render(<Dashboard reviews={[review(), review({ id: 'r2' })]} onGoToAnalyzer={vi.fn()} />)
    expect(screen.getByText('Top issue')).toBeInTheDocument()
    expect(screen.getAllByText('delivery').length).toBeGreaterThan(0)
  })

  it('counts the reviews needing a second look', () => {
    render(
      <Dashboard
        reviews={[review({ low_confidence: true }), review({ id: 'r2' })]}
        onGoToAnalyzer={vi.fn()}
      />,
    )
    // "Needs review" is also the stamp text, so target the KPI label exactly.
    const label = screen
      .getAllByText('Needs review')
      .find((el) => el.className.includes('eyebrow'))!
    expect(within(label.parentElement!).getByText('1')).toBeInTheDocument()
  })

  it('filters by sentiment', async () => {
    render(
      <Dashboard
        reviews={[
          review(),
          review({ id: 'r2', sentiment: { label: 'positive', confidence: 0.9 } }),
        ]}
        onGoToAnalyzer={vi.fn()}
      />,
    )
    await userEvent.selectOptions(screen.getByLabelText('Sentiment'), 'positive')
    await waitFor(() => expect(screen.getByText('1 matching this filter.')).toBeInTheDocument())
  })

  it('explains an over-filtered view instead of showing nothing', async () => {
    render(
      <Dashboard
        reviews={[review({ processed_at: new Date(Date.now() - 90 * 864e5).toISOString() })]}
        onGoToAnalyzer={vi.fn()}
      />,
    )
    await userEvent.selectOptions(screen.getByLabelText('Period'), '7')
    await waitFor(() =>
      expect(screen.getByText(/no reviews match these filters/i)).toBeInTheDocument(),
    )
  })

  it('renders an accessible table of recent reviews', () => {
    render(<Dashboard reviews={[review()]} onGoToAnalyzer={vi.fn()} />)
    expect(screen.getByRole('table')).toBeInTheDocument()
    expect(screen.getByRole('columnheader', { name: /review/i })).toBeInTheDocument()
  })

  it('pages through a long list rather than dumping everything', async () => {
    const many = Array.from({ length: 30 }, (_, i) =>
      review({ id: `r${i}`, text: `review number ${i}` }),
    )
    render(<Dashboard reviews={many} onGoToAnalyzer={vi.fn()} />)
    const more = screen.getByRole('button', { name: /show more/i })
    expect(more).toHaveTextContent('18 left')
    await userEvent.click(more)
    await waitFor(() => expect(screen.getByRole('button', { name: /6 left/i })).toBeInTheDocument())
  })
})

// --- Queue -----------------------------------------------------------------

describe('Queue', () => {
  const flagged = review({
    low_confidence: true,
    reasons: ['Sentiment prediction is below the confidence threshold.'],
    sentiment: { label: 'neutral', confidence: 0.44 },
  })

  it('says the queue is clear when nothing is flagged', () => {
    render(<Queue reviews={[review()]} onResolve={vi.fn()} onGoToAnalyzer={vi.fn()} />)
    expect(screen.getByText(/queue is clear/i)).toBeInTheDocument()
  })

  it('lists flagged reviews with the reason attached', () => {
    render(<Queue reviews={[flagged]} onResolve={vi.fn()} onGoToAnalyzer={vi.fn()} />)
    expect(screen.getByText(flagged.text)).toBeInTheDocument()
    expect(screen.getByText(/below the confidence threshold/i)).toBeInTheDocument()
  })

  it('offers confirm and correct actions', () => {
    render(<Queue reviews={[flagged]} onResolve={vi.fn()} onGoToAnalyzer={vi.fn()} />)
    expect(screen.getByRole('button', { name: /confirm neutral/i })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /^negative$/i })).toBeInTheDocument()
  })

  it('reports a correction with the chosen label', async () => {
    const onResolve = vi.fn()
    render(<Queue reviews={[flagged]} onResolve={onResolve} onGoToAnalyzer={vi.fn()} />)
    await userEvent.click(screen.getByRole('button', { name: /^negative$/i }))
    expect(onResolve).toHaveBeenCalledWith('r1', 'negative')
  })

  it('separates waiting from checked items', async () => {
    const resolved = review({
      id: 'r2',
      low_confidence: true,
      resolution: { by: 'human', sentiment: 'negative', at: new Date().toISOString() },
    })
    render(<Queue reviews={[flagged, resolved]} onResolve={vi.fn()} onGoToAnalyzer={vi.fn()} />)

    expect(screen.getByRole('tab', { name: /waiting/i })).toHaveAttribute('aria-selected', 'true')
    await userEvent.click(screen.getByRole('tab', { name: /checked/i }))
    await waitFor(() => expect(screen.getByText(/marked/i)).toBeInTheDocument())
  })

  it('does not offer actions on an already-checked item', async () => {
    const resolved = review({
      low_confidence: true,
      resolution: { by: 'human', sentiment: 'positive', at: new Date().toISOString() },
    })
    render(<Queue reviews={[resolved]} onResolve={vi.fn()} onGoToAnalyzer={vi.fn()} />)
    await userEvent.click(screen.getByRole('tab', { name: /checked/i }))
    await waitFor(() =>
      expect(screen.queryByRole('button', { name: /confirm/i })).not.toBeInTheDocument(),
    )
  })

  it('congratulates rather than showing an empty list when all are handled', async () => {
    const resolved = review({
      low_confidence: true,
      resolution: { by: 'human', sentiment: 'negative', at: new Date().toISOString() },
    })
    render(<Queue reviews={[resolved]} onResolve={vi.fn()} onGoToAnalyzer={vi.fn()} />)
    expect(screen.getByText(/all caught up/i)).toBeInTheDocument()
  })
})
