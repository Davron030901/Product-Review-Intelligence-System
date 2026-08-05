import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'

import { AppShell } from '../components/AppShell'
import { ConfidenceBar } from '../components/ConfidenceBar'
import { IssueTag } from '../components/IssueTag'
import { SentimentBadge } from '../components/SentimentBadge'
import { EmptyState, ErrorState, LoadingBlock } from '../components/States'
import { StatCard } from '../components/StatCard'
import { Stamp } from '../components/Stamp'

describe('SentimentBadge', () => {
  it('shows the label as words, not only colour', () => {
    render(<SentimentBadge label="negative" />)
    expect(screen.getByText('Negative')).toBeInTheDocument()
  })

  it('shows the confidence when given one', () => {
    render(<SentimentBadge label="positive" confidence={0.91} />)
    expect(screen.getByText('91%')).toBeInTheDocument()
  })

  it('hides a zero confidence rather than showing 0%', () => {
    render(<SentimentBadge label="unknown" confidence={0} />)
    expect(screen.queryByText('0%')).not.toBeInTheDocument()
  })

  it('renders the unknown state', () => {
    render(<SentimentBadge label="unknown" />)
    expect(screen.getByText('Unknown')).toBeInTheDocument()
  })
})

describe('IssueTag', () => {
  it('renders the category name', () => {
    render(<IssueTag category="delivery" />)
    expect(screen.getByText('delivery')).toBeInTheDocument()
  })

  it('shows confidence when supplied', () => {
    render(<IssueTag category="quality" confidence={0.87} />)
    expect(screen.getByText('87%')).toBeInTheDocument()
  })

  it('is not interactive without an onClick', () => {
    render(<IssueTag category="price" />)
    expect(screen.queryByRole('button')).not.toBeInTheDocument()
  })

  it('becomes a toggle button when clickable', async () => {
    const onClick = vi.fn()
    render(<IssueTag category="defect" onClick={onClick} selected />)
    const button = screen.getByRole('button')
    expect(button).toHaveAttribute('aria-pressed', 'true')
    await userEvent.click(button)
    expect(onClick).toHaveBeenCalledOnce()
  })

  it('explains the category on hover', () => {
    render(<IssueTag category="service" />)
    expect(screen.getByTitle(/support/i)).toBeInTheDocument()
  })
})

describe('ConfidenceBar', () => {
  it('exposes the value to assistive technology', () => {
    render(<ConfidenceBar value={0.42} label="Model confidence" />)
    const meter = screen.getByRole('meter')
    expect(meter).toHaveAttribute('aria-valuenow', '42')
    expect(meter).toHaveAttribute('aria-valuemin', '0')
    expect(meter).toHaveAttribute('aria-valuemax', '100')
  })

  it('warns when the value sits below the threshold', () => {
    render(<ConfidenceBar value={0.3} threshold={0.55} label="c" />)
    expect(screen.getByText(/below the 55% threshold/i)).toBeInTheDocument()
  })

  it('stays quiet when the value clears the threshold', () => {
    render(<ConfidenceBar value={0.9} threshold={0.55} label="c" />)
    expect(screen.queryByText(/below the/i)).not.toBeInTheDocument()
  })
})

describe('Stamp', () => {
  it('announces itself as a status', () => {
    render(<Stamp />)
    expect(screen.getByRole('status')).toHaveTextContent(/needs review/i)
  })

  it('accepts custom wording', () => {
    render(<Stamp label="Check this" />)
    expect(screen.getByText('Check this')).toBeInTheDocument()
  })
})

describe('StatCard', () => {
  it('renders label, value and detail', () => {
    render(<StatCard label="Negative" value="36%" detail="16 of 45" />)
    expect(screen.getByText('Negative')).toBeInTheDocument()
    expect(screen.getByText('36%')).toBeInTheDocument()
    expect(screen.getByText('16 of 45')).toBeInTheDocument()
  })
})

describe('state components', () => {
  it('LoadingBlock announces progress', () => {
    render(<LoadingBlock message="Loading things" />)
    expect(screen.getByRole('status')).toHaveTextContent(/loading things/i)
  })

  it('EmptyState explains what to do next', () => {
    render(<EmptyState title="Nothing here" body="Analyze a review to begin." />)
    expect(screen.getByText('Nothing here')).toBeInTheDocument()
    expect(screen.getByText(/analyze a review to begin/i)).toBeInTheDocument()
  })

  it('EmptyState can carry an action', () => {
    render(<EmptyState title="t" body="b" action={<button>Do it</button>} />)
    expect(screen.getByRole('button', { name: 'Do it' })).toBeInTheDocument()
  })

  it('ErrorState uses an alert role so it is announced', () => {
    render(<ErrorState message="Something broke" />)
    expect(screen.getByRole('alert')).toHaveTextContent('Something broke')
  })

  it('ErrorState retries on demand', async () => {
    const onRetry = vi.fn()
    render(<ErrorState message="broke" onRetry={onRetry} />)
    await userEvent.click(screen.getByRole('button', { name: /try again/i }))
    expect(onRetry).toHaveBeenCalledOnce()
  })

  it('ErrorState omits the retry button when there is nothing to retry', () => {
    render(<ErrorState message="broke" />)
    expect(screen.queryByRole('button')).not.toBeInTheDocument()
  })
})

describe('AppShell', () => {
  const renderShell = (props = {}) =>
    render(
      <AppShell screen="analyzer" onNavigate={vi.fn()} queueCount={0} {...props}>
        <p>content</p>
      </AppShell>,
    )

  it('renders its children', () => {
    renderShell()
    expect(screen.getByText('content')).toBeInTheDocument()
  })

  it('marks the current screen for assistive technology', () => {
    renderShell({ screen: 'dashboard' })
    const current = screen.getAllByRole('button', { current: 'page' })
    expect(current.length).toBeGreaterThan(0)
    expect(current[0]).toHaveTextContent(/overview/i)
  })

  it('navigates when a nav item is clicked', async () => {
    const onNavigate = vi.fn()
    renderShell({ onNavigate })
    await userEvent.click(screen.getAllByRole('button', { name: /overview/i })[0])
    expect(onNavigate).toHaveBeenCalledWith('dashboard')
  })

  it('shows the queue count when items are waiting', () => {
    renderShell({ queueCount: 7 })
    expect(screen.getAllByLabelText('7 flagged for review').length).toBeGreaterThan(0)
  })

  it('hides the queue badge when the queue is clear', () => {
    renderShell({ queueCount: 0 })
    expect(screen.queryByLabelText(/flagged for review/)).not.toBeInTheDocument()
  })

  it('caps a very large queue count', () => {
    renderShell({ queueCount: 250 })
    expect(screen.getAllByText('99+').length).toBeGreaterThan(0)
  })

  it('renders both desktop and mobile navigation landmarks', () => {
    renderShell()
    expect(screen.getAllByRole('navigation', { name: 'Main' })).toHaveLength(2)
  })
})
