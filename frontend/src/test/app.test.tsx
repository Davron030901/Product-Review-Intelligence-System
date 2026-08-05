import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it } from 'vitest'

import App from '../App'

/**
 * These exercise the thing unit tests cannot: that a review analyzed on one
 * screen actually reaches the other two.
 */

const go = async (name: RegExp) =>
  userEvent.click((await screen.findAllByRole('button', { name }))[0])

describe('App integration', () => {
  it('boots into the analyzer once the service check completes', async () => {
    render(<App />)
    await waitFor(() =>
      expect(screen.getByText(/what is this customer actually telling you/i)).toBeInTheDocument(),
    )
  })

  it('shows a loading state before the first paint', () => {
    render(<App />)
    expect(screen.getByRole('status')).toHaveTextContent(/connecting/i)
  })

  it('moves between all three screens', async () => {
    render(<App />)
    await waitFor(() => expect(screen.getByLabelText(/review text/i)).toBeInTheDocument())

    await go(/overview/i)
    await waitFor(() => expect(screen.getByText(/what keeps coming up/i)).toBeInTheDocument())

    await go(/review queue/i)
    await waitFor(() =>
      expect(screen.getByRole('heading', { name: 'Review queue' })).toBeInTheDocument(),
    )

    await go(/analyze/i)
    await waitFor(() => expect(screen.getByLabelText(/review text/i)).toBeInTheDocument())
  })

  it('seeds the dashboard so it is not empty on arrival', async () => {
    render(<App />)
    await go(/overview/i)
    await waitFor(() => expect(screen.getByText('Reviews analyzed')).toBeInTheDocument())
    expect(screen.queryByText(/no reviews analyzed yet/i)).not.toBeInTheDocument()
  })

  it('carries a newly analyzed review through to the dashboard', async () => {
    render(<App />)
    await waitFor(() => expect(screen.getByLabelText(/review text/i)).toBeInTheDocument())

    const unique = 'The courier lost my parcel entirely, absolutely terrible'
    await userEvent.type(screen.getByLabelText(/review text/i), unique)
    await userEvent.click(screen.getByRole('button', { name: /analyze review/i }))
    await waitFor(() => expect(screen.getByText('Negative')).toBeInTheDocument())

    await go(/overview/i)
    // The desktop table and the mobile card list are both in the DOM; CSS hides
    // one of them, so match all occurrences rather than expecting exactly one.
    await waitFor(() =>
      expect(screen.getAllByText(new RegExp(unique.slice(0, 30), 'i')).length).toBeGreaterThan(0),
    )
  })

  it('routes a low-confidence result into the queue and updates the badge', async () => {
    render(<App />)
    await waitFor(() => expect(screen.getByLabelText(/review text/i)).toBeInTheDocument())

    const before = Number(
      (await screen.findAllByLabelText(/flagged for review/i))[0].textContent,
    )

    await userEvent.type(screen.getByLabelText(/review text/i), 'meh')
    await userEvent.click(screen.getByRole('button', { name: /analyze review/i }))
    await waitFor(() => expect(screen.getByText(/why this was flagged/i)).toBeInTheDocument())

    await waitFor(async () => {
      const after = Number(
        (await screen.findAllByLabelText(/flagged for review/i))[0].textContent,
      )
      expect(after).toBe(before + 1)
    })
  })

  it('lets a person resolve a queued item, which clears it from waiting', async () => {
    render(<App />)
    await go(/review queue/i)
    await waitFor(() =>
      expect(screen.getByRole('heading', { name: 'Review queue' })).toBeInTheDocument(),
    )

    const waitingTab = screen.getByRole('tab', { name: /waiting/i })
    const before = Number(waitingTab.textContent!.replace(/\D/g, ''))
    expect(before).toBeGreaterThan(0)

    await userEvent.click(screen.getAllByRole('button', { name: /^negative$/i })[0])

    await waitFor(() => {
      const after = Number(
        screen.getByRole('tab', { name: /waiting/i }).textContent!.replace(/\D/g, ''),
      )
      expect(after).toBe(before - 1)
    })
    expect(screen.getByRole('tab', { name: /checked/i }).textContent).toMatch(/1/)
  })

  it('keeps the queue badge and the queue screen in agreement', async () => {
    render(<App />)
    await waitFor(() => expect(screen.getByLabelText(/review text/i)).toBeInTheDocument())
    const badge = Number((await screen.findAllByLabelText(/flagged for review/i))[0].textContent)

    await go(/review queue/i)
    await waitFor(() =>
      expect(screen.getByRole('heading', { name: 'Review queue' })).toBeInTheDocument(),
    )
    const tab = Number(
      screen.getByRole('tab', { name: /waiting/i }).textContent!.replace(/\D/g, ''),
    )
    expect(tab).toBe(badge)
  })

  it('does not lose analyzed reviews when navigating away and back', async () => {
    render(<App />)
    await waitFor(() => expect(screen.getByLabelText(/review text/i)).toBeInTheDocument())

    await userEvent.type(screen.getByLabelText(/review text/i), 'love it, perfect quality')
    await userEvent.click(screen.getByRole('button', { name: /analyze review/i }))
    await waitFor(() => expect(screen.getByText('Positive')).toBeInTheDocument())

    await go(/overview/i)
    await waitFor(() => expect(screen.getByText('Reviews analyzed')).toBeInTheDocument())
    const total = screen.getByText('Reviews analyzed').parentElement!.textContent

    await go(/analyze/i)
    await go(/overview/i)
    await waitFor(() =>
      expect(screen.getByText('Reviews analyzed').parentElement!.textContent).toBe(total),
    )
  })
})
