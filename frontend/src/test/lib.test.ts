import { describe, expect, it } from 'vitest'

import { CATEGORY_HEX, CATEGORY_MEANING, ISSUE_ORDER, SENTIMENT_STYLE } from '../lib/categories'
import { dayKey, pct, shortDate, timeAgo, truncate } from '../lib/format'

describe('pct', () => {
  it('formats a probability as a whole percentage', () => {
    expect(pct(0.91)).toBe('91%')
    expect(pct(0)).toBe('0%')
    expect(pct(1)).toBe('100%')
  })

  it('rounds rather than truncates', () => {
    expect(pct(0.916)).toBe('92%')
    expect(pct(0.914)).toBe('91%')
  })
})

describe('truncate', () => {
  it('leaves short strings alone', () => {
    expect(truncate('short', 20)).toBe('short')
  })

  it('adds an ellipsis when cutting', () => {
    const out = truncate('a'.repeat(50), 10)
    expect(out).toHaveLength(11) // 10 chars + ellipsis
    expect(out.endsWith('…')).toBe(true)
  })

  it('does not leave a trailing space before the ellipsis', () => {
    expect(truncate('hello world there', 6)).toBe('hello…')
  })

  it('handles the empty string', () => {
    expect(truncate('', 10)).toBe('')
  })
})

describe('timeAgo', () => {
  const ago = (ms: number) => new Date(Date.now() - ms).toISOString()

  it('describes the last minute as just now', () => {
    expect(timeAgo(ago(5_000))).toBe('just now')
  })

  it('reports minutes, hours and days', () => {
    expect(timeAgo(ago(5 * 60_000))).toBe('5m ago')
    expect(timeAgo(ago(3 * 3_600_000))).toBe('3h ago')
    expect(timeAgo(ago(3 * 86_400_000))).toBe('3d ago')
  })

  it('says yesterday rather than 1d ago', () => {
    expect(timeAgo(ago(26 * 3_600_000))).toBe('yesterday')
  })

  it('crosses the minute and hour boundaries correctly', () => {
    expect(timeAgo(ago(59 * 60_000))).toBe('59m ago')
    expect(timeAgo(ago(61 * 60_000))).toBe('1h ago')
  })
})

describe('dayKey / shortDate', () => {
  it('reduces a timestamp to a calendar day', () => {
    expect(dayKey('2026-07-30T22:14:00.000Z')).toBe('2026-07-30')
  })

  it('groups two times on the same day under one key', () => {
    expect(dayKey('2026-07-30T01:00:00Z')).toBe(dayKey('2026-07-30T23:00:00Z'))
  })

  it('renders a readable label', () => {
    expect(shortDate('2026-07-30')).toMatch(/Jul/)
  })
})

describe('category design mappings', () => {
  it('has a colour for every issue category', () => {
    ISSUE_ORDER.forEach((c) => {
      expect(CATEGORY_HEX[c]).toMatch(/^#[0-9A-Fa-f]{6}$/)
    })
  })

  it('has a plain-language meaning for every category', () => {
    ISSUE_ORDER.forEach((c) => {
      expect(CATEGORY_MEANING[c]).toBeTruthy()
    })
  })

  it('gives each category a distinct colour', () => {
    const hexes = ISSUE_ORDER.map((c) => CATEGORY_HEX[c])
    expect(new Set(hexes).size).toBe(hexes.length)
  })

  it('covers every sentiment state including unknown', () => {
    ;(['negative', 'neutral', 'positive', 'unknown'] as const).forEach((s) => {
      expect(SENTIMENT_STYLE[s].label).toBeTruthy()
      expect(SENTIMENT_STYLE[s].glyph).toBeTruthy()
    })
  })

  it('distinguishes sentiment by glyph, not only colour', () => {
    const glyphs = (['negative', 'neutral', 'positive'] as const).map(
      (s) => SENTIMENT_STYLE[s].glyph,
    )
    expect(new Set(glyphs).size).toBe(3)
  })
})
