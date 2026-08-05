import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { mockAnalyze, seedHistory } from '../api/mockData'


const RESULT_KEYS = [
  'sentiment', 'issues', 'low_confidence', 'reasons', 'input_category',
  'word_count', 'truncated', 'model_version', 'model_backend', 'processed_at',
]

describe('mockAnalyze — contract fidelity', () => {
  it('returns every key the real backend returns', async () => {
    const r = await mockAnalyze('Arrived late and crushed.', undefined, 0)
    RESULT_KEYS.forEach((k) => expect(r).toHaveProperty(k))
  })

  it('reports a confidence between 0 and 1', async () => {
    const r = await mockAnalyze('This is wonderful', undefined, 0)
    expect(r.sentiment.confidence).toBeGreaterThanOrEqual(0)
    expect(r.sentiment.confidence).toBeLessThanOrEqual(1)
  })

  it('detects a negative delivery complaint', async () => {
    const r = await mockAnalyze('Arrived two weeks late, terrible', undefined, 0)
    expect(r.sentiment.label).toBe('negative')
    expect(r.issues.map((i) => i.category)).toContain('delivery')
  })

  it('detects a positive review', async () => {
    const r = await mockAnalyze('Love it, great quality, perfect', undefined, 0)
    expect(r.sentiment.label).toBe('positive')
  })

  it('returns multiple issues for a multi-issue review', async () => {
    const r = await mockAnalyze('Arrived late and the box was crushed', undefined, 0)
    expect(r.issues.length).toBeGreaterThanOrEqual(2)
  })

  it('sorts issues by descending confidence', async () => {
    const r = await mockAnalyze('late, box crushed, broken, expensive', undefined, 0)
    const conf = r.issues.map((i) => i.confidence)
    expect(conf).toEqual([...conf].sort((a, b) => b - a))
  })
})

describe('mockAnalyze — abstention, matching the backend rules', () => {
  it('returns unknown for empty text', async () => {
    const r = await mockAnalyze('', undefined, 0)
    expect(r.sentiment.label).toBe('unknown')
    expect(r.low_confidence).toBe(true)
    expect(r.issues).toEqual([])
  })

  it('treats whitespace as empty', async () => {
    expect((await mockAnalyze('   \n ', undefined, 0)).sentiment.label).toBe('unknown')
  })

  it('flags a one-word review', async () => {
    const r = await mockAnalyze('meh', undefined, 0)
    expect(r.low_confidence).toBe(true)
    expect(r.reasons.join(' ')).toMatch(/word/i)
  })

  it('always attaches a reason when flagging', async () => {
    for (const text of ['', 'meh', 'ok']) {
      const r = await mockAnalyze(text, undefined, 0)
      if (r.low_confidence) expect(r.reasons.length).toBeGreaterThan(0)
    }
  })

  it('marks oversized input as truncated', async () => {
    const r = await mockAnalyze('a'.repeat(20001), undefined, 0)
    expect(r.truncated).toBe(true)
  })

  it('echoes the category back', async () => {
    expect((await mockAnalyze('nice', 'Dresses', 0)).input_category).toBe('Dresses')
  })

  it('survives hostile input', async () => {
    const nasty = ['😡😡', '<script>alert(1)</script>', "'; DROP TABLE x; --", '\u0000']
    for (const text of nasty) {
      const r = await mockAnalyze(text, undefined, 0)
      RESULT_KEYS.forEach((k) => expect(r).toHaveProperty(k))
    }
  })
})

describe('seedHistory', () => {
  it('produces a usable dashboard fixture', async () => {
    const rows = await seedHistory()
    expect(rows.length).toBeGreaterThan(20)
    rows.forEach((r) => {
      expect(r.id).toBeTruthy()
      expect(r.text).toBeTruthy()
    })
  })

  it('is sorted newest first', async () => {
    const rows = await seedHistory()
    const times = rows.map((r) => +new Date(r.processed_at))
    expect(times).toEqual([...times].sort((a, b) => b - a))
  })

  it('includes flagged reviews so the queue is not empty', async () => {
    const rows = await seedHistory()
    expect(rows.some((r) => r.low_confidence)).toBe(true)
  })

  it('spans several days so the trend chart has a shape', async () => {
    const rows = await seedHistory()
    const days = new Set(rows.map((r) => r.processed_at.slice(0, 10)))
    expect(days.size).toBeGreaterThan(3)
  })

  it('returns quickly enough not to block first paint', async () => {
    const t0 = Date.now()
    await seedHistory()
    expect(Date.now() - t0).toBeLessThan(1000)
  })
})

describe('api client — network behaviour', () => {
  const originalFetch = globalThis.fetch

  beforeEach(() => {
    vi.resetModules()
    vi.unstubAllEnvs()
  })

  afterEach(() => {
    globalThis.fetch = originalFetch
    vi.unstubAllEnvs()
    vi.useRealTimers()
  })

  async function loadClient() {
    vi.stubEnv('VITE_USE_MOCK', 'false')
    return (await import('../api/client')).api
  }

  it('strips a trailing slash from the configured base url', async () => {
    vi.stubEnv('VITE_API_BASE_URL', 'https://api.example.com/')
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ ok: true }), { status: 200 }),
    )
    globalThis.fetch = fetchMock as unknown as typeof fetch
    const api = await loadClient()
    await api.health()
    expect(fetchMock.mock.calls[0][0]).toBe('https://api.example.com/health')
  })

  it('falls back to the dev proxy path when no base url is set', async () => {
    vi.stubEnv('VITE_API_BASE_URL', '')
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ ok: true }), { status: 200 }),
    )
    globalThis.fetch = fetchMock as unknown as typeof fetch
    const api = await loadClient()
    await api.health()
    expect(fetchMock.mock.calls[0][0]).toBe('/api/health')
  })

  it('posts the review text as json', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ sentiment: {} }), { status: 200 }),
    )
    globalThis.fetch = fetchMock as unknown as typeof fetch
    const api = await loadClient()
    await api.analyze('arrived late', 'Tops')
    const [, init] = fetchMock.mock.calls[0]
    expect(init.method).toBe('POST')
    expect(JSON.parse(init.body)).toEqual({ text: 'arrived late', category: 'Tops' })
  })

  it('explains a 503 as a missing model rather than a raw status', async () => {
    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response('{}', { status: 503 }),
    ) as unknown as typeof fetch
    const api = await loadClient()
    await expect(api.analyze('x')).rejects.toThrow(/model/i)
  })

  it('surfaces the server detail on a 4xx', async () => {
    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ detail: 'Send at least one review.' }), { status: 422 }),
    ) as unknown as typeof fetch
    const api = await loadClient()
    await expect(api.analyzeBatch([])).rejects.toThrow('Send at least one review.')
  })

  it('reports an unreachable service in plain language', async () => {
    globalThis.fetch = vi.fn().mockRejectedValue(new TypeError('Failed to fetch')) as
      unknown as typeof fetch
    const api = await loadClient()
    // Checked by name, not `instanceof`: vi.resetModules() gives the
    // dynamically imported client its own ApiError class object, distinct from
    // the one imported at the top of this file.
    await expect(api.analyze('x')).rejects.toHaveProperty('name', 'ApiError')
    await expect(api.analyze('x')).rejects.toThrow(/reach/i)
  })

  it('mentions the idle-service cold start when a request times out', async () => {
    globalThis.fetch = vi.fn().mockRejectedValue(
      new DOMException('aborted', 'AbortError'),
    ) as unknown as typeof fetch
    const api = await loadClient()
    await expect(api.analyze('x')).rejects.toThrow(/too long|idle|suspend/i)
  })

  it('unwraps the batch envelope into a plain array', async () => {
    globalThis.fetch = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ results: [{ a: 1 }, { a: 2 }], count: 2 }), { status: 200 }),
    ) as unknown as typeof fetch
    const api = await loadClient()
    expect(await api.analyzeBatch(['a', 'b'])).toHaveLength(2)
  })

  it('uses the mock backend by default, with no network call', async () => {
    const fetchMock = vi.fn()
    globalThis.fetch = fetchMock as unknown as typeof fetch
    vi.resetModules()
    const { api } = await import('../api/client')
    const r = await api.analyze('arrived late')
    expect(fetchMock).not.toHaveBeenCalled()
    expect(r.model_backend).toBe('mock')
  })
})
