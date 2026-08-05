/**
 * The only place in the app that talks to the network.
 *
 * Set VITE_USE_MOCK=false once the backend is running; the default is mock so
 * the UI is developable and demoable without a trained model on hand.
 */
import { ApiError } from './types'
import type { AnalysisResult, HealthResponse } from './types'
import { mockAnalyze } from './mockData'

// In dev this stays '/api' and vite proxies it. In production set
// VITE_API_BASE_URL to the API origin. A trailing slash is stripped so
// both 'https://x.onrender.com' and 'https://x.onrender.com/' work.
const BASE = (import.meta.env.VITE_API_BASE_URL || '/api').replace(/\/+$/, '')
const USE_MOCK = import.meta.env.VITE_USE_MOCK !== 'false'

// Render's free tier suspends idle services; the first request after a sleep
// has to wait for a cold start, which can take the better part of a minute.
// Anything shorter than this would abort a request that was going to succeed.
const TIMEOUT_MS = 75_000

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), TIMEOUT_MS)
  let res: Response
  try {
    res = await fetch(`${BASE}${path}`, {
      ...init,
      signal: controller.signal,
      headers: { 'Content-Type': 'application/json', ...(init?.headers ?? {}) },
    })
  } catch (e) {
    if (e instanceof DOMException && e.name === 'AbortError') {
      throw new ApiError(
        'The analysis service took too long to respond. Free hosting suspends the ' +
          'service when idle — wait a moment and try again.',
      )
    }
    throw new ApiError("Can't reach the analysis service. Check that it's running.")
  } finally {
    clearTimeout(timer)
  }

  if (res.status === 503) {
    throw new ApiError('No model is loaded yet. Train one, then restart the service.', 503)
  }
  if (!res.ok) {
    const detail = await res.json().catch(() => null)
    throw new ApiError(detail?.detail ?? `Request failed (${res.status}).`, res.status)
  }
  return res.json() as Promise<T>
}

export const api = {
  usingMock: USE_MOCK,

  async analyze(text: string, category?: string): Promise<AnalysisResult> {
    if (USE_MOCK) return mockAnalyze(text, category)
    return request<AnalysisResult>('/analyze', {
      method: 'POST',
      body: JSON.stringify({ text, category: category ?? null }),
    })
  },

  async analyzeBatch(reviews: string[], category?: string): Promise<AnalysisResult[]> {
    if (USE_MOCK) return Promise.all(reviews.map((r) => mockAnalyze(r, category)))
    const body = await request<{ results: AnalysisResult[] }>('/analyze/batch', {
      method: 'POST',
      body: JSON.stringify({ reviews, category: category ?? null }),
    })
    return body.results
  },

  async health(): Promise<HealthResponse> {
    if (USE_MOCK) {
      return { status: 'ok', model_loaded: true, model_backend: 'mock', model_version: 'mock' }
    }
    return request<HealthResponse>('/health')
  },
}
