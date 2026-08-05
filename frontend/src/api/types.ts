/** Mirrors backend/src/api/schemas.py. Keep the two in sync. */

export type SentimentLabel = 'negative' | 'neutral' | 'positive' | 'unknown'

export type IssueCategory =
  | 'delivery' | 'packaging' | 'quality' | 'defect'
  | 'price' | 'service' | 'fit' | 'other'

export interface Sentiment {
  label: SentimentLabel
  confidence: number
}

export interface Issue {
  category: IssueCategory
  confidence: number
}

export interface AnalysisResult {
  sentiment: Sentiment
  issues: Issue[]
  low_confidence: boolean
  reasons: string[]
  input_category: string | null
  word_count: number
  truncated: boolean
  model_version: string
  model_backend: string
  processed_at: string
}

/** A result plus the text it came from, which the backend does not echo. */
export interface AnalyzedReview extends AnalysisResult {
  id: string
  text: string
  /** Set when a person has confirmed or corrected a flagged result. */
  resolution?: { by: 'human'; sentiment: SentimentLabel; at: string }
}

export interface HealthResponse {
  status: 'ok' | 'degraded'
  model_loaded: boolean
  model_backend?: string
  model_version?: string
}

export class ApiError extends Error {
  constructor(message: string, readonly status?: number) {
    super(message)
    this.name = 'ApiError'
  }
}
