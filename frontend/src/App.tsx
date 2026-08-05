import { useCallback, useEffect, useState } from 'react'
import { api } from './api/client'
import { seedHistory } from './api/mockData'
import type { AnalyzedReview, SentimentLabel } from './api/types'
import { AppShell, type Screen } from './components/AppShell'
import { ErrorState, LoadingBlock } from './components/States'
import { Analyzer } from './screens/Analyzer'
import { Dashboard } from './screens/Dashboard'
import { Queue } from './screens/Queue'

export default function App() {
  const [screen, setScreen] = useState<Screen>('analyzer')
  const [reviews, setReviews] = useState<AnalyzedReview[]>([])
  const [booting, setBooting] = useState(true)
  const [bootError, setBootError] = useState<string | null>(null)

  const boot = useCallback(async () => {
    setBooting(true)
    setBootError(null)
    try {
      const health = await api.health()
      if (!health.model_loaded) {
        setBootError('The service is running but no model is loaded. Train one, then restart it.')
      }
      // In mock mode, seed some history so the dashboard is not empty on arrival.
      if (api.usingMock) setReviews(await seedHistory())
    } catch (e) {
      setBootError(e instanceof Error ? e.message : 'Could not reach the analysis service.')
    } finally {
      setBooting(false)
    }
  }, [])

  useEffect(() => { void boot() }, [boot])

  const addReview = useCallback((r: AnalyzedReview) => {
    setReviews((prev) => [r, ...prev])
  }, [])

  const resolve = useCallback((id: string, sentiment: SentimentLabel) => {
    setReviews((prev) =>
      prev.map((r) =>
        r.id === id
          ? { ...r, resolution: { by: 'human' as const, sentiment, at: new Date().toISOString() } }
          : r,
      ),
    )
  }, [])

  const queueCount = reviews.filter((r) => r.low_confidence && !r.resolution).length

  return (
    <AppShell screen={screen} onNavigate={setScreen} queueCount={queueCount}>
      {booting ? (
        <LoadingBlock message="Connecting to the analysis service" />
      ) : (
        <>
          {bootError && (
            <div className="mb-6">
              <ErrorState message={bootError} onRetry={boot} />
            </div>
          )}
          {screen === 'analyzer' && <Analyzer onAnalyzed={addReview} />}
          {screen === 'dashboard' && (
            <Dashboard reviews={reviews} onGoToAnalyzer={() => setScreen('analyzer')} />
          )}
          {screen === 'queue' && (
            <Queue reviews={reviews} onResolve={resolve} onGoToAnalyzer={() => setScreen('analyzer')} />
          )}
        </>
      )}
    </AppShell>
  )
}
