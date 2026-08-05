export const pct = (n: number) => `${Math.round(n * 100)}%`

export function timeAgo(iso: string): string {
  const diff = Date.now() - new Date(iso).getTime()
  const mins = Math.floor(diff / 60000)
  if (mins < 1) return 'just now'
  if (mins < 60) return `${mins}m ago`
  const hrs = Math.floor(mins / 60)
  if (hrs < 24) return `${hrs}h ago`
  const days = Math.floor(hrs / 24)
  return days === 1 ? 'yesterday' : `${days}d ago`
}

export const dayKey = (iso: string) => new Date(iso).toISOString().slice(0, 10)

export const shortDate = (key: string) =>
  new Date(key).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })

export const truncate = (s: string, n = 120) =>
  s.length > n ? `${s.slice(0, n).trimEnd()}…` : s
