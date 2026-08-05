import type { ReactNode } from 'react'

export type Screen = 'analyzer' | 'dashboard' | 'queue'

const NAV: { id: Screen; label: string; hint: string; icon: ReactNode }[] = [
  {
    id: 'analyzer',
    label: 'Analyze',
    hint: 'Read one review',
    icon: (
      <path d="M4 5h16M4 11h16M4 17h9" strokeWidth="2" strokeLinecap="round" fill="none" stroke="currentColor" />
    ),
  },
  {
    id: 'dashboard',
    label: 'Overview',
    hint: 'See the pattern',
    icon: (
      <path d="M4 19V9m5 10V5m5 14v-7m5 7V8" strokeWidth="2" strokeLinecap="round" fill="none" stroke="currentColor" />
    ),
  },
  {
    id: 'queue',
    label: 'Review queue',
    hint: 'Check the flagged ones',
    icon: (
      <>
        <path d="M12 4 21 19H3z" strokeWidth="2" strokeLinejoin="round" fill="none" stroke="currentColor" />
        <path d="M12 10v4M12 16.5v.5" strokeWidth="2" strokeLinecap="round" stroke="currentColor" />
      </>
    ),
  },
]

export function AppShell({
  screen,
  onNavigate,
  queueCount,
  children,
}: {
  screen: Screen
  onNavigate: (s: Screen) => void
  queueCount: number
  children: ReactNode
}) {
  return (
    <div className="min-h-dvh flex flex-col lg:flex-row">
      {/* Desktop rail */}
      <nav
        aria-label="Main"
        className="hidden lg:flex flex-col w-64 shrink-0 bg-pine-deep text-paper px-5 py-7 shadow-rail"
      >
        <Brand />
        <ul className="mt-9 space-y-1.5 flex-1">
          {NAV.map((item) => {
            const active = screen === item.id
            return (
              <li key={item.id}>
                <button
                  type="button"
                  onClick={() => onNavigate(item.id)}
                  aria-current={active ? 'page' : undefined}
                  className={`w-full text-left flex items-start gap-3 px-3 py-3 rounded-md transition-colors
                    ${active ? 'bg-paper/12 text-paper' : 'text-paper/65 hover:bg-paper/[0.07] hover:text-paper'}`}
                >
                  <svg viewBox="0 0 24 24" className="w-5 h-5 mt-0.5 shrink-0">{item.icon}</svg>
                  <span className="flex-1">
                    <span className="flex items-center gap-2 font-medium">
                      {item.label}
                      {item.id === 'queue' && queueCount > 0 && <QueuePip count={queueCount} />}
                    </span>
                    <span className="block text-xs text-paper/50 mt-0.5">{item.hint}</span>
                  </span>
                </button>
              </li>
            )
          })}
        </ul>
        <p className="text-[11px] text-paper/40 leading-relaxed font-mono">
          Prototype. Every result is a suggestion with a measurable error rate, not a verdict.
        </p>
      </nav>

      {/* Mobile header */}
      <header className="lg:hidden bg-pine-deep text-paper px-4 py-4">
        <Brand compact />
      </header>

      <main className="flex-1 min-w-0 pb-24 lg:pb-0">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-10 py-6 sm:py-9">{children}</div>
      </main>

      {/* Mobile bottom tabs */}
      <nav
        aria-label="Main"
        className="lg:hidden fixed bottom-0 inset-x-0 z-20 bg-card border-t border-line
                   pb-[env(safe-area-inset-bottom)]"
      >
        <ul className="grid grid-cols-3">
          {NAV.map((item) => {
            const active = screen === item.id
            return (
              <li key={item.id}>
                <button
                  type="button"
                  onClick={() => onNavigate(item.id)}
                  aria-current={active ? 'page' : undefined}
                  className={`w-full min-h-[60px] flex flex-col items-center justify-center gap-1 px-1
                    ${active ? 'text-pine' : 'text-slate'}`}
                >
                  <span className="relative">
                    <svg viewBox="0 0 24 24" className="w-5 h-5">{item.icon}</svg>
                    {item.id === 'queue' && queueCount > 0 && (
                      <span className="absolute -top-1.5 -right-2.5">
                        <QueuePip count={queueCount} />
                      </span>
                    )}
                  </span>
                  <span className="text-[11px] font-medium">{item.label}</span>
                </button>
              </li>
            )
          })}
        </ul>
      </nav>
    </div>
  )
}

function QueuePip({ count }: { count: number }) {
  return (
    <span
      className="inline-flex items-center justify-center min-w-[20px] h-5 px-1.5 rounded-full
                 bg-signal text-paper font-mono text-[10px] font-semibold tabular-nums"
      aria-label={`${count} flagged for review`}
    >
      {count > 99 ? '99+' : count}
    </span>
  )
}

function Brand({ compact }: { compact?: boolean }) {
  return (
    <div className="flex items-center gap-3">
      <svg viewBox="0 0 32 32" className="w-8 h-8 shrink-0" aria-hidden="true">
        <rect x="2" y="6" width="28" height="7" rx="2" fill="currentColor" opacity="0.35" />
        <rect x="2" y="16" width="19" height="7" rx="2" fill="currentColor" opacity="0.7" />
        <rect x="2" y="25" width="11" height="5" rx="2" fill="currentColor" />
      </svg>
      <div>
        <p className="font-display font-extrabold tracking-tight leading-none text-lg">
          Review Intelligence
        </p>
        {!compact && (
          <p className="text-[11px] text-paper/55 mt-1 font-mono uppercase tracking-[0.14em]">
            Sorted, labelled, checked
          </p>
        )}
      </div>
    </div>
  )
}
