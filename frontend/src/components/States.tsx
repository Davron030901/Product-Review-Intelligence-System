/** Loading, empty and error states. Each says what happened and what to do next. */

export function Spinner({ className = 'w-5 h-5' }: { className?: string }) {
  return (
    <svg className={`animate-spin ${className}`} viewBox="0 0 24 24" aria-hidden="true">
      <circle cx="12" cy="12" r="9" stroke="currentColor" strokeWidth="2.5" fill="none" opacity="0.2" />
      <path d="M21 12a9 9 0 0 0-9-9" stroke="currentColor" strokeWidth="2.5" fill="none" strokeLinecap="round" />
    </svg>
  )
}

export function LoadingBlock({ message = 'Loading' }: { message?: string }) {
  return (
    <div className="flex items-center gap-3 text-slate py-14 justify-center" role="status">
      <Spinner />
      <span className="text-sm">{message}…</span>
    </div>
  )
}

export function EmptyState({
  title,
  body,
  action,
}: {
  title: string
  body: string
  action?: React.ReactNode
}) {
  return (
    <div className="text-center py-16 px-6">
      <svg viewBox="0 0 48 48" className="w-12 h-12 mx-auto text-line" aria-hidden="true">
        <rect x="6" y="10" width="36" height="28" rx="3" fill="none" stroke="currentColor" strokeWidth="2" />
        <path d="M6 18h36M15 26h18M15 32h11" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
      </svg>
      <h3 className="font-display font-bold text-lg mt-4">{title}</h3>
      <p className="text-slate text-sm mt-1.5 max-w-sm mx-auto">{body}</p>
      {action && <div className="mt-5">{action}</div>}
    </div>
  )
}

export function ErrorState({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return (
    <div
      role="alert"
      className="border border-cat-defect/35 bg-cat-defect/[0.06] rounded-lg p-5 flex flex-col
                 sm:flex-row sm:items-center gap-4"
    >
      <svg viewBox="0 0 24 24" className="w-6 h-6 text-cat-defect shrink-0" aria-hidden="true">
        <circle cx="12" cy="12" r="9.5" fill="none" stroke="currentColor" strokeWidth="2" />
        <path d="M12 7v6M12 16.4v.6" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" />
      </svg>
      <p className="text-sm text-ink flex-1">{message}</p>
      {onRetry && (
        <button type="button" onClick={onRetry} className="btn-ghost !min-h-[38px] text-sm shrink-0">
          Try again
        </button>
      )}
    </div>
  )
}
