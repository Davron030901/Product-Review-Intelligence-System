/**
 * The signature element: an inspection stamp.
 *
 * The whole product is a sorting line, and the one thing a sorting line does
 * that a dashboard doesn't is physically mark an item that a person must look
 * at. Amber is reserved for this and used nowhere else, so a flagged result
 * is recognisable from across the room without reading a word.
 */
export function Stamp({ label = 'Needs review', small = false }: { label?: string; small?: boolean }) {
  return (
    <span
      role="status"
      className={`inline-flex items-center gap-2 border-[2.5px] border-dashed border-signal
                  text-signal-ink bg-signal-wash font-display font-extrabold uppercase
                  -rotate-6 select-none animate-stamp-in
                  ${small ? 'text-[10px] tracking-[0.14em] px-2 py-0.5' : 'text-sm tracking-[0.18em] px-3.5 py-1.5'}`}
    >
      <svg viewBox="0 0 12 12" aria-hidden="true" className={small ? 'w-2.5 h-2.5' : 'w-3.5 h-3.5'}>
        <path d="M6 0.8 11.4 10.6H0.6z" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinejoin="round" />
        <path d="M6 4.4v2.6M6 8.5v.6" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
      </svg>
      {label}
    </span>
  )
}
