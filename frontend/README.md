# Product Review Intelligence — Frontend

React + TypeScript + Tailwind + Recharts. Three screens: analyze one review, see the pattern across all of them, and check the ones the model wasn't sure about.

## Run it

```bash
cd frontend
npm install
npm run dev          # http://localhost:5173
```

It starts in **mock mode** — a local fake backend with the same contract, so the UI is fully usable before the model exists. To talk to the real API:

```bash
# backend running on :8000
echo "VITE_USE_MOCK=false" > .env.local
npm run dev
```

Vite proxies `/api/*` to `http://localhost:8000`, so no CORS setup is needed in development. For a deployed build, set `VITE_API_BASE_URL` to the API origin.

## Design notes

**The idea.** This is a sorting line, not an analytics SaaS. Reviews come in unsorted, get labelled, and anything ambiguous gets physically marked for a person. The interface is built from that: label tags, a stamp, a queue.

**Palette.** Cool paper `#ECEEEA` over a faint sorting grid, near-black ink `#161A17`, deep pine `#23483C` for actions and the nav rail. Eight category hues, one per issue, used identically in tags, charts and KPI accents — so `defect` is the same red everywhere and becomes recognisable without reading.

**Amber is reserved.** `#B87611` appears only on "needs review". Nothing else in the app is allowed to use it, so a flagged item is identifiable at a glance.

**Type.** Bricolage Grotesque for headings (characterful, used sparingly), Public Sans for body, IBM Plex Mono for anything numeric — confidences, timestamps, counts — where tabular alignment matters more than warmth.

**Signature element: the stamp.** A dashed, rotated inspection mark on every low-confidence result. It's the one bold thing; everything around it stays quiet.

**Sentiment never relies on colour alone.** Each state carries a glyph (▼ ■ ▲) and a word, so it survives greyscale and colour-vision differences.

## Responsive behaviour

| | Analyzer | Dashboard | Queue |
|---|---|---|---|
| **< 640px** | Single column, input above result, full-width button | KPIs 2-up, filters stacked, charts reflow, table becomes a card list | Full-width cards, action buttons wrap |
| **640–1024px** | Category field and button share a row | KPIs 2-up, charts full width | Wider cards |
| **> 1024px** | Two columns, result sticky while you edit | Left nav rail, KPIs 4-up, charts side by side, real table | Constrained reading width |

Navigation is a left rail on desktop and a bottom tab bar on mobile, with a live count of items waiting in the queue. Touch targets are at least 44px throughout.

## Accessibility

Visible focus rings on every interactive element, semantic table markup with a caption, `aria-live` on the result region, `role="meter"` on confidence bars, `aria-current` on navigation, and `prefers-reduced-motion` fully respected.

## Structure

```
src/api/         types.ts (the backend contract), client.ts, mockData.ts
src/lib/         categories.ts (the colour/meaning mapping), format.ts
src/components/  AppShell, Stamp, IssueTag, SentimentBadge, ConfidenceBar, StatCard, States
src/screens/     Analyzer, Dashboard, Queue
src/App.tsx      screen routing + the shared review list
```

State lives in `App.tsx` and is passed down. There is no router and no state library — three screens don't need either, and adding them would be the kind of default choice this project is trying to avoid.

## Known gaps

- Analyzed reviews live in memory only; a refresh clears them. Persisting them is a backend concern (a reviews table), not a localStorage hack.
- Queue corrections are stored on the client and not yet sent anywhere. The endpoint to receive them is the obvious next backend task.
- Dark mode is not implemented.
