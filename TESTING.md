# Testing

328 tests across three layers: 210 backend (pytest), 108 frontend (vitest), 10 end-to-end (Playwright).

## Running them

```bash
# Backend — 210 tests, ~15s
cd backend
python -m pytest tests -q
python -m pytest tests -q --cov=src --cov-report=term-missing   # with coverage

# Frontend — 108 tests, ~10s
cd frontend
npm test
```

End-to-end needs both services up:

```bash
# terminal 1
cd backend && PYTHONPATH=$PWD ALLOWED_ORIGINS=http://localhost:4173 \
    uvicorn src.api.main:app --port 8000

# terminal 2
cd frontend && VITE_USE_MOCK=false VITE_API_BASE_URL=http://localhost:8000 \
    npm run build && npx vite preview --port 4173

# terminal 3
pip install playwright && playwright install chromium
python e2e/test_e2e.py
```

## What is covered

### Backend (210)

| File | Tests | Covers |
|---|---|---|
| `test_config.py` | 13 | Rating→sentiment mapping, taxonomy invariants, threshold ranges |
| `test_preprocess.py` | 8 | Cleaning, token counting, input-quality flags |
| `test_weak_labels.py` | 19 | Every category's lexicon fires, word boundaries, `other` logic, coverage report |
| `test_splits.py` | 16 | Grouped splitting, leakage detection, category holdout, determinism |
| `test_evaluate.py` | 16 | Metrics against hand-computed values, rare-label handling, JSON safety |
| `test_build_dataset.py` | 20 | Sample generation, real-CSV loading, dedup, metadata consistency |
| `test_train_baseline.py` | 18 | Full training run, artefact shape, report contents, leakage abort, reproducibility |
| `test_predictor.py` | 37 | Contract shape, abstention rules, hostile input, batch consistency |
| `test_api.py` | 51 | Every endpoint, validation, CORS, edge cases, OpenAPI |
| `test_failure_modes.py` | 12 | 503 with no model, safe 500s, corrupt artefacts, transformer fallback |

Coverage is 75%. The uncovered remainder is almost entirely `train_transformer.py`, which needs `torch` — it is exercised only to the extent that it imports cleanly and fails with install instructions when torch is absent.

### Frontend (108)

| File | Tests | Covers |
|---|---|---|
| `lib.test.ts` | 18 | Formatting helpers, category colour/meaning maps, distinct sentiment glyphs |
| `api.test.ts` | 27 | Mock backend contract fidelity, URL normalisation, error mapping, timeout, batch unwrapping |
| `components.test.tsx` | 28 | Every component, plus ARIA roles, meter values, nav landmarks |
| `screens.test.tsx` | 26 | Analyzer / Dashboard / Queue behaviour, empty and filtered states |
| `app.test.tsx` | 9 | Cross-screen state: analyze → dashboard → queue → resolved |

### End-to-end (10)

Real backend, real production build, real browser: analysis against the live API, confirmation the served model is not the mock, low-confidence stamping, chart rendering, filters, queue resolution, keyboard navigation, no horizontal overflow at 360px, and 44px touch targets.

## Bugs these tests caught

Four real defects, all fixed:

1. **Fractional star ratings were mislabelled.** `rating_to_sentiment` used `rating == 3` for neutral, so a 2.5-star review fell through to **positive**. Datasets with averaged or half-star ratings would have had their most ambiguous reviews silently labelled as the majority class. Now uses ranges.

2. **The `fit` lexicon missed its most common phrasings.** `runs (?:small|large|big)` did not match "runs very small" or "a bit too tight" — an intensifier between the verb and the size word defeated it. Now allows one optional word.

3. **Macro-F1 was averaged over the wrong class set.** sklearn defaults to averaging only over classes present in the data, so a class missing from a test fold quietly inflated the score — precisely when you most want to notice. Now passes `labels=classes` explicitly.

4. **Average precision reported 0.0 for unmeasurable labels.** A label with no positive examples got `0.0`, which reads as "the model failed" rather than "there was nothing to measure". Now returns `None`.

A fifth issue was found and fixed earlier: `seedHistory` awaited 44 artificially delayed calls in sequence, stalling first paint for roughly 18 seconds.

## What is deliberately not tested

- **Transformer training.** Requires `torch` and a GPU to be meaningful. Only the import guard is tested.
- **Label quality.** No test can tell you whether the weak labels match reality — that needs the hand-checked sample described in `backend/docs/labeling.md`. The test suite verifies the labelling *code* is correct, not that the labelling *scheme* is right. These are different claims and the distinction matters when presenting results.
- **Deployment.** Render and Vercel behaviour is verified by the checklist in `DEPLOY.md`, not automatically.

## Known non-issues

The browser console shows a 403 for Google Fonts in sandboxed environments without external network access. Font fallbacks are declared in `tailwind.config.js`, so the layout degrades gracefully.
