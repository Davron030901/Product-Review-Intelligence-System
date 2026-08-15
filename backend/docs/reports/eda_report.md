# Exploratory data analysis

**3,440 reviews**, 500 products, 5 categories.

## Sentiment balance

| Sentiment | Share | Count |
|---|---|---|
| negative | 79.1% | 2,720 |
| neutral | 0.1% | 2 |
| positive | 20.9% | 718 |

A model that always predicts *negative* scores **79.1% accuracy** while being useless. This is why macro-F1 leads every report in this project.

## Issue label coverage

| Category | Reviews | Share |
|---|---|---|
| delivery | 1,237 | 36.0% |
| packaging | 1,068 | 31.0% |
| quality | 913 | 26.5% |
| defect | 925 | 26.9% |
| price | 919 | 26.7% |
| service | 730 | 21.2% |
| fit | 986 | 28.7% |
| other | 0 | 0.0% |

- No label at all: **37** (1.1%) — the taxonomy's blind spot, or genuinely issue-free reviews.
- More than one label: **2,435** (70.8%) — the reason the issue task is multi-label rather than multi-class.
- Mean labels per review: **1.97**

## Review length

Median **12** words (p10 7, p90 17).

**10** reviews (0.3%) are under three words. These carry almost no signal, which is why the predictor abstains on them rather than guessing.

## Length by sentiment

| Sentiment | Median words |
|---|---|
| negative | 13 |
| neutral | 1 |
| positive | 9 |

If negative reviews are systematically longer, length becomes a proxy for sentiment and the model can lean on it instead of reading the words. Worth checking on real data.

## Negative rate by category

| Category | Reviews | Negative |
|---|---|---|
| Beauty | 701 | 77.6% |
| Dresses | 704 | 77.8% |
| Electronics | 665 | 80.5% |
| Home | 670 | 81.2% |
| Tops | 700 | 78.4% |

Large gaps here mean a single global model will serve some categories much better than others. See `transfer_report.md`.

## Leakage checks

- Duplicate review texts remaining: **0** (exact duplicates removed during build).
- Largest product by review count: **18** reviews — row-level splitting would place these on both sides of the split, which is why the split is grouped by `product_id`.
- Star rating is **not** a model feature; it defines the sentiment label, so using it as input would be circular.

## What this means for modelling

1. Report macro-F1, not accuracy — the class balance above makes accuracy uninformative.
2. Keep the issue task multi-label; a meaningful share of reviews raise more than one.
3. Keep an abstention path for very short reviews.
4. Split by product, and assert it.
5. Rare categories need per-label reporting; averaged away, they disappear.
