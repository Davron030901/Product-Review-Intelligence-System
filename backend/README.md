# Product Review Intelligence — Backend

Turns a raw product review into structured signal: sentiment, issue categories, and an honest confidence flag.

```
POST /analyze  {"text": "Arrived two weeks late and the box was crushed."}

{
  "sentiment": {"label": "negative", "confidence": 0.95},
  "issues": [{"category": "delivery", "confidence": 0.91},
             {"category": "packaging", "confidence": 0.74}],
  "low_confidence": false,
  "reasons": [],
  "model_version": "v1-baseline",
  "processed_at": "2026-07-31T09:12:44+00:00"
}
```

## Run it

```bash
cd backend
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m src.data.build_dataset      # builds data/processed/dataset.csv
python -m src.training.train_baseline # trains + evaluates, writes docs/reports/
python -m pytest tests -q
uvicorn src.api.main:app --reload --port 8000
```

Open http://localhost:8000/docs for the interactive API.

With Docker: `docker compose up --build` (train the model first — the image copies `models/`).

## Using a real dataset

Out of the box `build_dataset` generates a **bootstrap sample** so the pipeline runs on a fresh clone. It is templated text. **Never report metrics from it** — they will look near-perfect and mean nothing.

To use real data, download one of these and save it as `data/raw/reviews.csv`:

- **Women's E-Commerce Clothing Reviews** (Kaggle) — ~23k rows, clean, single domain. Column names already match the defaults.
- **Amazon Reviews** (e.g. Amazon Fine Food Reviews, or the McAuley Lab 2023 release) — much larger and multi-category, so it can answer the cross-category transfer question. Noisier.

Then:
```bash
python -m src.data.build_dataset --text-col "Review Text" --rating-col Rating \
    --product-col "Clothing ID" --category-col "Department Name"
```
Check the dataset license before committing anything to a public repo.

## Optional: train the transformer

```bash
pip install -r requirements-ml.txt
python -m src.training.train_transformer --epochs 3 --batch-size 16
```
One shared DistilBERT encoder, two heads (softmax sentiment + sigmoid multi-label issues). The API picks it up automatically if `models/transformer/model.pt` exists, and falls back to the baseline if it fails to load. **Compare it against `docs/reports/baseline_report.json` before claiming it is better** — on short review text TF-IDF is a serious opponent.

## Layout

```
src/config.py            taxonomy, thresholds, paths — the single source of truth
src/data/preprocess.py   cleaning + input-quality flags
src/data/weak_labels.py  the keyword lexicon that creates issue labels
src/data/build_dataset.py real CSV or bootstrap sample -> processed table
src/data/splits.py       group split by product + leakage assertions
src/training/            baseline (sklearn) and transformer (torch)
src/training/evaluate.py macro-first metrics, rare labels flagged
src/inference/predictor.py the JSON contract, plus the "I don't know" rules
src/api/                 FastAPI app + pydantic schemas
docs/                    labeling methodology, model card, filled brief tables
```

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| POST | `/analyze` | One review → structured result |
| POST | `/analyze/batch` | Up to 500 reviews → list of results |
| GET | `/taxonomy` | The label space, so the frontend doesn't hardcode it |
| GET | `/health` | Liveness + which model is loaded |

## Two things worth knowing before you read the metrics

1. **The issue labels are created, not given.** No public review dataset ships them. See `docs/labeling.md` for how they were derived and how well they hold up against a hand-checked sample. Every issue metric is only as good as that number.
2. **The model is allowed to abstain.** Empty, three-word, or non-English input returns `low_confidence: true` with a reason instead of a fabricated answer. That is a feature — the frontend routes these to a human review queue.

Known limitations are in `docs/model_card.md`.
