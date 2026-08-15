# Model comparison

All runs below trained on the same data (`f2684cbfe4af8666`, 3440 rows). 

| Run | Model | Sentiment macro-F1 | Issue macro-F1 | Issue micro-F1 |
|---|---|---|---|---|
| `b702a59af728` | baseline-tfidf-logreg | 0.6513 | 1.0000 | 1.0000 |

## Verdict

Only the baseline has been trained. Run `python -m src.training.train_transformer` to compare, or deploy the baseline and say so plainly.

## Cost, not just accuracy

| | TF-IDF + LogReg | DistilBERT multi-task |
|---|---|---|
| Training time | seconds, CPU | minutes to hours, GPU preferred |
| Model size | ~1 MB | ~250 MB |
| Inference | sub-millisecond | tens of ms on CPU |
| Free-tier hosting | comfortable | usually will not fit |
| Interpretability | inspect the coefficients | opaque |

A model is only better if it is better *after* these are counted. On short review text with keyword-generated labels, the baseline is a genuinely strong opponent: much of the signal is lexical, which is exactly what TF-IDF captures.

## Honest framing

Both models are scored against weak labels, so this table compares how well each reproduces a keyword lexicon — not which understands customers better. A transformer's real advantage (context, negation, sarcasm) is largely invisible to labels that were themselves produced by keyword matching. Measuring that advantage needs the human-checked set from `validate_labels.py` as the evaluation ground truth.
