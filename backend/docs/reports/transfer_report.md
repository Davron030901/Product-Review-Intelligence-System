# Cross-category transfer

> **These numbers measure nothing.** The dataset is the synthetic bootstrap sample, where product categories are assigned at random and carry no vocabulary of their own. With no real domain shift to detect, a drop of ~0.000 is the expected output of a working script, not evidence that the model transfers well. Rerun on a real multi-category dataset before citing anything below.

Trained on every category except one, evaluated on the unseen one.
Positive drop = the model performed worse on the category it had never seen.

| Held-out category | Rows | Sentiment macro-F1 (in → out) | Issue macro-F1 (in → out) | Issue drop |
|---|---|---|---|---|
| Beauty | 701 | 0.651 → 0.657 | 1.000 → 1.000 | +0.000 |
| Dresses | 704 | 0.645 → 0.645 | 1.000 → 1.000 | +0.000 |
| Electronics | 665 | 0.655 → 0.812 | 1.000 → 1.000 | +0.001 |
| Home | 670 | 0.652 → 0.986 | 1.000 → 1.000 | +0.000 |
| Tops | 700 | 0.651 → 0.653 | 1.000 → 1.000 | +0.000 |

## Summary

- Categories evaluated: **5**
- Mean issue macro-F1 drop: **+0.000**
- Mean sentiment macro-F1 drop: **-0.100**
- Transfers worst to: **Electronics**

## What this means

Two effects are bundled into the drop and cannot be separated by this experiment alone. First, genuine domain shift: complaint vocabulary differs between categories, and some labels (`fit`) barely exist outside apparel. Second, label noise: the keyword lexicon that generated the ground truth is domain-bound too, so the held-out labels are themselves less trustworthy.

The practical reading: **retrain per category domain rather than assuming one model serves all of them**, and treat any deployment into an unseen category as unvalidated until this table is rerun with that category held out.

Language transfer is a separate and worse problem, not measured here: training is English-only and non-English input is flagged rather than scored. See `model_card.md`.
