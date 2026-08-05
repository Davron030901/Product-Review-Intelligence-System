# Model card — Product Review Intelligence

## What it does
Takes English product review text, returns a sentiment label (negative / neutral / positive) and zero or more issue categories with confidence scores. Multi-label: one review can raise several issues.

## Intended use
A prototype decision-support tool for product and customer-experience teams: surfacing recurring themes across many reviews, and routing uncertain cases to a person. Student-scale capstone, not a production system.

## Out of scope
- Automatically acting on a review (issuing refunds, hiding listings, penalising sellers) without a human in the loop.
- Judging individual customers or sellers.
- Any language other than English.
- Any claim of accuracy beyond what `docs/reports/` actually shows.

## Inputs
Review text only. **Star rating is deliberately not a feature** — it is used to derive the sentiment label during training, and a review arriving at inference time may not have one. Feeding it in would be leakage and would inflate every metric.

Product category is accepted and echoed back but is not used by the model.

## Training data
Configurable. Default bootstrap sample is synthetic and for plumbing only. Real runs use a public review dataset (see README). Reviewer names and profile fields are dropped — the system needs the text, not the identity.

## Labels
Sentiment derived from star rating (1–2 negative, 3 neutral, 4–5 positive). Issue labels created by weak supervision — see `labeling.md`. **This is the largest single source of error in the system.**

## Evaluation
Grouped train/val/test split by `product_id`, so no product's reviews appear in two folds; the split is asserted, and training refuses to start if the assertion fails. Threshold selected on validation, metrics reported on test. Macro-F1 leads, per-label metrics reported, and labels with fewer than 25 positives are flagged rare and excluded from the headline macro figure — their scores are too noisy to act on.

## Known failure modes
- **Sarcasm and irony** are scored at face value.
- **Negation** ("no problems with delivery") can trigger a false issue label, inherited from the lexicon.
- **Very short reviews** ("meh", "fine") carry almost no signal; these are flagged low-confidence rather than guessed.
- **Non-English text** is detected only by a crude Latin-script heuristic, which will not distinguish English from Spanish or Uzbek. Predictions on non-English input are unreliable and flagged.
- **Domain shift.** A model trained on clothing reviews will not transfer cleanly to electronics: `fit` becomes meaningless, `defect` vocabulary changes entirely. Hold out a whole category (`holdout_category` in `src/data/splits.py`) to measure this instead of assuming it.
- **Class imbalance.** Positive reviews dominate, so accuracy is misleading and is never reported alone.

## Fairness and privacy
- No demographic attributes are used or inferred.
- Reviewer identifiers are dropped at ingestion.
- Worth checking and reporting: does per-label F1 vary systematically by product category or by review length? A model that works well on long reviews and poorly on short ones will systematically under-serve the customers who write briefly.
- The output should never be presented to a product team as objective truth about a seller. It is an aggregation aid with a measurable error rate.

## Abstention
The system returns `low_confidence: true` with human-readable reasons when input is empty, very short, non-Latin script, when sentiment confidence is below threshold, or when no issue clears the threshold. These are routed to the frontend's review queue. An honest "I don't know" is a correct answer.
