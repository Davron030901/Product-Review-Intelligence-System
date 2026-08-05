# How the issue labels were created

## The problem

The brief asks for issue categories (delivery, packaging, quality, defects, price, service). No public review dataset has them. Public datasets have a star rating and free text. So the labels have to be **created**, and the honest framing is: this project trains a model on labels of its own making, and the quality ceiling of the model is the quality of those labels.

## The taxonomy

`delivery, packaging, quality, defect, price, service, fit, other`

Choices worth defending:

- **quality vs defect are separate.** "The fabric feels thin" is a design/materials complaint the merchandising team owns. "It arrived cracked" is a single-unit failure that logistics or the supplier owns. Merging them would send both to the wrong team.
- **fit** is included because apparel is the likeliest dataset. Drop it for electronics or grocery data.
- **other** means "clearly a complaint, nothing else matched". It is deliberately *not* the same as an empty label set, which means "no issue detected". Conflating them would hide the taxonomy's own blind spots.
- Positive mentions are labelled too. "Shipping was fast" is a delivery signal. The product team wants to know what is working, not only what is broken.

## Pass 1 — keyword lexicon (implemented)

`src/data/weak_labels.py`. Word-boundary regexes per category, so `price` does not fire on `priceless`. Transparent and auditable: you can read exactly why any label was assigned.

Known failure modes, all real:
- **Negation.** "no delivery problems at all" fires `delivery` and looks like a complaint.
- **Sarcasm.** "brilliant, arrived in three pieces" is scored as positive by any lexicon.
- **Vocabulary gaps.** A complaint phrased in words not in the lexicon gets no label, and the model learns that pattern is issue-free.

## Pass 2 — LLM relabelling (recommended, not yet run)

Take a stratified sample (2–5k reviews) and label it with an LLM, few-shot, with the taxonomy definitions above in the prompt. Use it either to replace the lexicon labels on that sample, or to find disagreements worth inspecting. Ask for an explicit "no issue" option so the model can abstain.

Cost control: relabel a sample, not the whole set. Train on lexicon labels, evaluate against LLM labels, and report both numbers.

## Validation — the number that matters

Hand-check a stratified sample of 200–300 reviews. For each, record the true labels and compare against the lexicon.

Report:

| Metric | What it tells you |
|---|---|
| Per-label precision | Of reviews the lexicon called `delivery`, how many really were |
| Per-label recall | Of reviews really about delivery, how many the lexicon caught |
| No-label rate | How much of the data the taxonomy simply misses |
| Cohen's κ vs a second human | Whether the taxonomy is even well-defined enough for two people to agree |

**Until this table is filled with real numbers, every downstream metric is unverified.** If per-label precision is below ~0.7 for a category, the model is being trained to reproduce a mistake, and the right fix is the lexicon, not more epochs.

## What this means for the results

Model metrics measure agreement with the weak labels, not agreement with reality. Two separate numbers must be reported and never conflated:

1. Model vs weak labels (test set) — how well the model learned the labelling function.
2. Weak labels vs human judgement (validation sample) — how well the labelling function reflects the actual task.

Multiplying them gives the rough real-world ceiling.
