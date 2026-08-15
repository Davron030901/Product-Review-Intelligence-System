"""Exploratory data analysis.

Written as a script rather than a .ipynb so it runs in CI, diffs cleanly in
git, and cannot be committed with stale output cells. Run it and read the
console + docs/reports/eda_report.md.

    python notebooks/01_eda.py

Convert to a notebook if a mentor wants one:  jupytext --to notebook 01_eda.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.config import DATA_PROCESSED, ISSUE_LABELS, REPORTS_DIR

ISSUE_COLS = [f"issue_{i}" for i in ISSUE_LABELS]


def main():
    path = DATA_PROCESSED / "dataset.csv"
    if not path.exists():
        raise SystemExit("Run `python -m src.data.build_dataset` first.")

    df = pd.read_csv(path)
    df["text"] = df["text"].fillna("")
    lines = ["# Exploratory data analysis", ""]

    # --- shape -------------------------------------------------------------
    lines += [f"**{len(df):,} reviews**, {df['product_id'].nunique():,} products, "
              f"{df['category'].nunique()} categories.", ""]

    # --- class balance -----------------------------------------------------
    sent = df["sentiment"].value_counts(normalize=True).sort_index()
    lines += ["## Sentiment balance", "",
              "| Sentiment | Share | Count |", "|---|---|---|"]
    for label, share in sent.items():
        lines.append(f"| {label} | {share:.1%} | {int(share * len(df)):,} |")
    majority = sent.max()
    lines += ["",
              f"A model that always predicts *{sent.idxmax()}* scores "
              f"**{majority:.1%} accuracy** while being useless. This is why "
              f"macro-F1 leads every report in this project.", ""]

    # --- issue coverage ----------------------------------------------------
    lines += ["## Issue label coverage", "",
              "| Category | Reviews | Share |", "|---|---|---|"]
    for col in ISSUE_COLS:
        n = int(df[col].sum())
        lines.append(f"| {col.replace('issue_', '')} | {n:,} | {n / len(df):.1%} |")

    per_review = df[ISSUE_COLS].sum(axis=1)
    none = int((per_review == 0).sum())
    multi = int((per_review > 1).sum())
    lines += ["",
              f"- No label at all: **{none:,}** ({none / len(df):.1%}) — the "
              "taxonomy's blind spot, or genuinely issue-free reviews.",
              f"- More than one label: **{multi:,}** ({multi / len(df):.1%}) — "
              "the reason the issue task is multi-label rather than multi-class.",
              f"- Mean labels per review: **{per_review.mean():.2f}**", ""]

    # --- length ------------------------------------------------------------
    tokens = df["text"].str.split().str.len()
    lines += ["## Review length", "",
              f"Median **{int(tokens.median())}** words "
              f"(p10 {int(tokens.quantile(.1))}, p90 {int(tokens.quantile(.9))}).", ""]
    short = int((tokens < 3).sum())
    lines += [f"**{short:,}** reviews ({short / len(df):.1%}) are under three "
              "words. These carry almost no signal, which is why the predictor "
              "abstains on them rather than guessing.", ""]

    # --- length vs sentiment ----------------------------------------------
    lines += ["## Length by sentiment", "", "| Sentiment | Median words |",
              "|---|---|"]
    for label, group in df.groupby("sentiment"):
        lines.append(f"| {label} | {int(group['text'].str.split().str.len().median())} |")
    lines += ["",
              "If negative reviews are systematically longer, length becomes a "
              "proxy for sentiment and the model can lean on it instead of "
              "reading the words. Worth checking on real data.", ""]

    # --- per-category ------------------------------------------------------
    lines += ["## Negative rate by category", "",
              "| Category | Reviews | Negative |", "|---|---|---|"]
    for cat, group in df.groupby("category"):
        neg = (group["sentiment"] == "negative").mean()
        lines.append(f"| {cat} | {len(group):,} | {neg:.1%} |")
    lines += ["",
              "Large gaps here mean a single global model will serve some "
              "categories much better than others. See `transfer_report.md`.", ""]

    # --- leakage risks -----------------------------------------------------
    dupes = int(df["text"].duplicated().sum())
    top_product = df["product_id"].value_counts().iloc[0]
    lines += ["## Leakage checks", "",
              f"- Duplicate review texts remaining: **{dupes}** "
              "(exact duplicates removed during build).",
              f"- Largest product by review count: **{top_product}** reviews — "
              "row-level splitting would place these on both sides of the split, "
              "which is why the split is grouped by `product_id`.",
              f"- Star rating is **not** a model feature; it defines the sentiment "
              "label, so using it as input would be circular.", ""]

    # --- takeaways ---------------------------------------------------------
    lines += ["## What this means for modelling", "",
              "1. Report macro-F1, not accuracy — the class balance above makes "
              "accuracy uninformative.",
              "2. Keep the issue task multi-label; a meaningful share of reviews "
              "raise more than one.",
              "3. Keep an abstention path for very short reviews.",
              "4. Split by product, and assert it.",
              "5. Rare categories need per-label reporting; averaged away, they "
              "disappear."]

    out = REPORTS_DIR / "eda_report.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n[eda] wrote {out}")


if __name__ == "__main__":
    main()
