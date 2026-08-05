"""Build the modelling table.

Two sources, same output schema:

  1. A real public dataset placed at data/raw/reviews.csv. The expected
     columns are configurable below; the default matches the Women's
     E-Commerce Clothing Reviews file from Kaggle.
  2. A bootstrap sample generated locally, so the whole pipeline is runnable
     the moment the repo is cloned. It is clearly marked and MUST NOT be
     reported as a result -- it exists to prove the plumbing works.

Output: data/processed/dataset.csv with columns
    review_id, product_id, category, text, rating, sentiment, issue_*
"""
import argparse
import json
import random

import pandas as pd

from src.config import (DATA_PROCESSED, DATA_RAW, ISSUE_LABELS, RANDOM_SEED,
                        rating_to_sentiment)
from src.data.preprocess import clean_text, token_count
from src.data.weak_labels import coverage_report, label_frame

# Column mapping for the Kaggle clothing-reviews file. Override with --text-col
# etc. if you picked a different dataset.
DEFAULT_MAP = {
    "text": "Review Text",
    "rating": "Rating",
    "product_id": "Clothing ID",
    "category": "Department Name",
}

# --- bootstrap sample generator -------------------------------------------
_PHRASES = {
    "delivery": ["arrived four days late", "shipping was quick", "the courier never showed up",
                 "delivery took three weeks", "dispatch was same day"],
    "packaging": ["the box was crushed", "packaging was minimal", "beautifully wrapped",
                  "arrived in a torn envelope", "sealed properly"],
    "quality": ["the fabric feels cheap", "quality is excellent", "it fell apart after one wash",
                "well made and sturdy", "the material is very thin"],
    "defect": ["it arrived broken", "the zipper is faulty", "stopped working on day two",
               "there was a crack in the base", "missing parts in the box"],
    "price": ["far too expensive for what it is", "great value for money", "overpriced honestly",
              "worth the money", "not worth the price"],
    "service": ["customer service was rude", "support sorted it out fast", "no reply to my complaint",
                "the return process was painless", "the rep was unhelpful"],
    "fit": ["runs very small", "true to size", "too tight across the shoulders",
            "the length is perfect", "sizing is inconsistent"],
}
_OPENERS = ["", "Honestly, ", "Bought this last month. ", "First impression: ", "Second time ordering. "]
_CLOSERS = ["", " Would not order again.", " Very happy overall.", " Mixed feelings.",
            " Returning it.", " Recommend it."]
_THIN = ["meh", "ok", "fine", "good", "...", "no comment", "5 stars"]


def _generate_sample(n: int, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = random.Random(seed)
    cats = ["Tops", "Dresses", "Home", "Electronics", "Beauty"]
    rows = []
    for i in range(n):
        if rng.random() < 0.07:                      # thin / ambiguous reviews
            text, rating = rng.choice(_THIN), rng.choice([3, 4, 5])
        else:
            k = rng.choices([1, 2, 3], weights=[0.55, 0.33, 0.12])[0]
            topics = rng.sample(list(_PHRASES), k)
            parts = [rng.choice(_PHRASES[t]) for t in topics]
            text = rng.choice(_OPENERS) + ", ".join(parts).capitalize() + "." + rng.choice(_CLOSERS)
            negative_words = ("late", "never", "crushed", "torn", "cheap", "fell apart", "thin",
                              "broken", "faulty", "stopped", "crack", "missing", "expensive",
                              "overpriced", "not worth", "rude", "no reply", "unhelpful",
                              "small", "tight", "inconsistent")
            neg = sum(w in text.lower() for w in negative_words)
            rating = 1 if neg >= 2 else (2 if neg == 1 else rng.choice([4, 5, 5]))
        rows.append({
            "review_id": f"r{i:06d}",
            "product_id": f"p{rng.randint(1, max(n // 12, 2)):05d}",
            "category": rng.choice(cats),
            "text": text,
            "rating": rating,
        })
    return pd.DataFrame(rows)


def _load_real(path, colmap) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in colmap.values() if c not in df.columns]
    if missing:
        raise SystemExit(
            f"Columns {missing} not found in {path}.\n"
            f"Available: {list(df.columns)}\n"
            "Pass the right names with --text-col / --rating-col / --category-col."
        )
    out = pd.DataFrame({
        "review_id": [f"r{i:06d}" for i in range(len(df))],
        "product_id": df[colmap["product_id"]].astype(str) if colmap["product_id"] in df else "unknown",
        "category": df[colmap["category"]].astype(str) if colmap["category"] in df else "unknown",
        "text": df[colmap["text"]],
        "rating": pd.to_numeric(df[colmap["rating"]], errors="coerce"),
    })
    return out


def build(sample_size: int = 6000, colmap: dict = None) -> pd.DataFrame:
    colmap = colmap or DEFAULT_MAP
    raw_path = DATA_RAW / "reviews.csv"
    if raw_path.exists():
        print(f"[data] using real dataset at {raw_path}")
        df = _load_real(raw_path, colmap)
        source = str(raw_path)
    else:
        print(f"[data] no {raw_path} found -- generating a {sample_size}-row bootstrap sample.")
        print("[data] WARNING: bootstrap data is for plumbing only. Do not report metrics from it.")
        df = _generate_sample(sample_size)
        source = "bootstrap-sample"

    before = len(df)
    df["text"] = df["text"].map(clean_text)
    df = df[df["text"].str.len() > 0]
    df = df.dropna(subset=["rating"])
    df = df.drop_duplicates(subset=["text"])          # exact-duplicate removal
    df["n_tokens"] = df["text"].map(token_count)
    df["sentiment"] = df["rating"].map(rating_to_sentiment)
    df = label_frame(df, "text")

    rows, summary = coverage_report(df)
    meta = {
        "source": source,
        "rows_in": before,
        "rows_out": len(df),
        "dropped": before - len(df),
        "sentiment_distribution": df["sentiment"].value_counts(normalize=True).round(4).to_dict(),
        "issue_coverage": rows,
        "issue_summary": summary,
        "median_tokens": int(df["n_tokens"].median()),
    }
    out_path = DATA_PROCESSED / "dataset.csv"
    df.to_csv(out_path, index=False)
    (DATA_PROCESSED / "dataset_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[data] wrote {len(df)} rows -> {out_path}")
    print(json.dumps(meta, indent=2))
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build the processed review dataset.")
    ap.add_argument("--sample-size", type=int, default=6000)
    ap.add_argument("--text-col", default=DEFAULT_MAP["text"])
    ap.add_argument("--rating-col", default=DEFAULT_MAP["rating"])
    ap.add_argument("--product-col", default=DEFAULT_MAP["product_id"])
    ap.add_argument("--category-col", default=DEFAULT_MAP["category"])
    a = ap.parse_args()
    build(a.sample_size, {"text": a.text_col, "rating": a.rating_col,
                          "product_id": a.product_col, "category": a.category_col})
