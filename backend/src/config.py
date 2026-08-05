"""Central configuration: paths, taxonomy, thresholds.

Everything the pipeline needs to agree on lives here so training and
inference cannot drift apart.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "docs" / "reports"

for _d in (DATA_RAW, DATA_PROCESSED, MODELS_DIR, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- Label space -----------------------------------------------------------
SENTIMENT_LABELS = ["negative", "neutral", "positive"]

# Issue taxonomy. Multi-label: one review may raise several of these.
# "other" is the catch-all for text that is clearly a complaint but does not
# map to a known bucket; it is NOT the same as "no issue found".
ISSUE_LABELS = [
    "delivery",
    "packaging",
    "quality",
    "defect",
    "price",
    "service",
    "fit",
    "other",
]

# --- Star rating -> sentiment mapping --------------------------------------
def rating_to_sentiment(rating: float) -> str:
    """Map a star rating onto a sentiment band.

    Uses ranges rather than equality on 3: some datasets store averaged or
    half-star ratings, and `rating == 3` would let 2.5 fall through to
    "positive" -- silently mislabelling the most ambiguous reviews in the set
    as the majority class.
    """
    if rating <= 2:
        return "negative"
    if rating < 4:
        return "neutral"
    return "positive"


# --- Decision thresholds ---------------------------------------------------
# Probability at or above which an issue label is emitted.
ISSUE_THRESHOLD = 0.35
# Sentiment confidence below which the whole result is flagged for a human.
SENTIMENT_LOW_CONFIDENCE = 0.55
# Reviews shorter than this (in tokens) are treated as too thin to trust.
MIN_TOKENS_FOR_CONFIDENCE = 3

MODEL_VERSION = "v1-baseline"
RANDOM_SEED = 42
