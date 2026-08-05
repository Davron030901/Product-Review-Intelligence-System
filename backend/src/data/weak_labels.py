"""Weak supervision for the issue taxonomy.

Public review datasets ship a star rating but no issue labels, so we create
them. Pass 1 (this file) is a transparent keyword lexicon. Pass 2 (optional,
see docs/labeling.md) is an LLM relabel of a sample. Every label produced here
is a *hypothesis* that must be validated against a hand-checked sample before
it is trusted -- see scripts/validate_labels.py.

Design notes:
  - Patterns are word-boundary regexes, not substring checks, so "price" does
    not fire on "priceless".
  - A review with no firing pattern gets an EMPTY label set. That is a real
    outcome ("no issue detected"), not a failure.
  - Positive-sentiment reviews still get scanned: "shipping was fast" is a
    delivery mention, and the product team wants those too.
"""
import re
from collections import OrderedDict

from src.config import ISSUE_LABELS

# Ordered so the reported lexicon is stable across runs.
LEXICON = OrderedDict([
    ("delivery", [
        r"deliver\w*", r"shipp?\w*", r"arriv\w*", r"courier", r"postage",
        r"late", r"delay\w*", r"never came", r"tracking", r"dispatch\w*",
        r"took \w+ weeks?", r"took \w+ days?",
    ]),
    ("packaging", [
        r"packag\w*", r"box(?:es)?", r"wrapp?\w*", r"bubble wrap", r"sealed",
        r"unbox\w*", r"crushed", r"torn", r"envelope",
    ]),
    ("quality", [
        r"quality", r"cheap(?:ly)?", r"flims\w*", r"fabric", r"material",
        r"well made", r"poorly made", r"fell apart", r"thin", r"sturdy",
        r"durab\w*", r"craftsmanship",
    ]),
    ("defect", [
        r"broke\w*", r"defect\w*", r"faulty", r"crack\w*", r"damag\w*",
        r"stopped working", r"doesn'?t work", r"does not work", r"malfunction\w*",
        r"missing parts?", r"dead on arrival", r"leak\w*",
    ]),
    ("price", [
        r"price\b", r"pricey", r"overpriced", r"expensive", r"cost\w*",
        r"value for money", r"worth the money", r"not worth", r"cheap for",
        r"refund", r"bargain",
    ]),
    ("service", [
        r"customer (?:service|support|care)", r"support team", r"rep(?:resentative)?s?\b",
        r"unhelpful", r"rude", r"no (?:reply|response)", r"contacted them",
        r"return process", r"warranty", r"complaint",
    ]),
    ("fit", [
        r"\bfits?\b", r"fitting", r"size[ds]?\b", r"sizing", r"true to size", r"length",
        # An intensifier commonly sits between the verb and the size word
        # ("runs very small", "a bit too tight"), so allow one optional word.
        r"too (?:\w+ )?(?:small|large|big|tight|loose|long|short)",
        r"runs (?:\w+ )?(?:small|large|big|tight|narrow|wide)",
    ]),
])

COMPILED = {
    label: [re.compile(rf"\b{p}\b", re.IGNORECASE) for p in patterns]
    for label, patterns in LEXICON.items()
}


def label_review(text: str) -> list:
    """Return the issue labels whose lexicon fires on this review."""
    if not text:
        return []
    hits = [label for label, pats in COMPILED.items() if any(p.search(text) for p in pats)]
    return hits


def label_frame(df, text_col: str = "text"):
    """Add one binary column per issue label to a DataFrame."""
    import pandas as pd  # local import keeps this module importable standalone

    labels = df[text_col].fillna("").map(label_review)
    for issue in ISSUE_LABELS:
        if issue == "other":
            continue
        df[f"issue_{issue}"] = labels.map(lambda hits, i=issue: int(i in hits))
    # "other": a clearly negative review where nothing else fired.
    known = [f"issue_{i}" for i in ISSUE_LABELS if i != "other"]
    nothing_fired = df[known].sum(axis=1) == 0
    negative = df.get("sentiment", pd.Series(["neutral"] * len(df))) == "negative"
    df["issue_other"] = (nothing_fired & negative).astype(int)
    return df


def coverage_report(df):
    """How often each label fires -- the first sanity check on the lexicon."""
    cols = [f"issue_{i}" for i in ISSUE_LABELS]
    total = len(df)
    rows = []
    for c in cols:
        n = int(df[c].sum())
        rows.append({"label": c.replace("issue_", ""), "count": n,
                     "share": round(n / max(total, 1), 4)})
    unlabelled = int((df[cols].sum(axis=1) == 0).sum())
    return rows, {"total": total, "no_label": unlabelled,
                  "no_label_share": round(unlabelled / max(total, 1), 4)}
