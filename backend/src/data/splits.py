"""Train/val/test splitting with leakage control.

Reviews of the same product share vocabulary, seller behaviour and often
near-identical complaints. Splitting at the row level lets the model memorise
a product and score well for the wrong reason, so we split by product_id
(GroupShuffleSplit). One product's reviews land entirely in one fold.

Optionally, one whole product category is held out as a transfer set, which
is what answers the brief's "how well does this move to another category?"
question with a number instead of an opinion.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src.config import RANDOM_SEED


def group_split(df: pd.DataFrame, group_col: str = "product_id",
                val_size: float = 0.15, test_size: float = 0.15,
                seed: int = RANDOM_SEED):
    """Return (train, val, test) with no product appearing in two folds."""
    groups = df[group_col].astype(str).values

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    rest_idx, test_idx = next(gss.split(df, groups=groups))
    rest = df.iloc[rest_idx].reset_index(drop=True)
    test = df.iloc[test_idx].reset_index(drop=True)

    rel_val = val_size / (1 - test_size)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=rel_val, random_state=seed)
    tr_idx, va_idx = next(gss2.split(rest, groups=rest[group_col].astype(str).values))
    train = rest.iloc[tr_idx].reset_index(drop=True)
    val = rest.iloc[va_idx].reset_index(drop=True)
    return train, val, test


def holdout_category(df: pd.DataFrame, category: str, category_col: str = "category"):
    """Split off one entire category as a transfer-test set."""
    mask = df[category_col].astype(str).str.lower() == str(category).lower()
    if mask.sum() == 0:
        raise ValueError(f"category {category!r} not present in {category_col}")
    return df[~mask].reset_index(drop=True), df[mask].reset_index(drop=True)


def leakage_check(train: pd.DataFrame, test: pd.DataFrame,
                  group_col: str = "product_id", text_col: str = "text") -> dict:
    """Assert the split did what we think it did."""
    shared_groups = set(train[group_col]) & set(test[group_col])
    shared_text = set(train[text_col]) & set(test[text_col])
    return {
        "shared_products": len(shared_groups),
        "shared_exact_texts": len(shared_text),
        "train_rows": len(train),
        "test_rows": len(test),
        "clean": len(shared_groups) == 0 and len(shared_text) == 0,
    }


def label_distribution(df: pd.DataFrame, label_cols) -> dict:
    return {c.replace("issue_", ""): int(df[c].sum()) for c in label_cols}
