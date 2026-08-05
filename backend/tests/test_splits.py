"""Every reported metric is only meaningful if the split is clean, so these
tests treat leakage as a correctness bug rather than a quality concern."""
import pandas as pd
import pytest

from src.data.splits import (group_split, holdout_category, label_distribution,
                             leakage_check)


def make_df(n_products=40, per_product=5):
    rows = []
    cats = ["Tops", "Home", "Electronics"]
    for p in range(n_products):
        for r in range(per_product):
            rows.append({
                "product_id": f"p{p:03d}",
                "category": cats[p % len(cats)],
                "text": f"review {p}-{r} arrived late",
                "sentiment": ["negative", "neutral", "positive"][r % 3],
                "issue_delivery": r % 2,
                "issue_quality": (r + 1) % 2,
            })
    return pd.DataFrame(rows)


# --- group_split -----------------------------------------------------------

def test_split_returns_three_disjoint_folds():
    df = make_df()
    train, val, test = group_split(df)
    assert len(train) + len(val) + len(test) == len(df)


def test_no_product_appears_in_more_than_one_fold():
    """The whole point of the grouped split."""
    train, val, test = group_split(make_df())
    for a, b in ((train, val), (train, test), (val, test)):
        assert set(a["product_id"]).isdisjoint(set(b["product_id"]))


def test_split_is_deterministic_for_a_given_seed():
    df = make_df()
    a = group_split(df, seed=7)[0]["product_id"].tolist()
    b = group_split(df, seed=7)[0]["product_id"].tolist()
    assert a == b


def test_different_seeds_give_different_splits():
    df = make_df()
    a = set(group_split(df, seed=1)[2]["product_id"])
    b = set(group_split(df, seed=99)[2]["product_id"])
    assert a != b


def test_fold_sizes_are_roughly_the_requested_proportions():
    df = make_df(n_products=200, per_product=3)
    train, val, test = group_split(df, val_size=0.15, test_size=0.15)
    n = len(df)
    assert 0.60 <= len(train) / n <= 0.78
    assert 0.08 <= len(val) / n <= 0.22
    assert 0.08 <= len(test) / n <= 0.22


def test_returned_frames_have_a_clean_index():
    """Downstream code indexes positionally; a stale index causes silent misalignment."""
    for fold in group_split(make_df()):
        assert fold.index.tolist() == list(range(len(fold)))


def test_split_preserves_all_columns():
    df = make_df()
    train, _, _ = group_split(df)
    assert list(train.columns) == list(df.columns)


def test_single_review_per_product_still_splits():
    df = make_df(n_products=60, per_product=1)
    train, val, test = group_split(df)
    assert min(len(train), len(val), len(test)) > 0


# --- leakage_check ---------------------------------------------------------

def test_leakage_check_passes_on_a_clean_split():
    train, _, test = group_split(make_df())
    assert leakage_check(train, test)["clean"] is True


def test_leakage_check_catches_a_shared_product():
    df = make_df()
    train, _, test = group_split(df)
    dirty = pd.concat([test, train.head(3)], ignore_index=True)
    result = leakage_check(train, dirty)
    assert result["clean"] is False
    assert result["shared_products"] > 0


def test_leakage_check_catches_duplicated_text_across_folds():
    """Same text under a different product id is still leakage."""
    train, _, test = group_split(make_df())
    copied = train.head(2).copy()
    copied["product_id"] = "p999"
    dirty = pd.concat([test, copied], ignore_index=True)
    result = leakage_check(train, dirty)
    assert result["shared_exact_texts"] > 0
    assert result["clean"] is False


def test_leakage_check_reports_row_counts():
    train, _, test = group_split(make_df())
    r = leakage_check(train, test)
    assert r["train_rows"] == len(train)
    assert r["test_rows"] == len(test)


# --- holdout_category ------------------------------------------------------

def test_holdout_category_separates_one_category_entirely():
    rest, held = holdout_category(make_df(), "Home")
    assert set(held["category"]) == {"Home"}
    assert "Home" not in set(rest["category"])
    assert len(rest) + len(held) == len(make_df())


def test_holdout_category_is_case_insensitive():
    _, held = holdout_category(make_df(), "hOmE")
    assert len(held) > 0


def test_holdout_category_rejects_an_unknown_category():
    with pytest.raises(ValueError):
        holdout_category(make_df(), "Groceries")


# --- label_distribution ----------------------------------------------------

def test_label_distribution_counts_positives_per_label():
    df = make_df()
    dist = label_distribution(df, ["issue_delivery", "issue_quality"])
    assert dist["delivery"] == int(df["issue_delivery"].sum())
    assert dist["quality"] == int(df["issue_quality"].sum())
