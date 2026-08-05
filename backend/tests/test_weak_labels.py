"""The lexicon creates the issue labels the whole model learns from, so a
bug here is not a bug in a feature -- it is corrupted ground truth."""
import pandas as pd

from src.config import ISSUE_LABELS
from src.data.weak_labels import (COMPILED, LEXICON, coverage_report,
                                  label_frame, label_review)


# --- label_review ----------------------------------------------------------

def test_each_category_has_at_least_one_firing_example():
    """Guards against a regex typo silently disabling a whole category."""
    examples = {
        "delivery": "the parcel arrived late",
        "packaging": "the box was crushed",
        "quality": "the fabric feels cheap",
        "defect": "it arrived broken",
        "price": "far too expensive",
        "service": "customer service was rude",
        "fit": "runs very small",
    }
    for category, text in examples.items():
        assert category in label_review(text), f"{category} lexicon did not fire"


def test_multiple_categories_fire_together():
    hits = label_review("Arrived late, the box was crushed and the zipper is faulty")
    assert {"delivery", "packaging", "defect"} <= set(hits)


def test_no_labels_when_nothing_matches():
    assert label_review("hello there, thanks") == []


def test_empty_and_none_are_safe():
    assert label_review("") == []
    assert label_review(None) == []


def test_matching_is_case_insensitive():
    assert "delivery" in label_review("ARRIVED LATE")


def test_word_boundaries_prevent_substring_false_positives():
    assert "price" not in label_review("this gift is priceless")


def test_labels_returned_are_all_in_the_declared_taxonomy():
    hits = label_review("late crushed cheap broken expensive rude too small")
    assert set(hits) <= set(ISSUE_LABELS)


def test_lexicon_and_compiled_stay_in_sync():
    assert set(LEXICON) == set(COMPILED)


def test_known_limitation_negation_is_not_handled():
    """Documented in docs/labeling.md. Pinned so it fails loudly if someone
    'fixes' negation without updating the documentation."""
    assert "delivery" in label_review("no delivery problems at all")


# --- label_frame -----------------------------------------------------------

def make_df():
    return pd.DataFrame({
        "text": [
            "arrived late and the box was crushed",
            "love it, great quality",
            "terrible, nothing works",
            "",
        ],
        "sentiment": ["negative", "positive", "negative", "neutral"],
    })


def test_label_frame_adds_a_column_for_every_issue():
    df = label_frame(make_df())
    for issue in ISSUE_LABELS:
        assert f"issue_{issue}" in df.columns


def test_label_frame_columns_are_binary_ints():
    df = label_frame(make_df())
    for issue in ISSUE_LABELS:
        assert set(df[f"issue_{issue}"].unique()) <= {0, 1}


def test_label_frame_marks_the_expected_rows():
    df = label_frame(make_df())
    assert df.loc[0, "issue_delivery"] == 1
    assert df.loc[0, "issue_packaging"] == 1
    assert df.loc[1, "issue_quality"] == 1


def test_other_fires_only_for_negative_reviews_with_no_other_label():
    df = label_frame(make_df())
    # row 2: negative, no lexicon hit -> "other"
    assert df.loc[2, "issue_other"] == 1
    # row 0: negative but delivery/packaging fired -> not "other"
    assert df.loc[0, "issue_other"] == 0
    # row 3: empty and neutral -> not "other"
    assert df.loc[3, "issue_other"] == 0


def test_other_is_not_assigned_to_positive_reviews():
    df = label_frame(pd.DataFrame({
        "text": ["absolutely wonderful"], "sentiment": ["positive"],
    }))
    assert df.loc[0, "issue_other"] == 0


def test_label_frame_without_a_sentiment_column_does_not_crash():
    """build_dataset always supplies sentiment, but the fallback path exists."""
    df = label_frame(pd.DataFrame({"text": ["arrived late"]}))
    assert df.loc[0, "issue_delivery"] == 1


# --- coverage_report -------------------------------------------------------

def test_coverage_report_counts_match_the_columns():
    df = label_frame(make_df())
    rows, summary = coverage_report(df)
    by_label = {r["label"]: r["count"] for r in rows}
    assert by_label["delivery"] == int(df["issue_delivery"].sum())
    assert summary["total"] == len(df)


def test_coverage_report_shares_are_fractions():
    rows, summary = coverage_report(label_frame(make_df()))
    assert all(0.0 <= r["share"] <= 1.0 for r in rows)
    assert 0.0 <= summary["no_label_share"] <= 1.0


def test_coverage_report_counts_unlabelled_rows():
    df = label_frame(make_df())
    _, summary = coverage_report(df)
    # row 3 is empty -> no label at all
    assert summary["no_label"] >= 1


def test_coverage_report_handles_an_empty_frame_without_dividing_by_zero():
    empty = label_frame(pd.DataFrame({"text": [], "sentiment": []}))
    rows, summary = coverage_report(empty)
    assert summary["total"] == 0
    assert summary["no_label_share"] == 0.0
    assert all(r["share"] == 0.0 for r in rows)
