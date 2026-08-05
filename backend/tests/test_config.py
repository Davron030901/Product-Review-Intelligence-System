"""The taxonomy and thresholds are shared by training and inference, so a
change here silently changes both. These tests pin the contract."""
import pytest

from src.config import (ISSUE_LABELS, ISSUE_THRESHOLD, MIN_TOKENS_FOR_CONFIDENCE,
                        SENTIMENT_LABELS, SENTIMENT_LOW_CONFIDENCE,
                        rating_to_sentiment)


@pytest.mark.parametrize("rating,expected", [
    (1, "negative"), (2, "negative"),
    (3, "neutral"),
    (4, "positive"), (5, "positive"),
])
def test_rating_maps_to_expected_sentiment(rating, expected):
    assert rating_to_sentiment(rating) == expected


def test_rating_boundaries_are_where_we_think():
    """2 vs 3 and 3 vs 4 are the two decision boundaries."""
    assert rating_to_sentiment(2) != rating_to_sentiment(3)
    assert rating_to_sentiment(3) != rating_to_sentiment(4)


def test_float_ratings_do_not_fall_through():
    """Some datasets store averaged or half-star ratings."""
    assert rating_to_sentiment(2.5) == "neutral"
    assert rating_to_sentiment(4.5) == "positive"
    assert rating_to_sentiment(1.5) == "negative"


def test_out_of_range_ratings_still_return_a_label():
    assert rating_to_sentiment(0) == "negative"
    assert rating_to_sentiment(10) == "positive"


def test_every_rating_maps_into_the_declared_label_space():
    for r in (1, 2, 3, 4, 5):
        assert rating_to_sentiment(r) in SENTIMENT_LABELS


def test_labels_are_unique():
    assert len(ISSUE_LABELS) == len(set(ISSUE_LABELS))
    assert len(SENTIMENT_LABELS) == len(set(SENTIMENT_LABELS))


def test_other_is_present_as_the_catch_all():
    assert "other" in ISSUE_LABELS


def test_thresholds_are_probabilities():
    assert 0 < ISSUE_THRESHOLD < 1
    assert 0 < SENTIMENT_LOW_CONFIDENCE < 1


def test_min_tokens_is_a_small_positive_int():
    assert isinstance(MIN_TOKENS_FOR_CONFIDENCE, int)
    assert 1 <= MIN_TOKENS_FOR_CONFIDENCE <= 10
