"""The predictor defines the contract the frontend consumes, and the rules
for when the system is allowed to say "I don't know". Both are tested here
against input designed to break them."""
import pytest

from src.config import ISSUE_LABELS, SENTIMENT_LABELS
from src.inference.predictor import (MAX_CHARS, ModelNotTrained, ReviewPredictor,
                                     get_predictor)

REQUIRED_KEYS = {
    "sentiment", "issues", "low_confidence", "reasons", "input_category",
    "word_count", "truncated", "model_version", "model_backend", "processed_at",
}


@pytest.fixture(scope="module")
def predictor():
    try:
        return get_predictor()
    except ModelNotTrained:
        pytest.skip("no trained model; run train_baseline first")


# --- contract shape --------------------------------------------------------

def test_response_has_every_contract_key(predictor):
    result = predictor.predict("Arrived late and the box was crushed.")
    assert REQUIRED_KEYS <= set(result)


def test_sentiment_label_is_in_the_declared_space(predictor):
    result = predictor.predict("This is wonderful, I love it.")
    assert result["sentiment"]["label"] in SENTIMENT_LABELS + ["unknown"]


def test_confidence_is_a_probability(predictor):
    result = predictor.predict("The zipper broke immediately.")
    assert 0.0 <= result["sentiment"]["confidence"] <= 1.0


def test_issue_categories_are_in_the_taxonomy(predictor):
    result = predictor.predict("Late delivery, crushed box, broken item, rude support.")
    for issue in result["issues"]:
        assert issue["category"] in ISSUE_LABELS
        assert 0.0 <= issue["confidence"] <= 1.0


def test_issues_are_sorted_by_confidence_descending(predictor):
    result = predictor.predict("Arrived late, box crushed, item broken, too expensive.")
    confidences = [i["confidence"] for i in result["issues"]]
    assert confidences == sorted(confidences, reverse=True)


def test_processed_at_is_an_iso_timestamp(predictor):
    from datetime import datetime
    result = predictor.predict("Good product.")
    datetime.fromisoformat(result["processed_at"])


def test_result_is_json_serialisable(predictor):
    """FastAPI will serialise this; numpy floats would raise."""
    import json
    json.dumps(predictor.predict("Arrived broken and late."))


# --- abstention rules ------------------------------------------------------

def test_empty_text_returns_unknown_not_a_guess(predictor):
    result = predictor.predict("")
    assert result["sentiment"]["label"] == "unknown"
    assert result["low_confidence"] is True
    assert result["issues"] == []
    assert result["reasons"]


def test_whitespace_only_is_treated_as_empty(predictor):
    assert predictor.predict("      \n\t  ")["sentiment"]["label"] == "unknown"


def test_very_short_review_is_flagged(predictor):
    result = predictor.predict("meh")
    assert result["low_confidence"] is True
    assert any("word" in r.lower() for r in result["reasons"])


def test_non_latin_script_is_flagged(predictor):
    result = predictor.predict("Bu mahsulot juda yomon keldi va sifati past edi")
    assert result["low_confidence"] is True


def test_every_flag_carries_a_reason(predictor):
    """A flag with no explanation is useless in the review queue."""
    for text in ("", "meh", "ok"):
        result = predictor.predict(text)
        if result["low_confidence"]:
            assert len(result["reasons"]) > 0


def test_confident_result_has_no_reasons(predictor):
    result = predictor.predict(
        "Arrived two weeks late and the box was completely crushed. Returning it.")
    if not result["low_confidence"]:
        assert result["reasons"] == []


# --- hostile input ---------------------------------------------------------

@pytest.mark.parametrize("text", [
    "😡😡😡",
    "!!!???...",
    "<script>alert('x')</script>",
    "'; DROP TABLE reviews; --",
    "a" * 5000,
    "\x00\x01\x02 broken",
    "https://example.com/very/long/path?a=1&b=2",
    "ＡＲＲＩＶＥＤ　ＬＡＴＥ",
    "🚚 late 📦 crushed",
])
def test_hostile_input_does_not_crash(predictor, text):
    result = predictor.predict(text)
    assert REQUIRED_KEYS <= set(result)


def test_none_input_is_handled(predictor):
    result = predictor.predict(None)
    assert result["sentiment"]["label"] == "unknown"


def test_non_string_input_is_handled(predictor):
    assert predictor.predict(12345)["sentiment"]["label"] == "unknown"


def test_oversized_input_is_truncated_not_rejected(predictor):
    result = predictor.predict("broken " * (MAX_CHARS // 3))
    assert result["truncated"] is True
    assert REQUIRED_KEYS <= set(result)


def test_input_just_under_the_cap_is_not_marked_truncated(predictor):
    assert predictor.predict("a" * (MAX_CHARS - 10))["truncated"] is False


# --- behaviour -------------------------------------------------------------

def test_word_count_reflects_the_cleaned_text(predictor):
    assert predictor.predict("arrived broken today")["word_count"] == 3


def test_category_is_echoed_back(predictor):
    assert predictor.predict("Nice item", category="Dresses")["input_category"] == "Dresses"


def test_category_defaults_to_none(predictor):
    assert predictor.predict("Nice item")["input_category"] is None


def test_prediction_is_deterministic(predictor):
    text = "Arrived late and the box was crushed."
    a, b = predictor.predict(text), predictor.predict(text)
    assert a["sentiment"] == b["sentiment"]
    assert a["issues"] == b["issues"]


def test_delivery_complaint_surfaces_a_delivery_issue(predictor):
    result = predictor.predict(
        "The parcel arrived three weeks late and tracking never updated.")
    assert "delivery" in [i["category"] for i in result["issues"]]


def test_raising_the_threshold_yields_no_more_issues(predictor):
    text = "Arrived late, box crushed, item broken."
    lenient = predictor.predict(text, issue_threshold=0.1)
    strict = predictor.predict(text, issue_threshold=0.99)
    assert len(strict["issues"]) <= len(lenient["issues"])


# --- batch -----------------------------------------------------------------

def test_batch_returns_one_result_per_input(predictor):
    results = predictor.predict_batch(["broken on arrival", "love it", ""])
    assert len(results) == 3
    assert all(REQUIRED_KEYS <= set(r) for r in results)


def test_batch_matches_single_prediction(predictor):
    text = "Arrived late and damaged."
    assert predictor.predict_batch([text])[0]["sentiment"] == predictor.predict(text)["sentiment"]


def test_empty_batch_returns_empty_list(predictor):
    assert predictor.predict_batch([]) == []


# --- loading ---------------------------------------------------------------

def test_get_predictor_is_a_singleton():
    assert get_predictor() is get_predictor()


def test_missing_model_directory_raises_a_helpful_error(tmp_path):
    with pytest.raises(ModelNotTrained) as excinfo:
        ReviewPredictor(models_dir=tmp_path)
    assert "train_baseline" in str(excinfo.value)
