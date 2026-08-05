from src.data.preprocess import clean_text, input_quality_flags, token_count
from src.data.weak_labels import label_review


def test_clean_strips_html_and_urls():
    assert clean_text("<b>Nice</b> see https://x.com now") == "Nice see now"


def test_clean_handles_none_and_empty():
    assert clean_text(None) == ""
    assert clean_text("   ") == ""


def test_token_count():
    assert token_count("arrived broken today") == 3
    assert token_count("") == 0


def test_flags_detect_short_and_empty():
    assert input_quality_flags("")["empty"] is True
    assert input_quality_flags("meh")["too_short"] is True
    assert input_quality_flags("this arrived completely broken")["too_short"] is False


def test_flags_detect_non_latin_script():
    assert input_quality_flags("很好的产品非常满意")["non_latin_script"] is True


def test_weak_labels_are_multi_label():
    hits = label_review("Arrived late and the box was crushed")
    assert "delivery" in hits and "packaging" in hits


def test_weak_labels_respect_word_boundaries():
    assert "price" not in label_review("this gift is priceless to me")


def test_weak_labels_empty_when_nothing_fires():
    assert label_review("hello there") == []
