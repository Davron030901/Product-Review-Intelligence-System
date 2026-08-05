"""Dataset construction and training, tested end to end on a temporary
directory so the real data/ and models/ are never touched."""
import json

import pandas as pd
import pytest

from src.config import ISSUE_LABELS, SENTIMENT_LABELS
from src.data.build_dataset import _generate_sample, _load_real, build


# --- bootstrap sample generator --------------------------------------------

def test_generated_sample_has_the_expected_columns():
    df = _generate_sample(50)
    assert {"review_id", "product_id", "category", "text", "rating"} <= set(df.columns)


def test_generated_sample_has_the_requested_row_count():
    assert len(_generate_sample(120)) == 120


def test_generated_sample_is_deterministic():
    a = _generate_sample(40, seed=3)["text"].tolist()
    b = _generate_sample(40, seed=3)["text"].tolist()
    assert a == b


def test_generated_ratings_are_in_range():
    ratings = _generate_sample(200)["rating"]
    assert ratings.between(1, 5).all()


def test_generated_sample_contains_repeat_products():
    """Grouped splitting is pointless if every product is unique."""
    df = _generate_sample(300)
    assert df["product_id"].duplicated().any()


def test_generated_sample_includes_thin_reviews():
    """Short/ambiguous reviews must exist, or the abstention path is untested."""
    df = _generate_sample(500)
    assert (df["text"].str.split().str.len() <= 2).any()


# --- _load_real ------------------------------------------------------------

def test_load_real_maps_columns(tmp_path):
    src = tmp_path / "reviews.csv"
    pd.DataFrame({
        "Review Text": ["arrived late", "great"],
        "Rating": [1, 5],
        "Clothing ID": [10, 11],
        "Department Name": ["Tops", "Tops"],
    }).to_csv(src, index=False)
    out = _load_real(src, {"text": "Review Text", "rating": "Rating",
                           "product_id": "Clothing ID", "category": "Department Name"})
    assert list(out["text"]) == ["arrived late", "great"]
    assert len(out) == 2


def test_load_real_fails_loudly_on_missing_columns(tmp_path):
    src = tmp_path / "reviews.csv"
    pd.DataFrame({"body": ["x"], "stars": [5]}).to_csv(src, index=False)
    with pytest.raises(SystemExit) as excinfo:
        _load_real(src, {"text": "Review Text", "rating": "Rating",
                         "product_id": "Clothing ID", "category": "Department Name"})
    assert "Review Text" in str(excinfo.value)


def test_load_real_coerces_bad_ratings_to_nan(tmp_path):
    src = tmp_path / "reviews.csv"
    pd.DataFrame({
        "Review Text": ["a", "b"], "Rating": ["five", 4],
        "Clothing ID": [1, 2], "Department Name": ["Tops", "Tops"],
    }).to_csv(src, index=False)
    out = _load_real(src, {"text": "Review Text", "rating": "Rating",
                           "product_id": "Clothing ID", "category": "Department Name"})
    assert out["rating"].isna().sum() == 1


# --- build (integration) ---------------------------------------------------

@pytest.fixture(scope="module")
def built(tmp_path_factory, monkeypatch_module):
    """Build a dataset into a temp dir once and reuse it."""
    out = tmp_path_factory.mktemp("processed")
    monkeypatch_module.setattr("src.data.build_dataset.DATA_PROCESSED", out)
    monkeypatch_module.setattr("src.data.build_dataset.DATA_RAW", out / "raw")
    (out / "raw").mkdir(exist_ok=True)
    df = build(sample_size=800)
    return df, out


def test_build_writes_the_dataset_and_metadata(built):
    _, out = built
    assert (out / "dataset.csv").exists()
    assert (out / "dataset_meta.json").exists()


def test_build_produces_a_column_for_every_issue_label(built):
    df, _ = built
    for issue in ISSUE_LABELS:
        assert f"issue_{issue}" in df.columns


def test_build_assigns_only_valid_sentiments(built):
    df, _ = built
    assert set(df["sentiment"]) <= set(SENTIMENT_LABELS)


def test_build_removes_duplicate_texts(built):
    df, _ = built
    assert not df["text"].duplicated().any()


def test_build_drops_empty_text(built):
    df, _ = built
    assert (df["text"].str.strip().str.len() > 0).all()


def test_build_has_no_missing_ratings(built):
    df, _ = built
    assert df["rating"].notna().all()


def test_metadata_records_provenance_and_distribution(built):
    _, out = built
    meta = json.loads((out / "dataset_meta.json").read_text())
    for key in ("source", "rows_in", "rows_out", "dropped",
                "sentiment_distribution", "issue_coverage", "median_tokens"):
        assert key in meta


def test_metadata_row_counts_are_consistent(built):
    df, out = built
    meta = json.loads((out / "dataset_meta.json").read_text())
    assert meta["rows_out"] == len(df)
    assert meta["rows_in"] - meta["dropped"] == meta["rows_out"]


def test_sentiment_distribution_sums_to_one(built):
    _, out = built
    meta = json.loads((out / "dataset_meta.json").read_text())
    assert abs(sum(meta["sentiment_distribution"].values()) - 1.0) < 0.01


def test_build_labels_are_binary(built):
    df, _ = built
    for issue in ISSUE_LABELS:
        assert set(df[f"issue_{issue}"].unique()) <= {0, 1}


def test_most_rows_receive_at_least_one_label(built):
    """A lexicon that fires on almost nothing would make the issue task
    meaningless, and it would fail quietly rather than crash."""
    df, _ = built
    cols = [f"issue_{i}" for i in ISSUE_LABELS]
    labelled = (df[cols].sum(axis=1) > 0).mean()
    assert labelled > 0.5
