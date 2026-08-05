"""Training, run for real on a small dataset in a temp directory.

Slow relative to the rest of the suite (a few seconds), but this is the path
that produces every number in the report, so it is worth exercising rather
than mocking.
"""
import json

import joblib
import pandas as pd
import pytest

from src.config import ISSUE_LABELS
from src.data.build_dataset import _generate_sample
from src.data.preprocess import clean_text
from src.data.weak_labels import label_frame
from src.training.train_baseline import ISSUE_COLS, build_vectoriser, train


def make_dataset(tmp_path, n=900):
    df = _generate_sample(n, seed=11)
    df["text"] = df["text"].map(clean_text)
    df = df[df["text"].str.len() > 0].drop_duplicates(subset=["text"])
    df["sentiment"] = df["rating"].map(
        lambda r: "negative" if r <= 2 else ("neutral" if r < 4 else "positive"))
    df = label_frame(df, "text")
    path = tmp_path / "dataset.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture(scope="module")
def trained(tmp_path_factory, monkeypatch_module):
    tmp = tmp_path_factory.mktemp("train")
    reports = tmp / "reports"
    reports.mkdir()
    monkeypatch_module.setattr("src.training.train_baseline.REPORTS_DIR", reports)
    dataset = make_dataset(tmp)
    report = train(dataset_path=dataset, out_dir=tmp)
    return report, tmp, reports


# --- vectoriser ------------------------------------------------------------

def test_vectoriser_uses_bigrams():
    assert build_vectoriser().ngram_range == (1, 2)


def test_vectoriser_learns_a_vocabulary():
    v = build_vectoriser()
    v.fit(["arrived late", "arrived broken", "arrived late again"])
    assert len(v.vocabulary_) > 0


# --- artefact --------------------------------------------------------------

def test_training_writes_a_model_file(trained):
    _, tmp, _ = trained
    assert (tmp / "baseline.joblib").exists()


def test_artefact_contains_both_heads(trained):
    _, tmp, _ = trained
    art = joblib.load(tmp / "baseline.joblib")
    for key in ("sentiment_pipe", "issue_pipe", "issue_labels",
                "sentiment_classes", "model_type", "seed"):
        assert key in art


def test_artefact_issue_labels_are_in_the_taxonomy(trained):
    _, tmp, _ = trained
    art = joblib.load(tmp / "baseline.joblib")
    assert set(art["issue_labels"]) <= set(ISSUE_LABELS)


def test_saved_model_can_predict(trained):
    _, tmp, _ = trained
    art = joblib.load(tmp / "baseline.joblib")
    assert art["sentiment_pipe"].predict(["arrived broken and late"])[0] in {
        "negative", "neutral", "positive"}


def test_issue_probabilities_have_one_column_per_kept_label(trained):
    _, tmp, _ = trained
    art = joblib.load(tmp / "baseline.joblib")
    proba = art["issue_pipe"].predict_proba(["arrived late, box crushed"])
    assert proba.shape[1] == len(art["issue_labels"])


# --- report ----------------------------------------------------------------

def test_report_is_written_to_disk(trained):
    _, _, reports = trained
    assert (reports / "baseline_report.json").exists()


def test_report_file_is_valid_json(trained):
    _, _, reports = trained
    json.loads((reports / "baseline_report.json").read_text())


def test_report_contains_the_expected_sections(trained):
    report, _, _ = trained
    for key in ("sentiment", "issues_per_label", "issues_summary",
                "selected_threshold", "leakage_check", "rows", "train_seconds"):
        assert key in report


def test_report_confirms_a_clean_split(trained):
    report, _, _ = trained
    assert report["leakage_check"]["clean"] is True
    assert report["leakage_check"]["shared_products"] == 0


def test_row_counts_are_plausible(trained):
    report, _, _ = trained
    rows = report["rows"]
    assert rows["train"] > rows["val"]
    assert rows["train"] > rows["test"]
    assert all(v > 0 for v in rows.values())


def test_metrics_are_within_valid_ranges(trained):
    report, _, _ = trained
    s = report["sentiment"]
    assert 0.0 <= s["accuracy"] <= 1.0
    assert 0.0 <= s["macro_f1"] <= 1.0
    assert 0.0 <= report["issues_summary"]["micro_f1"] <= 1.0


def test_selected_threshold_is_in_the_sweep_grid(trained):
    report, _, _ = trained
    grid = [s["threshold"] for s in report["threshold_sweep_on_val"]]
    assert report["selected_threshold"] in grid


def test_model_beats_a_random_baseline(trained):
    """A sanity floor: three classes, so anything at or below chance means
    the pipeline is wired wrong, not that the model is weak."""
    report, _, _ = trained
    assert report["sentiment"]["accuracy"] > 0.4


# --- failure modes ---------------------------------------------------------

def test_training_aborts_when_the_split_leaks(tmp_path, monkeypatch):
    """The guard must stop training, not warn and continue."""
    dataset = make_dataset(tmp_path, n=400)
    monkeypatch.setattr(
        "src.training.train_baseline.leakage_check",
        lambda *a, **k: {"clean": False, "shared_products": 7,
                         "shared_exact_texts": 0, "train_rows": 1, "test_rows": 1},
    )
    with pytest.raises(SystemExit) as excinfo:
        train(dataset_path=dataset, out_dir=tmp_path)
    assert "Leakage" in str(excinfo.value)


def test_training_is_reproducible_for_a_fixed_seed(tmp_path, monkeypatch_module):
    reports = tmp_path / "r"
    reports.mkdir()
    monkeypatch_module.setattr("src.training.train_baseline.REPORTS_DIR", reports)
    dataset = make_dataset(tmp_path, n=400)
    a = train(dataset_path=dataset, out_dir=tmp_path, seed=5)
    b = train(dataset_path=dataset, out_dir=tmp_path, seed=5)
    assert a["sentiment"]["accuracy"] == b["sentiment"]["accuracy"]
    assert a["issues_summary"]["macro_f1"] == b["issues_summary"]["macro_f1"]


def test_issue_cols_cover_the_whole_taxonomy():
    assert ISSUE_COLS == [f"issue_{i}" for i in ISSUE_LABELS]
