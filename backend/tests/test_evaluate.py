"""Metrics are checked against cases whose correct answer is known by hand.

A metric bug is the worst kind here: it does not crash, it just reports a
number that is wrong, and the whole evaluation section of the report is built
on it.
"""
import numpy as np
import pandas as pd
import pytest

from src.training.evaluate import (RARE_SUPPORT, evaluate_all, evaluate_issues,
                                   evaluate_sentiment, threshold_sweep)


class FakePipe:
    """Stands in for a fitted sklearn pipeline with known outputs."""

    def __init__(self, preds=None, proba=None, classes=None):
        self._preds = preds
        self._proba = np.asarray(proba) if proba is not None else None
        self.named_steps = {"clf": type("C", (), {"classes_": np.array(classes)})()} \
            if classes is not None else {}

    def predict(self, X):
        return np.asarray(self._preds[: len(X)])

    def predict_proba(self, X):
        return self._proba[: len(X)]


# --- evaluate_sentiment ----------------------------------------------------

def test_perfect_sentiment_predictions_score_one():
    labels = ["negative", "positive", "neutral", "negative"]
    df = pd.DataFrame({"text": ["a", "b", "c", "d"], "sentiment": labels})
    artefact = {"sentiment_pipe": FakePipe(preds=labels,
                                           classes=["negative", "neutral", "positive"])}
    out = evaluate_sentiment(artefact, df)
    assert out["accuracy"] == 1.0
    assert out["macro_f1"] == 1.0


def test_sentiment_accuracy_is_computed_correctly():
    """3 of 4 correct -> 0.75, checked by hand."""
    truth = ["negative", "positive", "neutral", "negative"]
    preds = ["negative", "positive", "neutral", "positive"]
    df = pd.DataFrame({"text": list("abcd"), "sentiment": truth})
    artefact = {"sentiment_pipe": FakePipe(preds=preds,
                                           classes=["negative", "neutral", "positive"])}
    assert evaluate_sentiment(artefact, df)["accuracy"] == 0.75


def test_majority_class_predictor_has_high_accuracy_but_poor_macro_f1():
    """The exact reason macro-F1 leads the report instead of accuracy."""
    truth = ["positive"] * 9 + ["negative"]
    preds = ["positive"] * 10
    df = pd.DataFrame({"text": list("abcdefghij"), "sentiment": truth})
    artefact = {"sentiment_pipe": FakePipe(preds=preds,
                                           classes=["negative", "neutral", "positive"])}
    out = evaluate_sentiment(artefact, df)
    assert out["accuracy"] == 0.9
    assert out["macro_f1"] < 0.4


def test_confusion_matrix_shape_matches_class_count():
    df = pd.DataFrame({"text": ["a", "b"], "sentiment": ["negative", "positive"]})
    artefact = {"sentiment_pipe": FakePipe(preds=["negative", "positive"],
                                           classes=["negative", "neutral", "positive"])}
    cm = evaluate_sentiment(artefact, df)["confusion_matrix"]
    assert len(cm["matrix"]) == 3
    assert all(len(row) == 3 for row in cm["matrix"])


def test_per_class_support_sums_to_row_count():
    df = pd.DataFrame({"text": list("abcd"),
                       "sentiment": ["negative", "positive", "positive", "neutral"]})
    artefact = {"sentiment_pipe": FakePipe(preds=["negative"] * 4,
                                           classes=["negative", "neutral", "positive"])}
    out = evaluate_sentiment(artefact, df)
    assert sum(v["support"] for v in out["per_class"].values()) == 4


# --- evaluate_issues -------------------------------------------------------

def issue_artefact(proba, labels=("delivery", "quality")):
    return {"issue_pipe": FakePipe(proba=proba), "issue_labels": list(labels)}


def test_perfect_multilabel_predictions_score_one():
    df = pd.DataFrame({"text": ["a", "b"], "issue_delivery": [1, 0], "issue_quality": [0, 1]})
    artefact = issue_artefact([[0.9, 0.1], [0.1, 0.9]])
    per_label, summary = evaluate_issues(artefact, df, threshold=0.5)
    assert summary["micro_f1"] == 1.0
    assert per_label["delivery"]["f1"] == 1.0


def test_threshold_changes_the_predictions():
    df = pd.DataFrame({"text": ["a"], "issue_delivery": [1], "issue_quality": [1]})
    artefact = issue_artefact([[0.6, 0.4]])
    _, low = evaluate_issues(artefact, df, threshold=0.3)
    _, high = evaluate_issues(artefact, df, threshold=0.9)
    assert low["mean_labels_pred"] == 2.0
    assert high["mean_labels_pred"] == 0.0


def test_support_counts_true_positives_per_label():
    df = pd.DataFrame({"text": list("abc"),
                       "issue_delivery": [1, 1, 0], "issue_quality": [0, 0, 0]})
    artefact = issue_artefact([[0.9, 0.1]] * 3)
    per_label, _ = evaluate_issues(artefact, df, threshold=0.5)
    assert per_label["delivery"]["support"] == 2
    assert per_label["quality"]["support"] == 0


def test_rare_labels_are_flagged_and_excluded_from_the_headline_macro():
    """A label with tiny support must not quietly drag the headline number."""
    n = RARE_SUPPORT * 4
    df = pd.DataFrame({
        "text": ["x"] * n,
        "issue_delivery": [1] * n,                     # plentiful, predicted well
        "issue_quality": [1] + [0] * (n - 1),          # rare, predicted badly
    })
    artefact = issue_artefact([[0.9, 0.1]] * n)
    per_label, summary = evaluate_issues(artefact, df, threshold=0.5)
    assert per_label["quality"]["rare"] is True
    assert per_label["delivery"]["rare"] is False
    assert "quality" in summary["rare_labels"]
    assert summary["macro_f1_excluding_rare"] > summary["macro_f1"]


def test_mean_labels_true_detects_multilabel_collapse():
    """If the model predicts one label where reviews really raise two, this
    is the number that shows it."""
    df = pd.DataFrame({"text": ["a", "b"], "issue_delivery": [1, 1], "issue_quality": [1, 1]})
    artefact = issue_artefact([[0.9, 0.2], [0.9, 0.2]])
    _, summary = evaluate_issues(artefact, df, threshold=0.5)
    assert summary["mean_labels_true"] == 2.0
    assert summary["mean_labels_pred"] == 1.0


def test_exact_match_ratio_requires_every_label_correct():
    df = pd.DataFrame({"text": ["a", "b"], "issue_delivery": [1, 1], "issue_quality": [0, 1]})
    artefact = issue_artefact([[0.9, 0.1], [0.9, 0.1]])
    _, summary = evaluate_issues(artefact, df, threshold=0.5)
    assert summary["exact_match_ratio"] == 0.5


def test_average_precision_is_none_when_a_label_has_no_positives():
    df = pd.DataFrame({"text": ["a", "b"], "issue_delivery": [0, 0], "issue_quality": [1, 1]})
    artefact = issue_artefact([[0.5, 0.5], [0.5, 0.5]])
    per_label, _ = evaluate_issues(artefact, df, threshold=0.5)
    assert per_label["delivery"]["average_precision"] is None


# --- threshold_sweep -------------------------------------------------------

def test_sweep_returns_one_entry_per_grid_point():
    df = pd.DataFrame({"text": ["a"], "issue_delivery": [1], "issue_quality": [0]})
    artefact = issue_artefact([[0.7, 0.2]])
    sweep = threshold_sweep(artefact, df, grid=(0.2, 0.5, 0.8))
    assert [s["threshold"] for s in sweep] == [0.2, 0.5, 0.8]


def test_sweep_picks_a_sensible_threshold():
    df = pd.DataFrame({"text": list("abcd"),
                       "issue_delivery": [1, 1, 0, 0], "issue_quality": [0, 0, 1, 1]})
    artefact = issue_artefact([[0.8, 0.2], [0.8, 0.2], [0.2, 0.8], [0.2, 0.8]])
    sweep = threshold_sweep(artefact, df, grid=(0.1, 0.5, 0.95))
    best = max(sweep, key=lambda d: d["macro_f1"])
    assert best["threshold"] == 0.5


# --- evaluate_all ----------------------------------------------------------

def test_evaluate_all_selects_threshold_on_val_and_reports_on_test():
    val = pd.DataFrame({"text": list("abcd"), "sentiment": ["negative"] * 4,
                        "issue_delivery": [1, 1, 0, 0], "issue_quality": [0, 0, 1, 1]})
    test = val.copy()
    proba = [[0.8, 0.2], [0.8, 0.2], [0.2, 0.8], [0.2, 0.8]]
    artefact = {
        "sentiment_pipe": FakePipe(preds=["negative"] * 4,
                                   classes=["negative", "neutral", "positive"]),
        "issue_pipe": FakePipe(proba=proba),
        "issue_labels": ["delivery", "quality"],
    }
    out = evaluate_all(artefact, val, test)
    assert out["selected_threshold"] == out["issues_summary"]["threshold"]
    for key in ("sentiment", "sentiment_val", "issues_per_label",
                "issues_summary", "threshold_sweep_on_val"):
        assert key in out


def test_evaluate_all_output_is_json_serialisable():
    """The report is written to disk with json.dumps; numpy types break it."""
    import json
    df = pd.DataFrame({"text": ["a", "b"], "sentiment": ["negative", "positive"],
                       "issue_delivery": [1, 0], "issue_quality": [0, 1]})
    artefact = {
        "sentiment_pipe": FakePipe(preds=["negative", "positive"],
                                   classes=["negative", "neutral", "positive"]),
        "issue_pipe": FakePipe(proba=[[0.9, 0.1], [0.1, 0.9]]),
        "issue_labels": ["delivery", "quality"],
    }
    json.dumps(evaluate_all(artefact, df, df))
