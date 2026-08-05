"""Evaluation: macro-first, rare labels reported separately.

Accuracy on this data is a vanity metric -- positive reviews dominate and a
model that always says "positive" scores well. Everything below is macro-
averaged or per-label so a rare-but-expensive issue (defects) cannot be
hidden behind a common one.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score,
                             classification_report, confusion_matrix, f1_score,
                             precision_recall_fscore_support)

from src.config import ISSUE_LABELS, ISSUE_THRESHOLD

ISSUE_COLS = [f"issue_{i}" for i in ISSUE_LABELS]
# Labels with fewer than this many positives in the eval fold are reported but
# flagged: their metrics are too noisy to act on.
RARE_SUPPORT = 25


def evaluate_sentiment(artefact, df) -> dict:
    pipe = artefact["sentiment_pipe"]
    y_true = df["sentiment"].values
    y_pred = pipe.predict(df["text"])
    classes = list(pipe.named_steps["clf"].classes_)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=classes,
                                                 zero_division=0)
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        # labels=classes so the macro average is taken over the full declared
        # class space. Without it sklearn averages only over classes present in
        # this fold, which quietly inflates the score whenever a class is
        # missing from the test set -- exactly when you most want to notice.
        "macro_f1": round(float(f1_score(y_true, y_pred, labels=classes,
                                         average="macro", zero_division=0)), 4),
        "per_class": {
            c: {"precision": round(float(p[i]), 4), "recall": round(float(r[i]), 4),
                "f1": round(float(f[i]), 4), "support": int(s[i])}
            for i, c in enumerate(classes)
        },
        "confusion_matrix": {
            "labels": classes,
            "matrix": confusion_matrix(y_true, y_pred, labels=classes).tolist(),
        },
    }


def evaluate_issues(artefact, df, threshold: float = ISSUE_THRESHOLD) -> tuple:
    pipe = artefact["issue_pipe"]
    labels = artefact["issue_labels"]
    cols = [f"issue_{l}" for l in labels]
    y_true = df[cols].values
    proba = pipe.predict_proba(df["text"])
    y_pred = (proba >= threshold).astype(int)

    per_label = {}
    for i, label in enumerate(labels):
        support = int(y_true[:, i].sum())
        p, r, f, _ = precision_recall_fscore_support(
            y_true[:, i], y_pred[:, i], average="binary", zero_division=0)
        if support == 0:
            # sklearn returns 0.0 here (with a warning), which reads as
            # "the model failed" when the truth is "there was nothing to
            # measure". None keeps the two cases distinguishable in the report.
            ap = float("nan")
        else:
            try:
                ap = float(average_precision_score(y_true[:, i], proba[:, i]))
            except ValueError:
                ap = float("nan")
        per_label[label] = {
            "precision": round(float(p), 4),
            "recall": round(float(r), 4),
            "f1": round(float(f), 4),
            "average_precision": None if np.isnan(ap) else round(ap, 4),
            "support": support,
            "rare": support < RARE_SUPPORT,
        }

    summary = {
        "threshold": threshold,
        "micro_f1": round(float(f1_score(y_true, y_pred, average="micro",
                                         zero_division=0)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro",
                                         zero_division=0)), 4),
        "macro_f1_excluding_rare": round(float(np.mean(
            [m["f1"] for m in per_label.values() if not m["rare"]] or [0.0])), 4),
        "rare_labels": [l for l, m in per_label.items() if m["rare"]],
        "exact_match_ratio": round(float((y_pred == y_true).all(axis=1).mean()), 4),
        "mean_labels_true": round(float(y_true.sum(axis=1).mean()), 3),
        "mean_labels_pred": round(float(y_pred.sum(axis=1).mean()), 3),
    }
    return per_label, summary


def threshold_sweep(artefact, df, grid=(0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6)) -> list:
    """Pick the issue threshold on validation data, never on test."""
    labels = artefact["issue_labels"]
    y_true = df[[f"issue_{l}" for l in labels]].values
    proba = artefact["issue_pipe"].predict_proba(df["text"])
    out = []
    for t in grid:
        y_pred = (proba >= t).astype(int)
        out.append({
            "threshold": t,
            "macro_f1": round(float(f1_score(y_true, y_pred, average="macro",
                                             zero_division=0)), 4),
            "micro_f1": round(float(f1_score(y_true, y_pred, average="micro",
                                             zero_division=0)), 4),
        })
    return out


def evaluate_all(artefact, val_df, test_df) -> dict:
    sweep = threshold_sweep(artefact, val_df)
    best = max(sweep, key=lambda d: d["macro_f1"])
    per_label, summary = evaluate_issues(artefact, test_df, best["threshold"])
    return {
        "sentiment": evaluate_sentiment(artefact, test_df),
        "sentiment_val": evaluate_sentiment(artefact, val_df),
        "issues_per_label": per_label,
        "issues_summary": summary,
        "threshold_sweep_on_val": sweep,
        "selected_threshold": best["threshold"],
        "note": ("Threshold selected on validation, metrics reported on test. "
                 "Rare labels (support < %d) are flagged and excluded from "
                 "macro_f1_excluding_rare." % RARE_SUPPORT),
    }
