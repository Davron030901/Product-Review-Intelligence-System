"""Baseline: TF-IDF + Logistic Regression.

Two heads sharing one vectoriser:
  - sentiment: multinomial logistic regression (3 classes)
  - issues:    one-vs-rest logistic regression (multi-label)

This is the number the transformer has to beat. It trains in seconds, is
fully interpretable, and on short review text it is a genuinely strong
baseline -- do not assume it will lose.
"""
import argparse
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline

from src.config import (DATA_PROCESSED, ISSUE_LABELS, MODELS_DIR, RANDOM_SEED,
                        REPORTS_DIR)
from src.data.splits import group_split, leakage_check
from src.training.evaluate import evaluate_all
from src.training.tracking import (dataset_fingerprint, log_run,
                                   summarise_for_tracking)

ISSUE_COLS = [f"issue_{i}" for i in ISSUE_LABELS]


def build_vectoriser() -> TfidfVectorizer:
    return TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9,
        sublinear_tf=True,
        strip_accents="unicode",
        lowercase=True,
    )


def train(dataset_path=None, out_dir=None, seed: int = RANDOM_SEED) -> dict:
    dataset_path = dataset_path or DATA_PROCESSED / "dataset.csv"
    out_dir = out_dir or MODELS_DIR
    df = pd.read_csv(dataset_path)
    df["text"] = df["text"].fillna("")

    train_df, val_df, test_df = group_split(df, seed=seed)
    leak = leakage_check(train_df, test_df)
    if not leak["clean"]:
        raise SystemExit(f"Leakage detected, refusing to train: {leak}")

    t0 = time.time()
    # Sentiment head. class_weight balanced because positive reviews dominate.
    sentiment_pipe = Pipeline([
        ("tfidf", build_vectoriser()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced",
                                   random_state=seed)),
    ])
    sentiment_pipe.fit(train_df["text"], train_df["sentiment"])

    # Issue head. Shared vectoriser config, separate fit; OvR gives us
    # independent probabilities per label, which is what multi-label needs.
    issue_pipe = Pipeline([
        ("tfidf", build_vectoriser()),
        ("clf", OneVsRestClassifier(
            LogisticRegression(max_iter=2000, class_weight="balanced",
                               random_state=seed))),
    ])
    y_issue = train_df[ISSUE_COLS].values
    # Drop labels with no positive example in train -- OvR cannot fit them and
    # a silent all-zero classifier is worse than an explicit gap.
    usable = [i for i, c in enumerate(ISSUE_COLS) if y_issue[:, i].sum() > 0]
    dropped = [ISSUE_COLS[i].replace("issue_", "") for i in range(len(ISSUE_COLS)) if i not in usable]
    issue_pipe.fit(train_df["text"], y_issue[:, usable])
    train_secs = round(time.time() - t0, 2)

    artefact = {
        "sentiment_pipe": sentiment_pipe,
        "issue_pipe": issue_pipe,
        "issue_labels": [ISSUE_COLS[i].replace("issue_", "") for i in usable],
        "sentiment_classes": list(sentiment_pipe.named_steps["clf"].classes_),
        "dropped_issue_labels": dropped,
        "model_type": "tfidf-logreg",
        "seed": seed,
    }
    out_path = out_dir / "baseline.joblib"
    joblib.dump(artefact, out_path)

    report = evaluate_all(artefact, val_df, test_df)
    report.update({
        "model": "baseline-tfidf-logreg",
        "train_seconds": train_secs,
        "leakage_check": leak,
        "rows": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
        "dropped_issue_labels": dropped,
    })
    (REPORTS_DIR / "baseline_report.json").write_text(json.dumps(report, indent=2))

    run = log_run(
        name="baseline-tfidf-logreg",
        params={"vectoriser": "tfidf 1-2gram, min_df=2, sublinear",
                "classifier": "LogisticRegression(class_weight=balanced)",
                "issue_strategy": "one-vs-rest", "seed": seed,
                "selected_threshold": report["selected_threshold"]},
        metrics=summarise_for_tracking(report),
        dataset=dataset_fingerprint(dataset_path),
        artefacts=[str(out_path)],
        notes="Reference baseline. Any other model must beat these numbers on "
              "the same dataset hash to be worth deploying.",
    )
    report["run_id"] = run["run_id"]
    (REPORTS_DIR / "baseline_report.json").write_text(json.dumps(report, indent=2))

    print(f"[train] run_id={run['run_id']}")
    print(f"[train] saved model -> {out_path}")
    print(f"[train] saved report -> {REPORTS_DIR / 'baseline_report.json'}")
    print(json.dumps({k: report[k] for k in ("sentiment", "issues_summary")}, indent=2))
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    a = ap.parse_args()
    train(a.dataset, seed=a.seed)
