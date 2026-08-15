"""Cross-category transfer evaluation.

The brief asks how well the model would move to a different product category.
That question deserves a number, not an opinion, so this script trains on
every category except one and evaluates on the category it has never seen.

    python -m src.training.evaluate_transfer                # every category
    python -m src.training.evaluate_transfer --category Home

The gap between in-domain and held-out performance is the answer. Expect it to
be substantial: the vocabulary shifts (a "defect" in electronics reads nothing
like a "defect" in apparel), and the keyword lexicon that generated the labels
is itself domain-bound, so the held-out labels are noisier too. Both effects
push the same way and neither is a bug.
"""
import argparse
import json

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline

from src.config import (DATA_PROCESSED, ISSUE_LABELS, RANDOM_SEED, REPORTS_DIR)
from src.data.splits import group_split, holdout_category
from src.training.evaluate import evaluate_issues, evaluate_sentiment
from src.training.train_baseline import build_vectoriser
from src.training.tracking import dataset_fingerprint, log_run

ISSUE_COLS = [f"issue_{i}" for i in ISSUE_LABELS]
MIN_ROWS = 40


def _fit(train_df, seed):
    sentiment_pipe = Pipeline([
        ("tfidf", build_vectoriser()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced",
                                   random_state=seed)),
    ]).fit(train_df["text"], train_df["sentiment"])

    y = train_df[ISSUE_COLS].values
    usable = [i for i in range(len(ISSUE_COLS)) if y[:, i].sum() > 0]
    issue_pipe = Pipeline([
        ("tfidf", build_vectoriser()),
        ("clf", OneVsRestClassifier(LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=seed))),
    ]).fit(train_df["text"], y[:, usable])

    return {
        "sentiment_pipe": sentiment_pipe,
        "issue_pipe": issue_pipe,
        "issue_labels": [ISSUE_COLS[i].replace("issue_", "") for i in usable],
    }


def evaluate_one(df, category, seed=RANDOM_SEED, threshold=0.35) -> dict:
    """Train without `category`, then score on it and on an in-domain test set."""
    rest, held = holdout_category(df, category)
    if len(held) < MIN_ROWS or len(rest) < MIN_ROWS:
        return {"category": category, "skipped": True,
                "reason": f"too few rows (held={len(held)}, rest={len(rest)})"}

    train_df, _, in_domain_test = group_split(rest, seed=seed)
    artefact = _fit(train_df, seed)

    # Only score issue labels the model actually saw during training.
    labels = [l for l in artefact["issue_labels"]
              if held[f"issue_{l}"].sum() > 0 or True]
    artefact["issue_labels"] = labels

    in_sent = evaluate_sentiment(artefact, in_domain_test)
    out_sent = evaluate_sentiment(artefact, held)
    in_issues, in_summary = evaluate_issues(artefact, in_domain_test, threshold)
    out_issues, out_summary = evaluate_issues(artefact, held, threshold)

    return {
        "category": category,
        "skipped": False,
        "rows": {"train": len(train_df), "in_domain_test": len(in_domain_test),
                 "held_out": len(held)},
        "in_domain": {"sentiment_macro_f1": in_sent["macro_f1"],
                      "sentiment_accuracy": in_sent["accuracy"],
                      "issue_macro_f1": in_summary["macro_f1"],
                      "issue_micro_f1": in_summary["micro_f1"]},
        "held_out": {"sentiment_macro_f1": out_sent["macro_f1"],
                     "sentiment_accuracy": out_sent["accuracy"],
                     "issue_macro_f1": out_summary["macro_f1"],
                     "issue_micro_f1": out_summary["micro_f1"]},
        "drop": {
            "sentiment_macro_f1": round(
                in_sent["macro_f1"] - out_sent["macro_f1"], 4),
            "issue_macro_f1": round(
                in_summary["macro_f1"] - out_summary["macro_f1"], 4),
        },
        "held_out_per_label": out_issues,
    }


def run(dataset_path=None, category=None, seed=RANDOM_SEED) -> dict:
    dataset_path = dataset_path or DATA_PROCESSED / "dataset.csv"
    df = pd.read_csv(dataset_path)
    df["text"] = df["text"].fillna("")

    categories = [category] if category else sorted(df["category"].dropna().unique())
    results = [evaluate_one(df, c, seed) for c in categories]
    scored = [r for r in results if not r["skipped"]]

    summary = {}
    if scored:
        summary = {
            "categories_evaluated": len(scored),
            "mean_sentiment_drop": round(
                sum(r["drop"]["sentiment_macro_f1"] for r in scored) / len(scored), 4),
            "mean_issue_drop": round(
                sum(r["drop"]["issue_macro_f1"] for r in scored) / len(scored), 4),
            "worst_category": max(scored, key=lambda r: r["drop"]["issue_macro_f1"])["category"],
        }

    out = {"bootstrap_data": _is_bootstrap_data(),
           "results": results, "summary": summary,
           "note": ("Each row: trained on all other categories, evaluated on the "
                    "named one. 'drop' is in-domain minus held-out, so positive "
                    "means the model got worse when moved.")}

    path = REPORTS_DIR / "transfer_report.json"
    path.write_text(json.dumps(out, indent=2))
    _write_markdown(out)

    if scored:
        log_run(name="transfer-eval-baseline",
                params={"seed": seed, "categories": [r["category"] for r in scored]},
                metrics={"mean_issue_drop": summary["mean_issue_drop"],
                         "mean_sentiment_drop": summary["mean_sentiment_drop"]},
                dataset=dataset_fingerprint(dataset_path),
                notes="Leave-one-category-out. Answers brief question 6.")

    if _is_bootstrap_data():
        print("[transfer] WARNING: bootstrap data has randomly assigned categories, "
              "so there is no domain shift to measure. A near-zero drop here means "
              "the script works, not that the model transfers.")
    print(json.dumps(summary, indent=2))
    print(f"[transfer] wrote {path}")
    return out


def _is_bootstrap_data() -> bool:
    meta_path = DATA_PROCESSED / "dataset_meta.json"
    if not meta_path.exists():
        return False
    try:
        return json.loads(meta_path.read_text()).get("source") == "bootstrap-sample"
    except (json.JSONDecodeError, OSError):
        return False


BOOTSTRAP_WARNING = (
    "> **These numbers measure nothing.** The dataset is the synthetic bootstrap "
    "sample, where product categories are assigned at random and carry no "
    "vocabulary of their own. With no real domain shift to detect, a drop of "
    "~0.000 is the expected output of a working script, not evidence that the "
    "model transfers well. Rerun on a real multi-category dataset before citing "
    "anything below."
)


def _write_markdown(out: dict):
    lines = ["# Cross-category transfer", ""]
    if _is_bootstrap_data():
        lines += [BOOTSTRAP_WARNING, ""]
    lines += [
             "Trained on every category except one, evaluated on the unseen one.",
             "Positive drop = the model performed worse on the category it had "
             "never seen.", "",
             "| Held-out category | Rows | Sentiment macro-F1 (in → out) | Issue macro-F1 (in → out) | Issue drop |",
             "|---|---|---|---|---|"]
    for r in out["results"]:
        if r["skipped"]:
            lines.append(f"| {r['category']} | — | skipped | {r['reason']} | — |")
            continue
        lines.append(
            f"| {r['category']} | {r['rows']['held_out']} "
            f"| {r['in_domain']['sentiment_macro_f1']:.3f} → {r['held_out']['sentiment_macro_f1']:.3f} "
            f"| {r['in_domain']['issue_macro_f1']:.3f} → {r['held_out']['issue_macro_f1']:.3f} "
            f"| {r['drop']['issue_macro_f1']:+.3f} |")

    s = out["summary"]
    if s:
        lines += ["", "## Summary", "",
                  f"- Categories evaluated: **{s['categories_evaluated']}**",
                  f"- Mean issue macro-F1 drop: **{s['mean_issue_drop']:+.3f}**",
                  f"- Mean sentiment macro-F1 drop: **{s['mean_sentiment_drop']:+.3f}**",
                  f"- Transfers worst to: **{s['worst_category']}**"]

    lines += ["", "## What this means", "",
              "Two effects are bundled into the drop and cannot be separated by this "
              "experiment alone. First, genuine domain shift: complaint vocabulary "
              "differs between categories, and some labels (`fit`) barely exist "
              "outside apparel. Second, label noise: the keyword lexicon that "
              "generated the ground truth is domain-bound too, so the held-out "
              "labels are themselves less trustworthy.",
              "",
              "The practical reading: **retrain per category domain rather than "
              "assuming one model serves all of them**, and treat any deployment "
              "into an unseen category as unvalidated until this table is rerun "
              "with that category held out.",
              "",
              "Language transfer is a separate and worse problem, not measured "
              "here: training is English-only and non-English input is flagged "
              "rather than scored. See `model_card.md`."]
    (REPORTS_DIR / "transfer_report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--category", default=None,
                    help="Evaluate a single category instead of all of them.")
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    a = ap.parse_args()
    run(a.dataset, a.category, a.seed)
