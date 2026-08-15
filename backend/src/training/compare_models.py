"""Compare tracked runs and decide whether the complex model earned its place.

    python -m src.training.compare_models

Reads the experiment log, groups runs by dataset hash (runs on different data
are not comparable), and writes a side-by-side table with a recommendation.

The recommendation is deliberately conservative. A transformer that beats a
TF-IDF baseline by half a point of macro-F1 has not earned a 250MB dependency,
a GPU, and a hundred-fold increase in inference cost -- on a free-tier
deployment it may not even fit in RAM. The threshold below encodes that: the
complex model must win by a visible margin, not a rounding error.
"""
import argparse
import json

from src.config import REPORTS_DIR
from src.training.tracking import load_runs

# Minimum macro-F1 gain for a heavier model to be worth deploying.
MEANINGFUL_GAIN = 0.02

REPORT = REPORTS_DIR / "model_comparison.md"
REPORT_JSON = REPORTS_DIR / "model_comparison.json"


def _headline(run: dict) -> dict:
    m = run.get("metrics", {})
    return {
        "run_id": run["run_id"],
        "name": run["name"],
        "sentiment_macro_f1": m.get("sentiment_macro_f1"),
        "sentiment_accuracy": m.get("sentiment_accuracy"),
        "issue_macro_f1": m.get("issue_macro_f1"),
        "issue_micro_f1": m.get("issue_micro_f1"),
        "dataset_sha": run.get("dataset", {}).get("sha256"),
        "rows": run.get("dataset", {}).get("rows"),
        "timestamp": run.get("timestamp", ""),
    }


def compare() -> dict:
    runs = [_headline(r) for r in load_runs()
            if r.get("metrics", {}).get("issue_macro_f1") is not None]

    if not runs:
        raise SystemExit("No scored runs found. Train a model first.")

    # Only compare runs that used identical data.
    by_data = {}
    for r in runs:
        by_data.setdefault(r["dataset_sha"], []).append(r)
    sha, group = max(by_data.items(), key=lambda kv: len(kv[1]))

    def best(prefix):
        candidates = [r for r in group if r["name"].startswith(prefix)]
        return max(candidates, key=lambda r: r["issue_macro_f1"]) if candidates else None

    baseline = best("baseline")
    transformer = best("transformer")

    verdict, gain = _verdict(baseline, transformer)
    out = {
        "dataset_sha": sha,
        "comparable_runs": len(group),
        "ignored_runs_on_other_data": len(runs) - len(group),
        "baseline": baseline,
        "transformer": transformer,
        "issue_macro_f1_gain": gain,
        "meaningful_gain_threshold": MEANINGFUL_GAIN,
        "verdict": verdict,
    }
    REPORT_JSON.write_text(json.dumps(out, indent=2))
    _write_markdown(out, group)
    print(verdict)
    print(f"[compare] wrote {REPORT}")
    return out


def _verdict(baseline, transformer):
    if baseline and not transformer:
        return ("Only the baseline has been trained. Run "
                "`python -m src.training.train_transformer` to compare, or "
                "deploy the baseline and say so plainly."), None
    if transformer and not baseline:
        return "Only the transformer has been trained; train the baseline to compare.", None

    gain = round(transformer["issue_macro_f1"] - baseline["issue_macro_f1"], 4)
    if gain >= MEANINGFUL_GAIN:
        return (f"Transformer wins by {gain:+.4f} macro-F1, above the "
                f"{MEANINGFUL_GAIN} threshold. Deploying it is defensible, "
                f"provided the target host has the memory for it."), gain
    if gain <= -MEANINGFUL_GAIN:
        return (f"Baseline wins by {-gain:+.4f} macro-F1. Deploy the baseline; "
                f"the transformer is not earning its cost here."), gain
    return (f"Difference is {gain:+.4f} macro-F1, inside the ±{MEANINGFUL_GAIN} "
            f"noise band. Deploy the baseline: same accuracy, a fraction of the "
            f"cost, and it stays interpretable."), gain


def _write_markdown(out: dict, group: list):
    fmt = lambda v: f"{v:.4f}" if isinstance(v, (int, float)) else "—"
    lines = [
        "# Model comparison",
        "",
        f"All runs below trained on the same data (`{out['dataset_sha']}`, "
        f"{group[0]['rows']} rows). "
        + (f"{out['ignored_runs_on_other_data']} run(s) on other data were excluded — "
           "metrics from different datasets are not comparable."
           if out["ignored_runs_on_other_data"] else ""),
        "",
        "| Run | Model | Sentiment macro-F1 | Issue macro-F1 | Issue micro-F1 |",
        "|---|---|---|---|---|",
    ]
    for r in sorted(group, key=lambda r: r["issue_macro_f1"], reverse=True):
        lines.append(f"| `{r['run_id']}` | {r['name']} "
                     f"| {fmt(r['sentiment_macro_f1'])} | {fmt(r['issue_macro_f1'])} "
                     f"| {fmt(r['issue_micro_f1'])} |")

    lines += ["", "## Verdict", "", out["verdict"], "",
              "## Cost, not just accuracy", "",
              "| | TF-IDF + LogReg | DistilBERT multi-task |",
              "|---|---|---|",
              "| Training time | seconds, CPU | minutes to hours, GPU preferred |",
              "| Model size | ~1 MB | ~250 MB |",
              "| Inference | sub-millisecond | tens of ms on CPU |",
              "| Free-tier hosting | comfortable | usually will not fit |",
              "| Interpretability | inspect the coefficients | opaque |",
              "",
              "A model is only better if it is better *after* these are counted. "
              "On short review text with keyword-generated labels, the baseline is "
              "a genuinely strong opponent: much of the signal is lexical, which is "
              "exactly what TF-IDF captures.",
              "",
              "## Honest framing", "",
              "Both models are scored against weak labels, so this table compares "
              "how well each reproduces a keyword lexicon — not which understands "
              "customers better. A transformer's real advantage (context, negation, "
              "sarcasm) is largely invisible to labels that were themselves produced "
              "by keyword matching. Measuring that advantage needs the human-checked "
              "set from `validate_labels.py` as the evaluation ground truth."]
    REPORT.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    argparse.ArgumentParser(description="Compare tracked model runs.").parse_args()
    compare()
