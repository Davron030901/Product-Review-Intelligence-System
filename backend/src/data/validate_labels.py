"""Label-quality validation.

`docs/labeling.md` says the issue labels are unvalidated until a human checks a
sample. This turns that from a promise into a two-command task.

    # 1. draw a stratified sample and write a blank annotation sheet
    python -m src.data.validate_labels sample --n 250

    # 2. open docs/label_validation_sheet.csv, fill the `true_labels` column
    #    (comma-separated, blank = no issue), then:
    python -m src.data.validate_labels score

Step 2 is the part no script can do. Until it is done, every issue metric in
this project measures agreement with a keyword lexicon, not agreement with
reality -- and those are different claims.

Sampling is stratified by label so rare categories get enough rows to say
anything about. A uniform random sample of 250 would contain perhaps three
`service` examples, which supports no conclusion at all.
"""
import argparse
import json
from collections import Counter

import pandas as pd

from src.config import DATA_PROCESSED, ISSUE_LABELS, RANDOM_SEED, ROOT
from src.data.weak_labels import label_review

SHEET = ROOT / "docs" / "label_validation_sheet.csv"
RESULT = ROOT / "docs" / "reports" / "label_validation.json"
RESULT_MD = ROOT / "docs" / "reports" / "label_validation.md"
MIN_PER_LABEL = 15


def draw_sample(n: int = 250, seed: int = RANDOM_SEED) -> pd.DataFrame:
    df = pd.read_csv(DATA_PROCESSED / "dataset.csv")
    df["text"] = df["text"].fillna("")

    picked, seen = [], set()

    # Guarantee a floor per label, including the "nothing fired" bucket, so no
    # category is represented by too few rows to judge.
    for label in ISSUE_LABELS:
        col = f"issue_{label}"
        if col not in df.columns:
            continue
        pool = df[(df[col] == 1) & (~df.index.isin(seen))]
        take = pool.sample(min(MIN_PER_LABEL, len(pool)), random_state=seed) \
            if len(pool) else pool
        picked.append(take)
        seen.update(take.index)

    issue_cols = [f"issue_{i}" for i in ISSUE_LABELS if f"issue_{i}" in df.columns]
    unlabelled = df[(df[issue_cols].sum(axis=1) == 0) & (~df.index.isin(seen))]
    if len(unlabelled):
        take = unlabelled.sample(min(MIN_PER_LABEL * 2, len(unlabelled)),
                                 random_state=seed)
        picked.append(take)
        seen.update(take.index)

    # Fill the rest at random so the sample still reflects the real distribution.
    remaining = df[~df.index.isin(seen)]
    shortfall = max(n - len(seen), 0)
    if shortfall and len(remaining):
        picked.append(remaining.sample(min(shortfall, len(remaining)),
                                       random_state=seed))

    sample = pd.concat(picked).drop_duplicates(subset=["review_id"]) \
        .sample(frac=1, random_state=seed).reset_index(drop=True)

    sample["predicted_labels"] = sample["text"].map(
        lambda t: ",".join(label_review(t)))
    sample["true_labels"] = ""          # the human fills this in
    sample["notes"] = ""
    return sample[["review_id", "text", "predicted_labels", "true_labels", "notes"]]


def cmd_sample(n: int, seed: int):
    sample = draw_sample(n, seed)
    SHEET.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(SHEET, index=False)
    print(f"[validate] wrote {len(sample)} rows -> {SHEET}")
    print("[validate] Fill the `true_labels` column with comma-separated categories")
    print(f"[validate] Valid categories: {', '.join(ISSUE_LABELS)}")
    print("[validate] Leave blank when the review raises no issue at all.")
    print("[validate] Then run: python -m src.data.validate_labels score")


def _parse(cell) -> set:
    if pd.isna(cell) or not str(cell).strip():
        return set()
    return {p.strip().lower() for p in str(cell).split(",") if p.strip()}


def cmd_score():
    if not SHEET.exists():
        raise SystemExit(f"{SHEET} not found. Run the `sample` command first.")

    df = pd.read_csv(SHEET, keep_default_na=False)
    annotated = df[df["true_labels"].astype(str).str.strip() != ""]

    # A blank row is genuinely ambiguous: unannotated, or annotated as "no
    # issue"? Require an explicit marker so the two cannot be confused.
    explicit_none = df[df["true_labels"].astype(str).str.strip().str.lower() == "none"]
    annotated = pd.concat([annotated, explicit_none]).drop_duplicates(subset=["review_id"])

    if len(annotated) == 0:
        raise SystemExit(
            "No annotations found. Fill the `true_labels` column first.\n"
            "Use the literal word `none` for reviews that raise no issue, so an "
            "annotated-as-empty row is distinguishable from an unannotated one."
        )

    per_label, totals = {}, Counter()
    for label in ISSUE_LABELS:
        tp = fp = fn = 0
        for _, row in annotated.iterrows():
            pred = _parse(row["predicted_labels"])
            true = _parse(row["true_labels"]) - {"none"}
            in_pred, in_true = label in pred, label in true
            tp += in_pred and in_true
            fp += in_pred and not in_true
            fn += in_true and not in_pred
        precision = tp / (tp + fp) if (tp + fp) else None
        recall = tp / (tp + fn) if (tp + fn) else None
        f1 = (2 * precision * recall / (precision + recall)
              if precision and recall else None)
        per_label[label] = {
            "precision": round(precision, 3) if precision is not None else None,
            "recall": round(recall, 3) if recall is not None else None,
            "f1": round(f1, 3) if f1 is not None else None,
            "true_positives": tp, "false_positives": fp, "false_negatives": fn,
            "human_support": tp + fn,
        }
        totals["tp"] += tp
        totals["fp"] += fp
        totals["fn"] += fn

    exact = sum(_parse(r["predicted_labels"]) == (_parse(r["true_labels"]) - {"none"})
                for _, r in annotated.iterrows()) / len(annotated)
    micro_p = totals["tp"] / (totals["tp"] + totals["fp"]) if (totals["tp"] + totals["fp"]) else 0
    micro_r = totals["tp"] / (totals["tp"] + totals["fn"]) if (totals["tp"] + totals["fn"]) else 0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) else 0
    scored = [m["f1"] for m in per_label.values() if m["f1"] is not None]

    unreliable = [l for l, m in per_label.items()
                  if m["precision"] is not None and m["precision"] < 0.7]

    out = {
        "annotated_rows": len(annotated),
        "total_rows_in_sheet": len(df),
        "exact_match_rate": round(exact, 3),
        "micro_precision": round(micro_p, 3),
        "micro_recall": round(micro_r, 3),
        "micro_f1": round(micro_f1, 3),
        "macro_f1": round(sum(scored) / len(scored), 3) if scored else None,
        "per_label": per_label,
        "labels_below_precision_0_7": unreliable,
        "verdict": ("Lexicon is unreliable for: " + ", ".join(unreliable)
                    if unreliable else
                    "No label falls below 0.7 precision on this sample."),
    }
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps(out, indent=2))
    _write_markdown(out)
    print(json.dumps({k: out[k] for k in
                      ("annotated_rows", "micro_f1", "macro_f1", "verdict")}, indent=2))
    print(f"[validate] wrote {RESULT_MD}")


def _write_markdown(out: dict):
    lines = [
        "# Label validation",
        "",
        f"Human-checked sample of **{out['annotated_rows']}** reviews "
        f"(sheet contains {out['total_rows_in_sheet']}).",
        "",
        "This measures the keyword lexicon against human judgement. It is the "
        "ceiling on every issue-classification number in this project: a model "
        "that perfectly reproduces a 70%-accurate labelling function is, at best, "
        "70% accurate about reality.",
        "",
        "| Category | Precision | Recall | F1 | Human support |",
        "|---|---|---|---|---|",
    ]
    fmt = lambda v: f"{v:.3f}" if isinstance(v, (int, float)) else "—"
    for label, m in out["per_label"].items():
        lines.append(f"| {label} | {fmt(m['precision'])} | {fmt(m['recall'])} "
                     f"| {fmt(m['f1'])} | {m['human_support']} |")

    lines += [
        "",
        f"- Micro precision / recall / F1: **{out['micro_precision']} / "
        f"{out['micro_recall']} / {out['micro_f1']}**",
        f"- Macro F1: **{out['macro_f1']}**",
        f"- Exact label-set match: **{out['exact_match_rate']}**",
        "",
        "## Verdict",
        "",
        out["verdict"],
        "",
        "Where precision is below ~0.7 the model is being trained to reproduce a "
        "mistake, and the fix belongs in `src/data/weak_labels.py`, not in more "
        "training epochs. Where recall is low, the lexicon is missing vocabulary "
        "and the model will systematically under-report that category.",
        "",
        "## Reporting these numbers",
        "",
        "Two figures must be quoted separately and never multiplied away:",
        "",
        "1. **Model vs weak labels** — in `baseline_report.json`. How well the "
        "model learned the labelling function.",
        "2. **Weak labels vs humans** — this table. How well the labelling "
        "function reflects the actual task.",
    ]
    RESULT_MD.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Validate the weak issue labels.")
    sub = ap.add_subparsers(dest="command", required=True)
    s = sub.add_parser("sample", help="Draw a stratified sample to annotate.")
    s.add_argument("--n", type=int, default=250)
    s.add_argument("--seed", type=int, default=RANDOM_SEED)
    sub.add_parser("score", help="Score a filled-in annotation sheet.")
    a = ap.parse_args()

    if a.command == "sample":
        cmd_sample(a.n, a.seed)
    else:
        cmd_score()
