"""Tests for the modules added to close the brief's remaining requirements:
experiment tracking, transfer evaluation, model comparison and label
validation."""
import json

import pandas as pd
import pytest

from src.config import ISSUE_LABELS
from src.data.build_dataset import _generate_sample
from src.data.preprocess import clean_text
from src.data.weak_labels import label_frame


# --- tracking --------------------------------------------------------------

@pytest.fixture
def tracking(tmp_path, monkeypatch):
    import src.training.tracking as t
    monkeypatch.setattr(t, "RUNS_FILE", tmp_path / "runs.jsonl")
    monkeypatch.setattr(t, "LEADERBOARD", tmp_path / "leaderboard.md")
    return t


def test_log_run_appends_a_record(tracking):
    tracking.log_run("m1", {"seed": 1}, {"issue_macro_f1": 0.5})
    tracking.log_run("m2", {"seed": 2}, {"issue_macro_f1": 0.7})
    assert len(tracking.load_runs()) == 2


def test_each_run_gets_a_unique_id(tracking):
    a = tracking.log_run("m", {}, {})
    b = tracking.log_run("m", {}, {})
    assert a["run_id"] != b["run_id"]


def test_run_record_captures_provenance(tracking):
    rec = tracking.log_run("m", {"lr": 0.01}, {"issue_macro_f1": 0.5})
    for key in ("run_id", "timestamp", "git_commit", "python", "params", "metrics"):
        assert key in rec


def test_runs_survive_a_corrupt_line(tracking):
    tracking.log_run("m", {}, {"issue_macro_f1": 0.5})
    with tracking.RUNS_FILE.open("a") as fh:
        fh.write("{not json\n")
    tracking.log_run("m2", {}, {"issue_macro_f1": 0.6})
    assert len(tracking.load_runs()) == 2


def test_leaderboard_sorts_best_first(tracking):
    tracking.log_run("weak", {}, {"issue_macro_f1": 0.10})
    tracking.log_run("strong", {}, {"issue_macro_f1": 0.90})
    text = tracking.LEADERBOARD.read_text()
    assert text.index("strong") < text.index("weak")


def test_leaderboard_warns_about_incomparable_data(tracking):
    tracking.log_run("m", {}, {"issue_macro_f1": 0.5})
    assert "hash matches" in tracking.LEADERBOARD.read_text()


def test_dataset_fingerprint_detects_changed_content(tracking, tmp_path):
    path = tmp_path / "d.csv"
    path.write_text("a,b\n1,2\n")
    first = tracking.dataset_fingerprint(path)
    path.write_text("a,b\n1,3\n")
    assert tracking.dataset_fingerprint(path)["sha256"] != first["sha256"]


def test_dataset_fingerprint_is_stable_for_identical_content(tracking, tmp_path):
    p1, p2 = tmp_path / "a.csv", tmp_path / "b.csv"
    p1.write_text("x,y\n1,2\n")
    p2.write_text("x,y\n1,2\n")
    assert (tracking.dataset_fingerprint(p1)["sha256"]
            == tracking.dataset_fingerprint(p2)["sha256"])


def test_dataset_fingerprint_handles_a_missing_file(tracking, tmp_path):
    assert tracking.dataset_fingerprint(tmp_path / "nope.csv")["sha256"] is None


def test_summarise_extracts_headline_metrics(tracking):
    report = {"sentiment": {"accuracy": 0.8, "macro_f1": 0.7},
              "issues_summary": {"macro_f1": 0.6, "micro_f1": 0.65, "threshold": 0.35}}
    out = tracking.summarise_for_tracking(report)
    assert out["sentiment_macro_f1"] == 0.7
    assert out["issue_macro_f1"] == 0.6


def test_summarise_tolerates_a_partial_report(tracking):
    assert tracking.summarise_for_tracking({})["issue_macro_f1"] is None


def test_baseline_training_records_a_run():
    """The real pipeline must actually log, not just be capable of logging."""
    from src.config import REPORTS_DIR
    report_path = REPORTS_DIR / "baseline_report.json"
    if not report_path.exists():
        pytest.skip("baseline not trained in this environment")
    assert "run_id" in json.loads(report_path.read_text())


# --- transfer evaluation ---------------------------------------------------

def make_multi_category_df(n=900):
    df = _generate_sample(n, seed=21)
    df["text"] = df["text"].map(clean_text)
    df = df[df["text"].str.len() > 0].drop_duplicates(subset=["text"])
    df["sentiment"] = df["rating"].map(
        lambda r: "negative" if r <= 2 else ("neutral" if r < 4 else "positive"))
    return label_frame(df, "text")


def test_transfer_evaluates_a_single_category(tmp_path, monkeypatch):
    import src.training.evaluate_transfer as tr
    monkeypatch.setattr(tr, "REPORTS_DIR", tmp_path)
    df = make_multi_category_df()
    result = tr.evaluate_one(df, "Home")
    assert result["category"] == "Home"
    if not result["skipped"]:
        assert "in_domain" in result and "held_out" in result


def test_transfer_never_trains_on_the_held_out_category(monkeypatch):
    """The entire point: the evaluation category must be unseen."""
    from src.data.splits import holdout_category
    df = make_multi_category_df()
    rest, held = holdout_category(df, "Home")
    assert "Home" not in set(rest["category"])
    assert set(held["category"]) == {"Home"}


def test_transfer_drop_is_in_domain_minus_held_out():
    import src.training.evaluate_transfer as tr
    df = make_multi_category_df()
    r = tr.evaluate_one(df, "Tops")
    if not r["skipped"]:
        expected = round(r["in_domain"]["issue_macro_f1"] - r["held_out"]["issue_macro_f1"], 4)
        assert r["drop"]["issue_macro_f1"] == expected


def test_transfer_skips_a_category_with_too_few_rows():
    import src.training.evaluate_transfer as tr
    df = make_multi_category_df()
    tiny = df[df["category"] == "Home"].head(5)
    combined = pd.concat([df[df["category"] != "Home"], tiny], ignore_index=True)
    assert tr.evaluate_one(combined, "Home")["skipped"] is True


def test_transfer_flags_bootstrap_data_as_meaningless():
    """A near-zero drop on randomly-assigned categories must not read as success."""
    import src.training.evaluate_transfer as tr
    from src.config import DATA_PROCESSED
    meta = DATA_PROCESSED / "dataset_meta.json"
    if not meta.exists():
        pytest.skip("no dataset built")
    if json.loads(meta.read_text()).get("source") == "bootstrap-sample":
        assert tr._is_bootstrap_data() is True
        assert "measure nothing" in tr.BOOTSTRAP_WARNING


# --- model comparison ------------------------------------------------------

@pytest.fixture
def comparison(tmp_path, monkeypatch):
    import src.training.compare_models as cm
    import src.training.tracking as t
    monkeypatch.setattr(t, "RUNS_FILE", tmp_path / "runs.jsonl")
    monkeypatch.setattr(t, "LEADERBOARD", tmp_path / "lb.md")
    monkeypatch.setattr(cm, "REPORT", tmp_path / "cmp.md")
    monkeypatch.setattr(cm, "REPORT_JSON", tmp_path / "cmp.json")
    return cm, t


def log(t, name, f1, sha="abc"):
    t.log_run(name, {}, {"issue_macro_f1": f1, "sentiment_macro_f1": f1,
                         "issue_micro_f1": f1},
              dataset={"sha256": sha, "rows": 100})


def test_comparison_requires_at_least_one_scored_run(comparison):
    cm, _ = comparison
    with pytest.raises(SystemExit):
        cm.compare()


def test_comparison_reports_baseline_only(comparison):
    cm, t = comparison
    log(t, "baseline-tfidf", 0.6)
    assert "Only the baseline" in cm.compare()["verdict"]


def test_comparison_prefers_baseline_when_the_gain_is_noise(comparison):
    cm, t = comparison
    log(t, "baseline-tfidf", 0.700)
    log(t, "transformer-distilbert", 0.705)
    out = cm.compare()
    assert "noise band" in out["verdict"]
    assert "Deploy the baseline" in out["verdict"]


def test_comparison_accepts_the_transformer_on_a_real_gain(comparison):
    cm, t = comparison
    log(t, "baseline-tfidf", 0.60)
    log(t, "transformer-distilbert", 0.70)
    out = cm.compare()
    assert out["issue_macro_f1_gain"] == pytest.approx(0.10, abs=1e-6)
    assert "Transformer wins" in out["verdict"]


def test_comparison_reports_when_the_baseline_wins(comparison):
    cm, t = comparison
    log(t, "baseline-tfidf", 0.80)
    log(t, "transformer-distilbert", 0.60)
    assert "Baseline wins" in cm.compare()["verdict"]


def test_comparison_excludes_runs_on_different_data(comparison):
    """Metrics from different datasets are not comparable and must not be mixed."""
    cm, t = comparison
    log(t, "baseline-tfidf", 0.6, sha="aaa")
    log(t, "transformer-distilbert", 0.7, sha="aaa")
    log(t, "baseline-other", 0.99, sha="zzz")
    out = cm.compare()
    assert out["dataset_sha"] == "aaa"
    assert out["ignored_runs_on_other_data"] == 1


def test_comparison_writes_a_cost_table(comparison):
    cm, t = comparison
    log(t, "baseline-tfidf", 0.6)
    cm.compare()
    text = cm.REPORT.read_text()
    assert "Free-tier hosting" in text
    assert "Interpretability" in text


# --- label validation ------------------------------------------------------

def test_sample_covers_every_label_it_can(tmp_path, monkeypatch):
    import src.data.validate_labels as vl
    sample = vl.draw_sample(n=120)
    assert len(sample) > 0
    assert set(sample.columns) == {"review_id", "text", "predicted_labels",
                                   "true_labels", "notes"}


def test_sample_leaves_the_human_column_blank():
    import src.data.validate_labels as vl
    sample = vl.draw_sample(n=80)
    assert (sample["true_labels"] == "").all()


def test_sample_includes_reviews_with_no_predicted_label():
    """The taxonomy's blind spot has to be inspectable too."""
    import src.data.validate_labels as vl
    sample = vl.draw_sample(n=200)
    assert (sample["predicted_labels"] == "").any()


def test_sample_is_deterministic():
    import src.data.validate_labels as vl
    a = vl.draw_sample(n=100, seed=3)["review_id"].tolist()
    b = vl.draw_sample(n=100, seed=3)["review_id"].tolist()
    assert a == b


def test_scoring_computes_precision_and_recall(tmp_path, monkeypatch):
    import src.data.validate_labels as vl
    sheet = tmp_path / "sheet.csv"
    pd.DataFrame({
        "review_id": ["a", "b", "c"],
        "text": ["late", "crushed box", "fine"],
        "predicted_labels": ["delivery", "packaging", ""],
        "true_labels": ["delivery", "delivery", "none"],
        "notes": ["", "", ""],
    }).to_csv(sheet, index=False)
    monkeypatch.setattr(vl, "SHEET", sheet)
    monkeypatch.setattr(vl, "RESULT", tmp_path / "r.json")
    monkeypatch.setattr(vl, "RESULT_MD", tmp_path / "r.md")

    vl.cmd_score()
    out = json.loads((tmp_path / "r.json").read_text())
    # delivery: predicted once, true twice -> precision 1.0, recall 0.5
    assert out["per_label"]["delivery"]["precision"] == 1.0
    assert out["per_label"]["delivery"]["recall"] == 0.5
    # packaging: predicted once, never true -> precision 0.0
    assert out["per_label"]["packaging"]["precision"] == 0.0


def test_scoring_flags_unreliable_labels(tmp_path, monkeypatch):
    import src.data.validate_labels as vl
    sheet = tmp_path / "s.csv"
    pd.DataFrame({
        "review_id": ["a", "b"], "text": ["x", "y"],
        "predicted_labels": ["packaging", "packaging"],
        "true_labels": ["delivery", "delivery"], "notes": ["", ""],
    }).to_csv(sheet, index=False)
    monkeypatch.setattr(vl, "SHEET", sheet)
    monkeypatch.setattr(vl, "RESULT", tmp_path / "r.json")
    monkeypatch.setattr(vl, "RESULT_MD", tmp_path / "r.md")
    vl.cmd_score()
    out = json.loads((tmp_path / "r.json").read_text())
    assert "packaging" in out["labels_below_precision_0_7"]
    assert "unreliable" in out["verdict"]


def test_scoring_refuses_an_unannotated_sheet(tmp_path, monkeypatch):
    import src.data.validate_labels as vl
    sheet = tmp_path / "s.csv"
    pd.DataFrame({"review_id": ["a"], "text": ["x"], "predicted_labels": ["delivery"],
                  "true_labels": [""], "notes": [""]}).to_csv(sheet, index=False)
    monkeypatch.setattr(vl, "SHEET", sheet)
    with pytest.raises(SystemExit) as e:
        vl.cmd_score()
    assert "Fill the" in str(e.value)


def test_scoring_distinguishes_none_from_unannotated(tmp_path, monkeypatch):
    """A blank cell means 'not checked'; the word none means 'checked, no issue'."""
    import src.data.validate_labels as vl
    sheet = tmp_path / "s.csv"
    pd.DataFrame({
        "review_id": ["a", "b"], "text": ["x", "y"],
        "predicted_labels": ["", "delivery"],
        "true_labels": ["none", ""], "notes": ["", ""],
    }).to_csv(sheet, index=False)
    monkeypatch.setattr(vl, "SHEET", sheet)
    monkeypatch.setattr(vl, "RESULT", tmp_path / "r.json")
    monkeypatch.setattr(vl, "RESULT_MD", tmp_path / "r.md")
    vl.cmd_score()
    assert json.loads((tmp_path / "r.json").read_text())["annotated_rows"] == 1


def test_validation_markdown_separates_the_two_claims(tmp_path, monkeypatch):
    import src.data.validate_labels as vl
    sheet = tmp_path / "s.csv"
    pd.DataFrame({"review_id": ["a"], "text": ["late"],
                  "predicted_labels": ["delivery"], "true_labels": ["delivery"],
                  "notes": [""]}).to_csv(sheet, index=False)
    monkeypatch.setattr(vl, "SHEET", sheet)
    monkeypatch.setattr(vl, "RESULT", tmp_path / "r.json")
    monkeypatch.setattr(vl, "RESULT_MD", tmp_path / "r.md")
    vl.cmd_score()
    text = (tmp_path / "r.md").read_text()
    assert "Model vs weak labels" in text
    assert "Weak labels vs humans" in text


# --- EDA -------------------------------------------------------------------

def test_eda_script_runs_and_writes_a_report(tmp_path, monkeypatch):
    import runpy
    import sys
    from pathlib import Path
    from src.config import REPORTS_DIR

    script = Path(__file__).resolve().parents[1] / "notebooks" / "01_eda.py"
    assert script.exists(), "EDA script missing"
    sys.argv = [str(script)]
    runpy.run_path(str(script), run_name="__main__")
    assert (REPORTS_DIR / "eda_report.md").exists()


def test_eda_report_explains_why_accuracy_is_misleading():
    from src.config import REPORTS_DIR
    path = REPORTS_DIR / "eda_report.md"
    if not path.exists():
        pytest.skip("EDA not run")
    assert "macro-F1" in path.read_text()
