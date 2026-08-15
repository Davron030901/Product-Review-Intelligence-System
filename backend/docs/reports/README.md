# Generated reports

Everything here is produced by scripts. Do not edit by hand — rerun the command.

| File | Produced by |
|---|---|
| `eda_report.md` | `python notebooks/01_eda.py` |
| `baseline_report.json` | `python -m src.training.train_baseline` |
| `transformer_report.json` | `python -m src.training.train_transformer` |
| `transfer_report.md` / `.json` | `python -m src.training.evaluate_transfer` |
| `model_comparison.md` / `.json` | `python -m src.training.compare_models` |
| `label_validation.md` / `.json` | `python -m src.data.validate_labels score` |
| `runs.jsonl`, `leaderboard.md` | appended automatically by every training run |

`label_validation.md` is absent until a human has annotated the sample. That
absence is meaningful: it means the issue metrics elsewhere in this directory
are unvalidated.
