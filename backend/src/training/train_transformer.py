"""Main model: one shared DistilBERT encoder, two heads.

Multi-task on purpose. Sentiment and issue category are the same linguistic
judgement viewed twice ("arrived smashed" is both negative and a defect), so
a shared encoder gets more signal per example than two separate fine-tunes,
and it halves inference cost -- one forward pass returns both outputs.

Heads:
  sentiment -> 3-way softmax,  CrossEntropyLoss
  issues    -> N-way sigmoid,  BCEWithLogitsLoss (multi-label)
  total loss = ce + issue_loss_weight * bce

Requires torch + transformers. If you only need the pipeline to run, the
baseline is enough -- this script is the "beat the baseline" attempt.

    python -m src.training.train_transformer --epochs 3 --batch-size 16
"""
import argparse
import json
import time

import numpy as np
import pandas as pd

from src.config import (DATA_PROCESSED, ISSUE_LABELS, MODELS_DIR, RANDOM_SEED,
                        REPORTS_DIR, SENTIMENT_LABELS)
from src.data.splits import group_split, leakage_check

ISSUE_COLS = [f"issue_{i}" for i in ISSUE_LABELS]
DEFAULT_MODEL = "distilbert-base-uncased"


def _require_torch():
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as e:
        raise SystemExit(
            "torch and transformers are required for this script.\n"
            "  pip install torch transformers\n"
            f"(import failed: {e})"
        )


def build_model(base_model: str, n_issues: int):
    import torch
    import torch.nn as nn
    from transformers import AutoModel

    class MultiTaskReviewModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = AutoModel.from_pretrained(base_model)
            hidden = self.encoder.config.hidden_size
            self.dropout = nn.Dropout(0.1)
            self.sentiment_head = nn.Linear(hidden, len(SENTIMENT_LABELS))
            self.issue_head = nn.Linear(hidden, n_issues)

        def forward(self, input_ids, attention_mask):
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            # Mean-pool over real tokens; more stable than [CLS] on short text.
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            pooled = self.dropout(pooled)
            return self.sentiment_head(pooled), self.issue_head(pooled)

    return MultiTaskReviewModel()


def make_dataset(df, tokenizer, max_len: int):
    import torch

    class ReviewDataset(torch.utils.data.Dataset):
        def __init__(self, frame):
            self.texts = frame["text"].fillna("").tolist()
            self.sent = [SENTIMENT_LABELS.index(s) for s in frame["sentiment"]]
            self.issues = frame[ISSUE_COLS].values.astype("float32")

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, i):
            enc = tokenizer(self.texts[i], truncation=True, max_length=max_len,
                            padding="max_length", return_tensors="pt")
            return {
                "input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "sentiment": torch.tensor(self.sent[i], dtype=torch.long),
                "issues": torch.tensor(self.issues[i]),
            }

    return ReviewDataset(df)


def train(epochs=3, batch_size=16, lr=2e-5, max_len=192, base_model=DEFAULT_MODEL,
          issue_loss_weight=1.0, seed=RANDOM_SEED):
    _require_torch()
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer

    torch.manual_seed(seed)
    np.random.seed(seed)

    df = pd.read_csv(DATA_PROCESSED / "dataset.csv")
    df["text"] = df["text"].fillna("")
    train_df, val_df, test_df = group_split(df, seed=seed)
    leak = leakage_check(train_df, test_df)
    if not leak["clean"]:
        raise SystemExit(f"Leakage detected, refusing to train: {leak}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] device={device} rows={len(train_df)}")

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = build_model(base_model, len(ISSUE_COLS)).to(device)

    train_dl = DataLoader(make_dataset(train_df, tokenizer, max_len),
                          batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(make_dataset(val_df, tokenizer, max_len),
                        batch_size=batch_size)

    # Positive weighting so rare issue labels are not optimised away.
    pos = train_df[ISSUE_COLS].values.sum(axis=0)
    neg = len(train_df) - pos
    pos_weight = torch.tensor(np.clip(neg / np.clip(pos, 1, None), 1, 20),
                              dtype=torch.float32, device=device)

    ce = nn.CrossEntropyLoss()
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = max(len(train_dl) * epochs, 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=total_steps,
                                                pct_start=0.1)

    history, t0 = [], time.time()
    for epoch in range(epochs):
        model.train()
        running = 0.0
        for batch in train_dl:
            opt.zero_grad()
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            s_logits, i_logits = model(ids, mask)
            loss = ce(s_logits, batch["sentiment"].to(device)) + \
                issue_loss_weight * bce(i_logits, batch["issues"].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            running += loss.item()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for batch in val_dl:
                s_logits, _ = model(batch["input_ids"].to(device),
                                    batch["attention_mask"].to(device))
                pred = s_logits.argmax(-1).cpu()
                correct += (pred == batch["sentiment"]).sum().item()
                total += len(pred)
        entry = {"epoch": epoch + 1,
                 "train_loss": round(running / max(len(train_dl), 1), 4),
                 "val_sentiment_accuracy": round(correct / max(total, 1), 4)}
        history.append(entry)
        print(f"[train] {entry}")

    out_dir = MODELS_DIR / "transformer"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "model.pt")
    tokenizer.save_pretrained(out_dir)
    (out_dir / "meta.json").write_text(json.dumps({
        "base_model": base_model, "issue_labels": ISSUE_LABELS,
        "sentiment_labels": SENTIMENT_LABELS, "max_len": max_len,
        "epochs": epochs, "lr": lr, "seed": seed,
    }, indent=2))
    (REPORTS_DIR / "transformer_history.json").write_text(json.dumps({
        "history": history, "train_seconds": round(time.time() - t0, 1),
        "leakage_check": leak,
    }, indent=2))
    print(f"[train] saved -> {out_dir}")
    print("[train] now compare against docs/reports/baseline_report.json before "
          "claiming the transformer is better.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-len", type=int, default=192)
    ap.add_argument("--base-model", default=DEFAULT_MODEL)
    a = ap.parse_args()
    train(a.epochs, a.batch_size, a.lr, a.max_len, a.base_model)
