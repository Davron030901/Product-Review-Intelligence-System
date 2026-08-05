"""Inference: raw review text -> the JSON contract the frontend consumes.

Loads the transformer if it exists, otherwise the baseline. The output shape
is identical either way, so the frontend never needs to know which model is
serving.

The rule this module enforces: the system is allowed to say "I don't know".
Empty, three-word, or non-Latin-script input produces a structured response
with low_confidence=true and a reason, not a fabricated answer and not a 500.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np

from src.config import (ISSUE_LABELS, ISSUE_THRESHOLD, MIN_TOKENS_FOR_CONFIDENCE,
                        MODELS_DIR, MODEL_VERSION, SENTIMENT_LOW_CONFIDENCE)
from src.data.preprocess import clean_text, input_quality_flags, token_count

MAX_CHARS = 20000  # hard input cap; longer text is truncated, not rejected


class ModelNotTrained(RuntimeError):
    pass


class ReviewPredictor:
    def __init__(self, models_dir: Path = None):
        self.models_dir = Path(models_dir or MODELS_DIR)
        self.backend = None
        self._baseline = None
        self._transformer = None
        self._load()

    # --- loading ----------------------------------------------------------
    def _load(self):
        tdir = self.models_dir / "transformer"
        if (tdir / "model.pt").exists():
            try:
                self._load_transformer(tdir)
                self.backend = "transformer"
                return
            except Exception as e:  # fall back rather than fail to start
                print(f"[predict] transformer load failed ({e}); using baseline.")
        bpath = self.models_dir / "baseline.joblib"
        if bpath.exists():
            self._baseline = joblib.load(bpath)
            self.backend = "baseline"
            return
        raise ModelNotTrained(
            f"No model found in {self.models_dir}. Run:\n"
            "  python -m src.data.build_dataset\n"
            "  python -m src.training.train_baseline"
        )

    def _load_transformer(self, tdir: Path):
        import torch
        from transformers import AutoTokenizer

        from src.training.train_transformer import build_model

        meta = json.loads((tdir / "meta.json").read_text())
        model = build_model(meta["base_model"], len(meta["issue_labels"]))
        model.load_state_dict(torch.load(tdir / "model.pt", map_location="cpu"))
        model.eval()
        self._transformer = {
            "model": model,
            "tokenizer": AutoTokenizer.from_pretrained(str(tdir)),
            "meta": meta,
            "torch": torch,
        }

    @property
    def model_version(self) -> str:
        return f"{MODEL_VERSION}" if self.backend == "baseline" else "v1-transformer"

    # --- scoring ----------------------------------------------------------
    def _score_baseline(self, text: str):
        art = self._baseline
        sent_pipe = art["sentiment_pipe"]
        probs = sent_pipe.predict_proba([text])[0]
        classes = list(sent_pipe.named_steps["clf"].classes_)
        idx = int(np.argmax(probs))
        sentiment = {"label": classes[idx], "confidence": float(probs[idx])}
        issue_probs = art["issue_pipe"].predict_proba([text])[0]
        issues = dict(zip(art["issue_labels"], map(float, issue_probs)))
        return sentiment, issues

    def _score_transformer(self, text: str):
        t = self._transformer
        torch = t["torch"]
        enc = t["tokenizer"](text, truncation=True, max_length=t["meta"]["max_len"],
                             return_tensors="pt")
        with torch.no_grad():
            s_logits, i_logits = t["model"](enc["input_ids"], enc["attention_mask"])
        s_probs = torch.softmax(s_logits, dim=-1)[0].tolist()
        labels = t["meta"]["sentiment_labels"]
        idx = int(np.argmax(s_probs))
        sentiment = {"label": labels[idx], "confidence": float(s_probs[idx])}
        i_probs = torch.sigmoid(i_logits)[0].tolist()
        issues = dict(zip(t["meta"]["issue_labels"], map(float, i_probs)))
        return sentiment, issues

    # --- public API -------------------------------------------------------
    def predict(self, text: str, category: str = None,
                issue_threshold: float = ISSUE_THRESHOLD) -> dict:
        raw = text if isinstance(text, str) else ""
        truncated = len(raw) > MAX_CHARS
        cleaned = clean_text(raw[:MAX_CHARS])
        flags = input_quality_flags(cleaned)
        reasons = []

        if flags["empty"]:
            return self._unknown(["Review text is empty."], category, truncated)

        if self.backend == "transformer":
            sentiment, issue_probs = self._score_transformer(cleaned)
        else:
            sentiment, issue_probs = self._score_baseline(cleaned)

        issues = sorted(
            ({"category": c, "confidence": round(p, 4)}
             for c, p in issue_probs.items() if p >= issue_threshold),
            key=lambda d: d["confidence"], reverse=True,
        )

        n_tok = token_count(cleaned)
        if n_tok < MIN_TOKENS_FOR_CONFIDENCE:
            reasons.append(f"Review is only {n_tok} word(s) long.")
        if flags["non_latin_script"]:
            reasons.append("Text does not look like English; the model was trained "
                           "on English reviews only.")
        if sentiment["confidence"] < SENTIMENT_LOW_CONFIDENCE:
            reasons.append("Sentiment prediction is below the confidence threshold.")
        if not issues:
            reasons.append("No issue category scored above the threshold.")

        return {
            "sentiment": {"label": sentiment["label"],
                          "confidence": round(sentiment["confidence"], 4)},
            "issues": issues,
            "low_confidence": bool(reasons),
            "reasons": reasons,
            "input_category": category,
            "word_count": n_tok,
            "truncated": truncated,
            "model_version": self.model_version,
            "model_backend": self.backend,
            "processed_at": datetime.now(timezone.utc).isoformat(),
        }

    def predict_batch(self, texts, category: str = None) -> list:
        return [self.predict(t, category) for t in texts]

    def _unknown(self, reasons, category, truncated) -> dict:
        return {
            "sentiment": {"label": "unknown", "confidence": 0.0},
            "issues": [],
            "low_confidence": True,
            "reasons": reasons,
            "input_category": category,
            "word_count": 0,
            "truncated": truncated,
            "model_version": self.model_version,
            "model_backend": self.backend,
            "processed_at": datetime.now(timezone.utc).isoformat(),
        }


_PREDICTOR = None


def get_predictor() -> ReviewPredictor:
    """Process-wide singleton; the model is loaded once, not per request."""
    global _PREDICTOR
    if _PREDICTOR is None:
        _PREDICTOR = ReviewPredictor()
    return _PREDICTOR
