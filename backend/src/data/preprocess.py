"""Text cleaning shared by training and inference.

Kept deliberately light: the transformer path wants near-raw text, and
over-aggressive cleaning destroys signal the model can use (negation,
punctuation-as-emphasis). We normalise, we do not sanitise.
"""
import re
import unicodedata

_URL = re.compile(r"https?://\S+|www\.\S+")
_HTML = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")
# Latin letters, digits, and common punctuation. Used only to detect whether
# the text is plausibly English, never to strip characters.
_LATIN = re.compile(r"[a-zA-Z]")


def clean_text(text: str) -> str:
    """Normalise a raw review into model-ready text."""
    if text is None:
        return ""
    text = unicodedata.normalize("NFKC", str(text))
    text = _HTML.sub(" ", text)
    text = _URL.sub(" ", text)
    text = _WS.sub(" ", text).strip()
    return text


def token_count(text: str) -> int:
    return len([t for t in clean_text(text).split() if t])


def is_probably_english(text: str) -> bool:
    """Cheap heuristic, not a language detector.

    Returns False when fewer than 40% of characters are Latin letters, which
    catches Cyrillic/CJK/emoji-only input. Documented as a known limitation:
    it will not distinguish English from other Latin-script languages.
    """
    text = clean_text(text)
    if not text:
        return False
    letters = len(_LATIN.findall(text))
    return letters / max(len(text), 1) >= 0.4


def input_quality_flags(text: str) -> dict:
    """Structured reasons a prediction may be untrustworthy."""
    cleaned = clean_text(text)
    return {
        "empty": len(cleaned) == 0,
        "too_short": 0 < token_count(cleaned) < 3,
        "non_latin_script": bool(cleaned) and not is_probably_english(cleaned),
    }
