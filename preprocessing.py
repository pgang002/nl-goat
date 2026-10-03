"""
Text cleaning/normalization utilities shared across the extraction
pipeline.
"""
from __future__ import annotations

import re

_WHITESPACE_RE = re.compile(r"\s+")
_URL_RE = re.compile(r"https?://\S+")


def clean_text(text: str) -> str:
    """Basic normalization: strip URLs, collapse whitespace, trim."""
    if not text:
        return ""
    text = _URL_RE.sub("", text)
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


def sentence_split(text: str) -> list[str]:
    """
    Lightweight sentence splitter for advisory text. Good enough for the
    fairly formulaic sentence structure of CVE descriptions; swap for
    spaCy's sentencizer if advisory/blog text needs more robust splitting.
    """
    text = clean_text(text)
    if not text:
        return []
    # Split on '. ' but avoid breaking on version numbers like "2.3.32."
    parts = re.split(r"(?<!\d)\.\s+(?=[A-Z])", text)
    return [p.strip().rstrip(".") + "." for p in parts if p.strip()]
