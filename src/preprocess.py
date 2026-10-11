"""Text cleaning for the fake news project."""
import re

URL_RE = re.compile(r"http\S+|www\.\S+")
AGENCY_RE = re.compile(r"^.{0,40}\((reuters)\)\s*-?\s*", re.IGNORECASE)  # avoids dataset leakage
NON_ALPHA_RE = re.compile(r"[^a-z\s]")
SPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Lowercase, strip URLs/agency tags/punctuation/numbers, normalise spaces."""
    text = str(text)
    text = AGENCY_RE.sub("", text)
    text = text.lower()
    text = URL_RE.sub(" ", text)
    text = NON_ALPHA_RE.sub(" ", text)
    return SPACE_RE.sub(" ", text).strip()
