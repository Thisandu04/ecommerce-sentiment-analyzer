import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

NEGATIONS = {"not", "no", "nor", "never", "n't", "none", "nothing"}
STOP_WORDS = ENGLISH_STOP_WORDS - NEGATIONS

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    words = [w for w in text.split() if w not in STOP_WORDS]
    return " ".join(words)