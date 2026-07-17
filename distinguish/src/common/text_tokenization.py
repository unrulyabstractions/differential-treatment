"""Word tokenization for lexical counting.

Deliberately simple: lowercase word tokens (letters plus internal apostrophes),
matching what the marked-words literature operates on.
"""

from __future__ import annotations

import re
from collections import Counter

_WORD_PATTERN = re.compile(r"[a-z]+(?:'[a-z]+)?")
# Typographic apostrophes (U+2019, U+02BC) must not split contractions, or the
# two sets' quote styles masquerade as lexical differences.
_APOSTROPHE_TRANSLATION = str.maketrans({"\u2019": "'", "\u02bc": "'"})


def tokenize_words(text: str) -> list[str]:
    """Lowercase word tokens from a text."""
    return _WORD_PATTERN.findall(text.lower().translate(_APOSTROPHE_TRANSLATION))


def count_words(texts: list[str]) -> Counter:
    """Aggregate word counts over a list of texts."""
    counts: Counter = Counter()
    for text in texts:
        counts.update(tokenize_words(text))
    return counts
