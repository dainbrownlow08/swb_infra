"""Shared text helpers for the token-based extractors.

Extracted verbatim from the retired per-token rate extractors (repetition_rate,
filler_word_rate, pronoun_rate — deprecated *columns*, superseded by the
per-second variants). The trusted extractors import from here: tokenize and
count_repetitions (repetition family), DEFAULT_FILLERS and count_filler_hits
(filler_word_per_second), _get_nlp and strip_bracket_tokens (pronoun_per_second).
"""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable

SPACY_MODEL = "en_core_web_sm"

# Lazy module-level cache so each ProcessPoolExecutor worker loads spaCy once.
_NLP = None


def _get_nlp():
    global _NLP
    if _NLP is None:
        import spacy
        _NLP = spacy.load(SPACY_MODEL)
    return _NLP


def strip_bracket_tokens(text: str) -> str:
    """Remove whole-bracket tokens like [noise], [laughter] before tokenization.

    A whole-bracket token starts with '[' and ends with ']'. Inline markers
    like 'i[t]-' (partial words) do NOT start with '[' so they are kept.
    """
    return " ".join(
        t for t in text.split()
        if not (t.startswith("[") and t.endswith("]"))
    )


def tokenize(text: str) -> list[str]:
    """Lowercase whitespace-split with whole-bracket tokens stripped."""
    return [
        w for w in text.lower().split()
        if not (w.startswith("[") and w.endswith("]"))
    ]


def count_repetitions(words: list[str]) -> int:
    """Number of unique tokens that appear at least twice (legacy semantics)."""
    counts: dict[str, int] = {}
    reps = 0
    for w in words:
        if w in counts:
            counts[w] += 1
            if counts[w] == 2:
                reps += 1
        else:
            counts[w] = 1
    return reps


DEFAULT_FILLERS: frozenset[str] = frozenset({
    "um", "uh", "like", "you know", "i mean",
    "so", "well", "i guess", "basically", "er",
})


def count_filler_hits(words: list[str], fillers: Iterable[str]) -> int:
    by_len: dict[int, set[str]] = defaultdict(set)
    for f in fillers:
        by_len[len(f.split())].add(f)
    if not by_len:
        return 0
    max_len = max(by_len)

    hits = 0
    i = 0
    n = len(words)
    while i < n:
        matched = False
        for k in range(min(max_len, n - i), 0, -1):
            phrase = " ".join(words[i:i + k]) if k > 1 else words[i]
            if phrase in by_len.get(k, ()):
                hits += 1
                i += k
                matched = True
                break
        if not matched:
            i += 1
    return hits
