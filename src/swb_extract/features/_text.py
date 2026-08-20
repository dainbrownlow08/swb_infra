"""Shared text helpers for the token-based extractors.

Extracted verbatim from the retired per-token rate extractors (repetition_rate,
filler_word_rate, pronoun_rate — deprecated *columns*, superseded by the
per-second variants). The trusted extractors import from here: tokenize and
count_repetitions (repetition family), DEFAULT_FILLERS and count_filler_hits
(filler_word_per_second), _get_nlp and strip_bracket_tokens (pronoun_per_second).
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Iterable

SPACY_MODEL = "en_core_web_sm"

# ms98 laughed-word notation: [laughter-yeah] = the speaker SAID "yeah" while
# laughing. Unlike [laughter]/[noise] (non-speech events), the inner word is
# real produced speech, so tokenization unwraps it instead of dropping it
# (2026-08-19 fix; 10,275 tokens in 5,714 utterances were silently dropped
# before). The inner form is kept verbatim — it may itself be a partial word
# like "no[t]-", which the partial-word policy already keeps. Corpus is
# all-lowercase for these markers (verified); matched case-insensitively anyway.
_LAUGHED_WORD_RE = re.compile(r"^\[laughter-(.+)\]$", re.IGNORECASE)

# Lazy module-level cache so each ProcessPoolExecutor worker loads spaCy once.
_NLP = None


def _get_nlp():
    global _NLP
    if _NLP is None:
        import spacy
        _NLP = spacy.load(SPACY_MODEL)
    return _NLP


def _unwrap_or_drop(token: str) -> str | None:
    """Laughed word → its inner word; markup tokens → None; else kept.

    Markup = whole-bracket non-speech events ([noise], [laughter]) and ms98
    angle markers (<b_aside>, <e_aside> — span delimiters for off-phone asides,
    2026-08-19 fix: 300 tokens in 147 utts were counted as words/syllables).
    """
    m = _LAUGHED_WORD_RE.match(token)
    if m:
        return m.group(1)
    if token.startswith("[") and token.endswith("]"):
        return None
    if token.startswith("<") and token.endswith(">"):
        return None
    return token


def strip_bracket_tokens(text: str) -> str:
    """Remove non-speech bracket tokens ([noise], [laughter], …); unwrap laughed
    words ([laughter-yeah] → yeah). Inline markers like 'i[t]-' (partial words)
    do NOT start with '[' so they are kept. Case preserved (spaCy consumers)."""
    kept = (_unwrap_or_drop(t) for t in text.split())
    return " ".join(t for t in kept if t is not None)


def tokenize(text: str) -> list[str]:
    """Lowercase whitespace-split; non-speech brackets dropped, laughed words unwrapped."""
    kept = (_unwrap_or_drop(w) for w in text.lower().split())
    return [w for w in kept if w is not None]


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


# The 10-token filler allowlist, partitioned by construct (2026-08-20 split,
# audit §4E-c): filled pauses vs discourse markers — the standard disfluency-
# taxonomy distinction (the Treebank dysfl gold layer draws exactly this
# {F}/{D} line). Tannen does not mention filler words (tannen_feature_map row
# 12); no HI/HC sign is assumed for either side — the split exists so the two
# classes can take DIFFERENT loadings if the data wants them to, instead of
# sharing one coefficient in a pooled rate (§4E-c cancellation concern). The
# split extractors count each side separately. DEFAULT_FILLERS stays the union
# so the legacy combined column remains exactly reproducible (and the split
# columns must sum to it — the extraction gate). "er": 0 hits corpus-wide
# (dead entry, kept for list continuity).
FILLED_PAUSES: frozenset[str] = frozenset({"um", "uh", "er"})
DISCOURSE_MARKERS: frozenset[str] = frozenset({
    "like", "you know", "i mean", "so", "well", "i guess", "basically",
})
DEFAULT_FILLERS: frozenset[str] = FILLED_PAUSES | DISCOURSE_MARKERS


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
