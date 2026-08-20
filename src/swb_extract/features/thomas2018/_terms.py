"""Term normalisation for rept / repu.

"Before counting repeats we removed stopwords as well as 'um', 'uh', and 'uh-huh', and
stemmed what remained" (Thomas et al. 2018 §3.2). Tokens come from the shared in-house
tokenizer (markup dropped, laughed words unwrapped); the ms98 pronunciation-variant
suffix (``because_1``) is stripped so the base word meets the stopword list and the
stemmer. The stopword list is NLTK's English list frozen here verbatim (198 entries,
nltk 3.9.4, copied 2026-08-20) so the column cannot drift with an NLTK upgrade;
the stemmer is NLTK's Porter (code, no data download). A "term" is a stemmed TYPE, so a
word repeated twice inside one utterance counts once.

Corpus note: the paper's filler list is exactly um / uh / uh-huh (MISC's spellings).
Switchboard's frequent "um-hum" is not on it and so counts as a term — a repeated
"um-hum" across a listener's own consecutive lines is a repeat under the literal
definition. Sweepable (FILLERS); left literal on purpose, flagged in AUDIT.md §4E-h.
"""
from __future__ import annotations

import re

from ..inhouse._text import tokenize
from ..inhouse._turn_index import TextIndex
from ._io import Key

FILLERS = frozenset({"um", "uh", "uh-huh"})

STOPWORDS = frozenset("""
    a about above after again against ain all am an and any are aren aren't as at be
    because been before being below between both but by can couldn couldn't d did didn
    didn't do does doesn doesn't doing don don't down during each few for from further
    had hadn hadn't has hasn hasn't have haven haven't having he he'd he'll he's her
    here hers herself him himself his how i i'd i'll i'm i've if in into is isn isn't
    it it'd it'll it's its itself just ll m ma me mightn mightn't more most mustn
    mustn't my myself needn needn't no nor not now o of off on once only or other our
    ours ourselves out over own re s same shan shan't she she'd she'll she's should
    should've shouldn shouldn't so some such t than that that'll the their theirs them
    themselves then there these they they'd they'll they're they've this those through
    to too under until up ve very was wasn wasn't we we'd we'll we're we've were weren
    weren't what when where which while who whom why will with won won't wouldn
    wouldn't y you you'd you'll you're you've your yours yourself yourselves
""".split())

_VARIANT_SUFFIX_RE = re.compile(r"_\d+$")
_STEMMER = None


def stem(word: str) -> str:
    global _STEMMER
    if _STEMMER is None:
        from nltk.stem import PorterStemmer

        _STEMMER = PorterStemmer()
    return _STEMMER.stem(word)


def terms(text: str) -> set[str]:
    """Stemmed content terms of an utterance, per the paper's preprocessing."""
    out: set[str] = set()
    for w in tokenize(text):
        w = _VARIANT_SUFFIX_RE.sub("", w)
        if not w or w in FILLERS or w in STOPWORDS:
            continue
        out.add(stem(w))
    return out


def repeated_terms(cur_text: str, prev_text: str) -> int:
    return len(terms(cur_text) & terms(prev_text))


def build_repeat_index(text_idx: TextIndex) -> dict[Key, int]:
    """rept per utterance: |terms(utt) ∩ terms(the same side's previous utterance)|.

    Utterances of a side are ordered by utt_num (ms98 numbers lines chronologically);
    a side's first utterance has no previous one and is absent (→ blank downstream).
    """
    by_side: dict[tuple[int, str], list[int]] = {}
    for call_id, side, utt_num in text_idx:
        by_side.setdefault((call_id, side), []).append(utt_num)
    out: dict[Key, int] = {}
    for (call_id, side), utt_nums in by_side.items():
        prev: set[str] | None = None
        for utt_num in sorted(utt_nums):
            cur = terms(text_idx[(call_id, side, utt_num)])
            if prev is not None:
                out[(call_id, side, utt_num)] = len(cur & prev)
            prev = cur
    return out
