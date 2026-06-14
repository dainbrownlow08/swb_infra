"""Shared backchannel definition — the analysis-notebook allowlist, packaged.

This is the 38-token allowlist used by the PCA notebooks (pca_paper_aligned_
with_stndr / pca_speaker_level / pca_speaker_level_interactional), promoted
into the package so extractors and notebooks classify identically instead of
copy-pasting. An utterance is a backchannel iff it has at least one token and
EVERY token (lowercased, leading/trailing punctuation stripped) is in the list.

Known limits, kept deliberately for continuity with the notebooks:

- Bracket annotations survive as words ("[laughter]" → "laughter"), which is
  not in the list — so "yeah [laughter]" classifies as substantive. Acceptable
  for now; the laughter counter (AUDIT.md §3 fix 5) will measure brackets.
- This is a lexical proxy, not gold dialog-act annotation. Validation against
  NXT-Switchboard backchannel tags is planned (AUDIT.md §4 C12).
"""
from __future__ import annotations

import string

BACKCHANNEL_TOKENS = frozenset(
    {
        "yeah", "yea", "yep", "yup", "yes", "ya", "yah",
        "uh-huh", "uhhuh", "huh-uh", "um-hum", "umhum",
        "mm-hmm", "mm-hm", "mmhm", "mhm", "m-hm",
        "mm", "mmm", "hm", "hmm", "hmmm", "hum", "huh",
        "uh", "um", "er", "erm", "ah", "oh", "ooh", "aw",
        "okay", "ok", "right", "sure", "wow", "gosh",
    }
)


def is_backchannel(text: str) -> bool:
    """True iff every token of `text` is a backchannel token (and there is one)."""
    tokens = [w.strip(string.punctuation).lower() for w in str(text).split()]
    tokens = [w for w in tokens if w]
    return len(tokens) > 0 and all(w in BACKCHANNEL_TOKENS for w in tokens)
