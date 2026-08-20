"""Syllable Rate per utterance — syllables per second.

The legacy FESyllableRate.py is broken (undefined `self.word_re`, empty Node
helper) and unrunnable. The newer reference at
`Conversational-Styles/src/feature_extractors/syllable_rate_textstat.py` uses
textstat for syllable counting and divides by utterance duration:

  syllable_count = textstat.syllable_count(text)
  syllable_rate = syllable_count / duration

We follow the same approach with one correction: whole-bracket tokens
([noise], [laughter], [vocalized-noise], etc.) are stripped before textstat
sees them. textstat would otherwise syllabify the bracketed annotation
literally (e.g. `[laughter]` → 2 syllables for "laughter"), inflating the
syllable count on bracket-rich utterances. Stripping is consistent with our
filler / pronoun / repetition extractors.

Output: utterances_v2/features/syllable_rate.csv
Header: Utterance File Name,syllable_rate
"""
from __future__ import annotations

import csv
import os
import re
import ssl
from pathlib import Path

from ...manifest import MANIFEST_HEADER, manifest_path
from ._duration_lookup import build_duration_index, lookup_duration
from ._text import strip_bracket_tokens

FEATURE_NAME = "syllable_rate"
HEADER = ("Utterance File Name", "syllable_rate")  # lowercase per legacy column


def _ensure_cmudict() -> None:
    """textstat depends on NLTK's cmudict corpus; make sure it's available."""
    import nltk

    try:
        nltk.data.find("corpora/cmudict")
        return
    except LookupError:
        pass
    # Some environments have SSL cert issues with the default downloader.
    try:
        ssl._create_default_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    nltk.download("cmudict", quiet=True)


# ms98 pronunciation-variant suffix (them_1 = 'em, because_1 = 'cause, …).
# Stripped before lookup so the base word hits CMUdict; this counts the
# CITATION form — a deliberate approximation for reduced variants ('cause said
# with 1 syllable counts as because = 2; 2,532 tokens, +0.07% corpus-wide),
# accepted because no principled reduced-form syllable source exists.
_VARIANT_SUFFIX_RE = re.compile(r"_\d+$")


def count_syllables(text: str) -> int:
    """Strip markup tokens, then count syllables per hyphen-part via textstat
    (CMUdict vowel nuclei; pyphen fallback for OOV). Returns 0 for empty.

    2026-08-19 fix: counting per hyphen part sends each part through CMUdict —
    the whole-text path stripped hyphens, so "um-hum" became OOV "umhum" and
    pyphen guessed 1 syllable (13,744 tokens; 11,332 pure-backchannel
    utterances had their rate halved). um(1)+hum(1)=2 by dictionary instead.
    """
    import textstat

    cleaned = strip_bracket_tokens(text)
    total = 0
    for tok in cleaned.split():
        tok = _VARIANT_SUFFIX_RE.sub("", tok)
        for part in tok.split("-"):
            if part:
                total += int(textstat.syllable_count(part))
    return total


def compute_rate(text: str, duration: float | None) -> float | None:
    if duration is None or duration <= 0:
        return None
    return count_syllables(text) / duration


def _fmt(v: float | None) -> str:
    return "" if v is None else repr(v)


def write_syllable_rates(
    manifest_csv: Path,
    output_csv: Path,
    transcript_root: Path,
) -> int:
    _ensure_cmudict()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    durations = build_duration_index(transcript_root)
    n = 0
    missing = 0
    with open(manifest_csv, encoding="utf-8", newline="") as fin, open(
        output_csv, "w", encoding="utf-8", newline=""
    ) as fout:
        reader = csv.reader(fin)
        header = next(reader, None)
        if tuple(header or ()) != MANIFEST_HEADER:
            raise RuntimeError(
                f"unexpected manifest header in {manifest_csv}: {header!r}"
            )
        writer = csv.writer(fout, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(HEADER)
        for row in reader:
            if not row:
                continue
            rel, text = row[0], row[1]
            d = lookup_duration(durations, rel)
            if d is None:
                missing += 1
            writer.writerow([rel, _fmt(compute_rate(text, d))])
            n += 1
    if missing:
        print(f"  warning: {missing} rows had no transcript-derived duration")
    return n


def run(args) -> int:
    out_root = Path(args.out_root)
    n = write_syllable_rates(
        manifest_path(out_root),
        out_root / "features" / "syllable_rate.csv",
        transcript_root=Path(args.transcript_root),
    )
    print(f"wrote {n} syllable rate rows")
    return 0
