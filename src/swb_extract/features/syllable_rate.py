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
import ssl
from pathlib import Path

from ..manifest import MANIFEST_HEADER, manifest_path
from ._duration_lookup import build_duration_index, lookup_duration

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


def _strip_bracket_text(text: str) -> str:
    return " ".join(
        t for t in text.split()
        if not (t.startswith("[") and t.endswith("]"))
    )


def count_syllables(text: str) -> int:
    """Strip whole-bracket tokens, then count via textstat. Returns 0 for empty."""
    import textstat

    cleaned = _strip_bracket_text(text)
    if not cleaned:
        return 0
    return int(textstat.syllable_count(cleaned))


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
