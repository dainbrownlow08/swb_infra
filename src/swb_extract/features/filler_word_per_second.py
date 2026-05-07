"""Filler Words per Second per utterance.

Same count algorithm as filler_word_rate (phrase-aware sliding window with
bracket stripping), but divides by utterance duration in seconds instead of
token count. Captures "speech rate" interpretation of filler usage rather
than "filler proportion of words spoken."

  count = same as filler_word_rate.count_filler_hits
  duration = utterance.end - utterance.start (from cleaned ms98 transcript)
  rate = count / duration   (None if duration <= 0)

Output: utterances_v2/features/filler_word_per_second.csv
Header: Utterance File Name,Filler Words per Second
"""
from __future__ import annotations

import csv
from pathlib import Path

from ..manifest import MANIFEST_HEADER, manifest_path
from ._duration_lookup import build_duration_index, lookup_duration
from .filler_word_rate import DEFAULT_FILLERS, count_filler_hits, tokenize

FEATURE_NAME = "filler_word_per_second"
HEADER = ("Utterance File Name", "Filler Words per Second")


def compute_rate_per_second(text: str, duration: float | None) -> float | None:
    if duration is None or duration <= 0:
        return None
    words = tokenize(text)
    if not words:
        return 0.0
    return count_filler_hits(words, DEFAULT_FILLERS) / duration


def _fmt(v: float | None) -> str:
    return "" if v is None else repr(v)


def write_filler_words_per_second(
    manifest_csv: Path,
    output_csv: Path,
    transcript_root: Path,
) -> int:
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
            writer.writerow([rel, _fmt(compute_rate_per_second(text, d))])
            n += 1
    if missing:
        print(f"  warning: {missing} rows had no transcript-derived duration")
    return n


def run(args) -> int:
    out_root = Path(args.out_root)
    n = write_filler_words_per_second(
        manifest_path(out_root),
        out_root / "features" / "filler_word_per_second.csv",
        transcript_root=Path(args.transcript_root),
    )
    print(f"wrote {n} filler-words-per-second rows")
    return 0
