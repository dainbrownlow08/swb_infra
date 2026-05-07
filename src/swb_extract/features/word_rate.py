"""Word Rate per utterance — words per second.

Legacy `FEWordRate.py` used `len(text.split()) / duration` with no bracket
handling, so `[laughter]`, `[noise]`, etc. inflated the count. Our convention
(filler / pronoun / repetition / syllable extractors) strips whole-bracket
tokens before counting. We follow the same convention here.

Output: utterances_v2/features/word_rate.csv
Header: Utterance File Name,word_rate
"""
from __future__ import annotations

import csv
from pathlib import Path

from ..manifest import MANIFEST_HEADER, manifest_path
from ._duration_lookup import build_duration_index, lookup_duration

FEATURE_NAME = "word_rate"
HEADER = ("Utterance File Name", "word_rate")


def tokenize(text: str) -> list[str]:
    return [
        w for w in text.lower().split()
        if not (w.startswith("[") and w.endswith("]"))
    ]


def count_words(text: str) -> int:
    return len(tokenize(text))


def compute_rate(text: str, duration: float | None) -> float | None:
    if duration is None or duration <= 0:
        return None
    return count_words(text) / duration


def _fmt(v: float | None) -> str:
    return "" if v is None else repr(v)


def write_word_rates(
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
            writer.writerow([rel, _fmt(compute_rate(text, d))])
            n += 1
    if missing:
        print(f"  warning: {missing} rows had no transcript-derived duration")
    return n


def run(args) -> int:
    out_root = Path(args.out_root)
    n = write_word_rates(
        manifest_path(out_root),
        out_root / "features" / "word_rate.csv",
        transcript_root=Path(args.transcript_root),
    )
    print(f"wrote {n} word rate rows")
    return 0
