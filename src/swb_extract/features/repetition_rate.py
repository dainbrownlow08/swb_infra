"""Repetition Rate per utterance.

Same algorithm as legacy FERepetition.py, with one correction: whole-bracket
tokens (`[noise]`, `[laughter]`, ...) are stripped before counting. They are
not real words, and counting them can both manufacture fake repetitions
(when the same bracket appears twice) and dilute the denominator. This
matches the bracket-handling we use for filler_word_rate and pronoun_rate.

Algorithm:
  tokens = text.lower().split()
  tokens = [t for t in tokens if not whole_bracket(t)]
  count occurrences of each unique token
  repetitions = number of unique tokens whose count reaches 2
                (binary per word — 'the the the' counts as 1 repetition,
                 not 2; matches legacy)
  rate = repetitions / total_token_count   (or 0.0 if no tokens)

Output: utterances_v2/features/repetition_rate.csv
Header: Utterance File Name,Repetition Rate
"""
from __future__ import annotations

import csv
from contextlib import contextmanager
from pathlib import Path

from ..manifest import MANIFEST_HEADER, manifest_path

FEATURE_NAME = "repetition_rate"
HEADER = ("Utterance File Name", "Repetition Rate")


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


def compute_rate(text: str) -> float:
    words = tokenize(text)
    if not words:
        return 0.0
    return count_repetitions(words) / len(words)


def write_repetition_rates(
    manifest_csv: Path,
    output_csv: Path,
) -> int:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    n = 0
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
            writer.writerow([rel, compute_rate(text)])
            n += 1
    return n


def run(args) -> int:
    out_root = Path(args.out_root)
    n = write_repetition_rates(
        manifest_path(out_root),
        out_root / "features" / "repetition_rate.csv",
    )
    print(f"wrote {n} repetition rate rows")
    return 0
