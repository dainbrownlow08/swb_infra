"""Repetitions In Current Utterance — pair-count of repeated tokens.

For each utterance, count token-position pairs (i, j) with i < j where
tokens[i] == tokens[j]. Equivalently sum over distinct words w of C(n_w, 2),
where n_w is the count of word w in the utterance.

This preserves the legacy `FERepeats` per-utterance metric semantics while
correcting two legacy bugs:

- Whole-bracket tokens (`[noise]`, `[laughter]`, …) are stripped before
  counting. They are noises, not words, and would otherwise manufacture
  fake repetitions or inflate the pair count.
- The legacy `gensim` + NLTK lemmatizer stack is dropped in favour of the
  same `tokenize` helper that `repetition_rate.py` uses, so tokenization
  is consistent across the repetition family.

This metric is complementary to `Repetition Rate` (which counts unique
words reaching 2). For example, "the the the":
  - Repetition Rate count → 1 (one unique word reached 2)
  - Repetitions In Current Utterance → 3 (three pairs: 0-1, 0-2, 1-2)

Output: utterances_v2/features/repetitions_in_current.csv
Header: Utterance File Name,Repetitions In Current Utterance
"""
from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

from ..manifest import MANIFEST_HEADER, manifest_path
from .repetition_rate import tokenize

FEATURE_NAME = "repetitions_in_current"
HEADER = ("Utterance File Name", "Repetitions In Current Utterance")


def count_pair_repetitions(words: list[str]) -> int:
    """sum over distinct words w of C(count[w], 2)."""
    counts = Counter(words)
    return sum(n * (n - 1) // 2 for n in counts.values() if n >= 2)


def compute(text: str) -> int:
    return count_pair_repetitions(tokenize(text))


def write_repetitions_in_current(
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
            writer.writerow([rel, compute(text)])
            n += 1
    return n


def run(args) -> int:
    out_root = Path(args.out_root)
    n = write_repetitions_in_current(
        manifest_path(out_root),
        out_root / "features" / "repetitions_in_current.csv",
    )
    print(f"wrote {n} repetitions-in-current rows")
    return 0
