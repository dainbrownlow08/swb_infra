"""Filler Word Rate per utterance.

Per-utterance proportion of tokens matching a known filler set, with
phrase-aware multi-word matching ('you know', 'i mean', 'i guess') and
whole-bracket tokens like [noise], [laughter] stripped from both numerator
and denominator. See plan file for legacy-bug analysis and rationale.

Output: utterances_v2/features/filler_word_rate.csv
Header: Utterance File Name,Filler Word Rate
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from ..manifest import MANIFEST_HEADER, manifest_path

FEATURE_NAME = "filler_word_rate"
HEADER = ("Utterance File Name", "Filler Word Rate")

DEFAULT_FILLERS: frozenset[str] = frozenset({
    "um", "uh", "like", "you know", "i mean",
    "so", "well", "i guess", "basically", "er",
})


def tokenize(text: str) -> list[str]:
    return [
        w for w in text.lower().split()
        if not (w.startswith("[") and w.endswith("]"))
    ]


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


def compute_rate(text: str, fillers: Iterable[str] = DEFAULT_FILLERS) -> float:
    words = tokenize(text)
    if not words:
        return 0.0
    return count_filler_hits(words, fillers) / len(words)


def write_filler_rates(
    manifest_csv: Path,
    output_csv: Path,
    fillers: Iterable[str] = DEFAULT_FILLERS,
) -> int:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fillers = frozenset(fillers)
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
            writer.writerow([rel, compute_rate(text, fillers)])
            n += 1
    return n


def run(args) -> int:
    out_root = Path(args.out_root)
    n = write_filler_rates(
        manifest_path(out_root),
        out_root / "features" / "filler_word_rate.csv",
    )
    print(f"wrote {n} filler word rate rows")
    return 0
