"""Laughter and non-speech bracket events per utterance — Tannen dimension 9.

Tannen ref: Ch.7 dim 9 "Laughter (when, how much)" (PDF p. 202) and Table 6.1
(PDF p. 185); dim 8 "tolerance for noise vs. silence" gets the noise columns.

This closes the long-standing gap flagged in tannen_feature_map.md (item #17)
and AUDIT.md §3 fix 5: every other extractor STRIPS bracket annotations before
analysis, so laughter — Tannen's ninth dimension — was destroyed rather than
measured. This extractor counts the events first, from the manifest's raw
transcript text (which preserves brackets).

ms98 bracket forms observed in this corpus (surveyed 2026-06-10):

  [laughter]            12,402  — standalone laugh        → Laughter Count
  [laughter-<word>]     10,275  — word produced laughing  → Laughed Word Count
  [noise]                8,381  — non-speech noise        → Noise Count
  [vocalized-noise]      4,009  — vocal non-speech        → Vocalized Noise Count
  [<said>/<intended>]    ~rare  — pronunciation-variant annotations (these are
                                  real spoken words, not events) and anything
                                  else whole-bracketed → Other Bracket Count

Partial-word forms like ``i[t]-`` do not start with '[' and are not counted.
All counts are always defined; 0 is an honest measurement.

Output: utterances_v2/features/laughter.csv
Header: Utterance File Name,Laughter Count,Laughed Word Count,Noise Count,
        Vocalized Noise Count,Other Bracket Count
"""
from __future__ import annotations

import csv
from pathlib import Path

from ..manifest import MANIFEST_HEADER, manifest_path

FEATURE_NAME = "laughter"
HEADER = (
    "Utterance File Name",
    "Laughter Count",
    "Laughed Word Count",
    "Noise Count",
    "Vocalized Noise Count",
    "Other Bracket Count",
)

# (laughter, laughed_word, noise, vocalized_noise, other)
Counts = tuple[int, int, int, int, int]


def count_bracket_events(text: str) -> Counts:
    laughter = laughed = noise = vocalized = other = 0
    for tok in str(text).split():
        if not (tok.startswith("[") and tok.endswith("]")):
            continue
        if tok == "[laughter]":
            laughter += 1
        elif tok.startswith("[laughter-"):
            laughed += 1
        elif tok == "[noise]":
            noise += 1
        elif tok == "[vocalized-noise]":
            vocalized += 1
        else:
            other += 1
    return laughter, laughed, noise, vocalized, other


def write_laughter_counts(manifest_csv: Path, output_csv: Path) -> int:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    totals = [0, 0, 0, 0, 0]
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
            counts = count_bracket_events(row[1] if len(row) > 1 else "")
            writer.writerow([row[0], *(str(c) for c in counts)])
            totals = [t + c for t, c in zip(totals, counts)]
            n += 1
    print(
        "  totals: laughter={}, laughed-words={}, noise={}, "
        "vocalized-noise={}, other={}".format(*totals)
    )
    return n


def run(args) -> int:
    out_root = Path(args.out_root)
    n = write_laughter_counts(
        manifest_path(out_root),
        out_root / "features" / "laughter.csv",
    )
    print(f"wrote {n} laughter rows")
    return 0
