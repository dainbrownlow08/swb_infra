"""Personal Pronouns per Second per utterance — 1st/2nd person density.

**2026-08-19 v2 redesign (decision on record in FEATURES.md).** v1 counted
spaCy's whole PRON class, which the same-day audit showed is only ~50% 1st/2nd
person (~27% it/that/what/there thing-reference) — total pronominalization, not
Tannen dim 1 "relative personal focus of topic". v2 counts exactly the
1st/2nd-person closed class so the column IS the dim-1 lexical instrument
(Chafe-style involvement marker Tannen builds on).

Pronouns are a closed class, so no tagger is needed: the counter is a token
list match on the shared tokenizer's output (laughed words unwrapped, markup
stripped). This removes the spaCy model-version reproducibility surface from
the trusted line and ports to any corpus unchanged. Contracted forms match by
their pre-apostrophe base ("i'm", "you're", "we'll" → i/you/we); "y'all"
matches whole. Partial-word attempts ("i[t]-") do not match — by design, the
attempted word is "it".

Output: utterances_v2/features/pronoun_per_second.csv
Header: Utterance File Name,Personal Pronouns per Second
"""
from __future__ import annotations

import csv
from pathlib import Path

from ..manifest import MANIFEST_HEADER, manifest_path
from ._duration_lookup import build_duration_index, lookup_duration
from ._text import tokenize

FEATURE_NAME = "pronoun_per_second"
HEADER = ("Utterance File Name", "Personal Pronouns per Second")

PERSONAL_PRONOUNS = frozenset({
    "i", "me", "my", "mine", "myself",
    "we", "us", "our", "ours", "ourselves",
    "you", "your", "yours", "yourself", "yourselves", "y'all",
})


def count_personal_pronouns(text: str) -> int:
    n = 0
    for w in tokenize(text):
        if w in PERSONAL_PRONOUNS or w.split("'")[0] in PERSONAL_PRONOUNS:
            n += 1
    return n


def compute_rate_per_second(text: str, duration: float | None) -> float | None:
    if duration is None or duration <= 0:
        return None
    return count_personal_pronouns(text) / duration


def _fmt(v: float | None) -> str:
    return "" if v is None else repr(v)


def write_pronouns_per_second(
    manifest_csv: Path,
    output_csv: Path,
    transcript_root: Path,
) -> int:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    durations = build_duration_index(transcript_root)
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
            d = lookup_duration(durations, rel)
            writer.writerow([rel, _fmt(compute_rate_per_second(text, d))])
            n += 1
    return n


def run(args) -> int:
    out_root = Path(args.out_root)
    n = write_pronouns_per_second(
        manifest_path(out_root),
        out_root / "features" / "pronoun_per_second.csv",
        transcript_root=Path(args.transcript_root),
    )
    print(f"wrote {n} personal-pronoun rate rows")
    return 0
