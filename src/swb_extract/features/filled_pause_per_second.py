"""Filled Pauses per Second per utterance — the hesitation half of the filler split.

One of the two extractors that split the pooled Filler Words per Second column
(2026-08-20, audit §4E-c): its allowlist mixed filled pauses with discourse
markers — two distinct construct classes per the standard disfluency taxonomy
(the Treebank ``{F}``/``{D}`` distinction), pooled 41.8% vs 58.2% of the
183,496 corpus hits. A pooled rate forces both classes onto one shared
loading; IF they behave oppositely in the style space, their effects cancel
there. Whether they do is an open empirical question — Tannen never mentions
filler words (tannen_feature_map row 12), so no HI/HC sign is assumed for
either half; signs are read off the loadings and reconciled at W7. This
extractor counts only ``_text.FILLED_PAUSES``
(um / uh / er — hesitation noises, no lexical homographs, so this column is
unambiguous by construction); the sibling ``discourse_marker_per_second``
counts the rest. The two columns sum EXACTLY to the legacy combined column on
every row (the extraction gate), which stays available for replication.

Same machinery as the combined extractor: shared tokenizer (laughed-word
unwrap, bracket/angle drop), phrase-aware ``count_filler_hits``, guarded
trans-span duration.

  count = _text.count_filler_hits over FILLED_PAUSES only
  duration = utterance.end - utterance.start (guarded; None → blank)
  rate = count / duration   (None if duration <= 0)

Output: utterances_v2/features/filled_pause_per_second.csv
Header: Utterance File Name,Filled Pauses per Second
"""
from __future__ import annotations

import csv
from pathlib import Path

from ..manifest import MANIFEST_HEADER, manifest_path
from ._duration_lookup import build_duration_index, lookup_duration
from ._text import FILLED_PAUSES, count_filler_hits, tokenize

FEATURE_NAME = "filled_pause_per_second"
HEADER = ("Utterance File Name", "Filled Pauses per Second")


def compute_rate_per_second(text: str, duration: float | None) -> float | None:
    if duration is None or duration <= 0:
        return None
    words = tokenize(text)
    if not words:
        return 0.0
    return count_filler_hits(words, FILLED_PAUSES) / duration


def _fmt(v: float | None) -> str:
    return "" if v is None else repr(v)


def write_filled_pauses_per_second(
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
    n = write_filled_pauses_per_second(
        manifest_path(out_root),
        out_root / "features" / "filled_pause_per_second.csv",
        transcript_root=Path(args.transcript_root),
    )
    print(f"wrote {n} filled-pauses-per-second rows")
    return 0
