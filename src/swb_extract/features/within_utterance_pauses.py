"""Within-utterance pauses — strategic silences inside a single turn.

Tannen ref: Ch.2 feature 4d (PDF p.62) "strategic within-turn pauses" — the
High-Considerateness counterpart to latching. HI speakers minimize internal
pausing (fast, latched); HC speakers leave silences for the interlocutor.

From the word-level alignment of one utterance, compute silent gaps between
adjacent words:  gap_i = word[i+1].start - word[i].end  (negatives clamped to 0
for overlapping forced-alignment boundaries).

Columns (one CSV):
  Within Pause Total Sec  — sum of all positive inter-word gaps
  Within Pause Count      — number of gaps >= PAUSE_MIN_SEC (a real pause)
  Within Pause Rate       — Total Sec / utterance span (silence fraction)
  Max Within Pause Sec    — longest single inter-word gap

A 0- or 1-word utterance has no internal gaps → all four are 0. Empty cells are
written only when the utterance has no word-level rows at all (cannot place it).

PAUSE_MIN_SEC = 0.25 s — below this, gaps are ordinary inter-word transitions,
not perceptible pauses. Documented; easy to sweep.

Output: utterances_v2/features/within_utterance_pauses.csv
Header: Utterance File Name,Within Pause Total Sec,Within Pause Count,Within Pause Rate,Max Within Pause Sec
"""
from __future__ import annotations

import csv
from pathlib import Path

from ..manifest import MANIFEST_HEADER, manifest_path, parse_rel_path
from .word_align import WordIndex, build_word_index

FEATURE_NAME = "within_utterance_pauses"
HEADER = (
    "Utterance File Name",
    "Within Pause Total Sec",
    "Within Pause Count",
    "Within Pause Rate",
    "Max Within Pause Sec",
)

PAUSE_MIN_SEC = 0.25

PauseStats = tuple[float, int, float, float]


def compute_pauses(words: list[tuple[float, float, str]]) -> PauseStats | None:
    """(total_sec, count>=θ, rate, max_sec) from sorted (start, end, token) rows.

    None if there are no word rows. Zeros for 0/1-word utterances.
    """
    if not words:
        return None
    if len(words) == 1:
        return (0.0, 0, 0.0, 0.0)
    gaps: list[float] = []
    for i in range(len(words) - 1):
        g = words[i + 1][0] - words[i][1]
        if g > 0:
            gaps.append(g)
    total = sum(gaps)
    count = sum(1 for g in gaps if g >= PAUSE_MIN_SEC)
    # Rows are sorted by start, so the last-by-start word need not end last;
    # use the max end (matches fto's word-tight bounds) so rate never exceeds 1.
    span = max(e for _s, e, _t in words) - words[0][0]
    rate = total / span if span > 0 else 0.0
    mx = max(gaps) if gaps else 0.0
    return (total, count, rate, mx)


def _fmt_row(stats: PauseStats | None) -> list[str]:
    if stats is None:
        return ["", "", "", ""]
    total, count, rate, mx = stats
    return [repr(total), str(count), repr(rate), repr(mx)]


def write_within_utterance_pauses(
    manifest_csv: Path,
    output_csv: Path,
    transcript_root: Path,
) -> int:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    word_idx: WordIndex = build_word_index(transcript_root)
    n = 0
    n_missing = 0
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
            rel = row[0]
            try:
                key = parse_rel_path(rel)
            except ValueError:
                writer.writerow([rel, "", "", "", ""])
                n += 1
                n_missing += 1
                continue
            stats = compute_pauses(word_idx.get(key, []))
            if stats is None:
                n_missing += 1
            writer.writerow([rel, *_fmt_row(stats)])
            n += 1
    if n_missing:
        print(f"  empty cells (no word rows): {n_missing}")
    return n


def run(args) -> int:
    out_root = Path(args.out_root)
    n = write_within_utterance_pauses(
        manifest_path(out_root),
        out_root / "features" / "within_utterance_pauses.csv",
        transcript_root=Path(args.transcript_root),
    )
    print(f"wrote {n} within-utterance-pause rows")
    return 0
