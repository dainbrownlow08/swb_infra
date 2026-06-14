"""Overlap — true simultaneous speech across the two speakers.

Tannen ref: Ch.4 "Overlap and Pace" (PDF pp.113-122); Ch.7 dim 5a. Cooperative
overlap is a hallmark of High Involvement — a show of engagement, not an
interruption. We measure overlap at the WORD level (intersecting A-side and
B-side word intervals on the shared conversation clock), so silences inside an
utterance are never miscounted as simultaneous speech.

For the current utterance U (speaker S), with U's own word intervals merged
into S's phonation spans, against the OTHER speaker O's word intervals:

  Overlap Duration Sec — total time during U where O also has a word interval.
  Overlap Count        — number of O word intervals intersecting U's speech.
  Overlap Onset Flag   — 1 if O is already mid-word at U's first-word onset
                         (S "comes in over" O — the floor-taking / involvement
                         signal); else 0.

Empty cells are written only when U has no word-level rows (cannot place it).
Cooperative-vs-obstructive classification is deferred to a later pass.

Output: utterances_v2/features/overlap.csv
Header: Utterance File Name,Overlap Duration Sec,Overlap Count,Overlap Onset Flag
"""
from __future__ import annotations

import csv
from bisect import bisect_left
from pathlib import Path

from ..manifest import MANIFEST_HEADER, manifest_path, parse_rel_path
from .word_align import (
    SideIntervals,
    WordIndex,
    build_side_intervals,
    build_word_index,
    other_side,
)

FEATURE_NAME = "overlap"
HEADER = (
    "Utterance File Name",
    "Overlap Duration Sec",
    "Overlap Count",
    "Overlap Onset Flag",
)

OverlapStats = tuple[float, int, int]


def _merge(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Merge a sorted list of (start, end) into non-overlapping spans."""
    out: list[tuple[float, float]] = []
    for s, e in intervals:
        if out and s <= out[-1][1]:
            if e > out[-1][1]:
                out[-1] = (out[-1][0], e)
        else:
            out.append((s, e))
    return out


def _intersect_total(
    a: list[tuple[float, float]], b: list[tuple[float, float]]
) -> float:
    """Total measure of the intersection of two sorted interval lists."""
    i = j = 0
    tot = 0.0
    while i < len(a) and j < len(b):
        lo = max(a[i][0], b[j][0])
        hi = min(a[i][1], b[j][1])
        if hi > lo:
            tot += hi - lo
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return tot


def compute_overlap(
    own_words: list[tuple[float, float, str]],
    other_spans: list[tuple[float, float]],
    other_starts: list[float],
) -> OverlapStats | None:
    """Overlap of one utterance's speech against the other speaker's word spans.

    own_words: this utterance's (start, end, token) rows, sorted.
    other_spans / other_starts: the OTHER side's word (start, end) spans for the
    whole call, sorted, with a parallel list of their start times for bisect.
    """
    if not own_words:
        return None
    own = _merge([(s, e) for s, e, _ in own_words])
    u_start = own[0][0]
    u_end = own[-1][1]

    # Candidate other-side words: start before U ends AND end after U starts.
    hi = bisect_left(other_starts, u_end)
    candidates = [iv for iv in other_spans[:hi] if iv[1] > u_start]

    duration = _intersect_total(own, _merge(candidates))
    count = sum(1 for iv in candidates if _intersect_total(own, [iv]) > 0)
    onset = 1 if any(s <= u_start < e for s, e in candidates) else 0
    return (duration, count, onset)


def _fmt_row(stats: OverlapStats | None) -> list[str]:
    if stats is None:
        return ["", "", ""]
    duration, count, onset = stats
    return [repr(duration), str(count), str(onset)]


def write_overlap(
    manifest_csv: Path,
    output_csv: Path,
    transcript_root: Path,
) -> int:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    word_idx: WordIndex = build_word_index(transcript_root)
    side_intervals: SideIntervals = build_side_intervals(word_idx)
    # Precompute parallel start arrays for bisect, once per (call, side).
    starts_by_side = {
        key: [s for s, _ in spans] for key, spans in side_intervals.items()
    }

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
                call_id, side, utt_num = parse_rel_path(rel)
            except ValueError:
                writer.writerow([rel, "", "", ""])
                n += 1
                n_missing += 1
                continue
            own = word_idx.get((call_id, side, utt_num), [])
            o_side = other_side(side)
            stats = compute_overlap(
                own,
                side_intervals.get((call_id, o_side), []),
                starts_by_side.get((call_id, o_side), []),
            )
            if stats is None:
                n_missing += 1
            writer.writerow([rel, *_fmt_row(stats)])
            n += 1
    if n_missing:
        print(f"  empty cells (no word rows): {n_missing}")
    return n


def run(args) -> int:
    out_root = Path(args.out_root)
    n = write_overlap(
        manifest_path(out_root),
        out_root / "features" / "overlap.csv",
        transcript_root=Path(args.transcript_root),
    )
    print(f"wrote {n} overlap rows")
    return 0
