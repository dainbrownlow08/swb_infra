"""Shared duration lookup for per-second features.

Builds a `(call_id, side, utt_num) → duration_seconds` index by parsing
every cleaned ms98 transcript file. Used by per-second feature extractors
that need to divide a token/event count by utterance duration.

**Alignment-consistency guard (2026-08-19).** A trans line whose word
alignments extend outside its own [start, end] span has a provably wrong
duration — the 2026-08-19 word_rate audit found 9 such lines in 247,820
(worst case: 11.9 s of aligned words inside a claimed 2.7 s span, yielding
21.9 words/s). Those durations are excluded from the index, so `lookup_duration`
returns None and every per-second consumer writes a blank (unmeasurable) —
9 nulls instead of 9 absurd rates. Generic and corpus-portable: it validates
the source alignments, not Switchboard specifics.
"""
from __future__ import annotations

from pathlib import Path

from ..manifest import parse_rel_path
from ..transcripts import iter_transcript_paths, parse_transcript
from .word_align import build_word_index

DurationIndex = dict[tuple[int, str, int], float]

# Word span may exceed the trans span by at most this before the trans line's
# duration is declared invalid (matches the slicer's tolerance scale).
SPAN_TOLERANCE_SEC = 0.01


def build_duration_index(transcript_root: Path) -> DurationIndex:
    bounds: dict[tuple[int, str, int], tuple[float, float]] = {}
    for tpath in iter_transcript_paths(transcript_root):
        for u in parse_transcript(tpath):
            bounds[(u.call_id, u.side, u.utt_num)] = (u.start, u.end)
    word_idx = build_word_index(transcript_root)
    idx: DurationIndex = {}
    n_invalid = 0
    for key, (t0, t1) in bounds.items():
        words = word_idx.get(key)
        if words:
            ws = min(s for s, _e, _t in words)
            we = max(e for _s, e, _t in words)
            if ws < t0 - SPAN_TOLERANCE_SEC or we > t1 + SPAN_TOLERANCE_SEC:
                n_invalid += 1
                continue  # duration untrustworthy → absent → blank downstream
        idx[key] = t1 - t0
    if n_invalid:
        print(
            f"  duration guard: {n_invalid} trans lines with word alignments "
            f"outside their span — durations invalidated (blank downstream)"
        )
    return idx


def lookup_duration(idx: DurationIndex, rel_path: str) -> float | None:
    call_id, side, utt_num = parse_rel_path(rel_path)
    return idx.get((call_id, side, utt_num))
