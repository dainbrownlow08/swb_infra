"""Shared duration lookup for per-second features.

Builds a `(call_id, side, utt_num) → duration_seconds` index by parsing
every cleaned ms98 transcript file. Used by per-second feature extractors
that need to divide a token/event count by utterance duration.
"""
from __future__ import annotations

from pathlib import Path

from ..manifest import parse_rel_path
from ..transcripts import iter_transcript_paths, parse_transcript

DurationIndex = dict[tuple[int, str, int], float]


def build_duration_index(transcript_root: Path) -> DurationIndex:
    idx: DurationIndex = {}
    for tpath in iter_transcript_paths(transcript_root):
        for u in parse_transcript(tpath):
            idx[(u.call_id, u.side, u.utt_num)] = u.end - u.start
    return idx


def lookup_duration(idx: DurationIndex, rel_path: str) -> float | None:
    call_id, side, utt_num = parse_rel_path(rel_path)
    return idx.get((call_id, side, utt_num))
