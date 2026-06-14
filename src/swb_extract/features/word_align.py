"""Word-level alignment parser & index — shared infra for timing features.

The cleaned ms98 transcripts ship per-word timing in sibling ``*-word.text``
files, with the SAME 4-field space-separated layout as ``*-trans.text`` but one
row per word::

    sw2001A-ms98-a-0002 1.215250 1.724625 hi

This module parses those files into:

- ``build_word_index`` → ``{(call, side, utt_num): [(start, end, token), ...]}``
  with the rows of each utterance sorted by start time.
- ``build_side_intervals`` → ``{(call, side): [(start, end), ...]}`` — every
  word interval on a speaker's side, sorted; used by ``overlap`` to find the
  other speaker's speech quickly.

Both A and B sides share one conversation clock (verified), so A-side and
B-side times are directly comparable.

Malformed lines are skipped (mirrors ``turn_gap``'s tolerant trans parser).
Non-speech tokens are vanishingly rare in word.text; they are treated as
ordinary occupied-time tokens so a ``[noise]`` span is never mistaken for
silence.
"""
from __future__ import annotations

import re
from pathlib import Path

from ..transcripts import _ID_RE

# sw<CALL><SIDE>-ms98-a-word.text
_WORD_FILENAME_RE = re.compile(r"^sw(\d+)([AB])-ms98-a-word\.text$")

WordRow = tuple[float, float, str]                       # (start, end, token)
WordIndex = dict[tuple[int, str, int], list[WordRow]]    # (call, side, utt) → rows
SideIntervals = dict[tuple[int, str], list[tuple[float, float]]]  # (call, side) → spans


def iter_word_paths(transcript_root: Path):
    yield from sorted(transcript_root.glob("*/*/sw*-ms98-a-word.text"))


def _parse_word_line(
    line: str, expected_call: int, expected_side: str
) -> tuple[int, float, float, str] | None:
    parts = line.split(maxsplit=3)
    if len(parts) < 4:
        return None
    id_field, start_s, end_s, token = parts
    m = _ID_RE.match(id_field)
    if not m:
        return None
    call_id, side, utt_str = int(m.group(1)), m.group(2), m.group(3)
    if call_id != expected_call or side != expected_side:
        return None
    try:
        start = float(start_s)
        end = float(end_s)
    except ValueError:
        return None
    if end < start:
        return None
    return int(utt_str), start, end, token


def build_word_index(transcript_root: Path) -> WordIndex:
    """Per-(call, side, utt_num) list of (start, end, token), sorted by start."""
    idx: WordIndex = {}
    for path in iter_word_paths(transcript_root):
        m = _WORD_FILENAME_RE.match(path.name)
        if not m:
            continue
        call_id, side = int(m.group(1)), m.group(2)
        try:
            with open(path, encoding="utf-8") as f:
                for raw in f:
                    line = raw.strip()
                    if not line:
                        continue
                    parsed = _parse_word_line(line, call_id, side)
                    if parsed is None:
                        continue
                    utt_num, start, end, token = parsed
                    idx.setdefault((call_id, side, utt_num), []).append(
                        (start, end, token)
                    )
        except OSError:
            continue
    for key in idx:
        idx[key].sort()
    return idx


def build_side_intervals(word_index: WordIndex) -> SideIntervals:
    """Collapse the word index to per-(call, side) sorted (start, end) lists."""
    out: SideIntervals = {}
    for (call_id, side, _utt), rows in word_index.items():
        bucket = out.setdefault((call_id, side), [])
        for start, end, _tok in rows:
            bucket.append((start, end))
    for key in out:
        out[key].sort()
    return out


def other_side(side: str) -> str:
    return "B" if side == "A" else "A"
