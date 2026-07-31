"""Per-conversation chronological utterance indexes over the cleaned ms98 transcripts.

Shared machinery extracted verbatim from the retired turn_gap extractor (the
deprecated Turn Gap *measure* — gap vs chronological predecessor including
backchannels — was dropped; this parser/index layer is what its trusted
successors run on). Consumers: fto, repetitions_in_previous, rising_terminal.

The legacy ``transcripts.parse_transcript`` cannot be reused for the index here:
it raises on bad IDs (poisoning the whole file). This module parses each
transcript file line-by-line itself and silently skips malformed lines.
"""
from __future__ import annotations

from pathlib import Path

from ..transcripts import _FILENAME_RE, _ID_RE, iter_transcript_paths

WellFormed = tuple[int, float, float, str]  # (utt_num, start, end, text)
MergedEntry = tuple[str, int, float, float]  # (side, utt_num, start, end)
TurnGapIndex = dict[int, list[MergedEntry]]
TextIndex = dict[tuple[int, str, int], str]  # (call, side, utt_num) → transcript text


def _parse_one_file(path: Path) -> tuple[int, str, list[WellFormed]] | None:
    """Parse a single trans file. Malformed lines are silently dropped.

    Returns (call_id, side, well_formed_entries) on success, or None if the
    filename itself is unparseable. Each entry carries (utt_num, start, end, text).
    """
    fname_match = _FILENAME_RE.match(path.name)
    if not fname_match:
        return None
    expected_call = int(fname_match.group(1))
    expected_side = fname_match.group(2)

    entries: list[WellFormed] = []
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            parsed = _parse_one_line(line, expected_call, expected_side)
            if parsed is not None:
                entries.append(parsed)
    return expected_call, expected_side, entries


def _parse_one_line(
    line: str, expected_call: int, expected_side: str
) -> WellFormed | None:
    parts = line.split(maxsplit=3)
    if len(parts) < 4:
        return None
    id_field, start_s, end_s, text = parts
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
    return (int(utt_str), start, end, text)


def build_turn_gap_index(transcript_root: Path) -> TurnGapIndex:
    """Build per-conversation merged-chronological index of well-formed entries."""
    by_call: dict[int, list[MergedEntry]] = {}
    for tpath in iter_transcript_paths(transcript_root):
        try:
            parsed = _parse_one_file(tpath)
        except OSError:
            continue
        if parsed is None:
            continue
        call_id, side, entries = parsed
        bucket = by_call.setdefault(call_id, [])
        for utt_num, start, end, _text in entries:
            bucket.append((side, utt_num, start, end))
    for call_id, merged in by_call.items():
        # Sort by start time, then end time, then utt_num as a deterministic tiebreaker.
        merged.sort(key=lambda x: (x[2], x[3], x[1]))
    return by_call


def build_text_index(transcript_root: Path) -> TextIndex:
    """Per-utterance transcript text keyed on (call_id, side, utt_num).

    Built from the same line-by-line parser as `build_turn_gap_index` so the
    two indexes stay in sync (only well-formed lines appear in either).
    """
    idx: TextIndex = {}
    for tpath in iter_transcript_paths(transcript_root):
        try:
            parsed = _parse_one_file(tpath)
        except OSError:
            continue
        if parsed is None:
            continue
        call_id, side, entries = parsed
        for utt_num, _start, _end, text in entries:
            idx[(call_id, side, utt_num)] = text
    return idx
