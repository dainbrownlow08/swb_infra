"""Turn Gap per utterance — cross-speaker chronological.

Turn Gap = current.start - previous.end, where "previous" is the utterance
immediately preceding `current` in the conversation's chronological merge of
both sides (sorted by start time).

A cell is left empty only when we cannot calculate a meaningful gap:

- **First of conversation** — there is no previous utterance. Emitting
  `current.start` here would reintroduce the legacy t=0 inflation bug.
- **Current utterance not found in transcripts** — its (call, side, utt_num)
  is missing or malformed in the trans data, so we cannot place it.

Malformed lines elsewhere in the trans files do not block other rows: they are
simply dropped from the merged chronology. As long as the row's own data and
its merged-list predecessor both parsed cleanly, the gap is computed.

The legacy `transcripts.parse_transcript` cannot be reused for the index here:
it raises on bad IDs (poisoning the whole file). This module parses each
transcript file line-by-line itself and silently skips malformed lines.

Output: utterances_v2/features/turn_gap.csv
Header: Utterance File Name,Turn Gap
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

from ..manifest import MANIFEST_HEADER, manifest_path, parse_rel_path
from ..transcripts import _FILENAME_RE, _ID_RE, iter_transcript_paths

FEATURE_NAME = "turn_gap"
HEADER = ("Utterance File Name", "Turn Gap")

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


def lookup_turn_gap(idx: TurnGapIndex, rel_path: str) -> float | None:
    call_id, side, utt_num = parse_rel_path(rel_path)
    merged = idx.get(call_id)
    if not merged:
        return None
    cur_pos: int | None = None
    for i, e in enumerate(merged):
        if e[0] == side and e[1] == utt_num:
            cur_pos = i
            break
    if cur_pos is None or cur_pos == 0:
        return None
    prev = merged[cur_pos - 1]
    cur = merged[cur_pos]
    return cur[2] - prev[3]


def _fmt(v: float | None) -> str:
    return "" if v is None else repr(v)


def write_turn_gaps(
    manifest_csv: Path,
    output_csv: Path,
    transcript_root: Path,
) -> int:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    idx = build_turn_gap_index(transcript_root)
    n = 0
    n_first = 0
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
            v = lookup_turn_gap(idx, rel)
            if v is None:
                if _is_first_of_conversation(idx, rel):
                    n_first += 1
                else:
                    n_missing += 1
            writer.writerow([rel, _fmt(v)])
            n += 1
    if n_first or n_missing:
        print(
            f"  empty cells: first-of-conversation={n_first}, missing={n_missing}"
        )
    return n


def _is_first_of_conversation(idx: TurnGapIndex, rel_path: str) -> bool:
    try:
        call_id, side, utt_num = parse_rel_path(rel_path)
    except ValueError:
        return False
    merged = idx.get(call_id)
    if not merged:
        return False
    return merged[0][0] == side and merged[0][1] == utt_num


def run(args) -> int:
    out_root = Path(args.out_root)
    n = write_turn_gaps(
        manifest_path(out_root),
        out_root / "features" / "turn_gap.csv",
        transcript_root=Path(args.transcript_root),
    )
    print(f"wrote {n} turn gap rows")
    return 0
