"""Utterance boundaries + per-conversation chronology for wps, olap and poplen.

The paper's utterance is one line of transcript with the speech-to-text system's
timestamps. Switchboard's analogue is one ms98 trans line, but trans times carry
silence padding (fto.py: it biases every between-speaker offset negative), so an
utterance's start/end here are its first-word onset / last-word offset from the
sibling ``*-word.text``, falling back to the trans bounds when an utterance has no
word rows. One definition, shared by the three timing variables.

``onset_timing`` applies the paper's two turn-taking notions in start order,
tracking each side's running max end:

  olap   = 1 if the utterance begins while the partner's talk is still running
           ("begin while the other participant is still talking"), else 0.
  poplen = start − partner's running end, defined only when the partner's end is the
           most recent speech before this start — so non-alternation is allowed, and
           the gap belongs to the partner's NEXT utterance only. Blank when the
           utterance starts in overlap (no pause exists), when the speaker's own talk
           is more recent (an own-own gap), or before the partner has spoken at all.
"""
from __future__ import annotations

from pathlib import Path

from ..inhouse._turn_index import build_turn_gap_index
from ..inhouse.word_align import build_word_index, other_side
from ._io import Key

Bounds = dict[Key, tuple[float, float]]
Utt = tuple[float, float, str, int]  # (start, end, side, utt_num)
Timing = dict[Key, tuple[int, float | None]]  # key → (olap flag, poplen or None)


def build_bounds(transcript_root: Path) -> Bounds:
    """Word-tight (start, end) per utterance; trans bounds where word rows are missing."""
    words = build_word_index(transcript_root)
    bounds: Bounds = {}
    for call_id, merged in build_turn_gap_index(transcript_root).items():
        for side, utt_num, t0, t1 in merged:
            key = (call_id, side, utt_num)
            rows = words.get(key)
            bounds[key] = (rows[0][0], max(e for _s, e, _t in rows)) if rows else (t0, t1)
    return bounds


def build_chronology(bounds: Bounds) -> dict[int, list[Utt]]:
    chrono: dict[int, list[Utt]] = {}
    for (call_id, side, utt_num), (start, end) in bounds.items():
        chrono.setdefault(call_id, []).append((start, end, side, utt_num))
    for utts in chrono.values():
        utts.sort()
    return chrono


def onset_timing(call_id: int, utts: list[Utt]) -> Timing:
    """(olap, poplen) per utterance of one conversation; ``utts`` sorted by start."""
    last_end: dict[str, float] = {}
    out: Timing = {}
    for start, end, side, utt_num in utts:
        other = last_end.get(other_side(side))
        own = last_end.get(side)
        if other is None:
            olap, pause = 0, None  # the partner has not spoken yet
        elif other > start:
            olap, pause = 1, None  # partner still talking: an overlap, no pause
        elif own is not None and own > other:
            olap, pause = 0, None  # own talk is the more recent: an own-own gap
        else:
            olap, pause = 0, start - other  # the partner's end is the last speech before U
        out[(call_id, side, utt_num)] = (olap, pause)
        last_end[side] = end if own is None else max(own, end)
    return out


def build_onset_timing(transcript_root: Path) -> Timing:
    out: Timing = {}
    for call_id, utts in build_chronology(build_bounds(transcript_root)).items():
        out.update(onset_timing(call_id, utts))
    return out
