"""Pins the chronological-merge index in features/_turn_index.py.

Ported from switchboard's test_turn_gap.py. The deprecated Turn Gap *measure*
(gap vs merged-chronological predecessor) survives here only as a local test
oracle — asserting on gap values verifies the index's merge order and the
parser's malformed-line dropping in one arithmetic check.
"""
from pathlib import Path

import pytest

from swb_extract.features._turn_index import TurnGapIndex, build_turn_gap_index
from swb_extract.manifest import parse_rel_path


def _lookup_turn_gap(idx: TurnGapIndex, rel_path: str) -> float | None:
    """Test oracle: current.start - previous.end over the merged chronology."""
    call_id, side, utt_num = parse_rel_path(rel_path)
    merged = idx.get(call_id)
    if not merged:
        return None
    cur_pos = None
    for i, e in enumerate(merged):
        if e[0] == side and e[1] == utt_num:
            cur_pos = i
            break
    if cur_pos is None or cur_pos == 0:
        return None
    prev = merged[cur_pos - 1]
    cur = merged[cur_pos]
    return cur[2] - prev[3]


def _make_trans_root(tmp_path: Path, files: dict[str, str]) -> Path:
    """Create a synthetic transcript root with the given trans files."""
    root = tmp_path / "swb_ms98_transcriptions_cleaned"
    for fname, body in files.items():
        call = int(fname[2:6])
        nn = call // 100
        d = root / f"{nn:02d}" / f"{call:04d}"
        d.mkdir(parents=True, exist_ok=True)
        (d / fname).write_text(body, encoding="utf-8")
    return root


def test_build_index_merges_both_sides_chronologically(tmp_path):
    body_a = (
        "sw2001A-ms98-a-0002 0.0 1.0 hi\n"
        "sw2001A-ms98-a-0004 2.0 3.0 ok\n"
    )
    body_b = (
        "sw2001B-ms98-a-0001 0.5 1.5 yes\n"
        "sw2001B-ms98-a-0003 1.7 2.5 no\n"
    )
    root = _make_trans_root(
        tmp_path,
        {
            "sw2001A-ms98-a-trans.text": body_a,
            "sw2001B-ms98-a-trans.text": body_b,
        },
    )
    idx = build_turn_gap_index(root)
    merged = idx[2001]
    assert [(e[0], e[1]) for e in merged] == [
        ("A", 2),
        ("B", 1),
        ("B", 3),
        ("A", 4),
    ]


def test_first_of_conversation_returns_none(tmp_path):
    body_a = "sw2001A-ms98-a-0002 0.0 1.0 hi\n"
    body_b = "sw2001B-ms98-a-0001 0.5 1.5 yes\n"
    root = _make_trans_root(
        tmp_path,
        {
            "sw2001A-ms98-a-trans.text": body_a,
            "sw2001B-ms98-a-trans.text": body_b,
        },
    )
    idx = build_turn_gap_index(root)
    # A.U002 starts at 0.0 → first-of-conv → None
    assert _lookup_turn_gap(idx, "200/sw2001A-U0002.wav") is None
    # B.U001 starts at 0.5 → second; gap = 0.5 - 1.0 = -0.5 (overlap)
    assert _lookup_turn_gap(idx, "200/sw2001B-U0001.wav") == pytest.approx(-0.5)


def test_cross_speaker_gap_math(tmp_path):
    # A.U002: 0.0 → 1.0
    # B.U001: 1.5 → 2.0   gap = 1.5 - 1.0 = 0.5
    # A.U004: 2.5 → 3.0   gap = 2.5 - 2.0 = 0.5
    body_a = (
        "sw2001A-ms98-a-0002 0.0 1.0 hi\n"
        "sw2001A-ms98-a-0004 2.5 3.0 ok\n"
    )
    body_b = "sw2001B-ms98-a-0001 1.5 2.0 yes\n"
    root = _make_trans_root(
        tmp_path,
        {
            "sw2001A-ms98-a-trans.text": body_a,
            "sw2001B-ms98-a-trans.text": body_b,
        },
    )
    idx = build_turn_gap_index(root)
    assert _lookup_turn_gap(idx, "200/sw2001B-U0001.wav") == pytest.approx(0.5)
    assert _lookup_turn_gap(idx, "200/sw2001A-U0004.wav") == pytest.approx(0.5)


def test_unknown_rel_path_returns_none(tmp_path):
    body_a = "sw2001A-ms98-a-0002 0.0 1.0 hi\n"
    body_b = "sw2001B-ms98-a-0001 0.5 1.5 yes\n"
    root = _make_trans_root(
        tmp_path,
        {
            "sw2001A-ms98-a-trans.text": body_a,
            "sw2001B-ms98-a-trans.text": body_b,
        },
    )
    idx = build_turn_gap_index(root)
    assert _lookup_turn_gap(idx, "200/sw2001A-U9999.wav") is None


def test_only_one_side_present_still_resolves_within_that_side(tmp_path):
    # Side B missing entirely. Side A's utterances merge into a single-side
    # ordering. First A utterance is None; subsequent A utterances compute
    # against the previous A entry.
    body_a = (
        "sw2001A-ms98-a-0002 0.0 1.0 hi\n"
        "sw2001A-ms98-a-0004 2.0 3.0 ok\n"
    )
    root = _make_trans_root(tmp_path, {"sw2001A-ms98-a-trans.text": body_a})
    idx = build_turn_gap_index(root)
    assert _lookup_turn_gap(idx, "200/sw2001A-U0002.wav") is None  # first
    assert _lookup_turn_gap(idx, "200/sw2001A-U0004.wav") == pytest.approx(1.0)


def test_malformed_line_is_dropped_neighbors_unaffected(tmp_path):
    # A malformed line is silently dropped from the merge; surrounding
    # well-formed utterances still compute their cross-speaker gaps normally.
    body_a = (
        "sw2001A-ms98-a-0002 0.0 1.0 hi\n"
        "BADID-line 0.5 0.6 garbled\n"
        "sw2001A-ms98-a-0004 2.0 3.0 ok\n"
    )
    body_b = "sw2001B-ms98-a-0001 1.5 1.8 yes\n"
    root = _make_trans_root(
        tmp_path,
        {
            "sw2001A-ms98-a-trans.text": body_a,
            "sw2001B-ms98-a-trans.text": body_b,
        },
    )
    idx = build_turn_gap_index(root)
    # Merged order (well-formed only): A.U002(0), B.U001(1.5), A.U004(2.0)
    assert _lookup_turn_gap(idx, "200/sw2001A-U0002.wav") is None  # first
    assert _lookup_turn_gap(idx, "200/sw2001B-U0001.wav") == pytest.approx(0.5)
    assert _lookup_turn_gap(idx, "200/sw2001A-U0004.wav") == pytest.approx(0.2)


def test_short_line_is_dropped(tmp_path):
    body_a = (
        "sw2001A-ms98-a-0002 0.0 1.0 hi\n"
        "junk\n"
        "sw2001A-ms98-a-0004 2.0 3.0 ok\n"
    )
    body_b = "sw2001B-ms98-a-0001 1.5 1.8 yes\n"
    root = _make_trans_root(
        tmp_path,
        {
            "sw2001A-ms98-a-trans.text": body_a,
            "sw2001B-ms98-a-trans.text": body_b,
        },
    )
    idx = build_turn_gap_index(root)
    # Short line dropped; everything else computes
    assert _lookup_turn_gap(idx, "200/sw2001A-U0004.wav") == pytest.approx(0.2)


def test_non_numeric_times_dropped(tmp_path):
    body_a = (
        "sw2001A-ms98-a-0002 0.0 1.0 hi\n"
        "sw2001A-ms98-a-0004 NaNstart 3.0 ok\n"
        "sw2001A-ms98-a-0006 4.0 5.0 ok\n"
    )
    body_b = "sw2001B-ms98-a-0001 1.5 1.8 yes\n"
    root = _make_trans_root(
        tmp_path,
        {
            "sw2001A-ms98-a-trans.text": body_a,
            "sw2001B-ms98-a-trans.text": body_b,
        },
    )
    idx = build_turn_gap_index(root)
    # U0004 dropped; merged: A.U002, B.U001, A.U006
    assert _lookup_turn_gap(idx, "200/sw2001A-U0004.wav") is None  # missing
    assert _lookup_turn_gap(idx, "200/sw2001A-U0006.wav") == pytest.approx(2.2)


def test_id_mismatch_dropped(tmp_path):
    body_a = (
        "sw2001A-ms98-a-0002 0.0 1.0 hi\n"
        "sw2001B-ms98-a-0004 2.0 3.0 wrong-side\n"
        "sw2001A-ms98-a-0006 4.0 5.0 ok\n"
    )
    body_b = "sw2001B-ms98-a-0001 1.5 1.8 yes\n"
    root = _make_trans_root(
        tmp_path,
        {
            "sw2001A-ms98-a-trans.text": body_a,
            "sw2001B-ms98-a-trans.text": body_b,
        },
    )
    idx = build_turn_gap_index(root)
    # The mismatched line is dropped from side A's parse.
    assert _lookup_turn_gap(idx, "200/sw2001A-U0006.wav") == pytest.approx(2.2)


def test_missing_conversation_returns_none(tmp_path):
    root = tmp_path / "empty"
    root.mkdir()
    idx = build_turn_gap_index(root)
    assert _lookup_turn_gap(idx, "200/sw2001A-U0002.wav") is None
