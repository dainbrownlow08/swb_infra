import csv
from pathlib import Path

import pytest

from swb_extract.features.turn_gap import (
    HEADER,
    build_turn_gap_index,
    lookup_turn_gap,
    write_turn_gaps,
)
from swb_extract.manifest import (
    MANIFEST_HEADER,
    manifest_path,
    open_appender,
    write_row,
)


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
    assert lookup_turn_gap(idx, "200/sw2001A-U0002.wav") is None
    # B.U001 starts at 0.5 → second; gap = 0.5 - 1.0 = -0.5 (overlap)
    assert lookup_turn_gap(idx, "200/sw2001B-U0001.wav") == pytest.approx(-0.5)


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
    assert lookup_turn_gap(idx, "200/sw2001B-U0001.wav") == pytest.approx(0.5)
    assert lookup_turn_gap(idx, "200/sw2001A-U0004.wav") == pytest.approx(0.5)


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
    assert lookup_turn_gap(idx, "200/sw2001A-U9999.wav") is None


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
    assert lookup_turn_gap(idx, "200/sw2001A-U0002.wav") is None  # first
    assert lookup_turn_gap(idx, "200/sw2001A-U0004.wav") == pytest.approx(1.0)


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
    assert lookup_turn_gap(idx, "200/sw2001A-U0002.wav") is None  # first
    assert lookup_turn_gap(idx, "200/sw2001B-U0001.wav") == pytest.approx(0.5)
    assert lookup_turn_gap(idx, "200/sw2001A-U0004.wav") == pytest.approx(0.2)


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
    assert lookup_turn_gap(idx, "200/sw2001A-U0004.wav") == pytest.approx(0.2)


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
    assert lookup_turn_gap(idx, "200/sw2001A-U0004.wav") is None  # missing
    assert lookup_turn_gap(idx, "200/sw2001A-U0006.wav") == pytest.approx(2.2)


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
    assert lookup_turn_gap(idx, "200/sw2001A-U0006.wav") == pytest.approx(2.2)


def test_missing_conversation_returns_none(tmp_path):
    root = tmp_path / "empty"
    root.mkdir()
    idx = build_turn_gap_index(root)
    assert lookup_turn_gap(idx, "200/sw2001A-U0002.wav") is None


def test_write_turn_gaps_round_trip(tmp_path, transcript_root):
    out = tmp_path / "utterances_v2"
    mp = manifest_path(out)
    with open_appender(mp) as w:
        write_row(w, "200/sw2001A-U0002.wav", "hi")
        write_row(w, "200/sw2001A-U0004.wav", "um-hum")

    n = write_turn_gaps(
        mp,
        out / "features" / "turn_gap.csv",
        transcript_root=transcript_root,
    )
    assert n == 2

    rows = list(csv.reader((out / "features" / "turn_gap.csv").open()))
    assert tuple(rows[0]) == HEADER
    assert rows[1][0] == "200/sw2001A-U0002.wav"
    # First row: first utterance of conv 2001 chronologically? sw2001A-0002
    # starts at 0.977625 — verify by checking it's either empty (first) or a float.
    # In real conv 2001, the first utterance overall could be on either side.
    # Either way, U0004 must produce a value (its predecessor exists somewhere).
    assert rows[2][0] == "200/sw2001A-U0004.wav"
    assert rows[2][1] != ""
    float(rows[2][1])  # parses


def test_join_contract_against_manifest(tmp_path, transcript_root):
    out = tmp_path / "utterances_v2"
    mp = manifest_path(out)
    with open_appender(mp) as w:
        write_row(w, "200/sw2001A-U0002.wav", "hi")
        write_row(w, "200/sw2001A-U0004.wav", "um-hum")

    write_turn_gaps(
        mp, out / "features" / "turn_gap.csv",
        transcript_root=transcript_root,
    )
    feat = list(csv.reader((out / "features" / "turn_gap.csv").open()))
    manifest = list(csv.reader(mp.open()))
    assert feat[0] == list(HEADER)
    assert manifest[0] == list(MANIFEST_HEADER)
    assert len(feat) == len(manifest)
    for fr, mr in zip(feat[1:], manifest[1:]):
        assert fr[0] == mr[0]


def test_features_dispatch_routes_turn_gap(tmp_path, transcript_root):
    from swb_extract.cli import main

    out = tmp_path / "utterances_v2"
    mp = manifest_path(out)
    with open_appender(mp) as w:
        write_row(w, "200/sw2001A-U0002.wav", "hi")

    rc = main([
        "features", "turn_gap",
        "--out-root", str(out),
        "--transcript-root", str(transcript_root),
    ])
    assert rc == 0
    assert (out / "features" / "turn_gap.csv").exists()
