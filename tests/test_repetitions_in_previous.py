import csv
from pathlib import Path

import pytest

from swb_extract.features.repetitions_in_previous import (
    HEADER,
    count_cross_pair_matches,
    lookup_repetitions_in_previous,
    write_repetitions_in_previous,
)
from swb_extract.features.turn_gap import build_text_index, build_turn_gap_index
from swb_extract.manifest import (
    MANIFEST_HEADER,
    manifest_path,
    open_appender,
    write_row,
)


def _make_trans_root(tmp_path: Path, files: dict[str, str]) -> Path:
    root = tmp_path / "swb_ms98_transcriptions_cleaned"
    for fname, body in files.items():
        call = int(fname[2:6])
        nn = call // 100
        d = root / f"{nn:02d}" / f"{call:04d}"
        d.mkdir(parents=True, exist_ok=True)
        (d / fname).write_text(body, encoding="utf-8")
    return root


def test_count_cross_pair_matches_basic():
    # current="the cat the", previous="the the"
    # the appears 2x in current, 2x in previous → 2 * 2 = 4
    # cat appears 1x in current, 0x in previous → 0
    assert count_cross_pair_matches(["the", "cat", "the"], ["the", "the"]) == 4
    # No overlap
    assert count_cross_pair_matches(["a", "b"], ["c", "d"]) == 0
    # Empty
    assert count_cross_pair_matches([], ["the"]) == 0
    assert count_cross_pair_matches(["the"], []) == 0


def test_count_cross_pair_matches_mixed():
    # cur=["a","b","a"], prev=["a","b","b"] → a:2*1=2, b:1*2=2 → 4
    assert count_cross_pair_matches(["a", "b", "a"], ["a", "b", "b"]) == 4


def test_first_of_conversation_returns_none(tmp_path):
    body_a = "sw2001A-ms98-a-0002 0.0 1.0 hi the the\n"
    body_b = "sw2001B-ms98-a-0001 0.5 1.5 yes the\n"
    root = _make_trans_root(
        tmp_path,
        {
            "sw2001A-ms98-a-trans.text": body_a,
            "sw2001B-ms98-a-trans.text": body_b,
        },
    )
    merged = build_turn_gap_index(root)
    text = build_text_index(root)
    # A.U002 starts at 0.0 → first-of-conv → None
    assert lookup_repetitions_in_previous(merged, text, "200/sw2001A-U0002.wav") is None
    # B.U001 starts at 0.5 → previous is A.U002; both contain "the"
    # cur=["yes","the"], prev=["hi","the","the"] → "the":1*2=2 → 2
    assert lookup_repetitions_in_previous(merged, text, "200/sw2001B-U0001.wav") == 2


def test_cross_speaker_previous(tmp_path):
    # A.U002: "hi the cat" (0.0–1.0)
    # B.U001: "the cat" (1.5–2.0) → prev is A.U002 → "the":1*1+ "cat":1*1 = 2
    body_a = "sw2001A-ms98-a-0002 0.0 1.0 hi the cat\n"
    body_b = "sw2001B-ms98-a-0001 1.5 2.0 the cat\n"
    root = _make_trans_root(
        tmp_path,
        {
            "sw2001A-ms98-a-trans.text": body_a,
            "sw2001B-ms98-a-trans.text": body_b,
        },
    )
    merged = build_turn_gap_index(root)
    text = build_text_index(root)
    assert lookup_repetitions_in_previous(merged, text, "200/sw2001B-U0001.wav") == 2


def test_brackets_stripped_in_both_sides(tmp_path):
    # [noise] tokens stripped in both current and previous before counting.
    # current="[noise] the", previous="[noise] the" → only "the":1*1=1, not 2.
    body_a = "sw2001A-ms98-a-0002 0.0 1.0 [noise] the\n"
    body_b = "sw2001B-ms98-a-0001 1.5 2.0 [noise] the\n"
    root = _make_trans_root(
        tmp_path,
        {
            "sw2001A-ms98-a-trans.text": body_a,
            "sw2001B-ms98-a-trans.text": body_b,
        },
    )
    merged = build_turn_gap_index(root)
    text = build_text_index(root)
    assert lookup_repetitions_in_previous(merged, text, "200/sw2001B-U0001.wav") == 1


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
    merged = build_turn_gap_index(root)
    text = build_text_index(root)
    assert lookup_repetitions_in_previous(merged, text, "200/sw2001A-U9999.wav") is None


def test_malformed_line_dropped_neighbors_unaffected(tmp_path):
    # Malformed line in side A is dropped; surrounding rows still resolve.
    body_a = (
        "sw2001A-ms98-a-0002 0.0 1.0 the cat\n"
        "BADID-line 0.5 0.6 garbled\n"
        "sw2001A-ms98-a-0004 2.0 3.0 the dog\n"
    )
    body_b = "sw2001B-ms98-a-0001 1.5 1.8 the bird\n"
    root = _make_trans_root(
        tmp_path,
        {
            "sw2001A-ms98-a-trans.text": body_a,
            "sw2001B-ms98-a-trans.text": body_b,
        },
    )
    merged = build_turn_gap_index(root)
    text = build_text_index(root)
    # Merged order: A.U002, B.U001, A.U004
    # B.U001 prev=A.U002 → "the":1*1=1
    # A.U004 prev=B.U001 → "the":1*1=1
    assert lookup_repetitions_in_previous(merged, text, "200/sw2001B-U0001.wav") == 1
    assert lookup_repetitions_in_previous(merged, text, "200/sw2001A-U0004.wav") == 1


def test_write_round_trip(tmp_path, transcript_root):
    out = tmp_path / "utterances_v2"
    mp = manifest_path(out)
    with open_appender(mp) as w:
        write_row(w, "200/sw2001A-U0002.wav", "")
        write_row(w, "200/sw2001A-U0004.wav", "")

    n = write_repetitions_in_previous(
        mp,
        out / "features" / "repetitions_in_previous.csv",
        transcript_root=transcript_root,
    )
    assert n == 2

    rows = list(csv.reader((out / "features" / "repetitions_in_previous.csv").open()))
    assert tuple(rows[0]) == HEADER
    # U0004 must produce an integer (its predecessor exists somewhere in conv 2001)
    assert rows[2][0] == "200/sw2001A-U0004.wav"
    assert rows[2][1] != ""
    int(rows[2][1])  # parses as int


def test_join_contract_against_manifest(tmp_path, transcript_root):
    out = tmp_path / "utterances_v2"
    mp = manifest_path(out)
    with open_appender(mp) as w:
        write_row(w, "200/sw2001A-U0002.wav", "")
        write_row(w, "200/sw2001A-U0004.wav", "")

    write_repetitions_in_previous(
        mp, out / "features" / "repetitions_in_previous.csv",
        transcript_root=transcript_root,
    )
    feat = list(csv.reader((out / "features" / "repetitions_in_previous.csv").open()))
    manifest = list(csv.reader(mp.open()))
    assert feat[0] == list(HEADER)
    assert manifest[0] == list(MANIFEST_HEADER)
    assert len(feat) == len(manifest)
    for fr, mr in zip(feat[1:], manifest[1:]):
        assert fr[0] == mr[0]


def test_features_dispatch_routes_repetitions_in_previous(tmp_path, transcript_root):
    from swb_extract.cli import main

    out = tmp_path / "utterances_v2"
    mp = manifest_path(out)
    with open_appender(mp) as w:
        write_row(w, "200/sw2001A-U0002.wav", "")

    rc = main([
        "features", "repetitions_in_previous",
        "--out-root", str(out),
        "--transcript-root", str(transcript_root),
    ])
    assert rc == 0
    assert (out / "features" / "repetitions_in_previous.csv").exists()
