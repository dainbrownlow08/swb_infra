import csv
from pathlib import Path

import pytest

from swb_extract.features.backchannels import is_backchannel
from swb_extract.features.fto import (
    HEADER,
    build_fto_index,
    lookup_fto,
    write_ftos,
)
from swb_extract.manifest import manifest_path, open_appender, write_row


def _make_trans_root(tmp_path: Path, files: dict[str, str]) -> Path:
    root = tmp_path / "swb_ms98_transcriptions_cleaned"
    for fname, body in files.items():
        call = int(fname[2:6])
        nn = call // 100
        d = root / f"{nn:02d}" / f"{call:04d}"
        d.mkdir(parents=True, exist_ok=True)
        (d / fname).write_text(body, encoding="utf-8")
    return root


# sw2001 chronology (by start time), no word files (trans-bound fallback):
#   A0002[0.0,5.0]  "hello there how are you doing"        first substantive turn
#   B0004[2.0,2.4]  "uh-huh"                               listener backchannel
#   A0006[5.5,6.5]  "so we went to the lake"               continuation, onset 0.5
#   B0008[6.6,9.0]  "oh wow that sounds like a great trip" transfer, FTO/onset +0.1
#   A0010[8.8,10.0] "yeah it really was"                   transfer, FTO/onset -0.2
#   B0012[10.5,11.0] "right"                               listener backchannel
#   A0014[11.5,12.0] "anyway"                              continuation, onset 1.5
#   B0016[12.4,13.0] "we should do it again"               transfer vs 12.0 → +0.4
BODY_2001_A = (
    "sw2001A-ms98-a-0002 0.0 5.0 hello there how are you doing\n"
    "sw2001A-ms98-a-0006 5.5 6.5 so we went to the lake\n"
    "sw2001A-ms98-a-0010 8.8 10.0 yeah it really was\n"
    "sw2001A-ms98-a-0014 11.5 12.0 anyway\n"
)
BODY_2001_B = (
    "sw2001B-ms98-a-0004 2.0 2.4 uh-huh\n"
    "sw2001B-ms98-a-0008 6.6 9.0 oh wow that sounds like a great trip\n"
    "sw2001B-ms98-a-0012 10.5 11.0 right\n"
    "sw2001B-ms98-a-0016 12.4 13.0 we should do it again\n"
)

# sw2002: a substantive interjection contained in the holder's speech.
BODY_2002_A = (
    "sw2002A-ms98-a-0002 0.0 10.0 we always go up to the mountains in the summer\n"
    "sw2002A-ms98-a-0006 10.5 11.0 yes we do\n"
)
BODY_2002_B = (
    "sw2002B-ms98-a-0004 4.0 5.0 you do not say\n"
    "sw2002B-ms98-a-0008 11.2 12.0 that is wonderful\n"
)

# sw2003: the holder's own lexically-BC utterance extends the turn.
BODY_2003_A = (
    "sw2003A-ms98-a-0002 0.0 2.0 so we bought the house\n"
    "sw2003A-ms98-a-0004 2.5 2.8 yeah\n"
)
BODY_2003_B = "sw2003B-ms98-a-0006 3.0 4.0 congratulations that is great\n"

# sw2004: word-tight edges (trans bounds padded; word rows tight).
BODY_2004_A = "sw2004A-ms98-a-0002 0.0 5.0 okay so\n"
BODY_2004_B = "sw2004B-ms98-a-0004 4.5 6.0 right yeah exactly\n"
WORDS_2004_A = (
    "sw2004A-ms98-a-0002 0.3 0.6 okay\n"
    "sw2004A-ms98-a-0002 0.6 4.2 so\n"
)
WORDS_2004_B = (
    "sw2004B-ms98-a-0004 4.8 5.2 right\n"
    "sw2004B-ms98-a-0004 5.2 6.0 yeah\n"
)

ALL_FILES = {
    "sw2001A-ms98-a-trans.text": BODY_2001_A,
    "sw2001B-ms98-a-trans.text": BODY_2001_B,
    "sw2002A-ms98-a-trans.text": BODY_2002_A,
    "sw2002B-ms98-a-trans.text": BODY_2002_B,
    "sw2003A-ms98-a-trans.text": BODY_2003_A,
    "sw2003B-ms98-a-trans.text": BODY_2003_B,
    "sw2004A-ms98-a-trans.text": BODY_2004_A,
    "sw2004B-ms98-a-trans.text": BODY_2004_B,
    "sw2004A-ms98-a-word.text": WORDS_2004_A,
    "sw2004B-ms98-a-word.text": WORDS_2004_B,
}


def _idx(tmp_path):
    return build_fto_index(_make_trans_root(tmp_path, ALL_FILES))


def test_first_substantive_turn_has_no_fto(tmp_path):
    assert lookup_fto(_idx(tmp_path), "200/sw2001A-U0002.wav") == (None, None, 1, 0, 0)


def test_listener_backchannel(tmp_path):
    assert lookup_fto(_idx(tmp_path), "200/sw2001B-U0004.wav") == (None, None, 0, 1, 0)


def test_same_side_continuation_onset(tmp_path):
    fto, onset, initial, bc, interj = lookup_fto(
        _idx(tmp_path), "200/sw2001A-U0006.wav"
    )
    assert (fto, initial, bc, interj) == (None, 0, 0, 0)
    assert onset == pytest.approx(0.5)  # 5.5 − own previous end 5.0


def test_transfer_positive_gap(tmp_path):
    fto, onset, initial, bc, interj = lookup_fto(
        _idx(tmp_path), "200/sw2001B-U0008.wav"
    )
    assert (initial, bc, interj) == (1, 0, 0)
    assert fto == pytest.approx(0.1)
    assert onset == pytest.approx(0.1)  # turn-initial: onset == FTO


def test_transfer_overlap_negative(tmp_path):
    fto, onset, initial, bc, interj = lookup_fto(
        _idx(tmp_path), "200/sw2001A-U0010.wav"
    )
    assert (initial, bc, interj) == (1, 0, 0)
    assert fto == pytest.approx(-0.2)
    assert onset == pytest.approx(-0.2)


def test_backchannel_does_not_break_turn(tmp_path):
    idx = _idx(tmp_path)
    # A0014 continues A's turn across B's "right"; onset vs own A0010 end (10.0)
    fto, onset, initial, bc, interj = lookup_fto(idx, "200/sw2001A-U0014.wav")
    assert (fto, initial, bc, interj) == (None, 0, 0, 0)
    assert onset == pytest.approx(1.5)
    # ...so B0016 measures against A's extended turn end (12.0), not 10.0
    fto, onset, initial, bc, interj = lookup_fto(idx, "200/sw2001B-U0016.wav")
    assert (initial, bc, interj) == (1, 0, 0)
    assert fto == pytest.approx(0.4)


def test_contained_interjection_is_not_a_transfer(tmp_path):
    idx = _idx(tmp_path)
    fto, onset, initial, bc, interj = lookup_fto(idx, "200/sw2002B-U0004.wav")
    assert (fto, initial, bc, interj) == (None, 0, 0, 1)
    assert onset == pytest.approx(-6.0)  # 4.0 into the holder's turn ending 10.0
    # A's turn survives the interjection; A0006 is a continuation...
    fto, onset, initial, bc, interj = lookup_fto(idx, "200/sw2002A-U0006.wav")
    assert (fto, initial, bc, interj) == (None, 0, 0, 0)
    assert onset == pytest.approx(0.5)
    # ...and the next real transfer measures against A's full turn end (11.0)
    fto, onset, initial, bc, interj = lookup_fto(idx, "200/sw2002B-U0008.wav")
    assert (initial, bc, interj) == (1, 0, 0)
    assert fto == pytest.approx(0.2)


def test_holders_own_bc_token_extends_turn(tmp_path):
    idx = _idx(tmp_path)
    # A's own "yeah" mid-turn: lexically BC, but extends A's turn; onset 0.5
    fto, onset, initial, bc, interj = lookup_fto(idx, "200/sw2003A-U0004.wav")
    assert (fto, initial, bc, interj) == (None, 0, 1, 0)
    assert onset == pytest.approx(0.5)
    fto, onset, initial, bc, interj = lookup_fto(idx, "200/sw2003B-U0006.wav")
    assert (initial, bc, interj) == (1, 0, 0)
    assert fto == pytest.approx(0.2)  # vs word end 2.8, not first utt end 2.0


def test_word_tight_bounds(tmp_path):
    fto, onset, initial, bc, interj = lookup_fto(
        _idx(tmp_path), "200/sw2004B-U0004.wav"
    )
    assert (initial, bc, interj) == (1, 0, 0)
    assert fto == pytest.approx(0.6)  # 4.8 − 4.2 word-tight; trans would say −0.5


# sw2005: trans order and word order DISAGREE. A0002 has long leading trans
# silence so it sorts first by trans start (0.0), but its speech starts at 3.0;
# B0004's speech is 1.0–2.0, entirely before A's. Word-time ordering must make
# B the first turn and A a real transfer (not a contained interjection in A).
BODY_2005_A = "sw2005A-ms98-a-0002 0.0 5.0 okay so\n"
BODY_2005_B = "sw2005B-ms98-a-0004 1.0 2.0 what time\n"
WORDS_2005_A = (
    "sw2005A-ms98-a-0002 3.0 3.5 okay\n"
    "sw2005A-ms98-a-0002 3.5 5.0 so\n"
)
WORDS_2005_B = (
    "sw2005B-ms98-a-0004 1.0 1.5 what\n"
    "sw2005B-ms98-a-0004 1.5 2.0 time\n"
)


def test_word_time_ordering_overrides_trans_padding(tmp_path):
    root = _make_trans_root(
        tmp_path,
        {
            "sw2005A-ms98-a-trans.text": BODY_2005_A,
            "sw2005B-ms98-a-trans.text": BODY_2005_B,
            "sw2005A-ms98-a-word.text": WORDS_2005_A,
            "sw2005B-ms98-a-word.text": WORDS_2005_B,
        },
    )
    idx = build_fto_index(root)
    # B speaks first in word time → opens the conversation's first turn.
    assert lookup_fto(idx, "200/sw2005B-U0004.wav") == (None, None, 1, 0, 0)
    # A is then a genuine floor transfer (3.0 − 2.0 = 1.0), NOT an interjection
    # contained in A — which is what trans-order processing produced.
    fto, onset, initial, bc, interj = lookup_fto(idx, "200/sw2005A-U0002.wav")
    assert (initial, bc, interj) == (1, 0, 0)
    assert fto == pytest.approx(1.0)
    assert onset == pytest.approx(1.0)


def test_is_backchannel_variants():
    assert is_backchannel("Uh-huh.")
    assert is_backchannel("um-hum yeah")
    assert is_backchannel("Okay, right.")
    assert not is_backchannel("oh wow that sounds like a great trip")
    assert not is_backchannel("")
    assert not is_backchannel("   ")


def test_write_round_trip(tmp_path):
    root = _make_trans_root(tmp_path, ALL_FILES)
    out = tmp_path / "utterances_v2"
    mp = manifest_path(out)
    with open_appender(mp) as w:
        write_row(w, "200/sw2001A-U0002.wav", "hello there how are you doing")
        write_row(w, "200/sw2001B-U0008.wav", "oh wow that sounds like a great trip")
        write_row(w, "200/sw2001B-U0012.wav", "right")
        write_row(w, "200/sw2002B-U0004.wav", "you do not say")
        # Not in the trans files: unplaceable, BC flag from manifest transcript
        write_row(w, "200/sw2001A-U0099.wav", "um-hum")
    n = write_ftos(mp, out / "features" / "fto.csv", transcript_root=root)
    assert n == 5
    rows = list(csv.reader((out / "features" / "fto.csv").open()))
    assert tuple(rows[0]) == HEADER
    by_rel = {r[0]: tuple(r[1:]) for r in rows[1:]}
    assert by_rel["200/sw2001A-U0002.wav"] == ("", "", "1", "0", "0")
    fto_s, onset_s, initial, bc, interj = by_rel["200/sw2001B-U0008.wav"]
    assert float(fto_s) == pytest.approx(0.1)
    assert float(onset_s) == pytest.approx(0.1)
    assert (initial, bc, interj) == ("1", "0", "0")
    assert by_rel["200/sw2001B-U0012.wav"] == ("", "", "0", "1", "0")
    fto_s, onset_s, initial, bc, interj = by_rel["200/sw2002B-U0004.wav"]
    assert fto_s == "" and float(onset_s) == pytest.approx(-6.0)
    assert (initial, bc, interj) == ("0", "0", "1")
    assert by_rel["200/sw2001A-U0099.wav"] == ("", "", "", "1", "")
