import csv
from pathlib import Path

from swb_extract.features.fto import TRANSFER, build_turn_events
from swb_extract.features.overlap_split import (
    HEADER,
    OBSTRUCTIVE_WINDOW_SEC,
    classify_event,
    lookup_split,
    write_overlap_split,
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


# sw2003 — every overlap-event class in one conversation (trans-bound fallback):
#   A0002[0.0,6.0]    first substantive turn                     → (0,0)
#   B0004[2.0,2.5]    "uh-huh" DURING A's turn                   → coop (bc by definition)
#   A0006[6.5,8.0]    same-side continuation (turn end → 8.0)    → (0,0)
#   B0008[7.5,10.5]   transfer, onset 0.5s before A's end ≤ W    → OBSTRUCTIVE
#   A0010[9.0,9.6]    contained interjection into B's turn       → coop (floor retained)
#   B0012[10.8,14.5]  continuation (B's turn end → 14.5)         → (0,0)
#   A0014[13.0,16.0]  transfer, onset 1.5s before B's end > W    → coop (deep overlap,
#                     holder finished their turn anyway)
BODY_2003_A = (
    "sw2003A-ms98-a-0002 0.0 6.0 we drove up the coast for two weeks last summer\n"
    "sw2003A-ms98-a-0006 6.5 8.0 it was really something\n"
    "sw2003A-ms98-a-0010 9.0 9.6 you did\n"
    "sw2003A-ms98-a-0014 13.0 16.0 that is amazing tell me everything about it\n"
)
BODY_2003_B = (
    "sw2003B-ms98-a-0004 2.0 2.5 uh-huh\n"
    "sw2003B-ms98-a-0008 7.5 10.5 oh we did that exact trip two years ago\n"
    "sw2003B-ms98-a-0012 10.8 14.5 yes indeed we rented a little cabin right on the water\n"
)

# sw2004 — non-overlap cases: a backchannel AFTER the turn's running end and a
# positive-gap transfer are not overlap events.
BODY_2004_A = "sw2004A-ms98-a-0002 0.0 3.0 did you ever get out to the lake house\n"
BODY_2004_B = (
    "sw2004B-ms98-a-0004 3.5 4.0 uh-huh\n"
    "sw2004B-ms98-a-0006 5.0 7.0 we went there just last month\n"
)

# sw2005 — conversation-initial backchannel: no turn exists to overlap.
BODY_2005_A = "sw2005A-ms98-a-0002 2.0 5.0 so tell me about your garden\n"
BODY_2005_B = "sw2005B-ms98-a-0004 0.5 1.0 yeah\n"

ALL_FILES = {
    "sw2003A-ms98-a-trans.text": BODY_2003_A,
    "sw2003B-ms98-a-trans.text": BODY_2003_B,
    "sw2004A-ms98-a-trans.text": BODY_2004_A,
    "sw2004B-ms98-a-trans.text": BODY_2004_B,
    "sw2005A-ms98-a-trans.text": BODY_2005_A,
    "sw2005B-ms98-a-trans.text": BODY_2005_B,
}


def _idx(tmp_path):
    return build_turn_events(_make_trans_root(tmp_path, ALL_FILES))


def test_overlapping_backchannel_is_cooperative(tmp_path):
    assert lookup_split(_idx(tmp_path), "200/sw2003B-U0004.wav") == (1, 0)


def test_quick_yield_transfer_is_obstructive(tmp_path):
    assert lookup_split(_idx(tmp_path), "200/sw2003B-U0008.wav") == (0, 1)


def test_contained_interjection_is_cooperative(tmp_path):
    assert lookup_split(_idx(tmp_path), "200/sw2003A-U0010.wav") == (1, 0)


def test_deep_overlap_transfer_is_cooperative(tmp_path):
    # B kept the floor for 1.5 s (> W) after A came in, finishing the turn:
    # the overlap did not drive them off.
    assert lookup_split(_idx(tmp_path), "200/sw2003A-U0014.wav") == (1, 0)


def test_first_turn_and_continuations_are_not_events(tmp_path):
    idx = _idx(tmp_path)
    assert lookup_split(idx, "200/sw2003A-U0002.wav") == (0, 0)
    assert lookup_split(idx, "200/sw2003A-U0006.wav") == (0, 0)
    assert lookup_split(idx, "200/sw2003B-U0012.wav") == (0, 0)


def test_backchannel_after_turn_end_is_not_an_event(tmp_path):
    assert lookup_split(_idx(tmp_path), "200/sw2004B-U0004.wav") == (0, 0)


def test_positive_gap_transfer_is_not_an_event(tmp_path):
    assert lookup_split(_idx(tmp_path), "200/sw2004B-U0006.wav") == (0, 0)


def test_conversation_initial_backchannel_is_not_an_event(tmp_path):
    assert lookup_split(_idx(tmp_path), "200/sw2005B-U0004.wav") == (0, 0)


def test_unplaceable_is_none(tmp_path):
    assert lookup_split(_idx(tmp_path), "200/sw2099A-U0002.wav") is None


def test_window_boundary():
    # Synthetic transfer events either side of the pre-registered W = 1.0 s
    # (avoid exact-boundary float equality).
    ref = 10.0
    inside = (TRANSFER, ref - OBSTRUCTIVE_WINDOW_SEC + 0.01, 15.0, ref, 0)
    outside = (TRANSFER, ref - OBSTRUCTIVE_WINDOW_SEC - 0.01, 15.0, ref, 0)
    assert classify_event(inside) == (0, 1)
    assert classify_event(outside) == (1, 0)


def test_write_round_trip(tmp_path):
    root = _make_trans_root(tmp_path, ALL_FILES)
    out = tmp_path / "utterances_v2"
    mp = manifest_path(out)
    with open_appender(mp) as w:
        write_row(w, "200/sw2003B-U0004.wav", "")
        write_row(w, "200/sw2003B-U0008.wav", "")
        write_row(w, "200/sw2003A-U0014.wav", "")
        write_row(w, "200/sw2003A-U0002.wav", "")
        write_row(w, "200/sw2099A-U0002.wav", "")
    n = write_overlap_split(
        mp, out / "features" / "overlap_split.csv", transcript_root=root
    )
    assert n == 5
    rows = list(csv.reader((out / "features" / "overlap_split.csv").open()))
    assert tuple(rows[0]) == HEADER
    by_rel = {r[0]: (r[1], r[2]) for r in rows[1:]}
    assert by_rel["200/sw2003B-U0004.wav"] == ("1", "0")
    assert by_rel["200/sw2003B-U0008.wav"] == ("0", "1")
    assert by_rel["200/sw2003A-U0014.wav"] == ("1", "0")
    assert by_rel["200/sw2003A-U0002.wav"] == ("0", "0")
    assert by_rel["200/sw2099A-U0002.wav"] == ("", "")  # unplaceable
