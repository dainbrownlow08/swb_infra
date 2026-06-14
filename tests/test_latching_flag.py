import csv
from pathlib import Path

from swb_extract.features.fto import build_fto_index
from swb_extract.features.latching_flag import (
    HEADER,
    LATCH_MAX_SEC,
    lookup_latching,
    write_latching_flags,
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


# Same conversation shape as test_fto's sw2001/sw2002 (trans-bound fallback):
#   A0002[0.0,5.0]   first substantive turn            → "" (no transfer)
#   B0004[2.0,2.4]   "uh-huh" backchannel              → ""
#   A0006[5.5,6.5]   same-side continuation            → ""
#   B0008[6.6,9.0]   transfer, FTO +0.1                → 1 (latched)
#   A0010[8.8,10.0]  transfer, FTO −0.2 (overlap)      → 0
#   B0012[10.5,11.0] "right" backchannel               → ""
#   A0014[11.5,12.0] continuation across the BC        → ""
#   B0016[12.4,13.0] transfer, FTO +0.4                → 0 (gap too long)
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

# sw2002: contained substantive interjection — not a transfer → ""
BODY_2002_A = (
    "sw2002A-ms98-a-0002 0.0 10.0 we always go up to the mountains in the summer\n"
)
BODY_2002_B = "sw2002B-ms98-a-0004 4.0 5.0 you do not say\n"

ALL_FILES = {
    "sw2001A-ms98-a-trans.text": BODY_2001_A,
    "sw2001B-ms98-a-trans.text": BODY_2001_B,
    "sw2002A-ms98-a-trans.text": BODY_2002_A,
    "sw2002B-ms98-a-trans.text": BODY_2002_B,
}


def _idx(tmp_path):
    return build_fto_index(_make_trans_root(tmp_path, ALL_FILES))


def test_latch_on_small_positive_fto(tmp_path):
    assert lookup_latching(_idx(tmp_path), "200/sw2001B-U0008.wav") == 1


def test_overlap_is_not_latch(tmp_path):
    assert lookup_latching(_idx(tmp_path), "200/sw2001A-U0010.wav") == 0


def test_long_gap_is_not_latch(tmp_path):
    assert lookup_latching(_idx(tmp_path), "200/sw2001B-U0016.wav") == 0


def test_first_turn_undefined(tmp_path):
    assert lookup_latching(_idx(tmp_path), "200/sw2001A-U0002.wav") is None


def test_backchannel_undefined(tmp_path):
    assert lookup_latching(_idx(tmp_path), "200/sw2001B-U0004.wav") is None


def test_continuation_undefined(tmp_path):
    assert lookup_latching(_idx(tmp_path), "200/sw2001A-U0006.wav") is None
    assert lookup_latching(_idx(tmp_path), "200/sw2001A-U0014.wav") is None


def test_interjection_undefined(tmp_path):
    assert lookup_latching(_idx(tmp_path), "200/sw2002B-U0004.wav") is None


def test_tau_boundary():
    # Direct synthetic FTO index entries either side of the threshold.
    # (Avoid exact-boundary float equality.)
    under = {(2001, "B", 1): (LATCH_MAX_SEC - 0.01, LATCH_MAX_SEC - 0.01, 1, 0, 0)}
    over = {(2001, "B", 1): (LATCH_MAX_SEC + 0.01, LATCH_MAX_SEC + 0.01, 1, 0, 0)}
    assert lookup_latching(under, "200/sw2001B-U0001.wav") == 1
    assert lookup_latching(over, "200/sw2001B-U0001.wav") == 0


def test_write_round_trip(tmp_path):
    root = _make_trans_root(tmp_path, ALL_FILES)
    out = tmp_path / "utterances_v2"
    mp = manifest_path(out)
    with open_appender(mp) as w:
        write_row(w, "200/sw2001A-U0002.wav", "")
        write_row(w, "200/sw2001B-U0008.wav", "")
        write_row(w, "200/sw2001B-U0012.wav", "")
        write_row(w, "200/sw2001B-U0016.wav", "")
    n = write_latching_flags(
        mp, out / "features" / "latching_flag.csv", transcript_root=root
    )
    assert n == 4
    rows = list(csv.reader((out / "features" / "latching_flag.csv").open()))
    assert tuple(rows[0]) == HEADER
    by_rel = {r[0]: r[1] for r in rows[1:]}
    assert by_rel["200/sw2001A-U0002.wav"] == ""  # first turn: no transfer
    assert by_rel["200/sw2001B-U0008.wav"] == "1"
    assert by_rel["200/sw2001B-U0012.wav"] == ""  # backchannel
    assert by_rel["200/sw2001B-U0016.wav"] == "0"
