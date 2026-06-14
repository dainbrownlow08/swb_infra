import csv
from pathlib import Path

import pytest

from swb_extract.features.overlap import (
    HEADER,
    compute_overlap,
    write_overlap,
)
from swb_extract.manifest import manifest_path, open_appender, write_row


def _ov(own, other):
    starts = [s for s, _ in other]
    return compute_overlap(own, other, starts)


def test_no_own_words_none():
    assert _ov([], [(0.0, 1.0)]) is None


def test_simple_overlap_duration():
    # own [0,2] vs other [1,3] → 1.0s overlap, 1 word, onset 0 (other starts at 1>0)
    dur, count, onset = _ov([(0.0, 2.0, "x")], [(1.0, 3.0)])
    assert dur == pytest.approx(1.0)
    assert count == 1
    assert onset == 0


def test_onset_flag_when_other_holds_floor():
    # own starts at 1, other already speaking [0,1.5] → onset 1
    dur, count, onset = _ov([(1.0, 2.0, "x")], [(0.0, 1.5)])
    assert dur == pytest.approx(0.5)
    assert count == 1
    assert onset == 1


def test_no_overlap():
    assert _ov([(0.0, 1.0, "x")], [(2.0, 3.0)]) == (0.0, 0, 0)


def test_counts_multiple_other_words():
    # own [0,5] spans two other words
    dur, count, onset = _ov([(0.0, 5.0, "x")], [(1.0, 2.0), (3.0, 4.0)])
    assert dur == pytest.approx(2.0)
    assert count == 2


def test_internal_silence_not_counted():
    # own has two words with a gap [0,1] and [4,5]; other speaks [2,3] in the gap
    # → no simultaneous speech (other talks during own's silence)
    dur, count, onset = _ov([(0.0, 1.0, "a"), (4.0, 5.0, "b")], [(2.0, 3.0)])
    assert dur == pytest.approx(0.0)
    assert count == 0


def _make_word_root(tmp_path: Path, files: dict[str, str]) -> Path:
    root = tmp_path / "swb_ms98_transcriptions_cleaned"
    for fname, body in files.items():
        call = int(fname[2:6])
        nn = call // 100
        d = root / f"{nn:02d}" / f"{call:04d}"
        d.mkdir(parents=True, exist_ok=True)
        (d / fname).write_text(body, encoding="utf-8")
    return root


def test_write_round_trip_cross_side(tmp_path):
    # A says one word [1,3]; B says [2,4] → they overlap [2,3]=1.0s
    word_a = "sw2001A-ms98-a-0002 1.0 3.0 hello\n"
    word_b = "sw2001B-ms98-a-0001 2.0 4.0 yeah\n"
    root = _make_word_root(
        tmp_path,
        {
            "sw2001A-ms98-a-word.text": word_a,
            "sw2001B-ms98-a-word.text": word_b,
        },
    )
    out = tmp_path / "utterances_v2"
    mp = manifest_path(out)
    with open_appender(mp) as w:
        write_row(w, "200/sw2001A-U0002.wav", "")
        write_row(w, "200/sw2001B-U0001.wav", "")
    write_overlap(mp, out / "features" / "overlap.csv", transcript_root=root)
    rows = list(csv.reader((out / "features" / "overlap.csv").open()))
    assert tuple(rows[0]) == HEADER
    by_rel = {r[0]: r[1:] for r in rows[1:]}
    # A's utterance overlapped by B for 1.0s
    assert float(by_rel["200/sw2001A-U0002.wav"][0]) == pytest.approx(1.0)
    # B starts at 2.0 while A is mid-word [1,3] → onset flag 1
    assert by_rel["200/sw2001B-U0001.wav"][2] == "1"
