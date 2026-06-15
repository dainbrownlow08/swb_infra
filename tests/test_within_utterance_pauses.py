import csv
from pathlib import Path

import pytest

from swb_extract.features.within_utterance_pauses import (
    HEADER,
    PAUSE_MIN_SEC,
    compute_pauses,
    write_within_utterance_pauses,
)
from swb_extract.manifest import manifest_path, open_appender, write_row


def test_no_words_none():
    assert compute_pauses([]) is None


def test_single_word_zeros():
    assert compute_pauses([(0.0, 1.0, "yeah")]) == (0.0, 0, 0.0, 0.0)


def test_real_pause_counted():
    # gap 0.5 >= 0.25 → counted; span 2.0 → rate 0.25
    total, count, rate, mx = compute_pauses([(0.0, 1.0, "a"), (1.5, 2.0, "b")])
    assert total == pytest.approx(0.5)
    assert count == 1
    assert rate == pytest.approx(0.25)
    assert mx == pytest.approx(0.5)


def test_small_gap_not_counted():
    # gap 0.1 < 0.25 → not counted, but still in total
    total, count, rate, mx = compute_pauses([(0.0, 1.0, "a"), (1.1, 2.0, "b")])
    assert total == pytest.approx(0.1)
    assert count == 0
    assert mx == pytest.approx(0.1)


def test_overlapping_words_clamped():
    # negative gap (forced-alignment overlap) contributes 0
    assert compute_pauses([(0.0, 1.0, "a"), (0.9, 2.0, "b")]) == (0.0, 0, 0.0, 0.0)


def test_threshold_boundary():
    # gap exactly PAUSE_MIN_SEC counts
    total, count, _, _ = compute_pauses(
        [(0.0, 1.0, "a"), (1.0 + PAUSE_MIN_SEC, 2.0, "b")]
    )
    assert count == 1


def test_span_uses_max_end_not_last_word(tmp_path=None):
    # An early word ends LATE (overlapping alignment); the last-by-start word
    # ends earlier. Span must be max-end − first-start = 5.0, not 2.0, so the
    # 0.0-gap-but-late-end case can never push rate above 1.
    words = [(0.0, 5.0, "loooong"), (1.0, 2.0, "b")]  # sorted by start
    total, count, rate, mx = compute_pauses(words)
    assert total == pytest.approx(0.0)  # word[1].start 1.0 < word[0].end 5.0 → no gap
    assert rate == pytest.approx(0.0)
    assert rate <= 1.0


def test_rate_never_exceeds_one(tmp_path=None):
    # Pause present, but an early word ends past the last word's start.
    words = [(0.0, 4.0, "a"), (4.5, 4.6, "b")]  # gap 0.5; max end 4.6, span 4.6
    total, count, rate, mx = compute_pauses(words)
    assert total == pytest.approx(0.5)
    assert rate == pytest.approx(0.5 / 4.6)
    assert rate <= 1.0


def _make_word_root(tmp_path: Path, files: dict[str, str]) -> Path:
    root = tmp_path / "swb_ms98_transcriptions_cleaned"
    for fname, body in files.items():
        call = int(fname[2:6])
        nn = call // 100
        d = root / f"{nn:02d}" / f"{call:04d}"
        d.mkdir(parents=True, exist_ok=True)
        (d / fname).write_text(body, encoding="utf-8")
    return root


def test_write_round_trip(tmp_path):
    word_body = (
        "sw2001A-ms98-a-0002 0.0 1.0 hello\n"
        "sw2001A-ms98-a-0002 1.5 2.0 there\n"   # 0.5s pause
        "sw2001A-ms98-a-0004 5.0 5.4 yeah\n"    # single word → zeros
    )
    root = _make_word_root(tmp_path, {"sw2001A-ms98-a-word.text": word_body})
    out = tmp_path / "utterances_v2"
    mp = manifest_path(out)
    with open_appender(mp) as w:
        write_row(w, "200/sw2001A-U0002.wav", "")
        write_row(w, "200/sw2001A-U0004.wav", "")
        write_row(w, "200/sw2001A-U0099.wav", "")  # no word rows → empty cells
    n = write_within_utterance_pauses(
        mp, out / "features" / "within_utterance_pauses.csv", transcript_root=root
    )
    assert n == 3
    rows = list(csv.reader((out / "features" / "within_utterance_pauses.csv").open()))
    assert tuple(rows[0]) == HEADER
    by_rel = {r[0]: r[1:] for r in rows[1:]}
    assert by_rel["200/sw2001A-U0002.wav"][1] == "1"          # one pause
    assert by_rel["200/sw2001A-U0004.wav"] == ["0.0", "0", "0.0", "0.0"]
    assert by_rel["200/sw2001A-U0099.wav"] == ["", "", "", ""]
