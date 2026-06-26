import csv

from swb_extract.features.laughter import (
    HEADER,
    count_bracket_events,
    write_laughter_counts,
)
from swb_extract.manifest import manifest_path, open_appender, write_row


def test_standalone_laughter():
    assert count_bracket_events("[laughter] yeah [laughter]") == (2, 0, 0, 0, 0)


def test_laughed_words():
    assert count_bracket_events("[laughter-yeah] [laughter-oh] sure") == (0, 2, 0, 0, 0)


def test_noise_and_vocalized():
    assert count_bracket_events("[noise] well [vocalized-noise]") == (0, 0, 1, 1, 0)


def test_pronunciation_variant_is_other():
    assert count_bracket_events("the [tranged/changed] world") == (0, 0, 0, 0, 1)


def test_partial_words_not_counted():
    # i[t]- and beca[use]- don't start with '[' — they are speech, not events
    assert count_bracket_events("i[t]- was beca[use]- of that") == (0, 0, 0, 0, 0)


def test_plain_text_zero():
    assert count_bracket_events("just ordinary talk here") == (0, 0, 0, 0, 0)
    assert count_bracket_events("") == (0, 0, 0, 0, 0)


def test_mixed():
    assert count_bracket_events(
        "[laughter] i know [laughter-it] [noise] right [vocalized-noise]"
    ) == (1, 1, 1, 1, 0)


def test_write_round_trip(tmp_path):
    out = tmp_path / "utterances_v2"
    mp = manifest_path(out)
    with open_appender(mp) as w:
        write_row(w, "200/sw2001A-U0001.wav", "[laughter] oh that is funny")
        write_row(w, "200/sw2001A-U0002.wav", "[laughter-yeah] [laughter-really]")
        write_row(w, "200/sw2001A-U0003.wav", "no brackets here")
    n = write_laughter_counts(mp, out / "features" / "laughter.csv")
    assert n == 3
    rows = list(csv.reader((out / "features" / "laughter.csv").open()))
    assert tuple(rows[0]) == HEADER
    assert rows[1][1:] == ["1", "0", "0", "0", "0"]
    assert rows[2][1:] == ["0", "2", "0", "0", "0"]
    assert rows[3][1:] == ["0", "0", "0", "0", "0"]
