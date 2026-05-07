import csv

from swb_extract.manifest import (
    MANIFEST_HEADER,
    already_done_calls,
    manifest_path,
    open_appender,
    parse_rel_path,
    write_row,
)


def test_writes_header_once_and_appends(tmp_path):
    mp = manifest_path(tmp_path)
    with open_appender(mp) as w:
        write_row(w, "200/sw2001A-U0002.wav", "hi um yeah")
        write_row(w, "200/sw2001A-U0004.wav", "um-hum")
    with open_appender(mp) as w:
        write_row(w, "200/sw2001A-U0006.wav", "and is")

    rows = list(csv.reader(mp.open(encoding="utf-8")))
    assert tuple(rows[0]) == MANIFEST_HEADER
    assert len(rows) == 4  # header + 3 data rows
    assert rows[1] == ["200/sw2001A-U0002.wav", "hi um yeah"]


def test_resume_detects_existing_calls(tmp_path):
    mp = manifest_path(tmp_path)
    with open_appender(mp) as w:
        write_row(w, "200/sw2001A-U0002.wav", "x")
        write_row(w, "200/sw2001B-U0001.wav", "y")
        write_row(w, "211/sw2113A-U0007.wav", "z")
    done = already_done_calls(mp)
    assert done == {(2001, "A"), (2001, "B"), (2113, "A")}


def test_already_done_empty_when_no_file(tmp_path):
    assert already_done_calls(tmp_path / "missing.csv") == set()


def test_parse_rel_path_roundtrip():
    assert parse_rel_path("200/sw2001A-U0002.wav") == (2001, "A", 2)
    assert parse_rel_path("493/sw4940B-U1234.wav") == (4940, "B", 1234)


def test_parse_rel_path_rejects_garbage():
    import pytest

    with pytest.raises(ValueError):
        parse_rel_path("foo/bar.wav")
