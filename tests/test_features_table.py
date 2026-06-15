import csv
from pathlib import Path

import pytest

from swb_extract.features_table import TableBuildError, build_table
from swb_extract.manifest import manifest_path, open_appender, write_row


def _write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def _setup(tmp_path: Path):
    out = tmp_path / "utterances_v2"
    mp = manifest_path(out)
    with open_appender(mp) as w:
        write_row(w, "200/sw2001A-U0002.wav", "hello there")
        write_row(w, "200/sw2001B-U0004.wav", "uh-huh")
    fdir = out / "features"
    _write_csv(
        fdir / "alpha.csv",
        ["Utterance File Name", "Alpha"],
        [["200/sw2001A-U0002.wav", "1.5"], ["200/sw2001B-U0004.wav", ""]],
    )
    _write_csv(
        fdir / "beta.csv",
        ["Utterance File Name", "Beta One", "Beta Two"],
        [["200/sw2001A-U0002.wav", "0", "x"], ["200/sw2001B-U0004.wav", "1", "y"]],
    )
    return out, mp, fdir


def test_happy_path(tmp_path):
    out, mp, fdir = _setup(tmp_path)
    dst = out / "features_table.csv"
    n_rows, n_cols = build_table(mp, fdir, dst)
    assert (n_rows, n_cols) == (2, 5)
    rows = list(csv.reader(dst.open()))
    assert rows[0] == [
        "Utterance File Name", "Transcript", "Alpha", "Beta One", "Beta Two",
    ]
    assert rows[1] == ["200/sw2001A-U0002.wav", "hello there", "1.5", "0", "x"]
    assert rows[2] == ["200/sw2001B-U0004.wav", "uh-huh", "", "1", "y"]


def test_out_of_sync_key_raises(tmp_path):
    out, mp, fdir = _setup(tmp_path)
    _write_csv(
        fdir / "gamma.csv",
        ["Utterance File Name", "Gamma"],
        [["200/sw2001B-U0004.wav", "9"], ["200/sw2001A-U0002.wav", "8"]],  # reversed
    )
    with pytest.raises(TableBuildError, match="gamma.csv out of sync"):
        build_table(mp, fdir, out / "features_table.csv")


def test_missing_row_raises(tmp_path):
    out, mp, fdir = _setup(tmp_path)
    _write_csv(
        fdir / "gamma.csv",
        ["Utterance File Name", "Gamma"],
        [["200/sw2001A-U0002.wav", "9"]],  # second row missing
    )
    with pytest.raises(TableBuildError, match="gamma.csv out of sync"):
        build_table(mp, fdir, out / "features_table.csv")


def test_extra_row_raises(tmp_path):
    out, mp, fdir = _setup(tmp_path)
    _write_csv(
        fdir / "gamma.csv",
        ["Utterance File Name", "Gamma"],
        [
            ["200/sw2001A-U0002.wav", "9"],
            ["200/sw2001B-U0004.wav", "8"],
            ["200/sw2099A-U0001.wav", "7"],
        ],
    )
    with pytest.raises(TableBuildError, match="rows beyond the end"):
        build_table(mp, fdir, out / "features_table.csv")


def test_over_wide_row_raises(tmp_path):
    out, mp, fdir = _setup(tmp_path)
    # gamma declares ONE value column but a data row carries two → must error,
    # not silently truncate the stray cell.
    _write_csv(
        fdir / "gamma.csv",
        ["Utterance File Name", "Gamma"],
        [
            ["200/sw2001A-U0002.wav", "9", "stray"],
            ["200/sw2001B-U0004.wav", "8"],
        ],
    )
    with pytest.raises(TableBuildError, match="value cols, header declares"):
        build_table(mp, fdir, out / "features_table.csv")


def test_duplicate_column_raises(tmp_path):
    out, mp, fdir = _setup(tmp_path)
    _write_csv(
        fdir / "alpha2.csv",
        ["Utterance File Name", "Alpha"],  # collides with alpha.csv's column
        [["200/sw2001A-U0002.wav", "3"], ["200/sw2001B-U0004.wav", "4"]],
    )
    with pytest.raises(TableBuildError, match="duplicate column"):
        build_table(mp, fdir, out / "features_table.csv")
