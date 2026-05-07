import csv

import pytest

from swb_extract.cli import main
from swb_extract.features.token_count import (
    HEADER,
    count_tokens,
    run,
    tokenize,
    write_token_counts,
)
from swb_extract.manifest import (
    MANIFEST_HEADER,
    manifest_path,
    open_appender,
    write_row,
)


def test_tokenize_strips_whole_bracket():
    assert tokenize("yes [laughter] um") == ["yes", "um"]


def test_tokenize_lowercases():
    assert tokenize("UM Yeah BAsically") == ["um", "yeah", "basically"]


def test_count_tokens_basic():
    assert count_tokens("hi um yeah") == 3


def test_count_tokens_empty():
    assert count_tokens("") == 0


def test_count_tokens_brackets_only():
    assert count_tokens("[laughter] [noise]") == 0


def test_count_tokens_mixed():
    # tokens after strip: ['um', 'yeah']
    assert count_tokens("um [laughter] yeah") == 2


def test_write_token_counts_round_trip(tmp_path):
    out = tmp_path / "utterances_v2"
    mp = manifest_path(out)
    with open_appender(mp) as w:
        write_row(w, "200/sw2001A-U0002.wav", "um yeah um")
        write_row(w, "200/sw2001A-U0004.wav", "you know what i mean")
        write_row(w, "200/sw2001A-U0006.wav", "")

    out_csv = out / "features" / "token_count.csv"
    n = write_token_counts(mp, out_csv)
    assert n == 3

    rows = list(csv.reader(out_csv.open(encoding="utf-8")))
    assert tuple(rows[0]) == HEADER
    assert rows[1][0] == "200/sw2001A-U0002.wav"
    assert int(rows[1][1]) == 3
    assert int(rows[2][1]) == 5
    assert int(rows[3][1]) == 0


def test_join_contract_against_manifest(tmp_path):
    out = tmp_path / "utterances_v2"
    mp = manifest_path(out)
    with open_appender(mp) as w:
        write_row(w, "200/sw2001A-U0002.wav", "um")
        write_row(w, "200/sw2001B-U0003.wav", "yeah")
        write_row(w, "211/sw2113A-U0007.wav", "you know")

    feat_csv = out / "features" / "token_count.csv"
    write_token_counts(mp, feat_csv)

    manifest_rows = list(csv.reader(mp.open(encoding="utf-8")))
    feat_rows = list(csv.reader(feat_csv.open(encoding="utf-8")))
    assert manifest_rows[0] == list(MANIFEST_HEADER)
    assert feat_rows[0] == list(HEADER)
    assert len(manifest_rows) == len(feat_rows)
    for mrow, frow in zip(manifest_rows[1:], feat_rows[1:]):
        assert mrow[0] == frow[0]


def test_run_via_cli_args(tmp_path):
    out = tmp_path / "utterances_v2"
    mp = manifest_path(out)
    with open_appender(mp) as w:
        write_row(w, "200/sw2001A-U0002.wav", "um yeah")

    class A:
        pass
    a = A()
    a.out_root = str(out)
    rc = run(a)
    assert rc == 0
    assert (out / "features" / "token_count.csv").exists()


def test_features_dispatch_routes_token_count(tmp_path):
    out = tmp_path / "utterances_v2"
    mp = manifest_path(out)
    with open_appender(mp) as w:
        write_row(w, "200/sw2001A-U0002.wav", "um")
    rc = main(["features", "token_count", "--out-root", str(out)])
    assert rc == 0
    assert (out / "features" / "token_count.csv").exists()
