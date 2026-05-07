import csv

import pytest

from swb_extract.cli import main
from swb_extract.features.filler_word_rate import (
    DEFAULT_FILLERS,
    HEADER,
    compute_rate,
    count_filler_hits,
    run,
    tokenize,
    write_filler_rates,
)
from swb_extract.manifest import (
    MANIFEST_HEADER,
    manifest_path,
    open_appender,
    write_row,
)


def test_tokenize_strips_whole_bracket():
    assert tokenize("yes [laughter] um") == ["yes", "um"]


def test_tokenize_keeps_inline_brackets():
    # i[t]- is a partial-word marker, not a whole-bracket token
    assert tokenize("i[t]- yeah") == ["i[t]-", "yeah"]


def test_tokenize_lowercases():
    assert tokenize("UM Yeah BAsically") == ["um", "yeah", "basically"]


def test_count_single_word_fillers():
    assert count_filler_hits(["um", "yeah", "um"], DEFAULT_FILLERS) == 2


def test_count_phrase_filler():
    assert count_filler_hits(["yeah", "you", "know", "that"], DEFAULT_FILLERS) == 1


def test_count_overlapping_phrases_greedy():
    # 'i mean' (2) + 'i guess' (2) + 'so' (1) = 3 hits
    assert count_filler_hits(
        ["i", "mean", "i", "guess", "so"], DEFAULT_FILLERS
    ) == 3


def test_count_does_not_double_count():
    assert count_filler_hits(
        ["you", "know", "you", "know"], DEFAULT_FILLERS
    ) == 2


def test_count_empty_list():
    assert count_filler_hits([], DEFAULT_FILLERS) == 0


def test_compute_rate_empty_transcript():
    assert compute_rate("") == 0.0
    assert compute_rate("[noise] [laughter]") == 0.0


def test_compute_rate_basic():
    # "yeah you know i mean um like that"
    # tokens: ['yeah','you','know','i','mean','um','like','that']  (8)
    # hits: you know (1) + i mean (1) + um (1) + like (1) = 4
    assert compute_rate("yeah you know i mean um like that") == 4 / 8


def test_compute_rate_strips_brackets():
    # tokens after strip: ['um','yeah'] (2). hits: um (1). rate = 0.5
    assert compute_rate("um [laughter] yeah") == 0.5


def test_phrase_match_after_bracket_strip_collapses_adjacency():
    # Documented behavior: stripping removes positionally; 'you' and 'know'
    # become adjacent so 'you know' matches.
    assert compute_rate("you [noise] know") == 1 / 2


def test_write_filler_rates_round_trip(tmp_path):
    out = tmp_path / "utterances_v2"
    mp = manifest_path(out)
    with open_appender(mp) as w:
        write_row(w, "200/sw2001A-U0002.wav", "um yeah um")
        write_row(w, "200/sw2001A-U0004.wav", "you know what i mean")
        write_row(w, "200/sw2001A-U0006.wav", "")

    out_csv = out / "features" / "filler_word_rate.csv"
    n = write_filler_rates(mp, out_csv)
    assert n == 3

    rows = list(csv.reader(out_csv.open(encoding="utf-8")))
    assert tuple(rows[0]) == HEADER
    assert rows[1][0] == "200/sw2001A-U0002.wav"
    # 'um yeah um' → 2 hits / 3 tokens
    assert float(rows[1][1]) == pytest.approx(2 / 3)
    # 'you know what i mean' → 'you know' (1) + 'i mean' (1) = 2 / 5
    assert float(rows[2][1]) == pytest.approx(2 / 5)
    # empty → 0.0
    assert float(rows[3][1]) == 0.0


def test_join_contract_against_manifest(tmp_path):
    out = tmp_path / "utterances_v2"
    mp = manifest_path(out)
    with open_appender(mp) as w:
        write_row(w, "200/sw2001A-U0002.wav", "um")
        write_row(w, "200/sw2001B-U0003.wav", "yeah")
        write_row(w, "211/sw2113A-U0007.wav", "you know")

    feat_csv = out / "features" / "filler_word_rate.csv"
    write_filler_rates(mp, feat_csv)

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
    assert (out / "features" / "filler_word_rate.csv").exists()


def test_features_dispatch_routes_filler_word_rate(tmp_path):
    out = tmp_path / "utterances_v2"
    mp = manifest_path(out)
    with open_appender(mp) as w:
        write_row(w, "200/sw2001A-U0002.wav", "um")
    rc = main(["features", "filler_word_rate", "--out-root", str(out)])
    assert rc == 0
    assert (out / "features" / "filler_word_rate.csv").exists()


def test_unknown_feature_still_raises(tmp_path):
    with pytest.raises(NotImplementedError):
        main(["features", "myfeat", "--out-root", str(tmp_path)])
