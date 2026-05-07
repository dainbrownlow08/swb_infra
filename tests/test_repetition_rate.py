import csv

import pytest

from swb_extract.cli import main
from swb_extract.features.repetition_rate import (
    HEADER,
    compute_rate,
    count_repetitions,
    tokenize,
    write_repetition_rates,
)
from swb_extract.manifest import (
    MANIFEST_HEADER,
    manifest_path,
    open_appender,
    write_row,
)


def test_tokenize_strips_whole_bracket():
    assert tokenize("yes [laughter] um") == ["yes", "um"]
    assert tokenize("[noise] right") == ["right"]


def test_tokenize_keeps_inline_brackets():
    # i[t]- is a partial-word marker, not a whole-bracket token
    assert tokenize("i[t]- yeah") == ["i[t]-", "yeah"]


def test_tokenize_lowercases():
    assert tokenize("YEAH the THE") == ["yeah", "the", "the"]


def test_count_repetitions_basic():
    assert count_repetitions(["the", "the"]) == 1
    assert count_repetitions(["the", "dog", "the", "dog"]) == 2


def test_count_repetitions_binary_per_word():
    # Legacy semantics: count increments only on the SECOND occurrence,
    # so 'the the the' is 1 repetition (not 2).
    assert count_repetitions(["the", "the", "the"]) == 1
    assert count_repetitions(["the", "the", "the", "the", "the"]) == 1


def test_count_repetitions_no_repeats():
    assert count_repetitions(["the", "dog", "barks"]) == 0
    assert count_repetitions([]) == 0


def test_compute_rate_well_what_club_what_club():
    # Legacy reference: rate = 0.25
    # Tokens: ['well','what','club','what','club','are','you','with'] = 8
    # Repeats: what, club → 2
    assert compute_rate("well what club what club are you with") == 0.25


def test_compute_rate_strips_brackets_no_fake_repetition():
    # Legacy bug: "[laughter] [laughter] right" would count [laughter] as a
    # repetition. Stripped, only "right" remains → 0/1 = 0.0
    assert compute_rate("[laughter] [laughter] right") == 0.0


def test_compute_rate_strips_brackets_doesnt_dilute():
    # Without strip: "the [laughter] the dog" → 4 tokens, 1 repeat = 0.25
    # With strip:    "the the dog"             → 3 tokens, 1 repeat = 0.333...
    assert compute_rate("the [laughter] the dog") == pytest.approx(1 / 3)


def test_compute_rate_empty_returns_zero():
    assert compute_rate("") == 0.0


def test_compute_rate_all_brackets_returns_zero():
    assert compute_rate("[laughter] [noise]") == 0.0


def test_compute_rate_single_token_returns_zero():
    assert compute_rate("yeah") == 0.0


def test_write_repetition_rates_round_trip(tmp_path):
    out = tmp_path / "utterances_v2"
    mp = manifest_path(out)
    with open_appender(mp) as w:
        write_row(w, "200/sw2001A-U0002.wav", "well what club what club are you with")
        write_row(w, "200/sw2001A-U0004.wav", "yeah")
        write_row(w, "200/sw2001A-U0006.wav", "")

    n = write_repetition_rates(mp, out / "features" / "repetition_rate.csv")
    assert n == 3

    rows = list(csv.reader((out / "features" / "repetition_rate.csv").open(encoding="utf-8")))
    assert tuple(rows[0]) == HEADER
    assert rows[1][0] == "200/sw2001A-U0002.wav"
    assert float(rows[1][1]) == pytest.approx(0.25)
    assert float(rows[2][1]) == 0.0
    assert float(rows[3][1]) == 0.0


def test_join_contract_against_manifest(tmp_path):
    out = tmp_path / "utterances_v2"
    mp = manifest_path(out)
    with open_appender(mp) as w:
        write_row(w, "200/sw2001A-U0002.wav", "the the dog")
        write_row(w, "200/sw2001B-U0003.wav", "[noise] yeah yeah")
        write_row(w, "211/sw2113A-U0007.wav", "ok")

    write_repetition_rates(mp, out / "features" / "repetition_rate.csv")

    feat = list(csv.reader((out / "features" / "repetition_rate.csv").open(encoding="utf-8")))
    manifest = list(csv.reader(mp.open(encoding="utf-8")))
    assert feat[0] == list(HEADER)
    assert manifest[0] == list(MANIFEST_HEADER)
    assert len(feat) == len(manifest)
    for fr, mr in zip(feat[1:], manifest[1:]):
        assert fr[0] == mr[0]


def test_features_dispatch_routes_repetition_rate(tmp_path):
    out = tmp_path / "utterances_v2"
    mp = manifest_path(out)
    with open_appender(mp) as w:
        write_row(w, "200/sw2001A-U0002.wav", "the the")
    rc = main(["features", "repetition_rate", "--out-root", str(out)])
    assert rc == 0
    assert (out / "features" / "repetition_rate.csv").exists()
