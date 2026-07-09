import csv
from pathlib import Path

import pytest

from swb_extract.features.repetitions_in_current import (
    HEADER,
    compute,
    count_pair_repetitions,
    write_repetitions_in_current,
)
from swb_extract.manifest import (
    MANIFEST_HEADER,
    manifest_path,
    open_appender,
    write_row,
)


def test_count_pair_repetitions_basic():
    # Three "the"s → C(3, 2) = 3 pairs
    assert count_pair_repetitions(["the", "the", "the"]) == 3
    # Two "the"s → C(2, 2) = 1 pair
    assert count_pair_repetitions(["the", "cat", "the"]) == 1
    # No repeats
    assert count_pair_repetitions(["a", "b", "c"]) == 0
    # Empty
    assert count_pair_repetitions([]) == 0


def test_count_pair_repetitions_multiple_words():
    # two "the"s (1 pair) + two "cat"s (1 pair) = 2
    assert count_pair_repetitions(["the", "cat", "the", "cat"]) == 2
    # three "x" + two "y" = C(3,2) + C(2,2) = 3 + 1 = 4
    assert count_pair_repetitions(["x", "y", "x", "y", "x"]) == 4


def test_pair_count_quadratic_tail_exceeds_token_count():
    # Boundary pin (AUDIT.md C1 Tier 1): the pair count is quadratic in massed
    # repeats — C(n, 2) — so the column can legitimately exceed token_count.
    # 15 repeats of one word = 105 pairs from 15 tokens, which is why the
    # observed max 109 > max token_count 81 is BY DESIGN, not a counting bug.
    assert count_pair_repetitions(["uh"] * 15) == 105
    assert compute(" ".join(["uh"] * 15)) == 105


def test_compute_strips_whole_brackets():
    # Two [noise] tokens would otherwise become a fake repetition; they're stripped.
    assert compute("[noise] the the") == 1
    assert compute("[laughter] [laughter]") == 0
    assert compute("[noise] [noise]") == 0


def test_compute_lowercases():
    # "The" and "the" are the same word after lower-casing.
    assert compute("The the The") == 3


def test_compute_keeps_inline_partial_brackets():
    # Partial-word markers like "i[t]-" don't start with [ so they're kept as-is.
    # Two of them in a row would count as a repetition.
    assert compute("i[t]- i[t]-") == 1


def test_compute_empty_text():
    assert compute("") == 0
    assert compute("   ") == 0
    assert compute("[noise] [laughter]") == 0


def test_write_round_trip(tmp_path):
    out = tmp_path / "utterances_v2"
    mp = manifest_path(out)
    with open_appender(mp) as w:
        write_row(w, "200/sw2001A-U0002.wav", "the the cat the dog")
        write_row(w, "200/sw2001A-U0004.wav", "[noise] one two three")

    n = write_repetitions_in_current(
        mp,
        out / "features" / "repetitions_in_current.csv",
    )
    assert n == 2

    rows = list(csv.reader((out / "features" / "repetitions_in_current.csv").open()))
    assert tuple(rows[0]) == HEADER
    assert rows[1] == ["200/sw2001A-U0002.wav", "3"]  # C(3,2) for "the"
    assert rows[2] == ["200/sw2001A-U0004.wav", "0"]  # all unique


def test_join_contract_against_manifest(tmp_path):
    out = tmp_path / "utterances_v2"
    mp = manifest_path(out)
    with open_appender(mp) as w:
        write_row(w, "200/sw2001A-U0002.wav", "yeah yeah")

    write_repetitions_in_current(
        mp, out / "features" / "repetitions_in_current.csv",
    )
    feat = list(csv.reader((out / "features" / "repetitions_in_current.csv").open()))
    manifest = list(csv.reader(mp.open()))
    assert feat[0] == list(HEADER)
    assert manifest[0] == list(MANIFEST_HEADER)
    assert len(feat) == len(manifest)
    for fr, mr in zip(feat[1:], manifest[1:]):
        assert fr[0] == mr[0]


def test_features_dispatch_routes_repetitions_in_current(tmp_path):
    from swb_extract.cli import main

    out = tmp_path / "utterances_v2"
    mp = manifest_path(out)
    with open_appender(mp) as w:
        write_row(w, "200/sw2001A-U0002.wav", "yeah")

    rc = main([
        "features", "repetitions_in_current",
        "--out-root", str(out),
    ])
    assert rc == 0
    assert (out / "features" / "repetitions_in_current.csv").exists()
