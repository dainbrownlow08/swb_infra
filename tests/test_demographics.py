import csv
from pathlib import Path

import pytest

from swb_extract.features.demographics import (
    HEADER,
    Demographics,
    build_speaker_demos,
    derive_decade,
    derive_generation,
    load_caller_tab,
    load_conv_tab,
    normalize_region,
    normalize_sex,
    run,
    write_demographics,
)
from swb_extract.manifest import MANIFEST_HEADER, manifest_path, open_appender, write_row

REPO = Path(__file__).resolve().parent.parent
LDC_CALLER = REPO / "tables" / "caller_tab.csv"
LDC_CONV = REPO / "tables" / "conv_tab.csv"


@pytest.fixture(scope="session")
def ldc_tables():
    if not LDC_CALLER.is_file() or not LDC_CONV.is_file():
        pytest.skip("LDC tables missing")
    return LDC_CALLER, LDC_CONV


def test_derive_generation_cutoffs():
    assert derive_generation(1924) == "GI"
    assert derive_generation(1926) == "GI"
    assert derive_generation(1927) == "Silent"
    assert derive_generation(1944) == "Silent"
    assert derive_generation(1945) == "Baby_Boomer"
    assert derive_generation(1964) == "Baby_Boomer"
    assert derive_generation(1965) == "Generation_X"
    assert derive_generation(1975) == "Generation_X"


def test_derive_decade():
    assert derive_decade(1956) == "1950s"
    assert derive_decade(1937) == "1930s"
    assert derive_decade(1924) == "1920s"
    assert derive_decade(1970) == "1970s"


def test_normalize_sex():
    assert normalize_sex("MALE") == "male"
    assert normalize_sex("FEMALE") == "female"
    assert normalize_sex(" Male ") == "male"
    with pytest.raises(ValueError):
        normalize_sex("OTHER")


def test_normalize_region():
    assert normalize_region("NORTH MIDLAND") == "north_midland"
    assert normalize_region("NEW ENGLAND") == "new_england"
    assert normalize_region("NYC") == "nyc"
    assert normalize_region("UNK") == "unk"
    assert normalize_region("") == "unk"
    assert normalize_region("   ") == "unk"


def test_load_caller_tab(ldc_tables):
    caller, _ = ldc_tables
    callers = load_caller_tab(caller)
    # Speaker 1020 = call 2001 side A: female, NORTH MIDLAND, 1956, education=2
    assert callers[1020] == Demographics(
        gender="female",
        region="north_midland",
        year_born=1956,
        generation="Baby_Boomer",
        decade="1950s",
        education="2",
    )


def test_load_conv_tab(ldc_tables):
    _, conv = ldc_tables
    convs = load_conv_tab(conv)
    # Call 2001 → caller_from=1020, caller_to=1044
    assert convs[2001] == (1020, 1044)


def test_build_speaker_demos_for_sw2001(ldc_tables):
    caller, conv = ldc_tables
    demos = build_speaker_demos(caller, conv)
    a = demos[(2001, "A")]
    b = demos[(2001, "B")]
    assert a.gender == "female"
    assert a.region == "north_midland"
    assert a.year_born == 1956
    assert a.generation == "Baby_Boomer"
    # Different speaker on B
    assert b.year_born != a.year_born or b.region != a.region


def test_write_demographics_round_trip(tmp_path, ldc_tables):
    caller, conv = ldc_tables
    out = tmp_path / "utterances_v2"
    mp = manifest_path(out)
    with open_appender(mp) as w:
        write_row(w, "200/sw2001A-U0002.wav", "hi um yeah")
        write_row(w, "200/sw2001A-U0004.wav", "um-hum")
        write_row(w, "200/sw2001B-U0001.wav", "yeah")

    demos = build_speaker_demos(caller, conv)
    out_csv = out / "features" / "demographics.csv"
    n = write_demographics(mp, out_csv, demos)
    assert n == 3

    rows = list(csv.reader(out_csv.open(encoding="utf-8")))
    assert tuple(rows[0]) == HEADER
    assert len(rows) == 4
    assert rows[1][0] == "200/sw2001A-U0002.wav"
    assert rows[1][1] == "female"  # Gender


def test_join_contract_against_manifest(tmp_path, ldc_tables):
    """Demographics CSV must join cleanly to manifest by Utterance File Name."""
    caller, conv = ldc_tables
    out = tmp_path / "utterances_v2"
    mp = manifest_path(out)
    with open_appender(mp) as w:
        write_row(w, "200/sw2001A-U0002.wav", "x")
        write_row(w, "200/sw2001B-U0003.wav", "y")
        write_row(w, "211/sw2113A-U0007.wav", "z")

    demos = build_speaker_demos(caller, conv)
    feat_csv = out / "features" / "demographics.csv"
    write_demographics(mp, feat_csv, demos)

    manifest_rows = list(csv.reader(mp.open(encoding="utf-8")))
    feat_rows = list(csv.reader(feat_csv.open(encoding="utf-8")))

    assert manifest_rows[0] == list(MANIFEST_HEADER)
    assert feat_rows[0] == list(HEADER)
    # Same number of data rows, same key column, same order
    assert len(manifest_rows) == len(feat_rows)
    for mrow, frow in zip(manifest_rows[1:], feat_rows[1:]):
        assert mrow[0] == frow[0]


def test_run_via_cli_args(tmp_path, ldc_tables):
    """Smoke-test run() with a Namespace-like args object."""
    caller, conv = ldc_tables
    out = tmp_path / "utterances_v2"
    mp = manifest_path(out)
    with open_appender(mp) as w:
        write_row(w, "200/sw2001A-U0002.wav", "x")

    class A:
        pass
    a = A()
    a.caller_tab = str(caller)
    a.conv_tab = str(conv)
    a.out_root = str(out)
    rc = run(a)
    assert rc == 0
    assert (out / "features" / "demographics.csv").exists()
