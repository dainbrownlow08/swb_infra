import csv
from pathlib import Path

import pytest

from swb_extract.cli import main
from swb_extract.features.word_rate import (
    HEADER,
    compute_rate,
    count_words,
    run,
    tokenize,
    write_word_rates,
)
from swb_extract.manifest import (
    MANIFEST_HEADER,
    manifest_path,
    open_appender,
    write_row,
)

REPO = Path(__file__).resolve().parent.parent
TRANSCRIPT_ROOT = REPO / "swb_ms98_transcriptions_cleaned"


@pytest.fixture(scope="session")
def transcript_root():
    if not TRANSCRIPT_ROOT.is_dir():
        pytest.skip(f"missing transcript root: {TRANSCRIPT_ROOT}")
    return TRANSCRIPT_ROOT


def test_tokenize_strips_whole_bracket():
    assert tokenize("yes [laughter] um") == ["yes", "um"]


def test_tokenize_lowercases():
    assert tokenize("UM Yeah BAsically") == ["um", "yeah", "basically"]


def test_count_words_basic():
    assert count_words("hi um yeah") == 3


def test_count_words_empty():
    assert count_words("") == 0
    assert count_words("[laughter]") == 0


def test_compute_rate_basic():
    # 3 words / 1.5s = 2.0
    assert compute_rate("hi um yeah", 1.5) == 2.0


def test_compute_rate_strips_brackets():
    # tokens after strip: ['um','yeah'] (2). 2 / 1.0 = 2.0
    assert compute_rate("um [laughter] yeah", 1.0) == 2.0


def test_compute_rate_zero_or_none_duration_returns_none():
    assert compute_rate("oh", 0.0) is None
    assert compute_rate("oh", -1.0) is None
    assert compute_rate("oh", None) is None


def test_compute_rate_empty_text_returns_zero():
    assert compute_rate("", 5.0) == 0.0
    assert compute_rate("[laughter]", 5.0) == 0.0


def test_write_word_rates_round_trip(tmp_path, transcript_root):
    out = tmp_path / "utterances_v2"
    mp = manifest_path(out)
    with open_appender(mp) as w:
        write_row(w, "200/sw2001A-U0002.wav", "hi um yeah how are you")

    n = write_word_rates(
        mp,
        out / "features" / "word_rate.csv",
        transcript_root=transcript_root,
    )
    assert n == 1

    rows = list(csv.reader((out / "features" / "word_rate.csv").open()))
    assert tuple(rows[0]) == HEADER
    assert rows[1][0] == "200/sw2001A-U0002.wav"
    rate = float(rows[1][1])
    # conversational word rate: ~1-5 words/sec
    assert 0.0 < rate < 10.0


def test_join_contract_against_manifest(tmp_path, transcript_root):
    out = tmp_path / "utterances_v2"
    mp = manifest_path(out)
    with open_appender(mp) as w:
        write_row(w, "200/sw2001A-U0002.wav", "oh yeah")

    write_word_rates(
        mp, out / "features" / "word_rate.csv",
        transcript_root=transcript_root,
    )
    feat = list(csv.reader((out / "features" / "word_rate.csv").open()))
    manifest = list(csv.reader(mp.open()))
    assert feat[0] == list(HEADER)
    assert manifest[0] == list(MANIFEST_HEADER)
    assert len(feat) == len(manifest)
    for fr, mr in zip(feat[1:], manifest[1:]):
        assert fr[0] == mr[0]


def test_features_dispatch_routes_word_rate(tmp_path, transcript_root):
    out = tmp_path / "utterances_v2"
    mp = manifest_path(out)
    with open_appender(mp) as w:
        write_row(w, "200/sw2001A-U0002.wav", "yeah")

    rc = main([
        "features", "word_rate",
        "--out-root", str(out),
        "--transcript-root", str(transcript_root),
    ])
    assert rc == 0
    assert (out / "features" / "word_rate.csv").exists()
