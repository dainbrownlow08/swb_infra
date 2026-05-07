import csv
from pathlib import Path

import pytest

from swb_extract.features.syllable_rate import (
    HEADER,
    compute_rate,
    count_syllables,
    write_syllable_rates,
)
from swb_extract.manifest import (
    MANIFEST_HEADER,
    manifest_path,
    open_appender,
    write_row,
)

# textstat needs cmudict; make sure it's available before importing
pytest.importorskip("textstat")
from swb_extract.features.syllable_rate import _ensure_cmudict
_ensure_cmudict()

REPO = Path(__file__).resolve().parent.parent
TRANSCRIPT_ROOT = REPO / "swb_ms98_transcriptions_cleaned"


@pytest.fixture(scope="session")
def transcript_root():
    if not TRANSCRIPT_ROOT.is_dir():
        pytest.skip(f"missing transcript root: {TRANSCRIPT_ROOT}")
    return TRANSCRIPT_ROOT


def test_count_syllables_basic():
    # textstat: 'oh' = 1, 'huh' = 1
    assert count_syllables("oh") == 1
    assert count_syllables("huh") == 1


def test_count_syllables_empty():
    assert count_syllables("") == 0


def test_count_syllables_strips_whole_bracket():
    # "[laughter]" alone → after strip → "" → 0 (textstat would have said 2)
    assert count_syllables("[laughter]") == 0
    # "[noise] right" → after strip → "right" → 1 (textstat would have said 2)
    assert count_syllables("[noise] right") == 1


def test_count_syllables_keeps_inline_brackets():
    # i[t]- is a partial-word marker; doesn't start with [, kept verbatim
    assert count_syllables("i[t]-") >= 1  # textstat parses through the marker


def test_compute_rate_basic():
    # 'um yeah um' → textstat ≈ 3 syllables, duration 1.5s → rate = 2.0
    rate = compute_rate("um yeah um", 1.5)
    assert rate is not None
    assert 1.5 < rate < 2.5  # textstat may count slightly differently


def test_compute_rate_empty_duration_returns_none():
    assert compute_rate("oh", 0.0) is None
    assert compute_rate("oh", -1.0) is None
    assert compute_rate("oh", None) is None


def test_compute_rate_empty_text_returns_zero():
    assert compute_rate("", 5.0) == 0.0
    assert compute_rate("[laughter]", 5.0) == 0.0


def test_write_syllable_rates_round_trip(tmp_path, transcript_root):
    out = tmp_path / "utterances_v2"
    mp = manifest_path(out)
    with open_appender(mp) as w:
        # sw2001A-U0002: ms98 duration ≈ 10.58s, transcript has lots of words
        write_row(
            w,
            "200/sw2001A-U0002.wav",
            "hi um yeah i'd like to talk about how you dress for work and "
            "and um what do you normally what type of outfit do you "
            "normally have to wear",
        )

    n = write_syllable_rates(
        mp,
        out / "features" / "syllable_rate.csv",
        transcript_root=transcript_root,
    )
    assert n == 1

    rows = list(csv.reader((out / "features" / "syllable_rate.csv").open()))
    assert tuple(rows[0]) == HEADER
    assert rows[1][0] == "200/sw2001A-U0002.wav"
    rate = float(rows[1][1])
    # Conversational rate is ~3-5 syll/sec; this is a fast-spoken intro
    assert 2.0 < rate < 6.0, f"expected ~3-5 syll/sec, got {rate}"


def test_join_contract_against_manifest(tmp_path, transcript_root):
    out = tmp_path / "utterances_v2"
    mp = manifest_path(out)
    with open_appender(mp) as w:
        write_row(w, "200/sw2001A-U0002.wav", "oh yeah")

    write_syllable_rates(
        mp, out / "features" / "syllable_rate.csv",
        transcript_root=transcript_root,
    )
    feat = list(csv.reader((out / "features" / "syllable_rate.csv").open()))
    manifest = list(csv.reader(mp.open()))
    assert feat[0] == list(HEADER)
    assert manifest[0] == list(MANIFEST_HEADER)
    assert len(feat) == len(manifest)
    for fr, mr in zip(feat[1:], manifest[1:]):
        assert fr[0] == mr[0]


def test_features_dispatch_routes_syllable_rate(tmp_path, transcript_root):
    from swb_extract.cli import main

    out = tmp_path / "utterances_v2"
    mp = manifest_path(out)
    with open_appender(mp) as w:
        write_row(w, "200/sw2001A-U0002.wav", "yeah")

    rc = main([
        "features", "syllable_rate",
        "--out-root", str(out),
        "--transcript-root", str(transcript_root),
    ])
    assert rc == 0
    assert (out / "features" / "syllable_rate.csv").exists()
